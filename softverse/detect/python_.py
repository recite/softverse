"""Python import detection via the standard library's own parser.

``ast`` is not a reimplementation of Python's grammar, it *is* Python's grammar,
so for the syntactic question -- does this file import X -- it is ground truth
rather than a peer. That makes Python the strongest of the three languages for
the validation story: precision and recall on extraction are exactly 1 by
construction, and what remains uncertain is only the construct question (is a
static import evidence of use).

v1 also used ``ast``, and still got four things wrong. Each is fixed here:

1. **Relative imports became fake packages.** ``node.level`` was never checked,
   so ``from .helpers import f`` emitted ``helpers``. The v1 tally is full of
   these: ``utils.plots``, ``odds_helper``, ``plot_common``, ``src.external.*``.
2. **Submodules were double counted.** Both the top level and the full dotted
   path were emitted, so ``sklearn`` and ``sklearn.model_selection`` were two
   separate "packages", inflating the unique-package count and the rankings.
3. **A SyntaxError silently produced zero imports** for the whole file. Every
   Python 2 script in the corpus therefore contributed nothing, invisibly. They
   tokenize fine even though they do not parse, so a token-stream fallback
   recovers them -- and because it works on tokens, strings and comments are
   still excluded structurally. It is not a regex of last resort.
4. **stdlib was decided by the analyst's machine** via ``find_spec``. Fixed in
   the resolver, which uses ``sys.stdlib_module_names``.
"""

from __future__ import annotations

import ast
import io
import tokenize

from softverse.detect.types import ExtractResult, Mention, ParseReport
from softverse.model.enums import Construct, Language, ParseStatus

#: Calls that import by name at runtime.
_DYNAMIC_IMPORTERS = {"import_module", "__import__"}


def _snippet(lines: list[str], line: int) -> str:
    return lines[line - 1].strip()[:300] if 0 < line <= len(lines) else ""


def _conditional(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether the import sits inside a conditional or exception handler."""
    current = parents.get(node)
    depth = 0
    while current is not None and depth < 8:
        if isinstance(current, ast.If | ast.Try | ast.ExceptHandler):
            return True
        current = parents.get(current)
        depth += 1
    return False


def _rebound_names(tree: ast.Module) -> set[str]:
    """Names the file assigns to, which therefore cannot be trusted as aliases.

    `import numpy as np` followed anywhere by `np = load(...)` means `np.foo`
    is no longer numpy, and nothing in the source says at which line the
    meaning changed. Dropping the alias entirely is the conservative reading:
    it loses the calls that really were numpy rather than inventing calls that
    were not.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.For | ast.AsyncFor):
            targets = [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets = [node.optional_vars]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            bound.add(node.name)
            arguments = node.args
            bound.update(
                a.arg
                for a in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    arguments.vararg,
                    arguments.kwarg,
                ]
                if a is not None
            )
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    bound.add(sub.id)
    return bound


def _alias_map(tree: ast.Module) -> dict[str, str]:
    """Local name -> top-level package, from this file's own `import`s.

    Only `import x` and `import x as y`. `from pandas import read_csv` binds a
    function rather than a module, and its call site is a bare `read_csv(f)`
    already counted at the import; adding the call would count one use twice.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            package = alias.name.split(".")[0]
            # `import os.path` with no `as` binds `os`, not `os.path`.
            aliases[alias.asname or package] = package
    for name in _rebound_names(tree):
        aliases.pop(name, None)
    return aliases


def _attribute_path(node: ast.expr) -> tuple[str, list[str]] | None:
    """`np.linalg.norm` -> ("np", ["linalg", "norm"]). None if not rooted."""
    attributes: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        attributes.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or not attributes:
        return None
    return current.id, list(reversed(attributes))


def _walk_ast(tree: ast.Module, lines: list[str]) -> list[Mention]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    aliases = _alias_map(tree)
    mentions: list[Mention] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mentions.append(
                    Mention(
                        # Top-level module only. Emitting the dotted path too is
                        # what made sklearn and sklearn.model_selection separate
                        # "packages" in v1.
                        raw_name=alias.name.split(".")[0],
                        construct=Construct.IMPORT,
                        line=node.lineno,
                        col=node.col_offset,
                        byte_start=0,
                        byte_end=0,
                        snippet=_snippet(lines, node.lineno),
                        is_conditional=_conditional(node, parents),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # `from . import x` / `from .helpers import f`. A local module,
                # never a distribution. v1 ignored node.level and invented a
                # package for every one of these.
                mentions.append(
                    Mention(
                        raw_name=node.module or ".",
                        construct=Construct.LOCAL_RELATIVE,
                        line=node.lineno,
                        col=node.col_offset,
                        byte_start=0,
                        byte_end=0,
                        snippet=_snippet(lines, node.lineno),
                        is_conditional=_conditional(node, parents),
                    )
                )
            elif node.module:
                # One mention per imported name, because `from pandas import
                # read_csv, DataFrame` is two things the code uses and a fifth
                # of these statements import more than one name. The package
                # repeats across them; deposit and file counts collapse it, so
                # only the raw mention count sees the difference.
                for alias in node.names or [None]:
                    mentions.append(
                        Mention(
                            raw_name=node.module.split(".")[0],
                            construct=Construct.IMPORT_FROM,
                            line=node.lineno,
                            col=node.col_offset,
                            byte_start=0,
                            byte_end=0,
                            snippet=_snippet(lines, node.lineno),
                            is_conditional=_conditional(node, parents),
                            # `from x import *` has the name `*`, which names
                            # no function, so it is recorded as no function.
                            called_function=(
                                alias.name
                                if alias is not None and alias.name != "*"
                                else None
                            ),
                        )
                    )
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if (
                isinstance(func, ast.Attribute)
                and name not in _DYNAMIC_IMPORTERS
                and (path := _attribute_path(func)) is not None
                and path[0] in aliases
            ):
                # `pd.read_csv(f)` after `import pandas as pd`. The same thing
                # R's `dplyr::select` is: the source naming the package at the
                # call site rather than only at the load. Calls on objects,
                # `df.head()` and `self.fit()`, have no root in the alias map
                # and produce nothing, which is the behaviour that keeps this
                # from becoming a list of every method in the file.
                mentions.append(
                    Mention(
                        raw_name=aliases[path[0]],
                        construct=Construct.ATTRIBUTE_CALL,
                        line=node.lineno,
                        col=node.col_offset,
                        byte_start=0,
                        byte_end=0,
                        snippet=_snippet(lines, node.lineno),
                        called_function=".".join(path[1]),
                        is_conditional=_conditional(node, parents),
                    )
                )
            if name in _DYNAMIC_IMPORTERS and node.args:
                first = node.args[0]
                literal = (
                    first.value
                    if isinstance(first, ast.Constant) and isinstance(first.value, str)
                    else None
                )
                mentions.append(
                    Mention(
                        raw_name=(literal or "<dynamic>").split(".")[0],
                        construct=Construct.DYNAMIC_IMPORT,
                        line=node.lineno,
                        col=node.col_offset,
                        byte_start=0,
                        byte_end=0,
                        snippet=_snippet(lines, node.lineno),
                        # A non-literal argument is unknowable statically.
                        is_dynamic=literal is None,
                    )
                )
    return mentions


def _tokenize_fallback(source: str, lines: list[str]) -> list[Mention]:
    """Recover imports from a file that does not parse.

    Python 2 scripts (``print "x"``) tokenize cleanly even though ``ast`` will
    not accept them. Because this reads the token stream rather than raw text,
    strings and comments are still excluded structurally -- an import inside a
    docstring produces STRING tokens, never NAME 'import'.
    """
    mentions: list[Mention] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return mentions

    i = 0
    while i < len(tokens):
        token = tokens[i]
        at_line_start = token.start[1] == 0 or (
            i > 0
            and tokens[i - 1].type
            in {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT}
        )
        if token.type == tokenize.NAME and at_line_start:
            if token.string == "import":
                for name, line in _names_after(tokens, i + 1):
                    mentions.append(
                        Mention(
                            raw_name=name.split(".")[0],
                            construct=Construct.IMPORT,
                            line=line,
                            col=0,
                            byte_start=0,
                            byte_end=0,
                            snippet=_snippet(lines, line),
                        )
                    )
            elif token.string == "from":
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt and nxt.type == tokenize.NAME:
                    mentions.append(
                        Mention(
                            raw_name=nxt.string.split(".")[0],
                            construct=Construct.IMPORT_FROM,
                            line=nxt.start[0],
                            col=0,
                            byte_start=0,
                            byte_end=0,
                            snippet=_snippet(lines, nxt.start[0]),
                        )
                    )
                elif nxt and nxt.type == tokenize.OP and nxt.string == ".":
                    mentions.append(
                        Mention(
                            raw_name=".",
                            construct=Construct.LOCAL_RELATIVE,
                            line=nxt.start[0],
                            col=0,
                            byte_start=0,
                            byte_end=0,
                            snippet=_snippet(lines, nxt.start[0]),
                        )
                    )
        i += 1
    return mentions


def _names_after(tokens: list[tokenize.TokenInfo], start: int) -> list[tuple[str, int]]:
    """Dotted names in an ``import a.b, c as d`` clause."""
    out: list[tuple[str, int]] = []
    current: list[str] = []
    line = tokens[start].start[0] if start < len(tokens) else 0
    i = start
    while i < len(tokens):
        token = tokens[i]
        if token.type in {tokenize.NEWLINE, tokenize.NL} or (
            token.type == tokenize.OP and token.string == ";"
        ):
            break
        if token.type == tokenize.NAME:
            if token.string == "as":
                i += 2
                continue
            current.append(token.string)
        elif token.type == tokenize.OP and token.string == ".":
            current.append(".")
        elif token.type == tokenize.OP and token.string == ",":
            if current:
                out.append(("".join(current), line))
            current = []
        i += 1
    if current:
        out.append(("".join(current), line))
    return [(n, line) for n, line in out if n and n[0].isalpha() or n.startswith("_")]


def extract(source: str | bytes, filename: str = "<source>") -> ExtractResult:
    """Extract Python import mentions."""
    text = source.decode("utf-8", "replace") if isinstance(source, bytes) else source
    lines = text.splitlines()

    try:
        tree = ast.parse(text, filename=filename)
    except SyntaxError as exc:
        mentions = _tokenize_fallback(text, lines)
        return ExtractResult(
            mentions=mentions,
            report=ParseReport(
                # Not "zero imports". The distinction is the whole point: v1
                # made a Python 2 file and a file with no imports identical.
                status=ParseStatus.FALLBACK if mentions else ParseStatus.SYNTAX_ERROR,
                language=Language.PYTHON,
                bytes_total=len(text),
                detail=f"SyntaxError: {exc.msg} (line {exc.lineno})",
            ),
        )
    except ValueError as exc:  # e.g. source containing null bytes
        return ExtractResult(
            mentions=[],
            report=ParseReport(
                status=ParseStatus.DECODE_ERROR,
                language=Language.PYTHON,
                bytes_total=len(text),
                detail=str(exc),
            ),
        )

    mentions = _walk_ast(tree, lines)
    mentions.sort(key=lambda m: (m.line, m.raw_name))
    return ExtractResult(
        mentions=mentions,
        report=ParseReport(
            status=ParseStatus.OK, language=Language.PYTHON, bytes_total=len(text)
        ),
    )

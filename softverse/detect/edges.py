"""Which file a script pulls in, so a deposit's scripts form a graph.

The paper's central validity threat is that a static reference is not a run: a
deposit can load a package in a superseded script nobody executes. Bounding
that needs the deposit's call graph -- which scripts a master script reaches
through ``source()``, ``do``, ``run``, ``include`` or a local import -- so the
tally can be recomputed over reachable files alone and the two compared.

The edges are read with each language's own parser rather than by regex, for
the reason :mod:`softverse.detect.r` gives: ``cat("do setup.do")`` is string
content, not a command, and a scanner that cannot tell those apart would
inflate the graph exactly where deposits talk about their own code in comments.

Only literal paths are recoverable. ``do "`path'/clean.do"`` names a macro
whose value is unknown statically, and those are counted (`unresolved`) rather
than guessed at, because a graph that silently drops edges makes files look
unreachable when the truth is that we could not follow the edge.
"""

from __future__ import annotations

import ast
import posixpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tree_sitter_language_pack import get_parser

from softverse.model.enums import Language
from softverse.stata.lexer import lex

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tree_sitter import Node

#: R calls whose first argument names a file to run.
_R_SOURCERS = {"source", "sys.source", "sourceCpp", "knit", "render", "spin"}

#: Stata commands that run another file. `do` and `run` differ only in whether
#: the lines echo; `include` splices the text in.
_STATA_SOURCERS = {"do", "run", "include", "doedit"}

#: A macro reference, `$global` or `` `local' ``: an edge we cannot follow.
_STATA_MACRO = re.compile(r"[$`]")

#: IPython's `%run script.py`, which is how notebooks chain.
_MAGIC_RUN = re.compile(r"^\s*[%!]\s*run\s+(?:-\S+\s+)*(\S+)", re.MULTILINE)

#: Any line magic or shell escape, which `ast` cannot read.
_MAGIC = re.compile(r"^\s*[%!].*$", re.MULTILINE)

#: The extension each language's files usually carry, for targets written
#: without one: `do clean` runs `clean.do`.
_DEFAULT_EXTENSIONS = {
    Language.R: (".R", ".r"),
    Language.STATA: (".do", ".ado"),
    Language.PYTHON: (".py",),
    Language.RMARKDOWN: (".Rmd", ".rmd"),
}


@dataclass(frozen=True)
class Edges:
    """Include targets found in one file, followable and not.

    Attributes:
        targets: Literal paths the file runs, as written. A target that
            matches no file in the deposit is a hole in the graph.
        imports: Names that *may* be a file in the deposit -- a Python import,
            which names a local module or an installed package and looks the
            same either way. One that matches nothing is an installed package,
            not a hole.
        n_unresolved: Targets naming a macro or a computed string, which cannot
            be followed statically and are counted instead of guessed.
    """

    targets: list[str]
    imports: list[str] = field(default_factory=list)
    n_unresolved: int = 0


def includes(text: str, language: Language) -> Edges:
    """Files ``text`` runs, by the rules of ``language``.

    Args:
        text: The decoded source.
        language: Which grammar to read it with.

    Returns:
        The literal targets, and how many edges named something dynamic.
    """
    if language is Language.R or language is Language.RMARKDOWN:
        return _r_includes(text)
    if language is Language.STATA:
        return _stata_includes(text)
    if language is Language.PYTHON:
        return _python_includes(text)
    return Edges([])


def _r_includes(text: str) -> Edges:
    data = text.encode("utf-8")
    tree = get_parser("r").parse(data)
    targets: list[str] = []
    unresolved = 0
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in {"ERROR", "MISSING"}:
            continue
        if node.type == "call" and _r_callee(node, data) in _R_SOURCERS:
            value = _r_file_argument(node, data)
            literal = _r_string(value, data) if value is not None else None
            if literal:
                targets.append(literal)
            elif value is not None:
                # `source(paste0(dir, "clean.R"))`: an edge that exists and
                # cannot be followed.
                unresolved += 1
        stack.extend(node.children)
    return Edges(targets, n_unresolved=unresolved)


def _r_callee(call: Node, source: bytes) -> str | None:
    """Function name of a call, `pkg::fn` reduced to `fn`."""
    fn = call.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "namespace_operator":
        fn = fn.child_by_field_name("rhs")
        if fn is None:
            return None
    if fn.type != "identifier":
        return None
    return source[fn.start_byte : fn.end_byte].decode("utf-8", "replace")


def _r_file_argument(call: Node, source: bytes) -> Node | None:
    """The `file =` argument, or the first positional one."""
    arguments = call.child_by_field_name("arguments")
    if arguments is None:
        return None
    positional = None
    for argument in (c for c in arguments.children if c.type == "argument"):
        value = argument.child_by_field_name("value")
        if value is None:
            continue
        name = argument.child_by_field_name("name")
        if name is not None:
            if source[name.start_byte : name.end_byte] == b"file":
                return value
        elif positional is None:
            positional = value
    return positional


def _r_string(node: Node, source: bytes) -> str | None:
    """Literal text of a string node, or None for anything computed."""
    if node.type != "string":
        return None
    content = node.child_by_field_name("content")
    if content is None:
        return ""
    return source[content.start_byte : content.end_byte].decode("utf-8", "replace")


def _stata_includes(text: str) -> Edges:
    targets: list[str] = []
    unresolved = 0
    for statement in lex(text):
        if statement.in_mata or not statement.command:
            continue
        if statement.command.lower() not in _STATA_SOURCERS:
            continue
        operand = next((t for t in statement.operands if t not in {",", ";"}), None)
        if operand is None:
            continue
        literal = operand.strip("\"'")
        if _STATA_MACRO.search(literal):
            unresolved += 1
        elif literal:
            targets.append(literal)
    return Edges(targets, n_unresolved=unresolved)


def _python_includes(text: str) -> Edges:
    targets = [match.group(1).strip("\"'") for match in _MAGIC_RUN.finditer(text)]
    imports: list[str] = []
    unresolved = 0
    try:
        # Notebook magics are not Python and would make the whole file
        # unparseable, costing every import in a file that uses one.
        tree = ast.parse(_MAGIC.sub("pass", text))
    except SyntaxError:
        return Edges(targets, imports, n_unresolved=unresolved)
    for node in ast.walk(tree):
        # A relative import names a module in the deposit: `from . import clean`
        # and `import clean` both point at `clean.py` next to the importer. The
        # resolver checks whether such a file exists, so a name that is really
        # an installed package costs nothing here.
        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.replace(".", "/"))
            else:
                imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name.replace(".", "/") for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "exec"
        ):
            unresolved += 1
    return Edges(targets, imports, n_unresolved=unresolved)


def resolve(target: str, from_path: str, paths: Iterable[str]) -> str | None:
    """Which of a deposit's files ``target`` names, read from ``from_path``.

    Two passes, in order of confidence: the path as written, relative to the
    including file and then to the deposit root; failing that, a unique match
    on the basename, because deposits are routinely rearranged after the code
    was written and `source("code/clean.R")` then sits beside `clean.R`. A
    basename matching two files resolves to neither.

    Args:
        target: The literal path as the script wrote it.
        from_path: Deposit-relative path of the file that runs it.
        paths: Every deposit-relative path in the deposit.

    Returns:
        The matching deposit-relative path, or None.
    """
    by_lower = {p.lower(): p for p in paths}
    cleaned = target.replace("\\", "/").strip().lstrip("./")
    if not cleaned:
        return None
    directory = posixpath.dirname(from_path)
    candidates: list[str] = []
    for base in (posixpath.join(directory, cleaned) if directory else cleaned, cleaned):
        normalized = posixpath.normpath(base).lstrip("/")
        candidates.append(normalized)
        candidates.extend(
            normalized + extension
            for extensions in _DEFAULT_EXTENSIONS.values()
            for extension in extensions
        )
    for candidate in candidates:
        if hit := by_lower.get(candidate.lower()):
            return hit

    stem = posixpath.basename(cleaned).lower()
    stems = [
        p
        for lower, p in by_lower.items()
        if posixpath.basename(lower) == stem
        or any(
            posixpath.basename(lower) == stem + extension.lower()
            for extensions in _DEFAULT_EXTENSIONS.values()
            for extension in extensions
        )
    ]
    return stems[0] if len(stems) == 1 else None

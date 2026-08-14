"""R package detection by walking the syntax tree.

The design point, and the reason this is not another regex: **strings and
comments are excluded structurally, not by pre-stripping.** In the tree-sitter
grammar, ``cat("run library(dplyr) first")`` is a ``call`` whose argument's
value is a ``string`` node; the bytes ``library(dplyr)`` live inside
``string_content`` and never form a ``call``. A walk that visits ``call`` nodes
therefore cannot see them. Likewise ``cppFunction("std::vector<int> f()")``
produces no ``namespace_operator``, because those bytes are string content too.

v1 attacked the same problem by deleting comments with a hand-rolled scanner and
then joining every line into one string before running six regexes over it. It
matched inside string literals (``std`` from embedded C++), it matched the loop
variable in ``library(pkg, character.only=TRUE)`` while losing the real package
list, and it missed ``requireNamespace("x", quietly = TRUE)`` -- the common form
-- because its pattern demanded a closing paren right after the quote. None of
those failure modes is reachable here.

Grammar node types this walker depends on, verified against
``tree-sitter-language-pack`` 1.14.3::

    call               fields: function, arguments
    arguments          children: argument (fields: name, value)
    namespace_operator fields: lhs, operator (:: or :::), rhs
    string             fields: content (string_content)

``test_r_grammar_contract`` asserts these still exist, so a grammar upgrade
fails loudly instead of silently returning zero mentions for every R file.
"""

from __future__ import annotations

from functools import cache

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from softverse.detect.types import ExtractResult, Mention, ParseReport
from softverse.model.enums import Construct, Language, ParseStatus

#: Callee name -> construct, for calls whose first argument names a package.
_LOADERS: dict[str, Construct] = {
    "library": Construct.LIBRARY,
    "require": Construct.REQUIRE,
    "requireNamespace": Construct.NAMESPACE_CHECK,
    "loadNamespace": Construct.NAMESPACE_CHECK,
    "attachNamespace": Construct.NAMESPACE_CHECK,
    "isNamespaceLoaded": Construct.NAMESPACE_CHECK,
}

#: pacman-style loaders, where *every* positional argument is a package.
_MULTI_LOADERS = {"p_load", "p_load_gh", "p_load_current_gh", "shelf"}

#: Installation, not use. Recorded separately so the headline tally can exclude
#: it: installing a package is not evidence the analysis ran on it.
_INSTALLERS = {
    "install.packages",
    "install_github",
    "install_gitlab",
    "install_bitbucket",
    "install_version",
    "install_cran",
    "p_install",
}

#: Calls naming a package, mapped to the keyword that carries it.
#:
#: The two groups are not interchangeable. For ``data()`` and ``vignette()`` the
#: *first positional* argument is a dataset or vignette name -- ``data("World")``
#: loads a dataset, not a package called World -- so only the explicit
#: ``package=`` keyword may be read. Falling back to the first positional here
#: produced exactly that false positive, caught by the renv cross-check.
#: For ``packageVersion()`` and friends the first positional *is* the package.
_PACKAGE_KEYWORD_ONLY = {
    "data": "package",
    "vignette": "package",
    "system.file": "package",
    "example": "package",
    "help": "package",
}

_PACKAGE_FIRST_ARG = {
    "packageVersion",
    "packageDescription",
    "citation",
    "getNamespace",
    "asNamespace",
    "packageName",
}

_PACKAGE_ARG = set(_PACKAGE_KEYWORD_ONLY) | _PACKAGE_FIRST_ARG

#: Nodes that make an enclosing call conditional.
_CONDITIONAL_PARENTS = {"if_statement", "if", "call"}
_CONDITIONAL_CALLEES = {"tryCatch", "try", "suppressWarnings", "suppressMessages"}


@cache
def _parser():
    return get_parser("r")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", "replace")


def _callee_name(call: Node, source: bytes) -> tuple[str | None, str | None]:
    """Return ``(callee, namespace)`` for a call node.

    ``pacman::p_load(x)`` yields ``("p_load", "pacman")``; the namespace is
    itself a package mention, handled by the namespace_operator walk.
    """
    fn = call.child_by_field_name("function")
    if fn is None:
        return None, None
    if fn.type == "identifier":
        return _text(fn, source), None
    if fn.type == "namespace_operator":
        lhs = fn.child_by_field_name("lhs")
        rhs = fn.child_by_field_name("rhs")
        return (
            _text(rhs, source) if rhs else None,
            _text(lhs, source) if lhs else None,
        )
    return None, None


def _string_value(node: Node, source: bytes) -> str | None:
    """Literal text of a ``string`` node, or ``None`` if it is not one."""
    if node.type != "string":
        return None
    content = node.child_by_field_name("content")
    return _text(content, source) if content is not None else ""


def _arguments(call: Node) -> list[Node]:
    args = call.child_by_field_name("arguments")
    if args is None:
        return []
    return [c for c in args.children if c.type == "argument"]


def _named_arg(args: list[Node], source: bytes, name: str) -> Node | None:
    for arg in args:
        key = arg.child_by_field_name("name")
        if key is not None and _text(key, source) == name:
            return arg.child_by_field_name("value")
    return None


def _is_truthy(node: Node | None, source: bytes) -> bool:
    if node is None:
        return False
    return _text(node, source).strip() in {"TRUE", "T", "true"}


def _snippet(source: bytes, node: Node) -> str:
    """The source line containing ``node``, trimmed.

    This is what makes a published number checkable by eye without re-running
    anything, and what makes a human spot-check cost seconds instead of minutes.
    """
    start = source.rfind(b"\n", 0, node.start_byte) + 1
    end = source.find(b"\n", node.end_byte)
    end = len(source) if end == -1 else end
    return source[start:end].decode("utf-8", "replace").strip()[:300]


def _conditional(node: Node, source: bytes) -> bool:
    parent = node.parent
    depth = 0
    while parent is not None and depth < 6:
        if parent.type in {"if_statement", "if"}:
            return True
        if parent.type == "call":
            callee, _ = _callee_name(parent, source)
            if callee in _CONDITIONAL_CALLEES:
                return True
        parent = parent.parent
        depth += 1
    return False


def _mention(
    name: str,
    construct: Construct,
    node: Node,
    source: bytes,
    *,
    dynamic: bool = False,
) -> Mention:
    return Mention(
        raw_name=name,
        construct=construct,
        line=node.start_point[0] + 1,
        col=node.start_point[1],
        byte_start=node.start_byte,
        byte_end=node.end_byte,
        snippet=_snippet(source, node),
        is_dynamic=dynamic,
        is_conditional=_conditional(node, source),
    )


def _handle_loader(
    call: Node, source: bytes, callee: str, construct: Construct
) -> list[Mention]:
    """``library(x)`` / ``require("x")`` / ``requireNamespace("x", quietly=TRUE)``.

    Matched by node shape, so the trailing-argument form v1's regex missed
    (292 files in the corpus contain ``requireNamespace``) is handled with no
    special case.
    """
    args = _arguments(call)
    if not args:
        return []
    first = args[0].child_by_field_name("value")
    if first is None:
        return []

    if (literal := _string_value(first, source)) is not None:
        return [_mention(literal, construct, first, source)]

    if first.type == "identifier":
        # An unquoted identifier is R's non-standard evaluation: normally the
        # package name itself. But with `character.only = TRUE` it is a
        # *variable* holding the name, and the real package is unknowable here.
        # v1 emitted the variable (`pkg`, `p`, `lib`) as a package and lost the
        # actual list; we record the mention as dynamic and resolve nothing.
        #
        # `requireNamespace` and friends have no such evaluation to begin
        # with: their `package` argument is documented as a character string,
        # so `requireNamespace(pkg)` is a variable every time and no
        # `character.only` is needed to say so. Only `library` and `require`
        # take a symbol.
        if construct is Construct.NAMESPACE_CHECK or _is_truthy(
            _named_arg(args, source, "character.only"), source
        ):
            return [
                _mention(
                    _text(first, source),
                    Construct.DYNAMIC_UNRESOLVED,
                    first,
                    source,
                    dynamic=True,
                )
            ]
        return [_mention(_text(first, source), construct, first, source)]

    # Anything else -- an index, a call, a paste() -- is dynamic by nature.
    return [
        _mention(
            _text(first, source)[:80],
            Construct.DYNAMIC_UNRESOLVED,
            first,
            source,
            dynamic=True,
        )
    ]


def _handle_multi(call: Node, source: bytes) -> list[Mention]:
    """``p_load(dplyr, ggplot2, install = FALSE)``.

    Only positional arguments are packages. v1 split on commas and stripped from
    the first ``=``, so ``install``, ``update`` and ``dependencies`` were all
    tallied as CRAN packages.
    """
    out: list[Mention] = []
    for arg in _arguments(call):
        if arg.child_by_field_name("name") is not None:
            continue
        value = arg.child_by_field_name("value")
        if value is None:
            continue
        if (literal := _string_value(value, source)) is not None:
            out.append(_mention(literal, Construct.P_LOAD, value, source))
        elif value.type == "identifier":
            out.append(_mention(_text(value, source), Construct.P_LOAD, value, source))
    return out


def _handle_installer(call: Node, source: bytes, callee: str) -> list[Mention]:
    out: list[Mention] = []
    for arg in _arguments(call):
        if arg.child_by_field_name("name") is not None:
            continue
        value = arg.child_by_field_name("value")
        if value is None:
            continue
        literal = _string_value(value, source)
        if literal is None:
            continue
        if callee.startswith("install_") and "/" in literal:
            # "user/repo" or "user/repo@ref" or "user/repo/subdir"
            repo = literal.split("/")[1].split("@")[0]
            out.append(_mention(repo, Construct.INSTALL, value, source))
        else:
            out.append(_mention(literal, Construct.INSTALL, value, source))
        break
    return out


def _handle_package_arg(call: Node, source: bytes, callee: str) -> list[Mention]:
    args = _arguments(call)
    if keyword := _PACKAGE_KEYWORD_ONLY.get(callee):
        # Keyword or nothing: the first positional is a dataset/vignette name.
        value = _named_arg(args, source, keyword)
    elif args and args[0].child_by_field_name("name") is None:
        value = args[0].child_by_field_name("value")
    else:
        value = None
    if value is None:
        return []
    literal = _string_value(value, source)
    if literal is not None:
        return [_mention(literal, Construct.PACKAGE_ARG, value, source)]
    if value.type == "identifier":
        # A bare symbol here is a variable, not a package. These functions
        # take a character scalar, so `packageVersion(pkg)` inside a helper is
        # the parameter, and recording its name as a package made `pkg` (90
        # mentions) and `package` (28) the two largest unresolved R names in
        # the corpus -- which reads as thin registry coverage rather than as
        # somebody's install-if-missing loop. Same treatment as
        # `library(pkg, character.only = TRUE)`, which was already handled;
        # the mention is kept, because dropping it would hide the idiom
        # instead of naming it.
        return [
            _mention(
                _text(value, source),
                Construct.DYNAMIC_UNRESOLVED,
                value,
                source,
                dynamic=True,
            )
        ]
    return []


def extract(source: str | bytes) -> ExtractResult:
    """Extract R package mentions from source text."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    tree = _parser().parse(data)
    root = tree.root_node

    mentions: list[Mention] = []
    error_nodes = 0
    bytes_in_error = 0

    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in {"ERROR", "MISSING"}:
            error_nodes += 1
            bytes_in_error += node.end_byte - node.start_byte
            # Do not descend into an error region; its shape is unreliable.
            continue

        if node.type == "namespace_operator":
            lhs = node.child_by_field_name("lhs")
            operator = node.child_by_field_name("operator")
            if lhs is not None and lhs.type == "identifier":
                internal = operator is not None and _text(operator, data) == ":::"
                mentions.append(
                    _mention(
                        _text(lhs, data),
                        Construct.NAMESPACE_INTERNAL
                        if internal
                        else Construct.NAMESPACE_OP,
                        lhs,
                        data,
                    )
                )

        elif node.type == "call":
            callee, _namespace = _callee_name(node, data)
            if callee is not None:
                if construct := _LOADERS.get(callee):
                    mentions.extend(_handle_loader(node, data, callee, construct))
                elif callee in _MULTI_LOADERS:
                    mentions.extend(_handle_multi(node, data))
                elif callee in _INSTALLERS:
                    mentions.extend(_handle_installer(node, data, callee))
                elif callee in _PACKAGE_ARG:
                    mentions.extend(_handle_package_arg(node, data, callee))

        stack.extend(node.children)

    if error_nodes:
        # tree-sitter always returns a tree, so "parsed" is not "parsed well".
        # Files that are mostly error are flagged, not silently trusted.
        status = (
            ParseStatus.SYNTAX_ERROR
            if bytes_in_error > 0.2 * max(len(data), 1)
            else ParseStatus.OK_WITH_ERRORS
        )
    else:
        status = ParseStatus.OK

    mentions.sort(key=lambda m: m.byte_start)
    return ExtractResult(
        mentions=mentions,
        report=ParseReport(
            status=status,
            language=Language.R,
            error_nodes=error_nodes,
            bytes_total=len(data),
            bytes_in_error=bytes_in_error,
        ),
    )

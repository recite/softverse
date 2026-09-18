"""Adapt the Stata lexer to the extractor interface.

The lexing lives in :mod:`softverse.stata.lexer`; this turns its statements into
:class:`~softverse.detect.types.Mention` rows so Stata flows through the same
pipeline as R and Python.

One decision worth naming: **every command token becomes a mention**, including
builtins, and the resolver decides what it is. The alternative -- filtering
builtins here -- would throw away the denominator. Knowing that a deposit issued
1,600 builtin commands and 3 package commands is the difference between "this
analysis barely uses packages" and "we found almost nothing", and v1 could not
tell those apart for any language.
"""

from __future__ import annotations

import re

from softverse.detect.types import ExtractResult, Mention, ParseReport
from softverse.model.enums import Construct, Language, ParseStatus
from softverse.stata.lexer import lex, local_programs

#: Commands whose operand names a package rather than being one: provisioning
#: and inquiry, which are recorded but excluded from the headline tally.
_OPERAND_COMMANDS = frozenset({"ssc", "net", "github", "findit", "which", "search"})

#: `ssc` and `net` subcommands that take a package name, by what they mean.
#: Everything else -- `ssc hot`, `net from URL`, `net set ado` -- names no
#: package, and reading the word after it as one recorded `hot` and `new`.
_PROVISION = ("install", "get")
_INQUIRE = ("describe", "type", "copy")

_PACKAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_FROM = re.compile(r"from\(\s*\"?([^\")\s]+)", re.IGNORECASE)


def _subcommand(word: str, names: tuple[str, ...]) -> bool:
    """Whether ``word`` is one of ``names`` or a Stata abbreviation of it."""
    word = word.lower()
    return any(
        name.startswith(word) and len(word) >= min(3, len(name)) for name in names
    ) or (word == "d" and "describe" in names)


#: `egen newvar = fn(...)`, with an optional storage type before the name.
_EGEN_FUNCTION = re.compile(r"=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_URL = re.compile(r"[a-z]+://\S+", re.IGNORECASE)


def _net_location(tokens: list[str]) -> str | None:
    """The site `net from URL` or `net cd URL` moves to, else None."""
    if tokens[0].lower() != "net" or len(tokens) < 3:
        return None
    if tokens[1].lower() not in {"from", "cd"}:
        return None
    found = _URL.search(tokens[2].strip('"'))
    return found.group(0).rstrip(",") if found else None


def _provisioned(tokens: list[str]) -> tuple[str, Construct, str | None] | None:
    """The package a provisioning or inquiry statement names.

    Args:
        tokens: The statement from its command word on.

    Returns:
        ``(package, construct, remote)``, or None when the statement names none.
    """
    head, rest = tokens[0].lower(), tokens[1:]
    if not rest:
        return None
    if head in {"findit", "which", "search"}:
        construct, operand = Construct.STATA_WHICH, rest[0]
    elif _subcommand(rest[0], _PROVISION) and len(rest) > 1:
        construct, operand = Construct.STATA_INSTALL, rest[1]
    elif head != "github" and _subcommand(rest[0], _INQUIRE) and len(rest) > 1:
        construct, operand = Construct.STATA_WHICH, rest[1]
    else:
        return None
    # The option comma is rarely set off by a space: `ssc install reghdfe,
    # replace` is the usual spelling, and splitting on whitespace alone left
    # `reghdfe,` -- not an identifier -- so 4,924 of the corpus's 12,565
    # install lines were dropped without a trace.
    operand = operand.split(",", 1)[0]
    remote = None
    if head == "github":
        # haghish's `github install user/repo`.
        remote = f"github.com/{operand}"
        operand = operand.rsplit("/", 1)[-1]
    elif head == "net" and (source := _FROM.search(" ".join(rest))):
        remote = source.group(1)
    if not _PACKAGE.fullmatch(operand):
        return None
    return operand, construct, remote


def _egen_functions(source: str, lines: list[str]) -> list[Mention]:
    """The functions `egen` is asked for, as the files that implement them.

    `egen n = nvals(x)` runs `_gnvals.ado`, and that file is what a package
    ships: `egenmore` provides dozens of them and no command of its own. Read
    as commands only, every such call was a call to official `egen`, and
    `egenmore`, used in over four hundred deposits, was credited with none.
    The mention names the file, `_gnvals`, so the index resolves it like any
    other helper: to official Stata where Stata ships it, to its package where
    a package does.
    """
    out = []
    for statement in lex(source):
        if statement.in_mata or (statement.command or "").lower() != "egen":
            continue
        called = _EGEN_FUNCTION.search(statement.text)
        if called is None:
            continue
        out.append(
            Mention(
                raw_name=f"_g{called.group(1)}",
                called_function=called.group(1),
                construct=Construct.STATA_COMMAND,
                line=statement.line,
                col=statement.col,
                byte_start=0,
                byte_end=0,
                snippet=lines[statement.line - 1].strip()[:300]
                if 0 < statement.line <= len(lines)
                else statement.text[:300],
            )
        )
    return out


def extract(source: str) -> ExtractResult:
    """Extract Stata command mentions from a do-file or ado-file."""
    lines = source.splitlines()
    locals_defined = local_programs(source)
    mentions: list[Mention] = []
    macro_commands = 0
    # `net from URL` sets where the `net install` lines after it fetch from,
    # and that two-line form is how suites served from an author's site --
    # SPost, the UCLA utilities -- are usually installed.
    net_site: str | None = None

    for statement in lex(source):
        if statement.in_mata:
            # Mata is a different language with its own namespace.
            continue
        if statement.is_macro_command:
            # `cmd' varlist -- unknowable statically. Counted, never guessed.
            macro_commands += 1
            continue
        if not statement.command:
            continue

        snippet = (
            lines[statement.line - 1].strip()[:300]
            if 0 < statement.line <= len(lines)
            else statement.text[:300]
        )
        head = statement.command.lower()

        tokens = statement.text.split()
        if head in {"if", "else"}:
            # `if _rc ssc install reghdfe, replace` -- install-if-missing on
            # one line. The command word is `if`, so the install was recorded
            # as a call to a builtin and the dependency it states was lost.
            start = next(
                (i for i, t in enumerate(tokens) if i and t.lower() in {"ssc", "net"}),
                None,
            )
            tokens = tokens[start:] if start else tokens
        if tokens[0].lower() in _OPERAND_COMMANDS:
            net_site = _net_location(tokens) or net_site
            if found := _provisioned(tokens):
                name, construct, remote = found
                if tokens[0].lower() == "net" and construct is Construct.STATA_INSTALL:
                    remote = remote or net_site
                mentions.append(
                    Mention(
                        raw_name=name,
                        construct=construct,
                        line=statement.line,
                        col=statement.col,
                        byte_start=0,
                        byte_end=0,
                        snippet=snippet,
                        remote=remote,
                    )
                )
            if head not in {"if", "else"}:
                continue

        mentions.append(
            Mention(
                raw_name=statement.command,
                # In Stata the command *is* the call, so this repeats
                # `raw_name` on purpose. It means "which functions of this
                # package does published code use" is one query across all
                # three languages instead of three.
                called_function=statement.command,
                construct=Construct.LOCAL_PROGRAM
                if head in {p.lower() for p in locals_defined}
                else Construct.STATA_COMMAND,
                line=statement.line,
                col=statement.col,
                byte_start=0,
                byte_end=0,
                snippet=snippet,
                is_conditional=bool(
                    {"capture", "cap", "captur"} & set(statement.prefixes)
                ),
            )
        )

    return ExtractResult(
        mentions=mentions + _egen_functions(source, lines),
        report=ParseReport(
            status=ParseStatus.OK,
            language=Language.STATA,
            bytes_total=len(source),
            detail=(
                f"{macro_commands} macro-named commands (unresolvable)"
                if macro_commands
                else None
            ),
        ),
    )

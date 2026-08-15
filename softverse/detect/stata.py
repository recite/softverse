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

from softverse.detect.types import ExtractResult, Mention, ParseReport
from softverse.model.enums import Construct, Language, ParseStatus
from softverse.stata.lexer import lex, local_programs

#: Second-position operands worth capturing: provisioning and inquiry, which are
#: recorded but excluded from the headline tally.
_OPERAND_COMMANDS = {
    "ssc": Construct.STATA_INSTALL,
    "net": Construct.STATA_INSTALL,
    "findit": Construct.STATA_WHICH,
    "which": Construct.STATA_WHICH,
    "search": Construct.STATA_WHICH,
}


def extract(source: str) -> ExtractResult:
    """Extract Stata command mentions from a do-file or ado-file."""
    lines = source.splitlines()
    locals_defined = local_programs(source)
    mentions: list[Mention] = []
    macro_commands = 0

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

        if head in _OPERAND_COMMANDS:
            operands = statement.operands
            # `ssc install foo` -- skip the subcommand to reach the package.
            target = next(
                (
                    token
                    for token in operands
                    if token.lower() not in {"install", "inst", "from", "describe", "d"}
                ),
                None,
            )
            if target and target.replace("_", "").isalnum():
                mentions.append(
                    Mention(
                        raw_name=target,
                        construct=_OPERAND_COMMANDS[head],
                        line=statement.line,
                        col=statement.col,
                        byte_start=0,
                        byte_end=0,
                        snippet=snippet,
                    )
                )
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
        mentions=mentions,
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

"""A Stata lexer, because no parser grammar exists.

tree-sitter has grammars for R, Python, Julia, MATLAB and SAS, but none for
Stata, so this is the one language where structural parsing is not available and
the tokenizer has to be written by hand.

The governing insight is that **the first token of a Stata command line is the
command**. That single fact removes v1's entire false-positive class: v1 searched
for a hardcoded list of names anywhere in the text, so ``gen diff = y2 - y1``
reported ``diff``, ``* did the main regression`` reported ``did``, and across the
corpus it reported the English words ``is`` (88 deposits) and ``we`` (57) as
Stata packages. Reading only the leading token, ``gen diff = ...`` yields
``gen``, and a comment yields nothing.

Getting to that leading token correctly is the whole job, and Stata makes it
harder than it sounds:

- ``*`` starts a comment **only** as the first non-blank character of a
  statement; elsewhere it is multiplication. 81% of the corpus's ``.do`` files
  contain ``*`` comments, and v1 never stripped them.
- ``///`` continues a line, and a ``/*`` that spans a line end does too. 40% of
  files use ``///``.
- ``#delimit ;`` switches the statement terminator mid-file. 7% of files do
  this, and mis-handling it mis-tokenizes everything after the switch.
- Prefixes stack: ``quietly bysort id: reghdfe y x`` has three tokens before
  the command.
- Commands can be built from macros (``` `cmd' varlist ```), which are
  unresolvable and must be counted as such rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Prefix commands, which stand before the real command. Includes Stata's
#: standard abbreviations, since `qui`, `noi` and `cap` are what people type.
PREFIX_COMMANDS = frozenset(
    {
        "quietly",
        "quietl",
        "quiet",
        "quie",
        "qui",
        "noisily",
        "noisil",
        "noisi",
        "nois",
        "noi",
        # `n` is the shortest abbreviation of `noisily`, and it was the one
        # missing: `cap n estadd ysumm` recorded `n` as the command and lost
        # `estadd` with it. 750 mentions across 14 deposits, the second-largest
        # unresolved Stata name in the corpus. Safe because there is no Stata
        # command `n` -- StataCorp's help server answers for it with
        # `[U] 13.4 System variables`, the page documenting `_n`.
        "n",
        "capture",
        "captur",
        "captu",
        "capt",
        "cap",
        "by",
        "bysort",
        "bys",
        "xi",
        "svy",
        "mi",
        "fvset",
        "bootstrap",
        "bs",
        "jackknife",
        "jknife",
        "permute",
        "simulate",
        "statsby",
        "rolling",
        "nestreg",
        "stepwise",
        "sw",
        "version",
        "vers",
        "ver",
    }
)

#: Prefixes taking a colon-terminated argument list (`bysort id year:`).
_COLON_PREFIXES = frozenset(
    {
        "by",
        "bysort",
        "bys",
        "svy",
        "mi",
        "xi",
        "statsby",
        "rolling",
        "bootstrap",
        "bs",
        "jackknife",
        "jknife",
        "permute",
        "simulate",
        "nestreg",
        "stepwise",
        "sw",
        "version",
        "vers",
        "ver",
    }
)

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Legal spellings of `program`. Matching any command merely *starting* with
#: `pr` also caught `predict`, `probit` and `preserve`, so `predict yhat`
#: declared a local program called `yhat`. StataCorp's help server resolves
#: `pr`, `pro` and `prog` to `[P] program`; the others are their own commands.
_PROGRAM_FORMS = frozenset({"pr", "pro", "prog", "progr", "progra", "program"})
#: A token containing a macro reference cannot be resolved to a command name.
_MACRO = re.compile(r"[`$]")


@dataclass
class Statement:
    """One Stata statement, reduced to what resolution needs."""

    command: str | None
    line: int
    col: int
    text: str
    prefixes: list[str] = field(default_factory=list)
    #: The command word came from a macro, so it is unknowable statically.
    is_macro_command: bool = False
    #: Inside a `mata:` ... `end` block, which is a different language.
    in_mata: bool = False

    @property
    def operands(self) -> list[str]:
        """Tokens after the command word, whitespace-split."""
        return self.text.split()[1:] if self.text else []


def strip_comments(source: str) -> str:
    """Remove Stata comments and join continuations, preserving line count.

    Handles, in one pass over the characters so that context is never lost:

    - ``/* ... */`` blocks, which nest in Stata
    - ``//`` to end of line, but not inside a string and not as part of ``///``
    - ``*`` at the start of a statement only
    - ``///`` continuations, which splice the next line onto this one
    - double quotes and Stata's compound ``` `"..."' ``` quotes, inside which
      none of the above apply

    Newlines are preserved (as newlines or spaces) so reported line numbers
    still refer to the original file.
    """
    out: list[str] = []
    i, n = 0, len(source)
    depth = 0  # /* */ nesting
    in_string = False
    in_compound = 0  # `"..."' nesting
    at_statement_start = True

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if depth:
            if ch == "/" and nxt == "*":
                depth += 1
                i += 2
                continue
            if ch == "*" and nxt == "/":
                depth -= 1
                i += 2
                continue
            # Keep newlines so line numbers survive a multi-line comment.
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if in_string or in_compound:
            out.append(ch)
            if in_compound:
                if ch == "`" and nxt == '"':
                    in_compound += 1
                    out.append(nxt)
                    i += 2
                    continue
                if ch == '"' and nxt == "'":
                    in_compound -= 1
                    out.append(nxt)
                    i += 2
                    continue
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "`" and nxt == '"':
            in_compound = 1
            out.append(ch)
            out.append(nxt)
            i += 2
            at_statement_start = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            at_statement_start = False
            continue

        if ch == "/" and nxt == "*":
            depth = 1
            i += 2
            continue
        if ch == "/" and nxt == "/":
            # `///` splices the next physical line onto this one.
            if source[i : i + 3] == "///":
                j = source.find("\n", i)
                if j == -1:
                    break
                out.append(" ")
                i = j + 1
                continue
            j = source.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "*" and at_statement_start:
            j = source.find("\n", i)
            i = n if j == -1 else j
            continue

        if ch == "\n":
            at_statement_start = True
        elif not ch.isspace():
            at_statement_start = False
        out.append(ch)
        i += 1

    return "".join(out)


#: Characters that, immediately before a brace, mark it as part of a word
#: rather than a block delimiter. Stata's own block braces always follow
#: whitespace or a closing parenthesis -- `if x==1 {`, `if (x==1){` -- so
#: nothing legitimate is suppressed, while `\multicolumn{3}{c}` is protected.
_WORD_BEFORE_BRACE = frozenset("_\\}")


def _brace_is_word_internal(cleaned: str, i: int, buffer: list[str]) -> bool:
    """True when the brace at ``i`` belongs to a word, not to a block.

    Quoted and parenthesised braces are already handled by the string and paren
    guards. This covers the remaining case: raw LaTeX at the top level, which
    `texdoc`'s `tex` command takes unquoted --

        tex & \\multicolumn{3}{c}{Full Sample} &

    -- where splitting on the braces produces fragments led by `3` and `c`, and
    `c` is the sole command of the package `fastcd`.
    """
    if not buffer or i == 0:
        # Nothing precedes it in this statement, so it cannot be word-internal.
        return False
    previous = cleaned[i - 1]
    return previous.isalnum() or previous in _WORD_BEFORE_BRACE


def _split_statements(cleaned: str) -> list[tuple[int, int, str]]:
    """Split cleaned source into ``(line, col, text)``, honouring ``#delimit``.

    Under ``#delimit ;`` a statement ends at a semicolon and newlines are mere
    whitespace; under ``#delimit cr`` (the default) a statement ends at a
    newline. Files switch between the two mid-stream.
    """
    statements: list[tuple[int, int, str]] = []
    semicolon_mode = False
    buffer: list[str] = []
    start_line = 1
    line = 1
    # String state must be tracked here as well as in strip_comments, which
    # deliberately preserves string contents. Without it, the braces inside a
    # LaTeX option -- esttab's prehead("\begin{tabular}{l*{4}{c}}") -- split one
    # statement into several, and the fragments' leading tokens ("c", "tabular")
    # were reported as commands. Measured on the corpus, that alone attributed
    # 144 deposits to `fastcd`, a package whose only command is named `c`.
    in_string = False
    in_compound = 0
    #: Parenthesis depth, so a brace inside an option list is not a delimiter.
    paren = 0

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        text = "".join(buffer).strip()
        if text:
            statements.append((start_line, 0, text))
        buffer = []
        start_line = end_line

    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        nxt = cleaned[i + 1] if i + 1 < n else ""

        if in_string or in_compound:
            if ch == "\n":
                line += 1
            buffer.append(ch)
            if in_compound:
                if ch == "`" and nxt == '"':
                    in_compound += 1
                    buffer.append(nxt)
                    i += 2
                    continue
                if ch == '"' and nxt == "'":
                    in_compound -= 1
                    buffer.append(nxt)
                    i += 2
                    continue
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == "`" and nxt == '"':
            in_compound = 1
            buffer.append(ch)
            buffer.append(nxt)
            i += 2
            continue
        if ch == '"':
            in_string = True
            buffer.append(ch)
            i += 1
            continue

        if ch == "\n":
            line += 1
            if semicolon_mode:
                buffer.append(" ")
            else:
                flush(line)
            i += 1
            continue

        # A #delimit directive is itself terminated by the *current* delimiter.
        if cleaned.startswith("#delimit", i) or cleaned.startswith("#d ", i):
            end = cleaned.find("\n", i)
            end = n if end == -1 else end
            directive = cleaned[i:end]
            semicolon_mode = ";" in directive
            flush(line)
            i = end
            continue

        # `${name}` is a global macro reference, and its braces are part of an
        # expression rather than a block delimiter. Splitting on them shattered
        # statements like `g d${_i}Lco = f${_i}.lnL - l.lnL` and reported the
        # macro name as a command: `_i`, `ind`, `output` and `Level` all reached
        # the top of the unresolved list this way. Same failure as the LaTeX
        # braces inside esttab options, arriving through a different syntax.
        if ch == "$" and nxt == "{":
            end = cleaned.find("}", i)
            if end != -1:
                buffer.append(cleaned[i : end + 1])
                i = end + 1
                continue

        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)

        if ch == ";" and semicolon_mode and paren == 0:
            flush(line)
            i += 1
            continue

        # Braces delimit blocks only at the top level. Inside an option list
        # they are data, not syntax.
        if (
            ch in "{}"
            and paren == 0
            and not _brace_is_word_internal(cleaned, i, buffer)
        ):
            flush(line)
            i += 1
            continue

        buffer.append(ch)
        i += 1

    flush(line)
    return statements


def _strip_prefixes(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Peel stacked prefix commands off the front. Returns ``(prefixes, rest)``."""
    prefixes: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i].lower().rstrip(":")
        if token not in PREFIX_COMMANDS:
            break
        prefixes.append(token)
        i += 1
        if token in _COLON_PREFIXES:
            # Skip the prefix's own arguments up to and including the colon.
            while i < len(tokens) and not tokens[i - 1].endswith(":"):
                if tokens[i].startswith(":"):
                    tokens[i] = tokens[i][1:]
                    if not tokens[i]:
                        i += 1
                    break
                if tokens[i].endswith(":"):
                    i += 1
                    break
                i += 1
    return prefixes, tokens[i:]


def _clean_token(token: str) -> str:
    """Strip trailing punctuation and leading dots from a command token.

    ``esttab, se`` tokenizes as ``esttab,`` -- the comma belongs to the option
    list, not the name.
    """
    return token.strip().lstrip(".").rstrip(",;:")


def lex(source: str) -> list[Statement]:
    """Tokenize Stata source into statements with their command words."""
    statements: list[Statement] = []
    in_mata = False

    for line, col, text in _split_statements(strip_comments(source)):
        tokens = text.split()
        if not tokens:
            continue

        prefixes, rest = _strip_prefixes(list(tokens))
        if not rest:
            continue

        raw = _clean_token(rest[0])
        head = raw.lower()

        # `mata:` opens a block in a different language; `end` closes it.
        if head in {"mata", "mata:"}:
            in_mata = True
        elif in_mata and head == "end":
            in_mata = False
            continue

        is_macro = bool(_MACRO.search(raw))
        command = raw if _IDENT.match(raw) else None

        statements.append(
            Statement(
                command=None if is_macro else command,
                line=line,
                col=col,
                text=" ".join(rest),
                prefixes=prefixes,
                is_macro_command=is_macro,
                in_mata=in_mata and head not in {"mata", "mata:"},
            )
        )
    return statements


def local_programs(source: str) -> set[str]:
    """Names defined by ``program define`` in this source.

    Subtracted before resolution: an author's helper program is not a package,
    and v1 had no way to tell the difference.
    """
    names: set[str] = set()
    for statement in lex(source):
        if not statement.command or statement.command.lower() not in _PROGRAM_FORMS:
            continue
        tokens = statement.text.split()
        rest = [t for t in tokens[1:] if t.lower() not in {"define", "def", "drop"}]
        if not rest:
            continue
        # `program mycmd, rclass` -- the options are attached to the name by a
        # comma, and an `.ado` file almost always declares a class (`sclass`,
        # `rclass`, `eclass`, `sortpreserve`). Leaving the comma on the token
        # failed the identifier match, so the program was never recorded as
        # locally defined and every call to it resolved to `unknown`. One
        # deposit that vendors Stata's own `ado/base` tree alone contributed
        # 56% of the corpus's unresolved Stata mentions this way.
        candidate = rest[0].split(",", 1)[0]
        if _IDENT.match(candidate):
            names.add(candidate)
    return names

"""Shared types for extractors.

The contract every extractor obeys: return mentions *and* a report. No extractor
may return an empty list without a :class:`ParseReport` explaining why. v1
returned ``None`` for "no packages here", "syntax error", "could not decode" and
"unsupported file type" alike, which is why 26,681 files vanished from its
outputs with no way to tell which case each was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from softverse.model.enums import Construct, Language, ParseStatus


@dataclass(frozen=True)
class Mention:
    """One syntactic reference to a package, with its location in source."""

    raw_name: str
    construct: Construct
    line: int
    col: int
    byte_start: int
    byte_end: int
    snippet: str = ""
    #: True when the name cannot be known statically -- an R
    #: `character.only=TRUE` loop variable, or a macro-named Stata command.
    #: Recorded rather than guessed at.
    is_dynamic: bool = False
    #: Inside `if`, `tryCatch`, `try/except` or `capture`.
    is_conditional: bool = False
    cell_index: int | None = None
    chunk_label: str | None = None
    #: The language this mention is *in*, when it differs from the file's.
    #: A `.Rmd` can hold Python chunks and a notebook can run an R kernel, so
    #: resolving against the file's language would send every mention in a
    #: literate document to the wrong registry and return UNKNOWN for all of
    #: them. Left None for single-language files, where the file's language is
    #: the mention's language.
    language: Language | None = None
    #: The thing actually called, when the source says. `dplyr::select(x)` is
    #: a mention of `dplyr` whose `called_function` is `select`; a Stata
    #: `reghdfe y x` is a mention of `reghdfe` whose called function is the
    #: same word, because in Stata the command *is* the call.
    #:
    #: Recording it for Stata is redundant and deliberate. Without it a
    #: consumer has to know that one language puts the call in `raw_name` and
    #: another puts the package there, and "which functions of this package
    #: does published code use" becomes three different queries.
    #:
    #: None where the source does not say: `library(dplyr)` names a package
    #: and no function, and a bare `select(x)` names a function whose package
    #: needs scope resolution this extractor does not attempt.
    called_function: str | None = None


@dataclass
class ParseReport:
    """Why a file yielded the mentions it did -- including none."""

    status: ParseStatus
    language: Language
    #: tree-sitter is error-tolerant and always returns a tree, so a parse that
    #: "succeeded" can still be mostly ERROR nodes. Counted, not assumed away.
    error_nodes: int = 0
    bytes_total: int = 0
    bytes_in_error: int = 0
    detail: str | None = None

    @property
    def error_fraction(self) -> float:
        return self.bytes_in_error / self.bytes_total if self.bytes_total else 0.0


@dataclass
class ExtractResult:
    """What an extractor returns."""

    mentions: list[Mention] = field(default_factory=list)
    report: ParseReport | None = None

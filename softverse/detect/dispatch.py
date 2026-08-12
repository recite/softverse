"""Route a file to its extractor, and decode it honestly.

Two v1 defects live here rather than in the extractors:

**Extension matching was case-sensitive.** ``file_path.suffix in extensions``
with a list containing ``".R"`` silently skips every ``.r`` file, and ``.DO``,
``.PY`` and ``.IPYNB`` were missed always.

**Encoding fell back to latin-1 for the whole file.** latin-1 cannot fail, so a
UTF-16 file decodes into NUL-interleaved text that matches nothing and yields
zero mentions -- indistinguishable from a file with no imports. Nothing counted
how often that happened, so the false-negative rate was unobservable. Here the
BOM is checked first, then UTF-8, then a detector, and the outcome is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

from softverse.detect import python_, r, stata
from softverse.detect.types import ExtractResult, ParseReport
from softverse.model.enums import Language, ParseStatus

#: Extension -> language. Lowercased before lookup, so case never matters.
EXTENSION_LANGUAGE: dict[str, Language] = {
    ".r": Language.R,
    ".rscript": Language.R,
    ".rmd": Language.RMARKDOWN,
    ".qmd": Language.RMARKDOWN,
    ".rnw": Language.RMARKDOWN,
    ".py": Language.PYTHON,
    ".ipynb": Language.NOTEBOOK,
    ".do": Language.STATA,
    ".ado": Language.STATA,
    ".mata": Language.MATA,
    ".jl": Language.JULIA,
    ".m": Language.MATLAB,
    ".sas": Language.SAS,
    ".sps": Language.SPSS,
    ".sh": Language.SHELL,
    ".bash": Language.SHELL,
    ".sql": Language.SQL,
}

#: Languages we currently have an extractor for. The rest are recorded as
#: present (a real, defensible claim about language use) but not attributed to
#: packages. v1 read .sas/.jl/.m into memory and silently discarded them.
IMPLEMENTED = {Language.R, Language.PYTHON, Language.STATA}

_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)


@dataclass
class Decoded:
    text: str
    encoding: str
    confidence: float
    fallback: bool


def language_of(path: Path | str) -> Language:
    """Language implied by the extension, case-insensitively."""
    return EXTENSION_LANGUAGE.get(Path(path).suffix.lower(), Language.UNKNOWN)


def looks_binary(data: bytes) -> bool:
    """A NUL byte in the first block means this is not source text.

    Checked before decoding so a binary file becomes ``skipped_binary`` rather
    than a successful decode producing zero mentions.
    """
    return b"\x00" in data[:8192] and not any(data.startswith(bom) for bom, _ in _BOMS)


def decode(data: bytes) -> Decoded:
    """Decode source bytes, recording how confident we are.

    Order matters: a BOM is definitive, UTF-8 covers almost everything else, and
    only then do we guess. Guessing is *recorded*, not hidden, so files whose
    mentions rest on a guess can be counted and excluded from strong claims.
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return Decoded(data.decode(encoding, "replace"), encoding, 1.0, False)
    try:
        return Decoded(data.decode("utf-8"), "utf-8", 1.0, False)
    except UnicodeDecodeError:
        pass
    if match := from_bytes(data).best():
        return Decoded(
            str(match),
            match.encoding,
            # charset-normalizer reports chaos, where lower is better.
            max(0.0, 1.0 - float(match.chaos)),
            True,
        )
    return Decoded(data.decode("latin-1", "replace"), "latin-1", 0.0, True)


def extract_file(path: Path) -> tuple[ExtractResult, Decoded | None, Language]:
    """Read, decode and extract one file. Never returns a bare empty list."""
    language = language_of(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return (
            ExtractResult(
                report=ParseReport(ParseStatus.DECODE_ERROR, language, detail=str(exc))
            ),
            None,
            language,
        )

    if not data.strip():
        return (
            ExtractResult(report=ParseReport(ParseStatus.SKIPPED_EMPTY, language)),
            None,
            language,
        )
    if looks_binary(data):
        return (
            ExtractResult(report=ParseReport(ParseStatus.SKIPPED_BINARY, language)),
            None,
            language,
        )
    if language not in IMPLEMENTED:
        return (
            ExtractResult(
                report=ParseReport(ParseStatus.UNSUPPORTED_LANGUAGE, language)
            ),
            None,
            language,
        )

    decoded = decode(data)
    if language is Language.R:
        result = r.extract(decoded.text)
    elif language is Language.STATA:
        result = stata.extract(decoded.text)
    else:
        result = python_.extract(decoded.text, str(path))

    # A guessed encoding is grounds for suspicion even when parsing succeeded.
    if decoded.fallback and result.report and result.report.status is ParseStatus.OK:
        result.report.status = ParseStatus.DECODE_FALLBACK
    return result, decoded, language

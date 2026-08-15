"""Extract code from literate documents: .Rmd/.qmd/.Rnw and .ipynb.

These are collected by every source and were parsed by none of them -- v1 read
them into memory and dropped them, so 1,159 .Rmd files across 31 deposits and
295 notebooks across 46 contributed nothing while looking like deposits with no
code.

Two formats, both of which embed one language inside another:

**knitr documents** (.Rmd, .qmd, .Rnw) interleave prose with fenced chunks. The
fence grammar is small and specified, so a splitter is more honest here than
bending a Markdown parser: knitr fences are not CommonMark info strings, and
Quarto adds ``#|`` cell options. Line offsets are tracked so a mention's line
number refers to the *document*, not to the chunk -- otherwise every snippet
points at the wrong place and the auditability the schema promises is fiction.

**Jupyter notebooks** are JSON. The kernel language is read from the metadata
rather than assumed to be Python, because R and Stata kernels both exist in this
corpus. Magics (``%pip install``, ``!pip``) are stripped before parsing, since
they are not valid Python, but ``pip install`` lines are still captured -- as
installation, which is not the same as use.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from softverse.detect import python_, r, stata
from softverse.detect.types import ExtractResult, Mention, ParseReport
from softverse.model.enums import Construct, Language, ParseStatus

#: ```{r label, opts} / ```{python} -- the engine is the first word.
_CHUNK_OPEN = re.compile(r"^\s*(?:```+|~~~+)\s*\{([^}]*)\}\s*$")
_CHUNK_CLOSE = re.compile(r"^\s*(?:```+|~~~+)\s*$")
#: Sweave/.Rnw chunks use a different delimiter entirely.
_RNW_OPEN = re.compile(r"^\s*<<([^>]*)>>=\s*$")
_RNW_CLOSE = re.compile(r"^\s*@\s*$")
#: Inline `r expr` spans, which really do load packages in practice.
_INLINE_R = re.compile(r"`r\s+([^`]+)`")
#: %pip install x / !pip install x / %conda install x
_MAGIC_INSTALL = re.compile(
    r"^\s*[%!]\s*(?:pip|conda|mamba)\s+install\s+(?:-\S+\s+)*(.+)$"
)

#: Kernels we can read. A kernel that is not in here is not parsed, because
#: the fallback used to be the Python extractor: a Julia notebook's
#: `import Ipopt` is valid Python, so it parsed cleanly and was recorded as a
#: Julia package, which is how one Julia "package" entered a corpus in which
#: no `.jl` file is ever parsed. A Stata kernel went the same way.
_CELL_EXTRACTOR = {
    Language.R: r.extract,
    Language.PYTHON: python_.extract,
    Language.STATA: stata.extract,
}

_ENGINE_LANGUAGE = {
    "r": Language.R,
    "python": Language.PYTHON,
    "py": Language.PYTHON,
    "stata": Language.STATA,
    "julia": Language.JULIA,
    "bash": Language.SHELL,
    "sh": Language.SHELL,
}


def _engine_of(header: str) -> tuple[Language | None, str | None]:
    """``r fig1, echo=FALSE`` -> (R, "fig1")."""
    parts = header.strip().split(",", 1)[0].split()
    if not parts:
        return None, None
    engine = _ENGINE_LANGUAGE.get(parts[0].strip().lower())
    label = parts[1].strip() if len(parts) > 1 else None
    return engine, label


def split_chunks(source: str) -> list[tuple[Language, str, int, str | None]]:
    """Split a knitr document into ``(language, code, first_line, label)``.

    ``first_line`` is the document line of the chunk's first code line, so
    mentions can be reported against the document rather than the chunk.
    """
    lines = source.splitlines()
    chunks: list[tuple[Language, str, int, str | None]] = []
    engine: Language | None = None
    label: str | None = None
    buffer: list[str] = []
    start = 0
    closer: re.Pattern[str] | None = None

    for number, line in enumerate(lines, 1):
        if closer is None:
            if match := _CHUNK_OPEN.match(line):
                engine, label = _engine_of(match.group(1))
                buffer, start, closer = [], number + 1, _CHUNK_CLOSE
            elif match := _RNW_OPEN.match(line):
                engine, label = Language.R, (match.group(1).split(",")[0] or None)
                buffer, start, closer = [], number + 1, _RNW_CLOSE
            continue
        if closer.match(line):
            if engine is not None and buffer:
                chunks.append((engine, "\n".join(buffer), start, label))
            closer, engine, label, buffer = None, None, None, []
            continue
        buffer.append(line)

    # An unterminated final chunk is still code worth reading.
    if closer is not None and engine is not None and buffer:
        chunks.append((engine, "\n".join(buffer), start, label))
    return chunks


def _shift(
    mentions: list[Mention],
    offset: int,
    label: str | None,
    language: Language | None = None,
) -> list[Mention]:
    """Re-base chunk-relative line numbers onto the document.

    `replace` rather than a field-by-field rebuild. Listing every field here
    means a field added to `Mention` is silently dropped from every mention in
    a literate document until somebody remembers this function, and there is
    no error when they do not: the data is simply null for `.Rmd` and
    `.ipynb` and correct everywhere else.
    """
    return [
        replace(
            m,
            line=m.line + offset - 1,
            chunk_label=label,
            language=language,
        )
        for m in mentions
    ]


def extract_rmarkdown(source: str) -> ExtractResult:
    """Extract from a knitr document by routing each chunk to its extractor."""
    mentions: list[Mention] = []
    engines: set[str] = set()

    for engine, code, first_line, label in split_chunks(source):
        engines.add(str(engine))
        if engine is Language.R:
            result = r.extract(code)
        elif engine is Language.PYTHON:
            result = python_.extract(code)
        elif engine is Language.STATA:
            result = stata.extract(code)
        else:
            continue
        mentions.extend(_shift(result.mentions, first_line, label, engine))

    # Inline `r pkg::fun(x)` spans load packages just as chunk code does.
    for number, line in enumerate(source.splitlines(), 1):
        for expression in _INLINE_R.findall(line):
            inline = r.extract(expression)
            mentions.extend(_shift(inline.mentions, number, "inline", Language.R))

    mentions.sort(key=lambda m: m.line)
    return ExtractResult(
        mentions=mentions,
        report=ParseReport(
            status=ParseStatus.OK,
            language=Language.RMARKDOWN,
            bytes_total=len(source),
            detail=f"engines: {','.join(sorted(engines))}" if engines else "no chunks",
        ),
    )


def _notebook_language(payload: dict) -> Language:
    """Kernel language from the metadata. Not every notebook is Python."""
    metadata = payload.get("metadata") or {}
    name = (
        (metadata.get("kernelspec") or {}).get("language")
        or (metadata.get("language_info") or {}).get("name")
        or "python"
    )
    return _ENGINE_LANGUAGE.get(str(name).lower(), Language.PYTHON)


def _cell_source(cell: dict) -> str:
    source = cell.get("source") or cell.get("input") or ""
    return "".join(source) if isinstance(source, list) else str(source)


def extract_notebook(source: str) -> ExtractResult:
    """Extract from a Jupyter notebook, cell by cell."""
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        return ExtractResult(
            report=ParseReport(
                status=ParseStatus.SYNTAX_ERROR,
                language=Language.NOTEBOOK,
                bytes_total=len(source),
                detail=f"invalid notebook JSON: {exc}",
            )
        )

    language = _notebook_language(payload)
    extractor = _CELL_EXTRACTOR.get(language)
    if extractor is None:
        return ExtractResult(
            report=ParseReport(
                status=ParseStatus.UNSUPPORTED_LANGUAGE,
                language=language,
                bytes_total=len(source),
                detail=f"no extractor for a {language} kernel",
            )
        )

    # nbformat 3 nested cells under `worksheets`; v1 read only `cells` and so
    # returned nothing for older notebooks without saying so.
    cells = payload.get("cells")
    if cells is None:
        cells = [
            cell
            for sheet in (payload.get("worksheets") or [])
            for cell in (sheet.get("cells") or [])
        ]

    mentions: list[Mention] = []
    failed_cells = 0
    for index, cell in enumerate(cells or []):
        if cell.get("cell_type") not in {"code", None}:
            continue
        code = _cell_source(cell)
        if not code.strip():
            continue

        cleaned: list[str] = []
        for line in code.splitlines():
            if install := _MAGIC_INSTALL.match(line):
                for token in install.group(1).split():
                    if token and not token.startswith("-"):
                        mentions.append(
                            Mention(
                                raw_name=token.split("==")[0].split(">")[0],
                                construct=Construct.SHELL_INSTALL,
                                line=index,
                                col=0,
                                byte_start=0,
                                byte_end=0,
                                snippet=line.strip()[:300],
                                cell_index=index,
                                language=language,
                            )
                        )
                continue
            # Magics and shell escapes are not valid source; dropping the line
            # keeps one bad line from costing the whole cell.
            cleaned.append("" if line.lstrip().startswith(("%", "!")) else line)

        body = "\n".join(cleaned)
        result = extractor(body)
        if result.report and result.report.status in {
            ParseStatus.SYNTAX_ERROR,
            ParseStatus.DECODE_ERROR,
        }:
            # Per-cell isolation: one unparseable cell must not void the
            # notebook, which is what happens when the whole file is parsed.
            failed_cells += 1
        for mention in result.mentions:
            # `replace`, for the reason given in `_shift`.
            mentions.append(replace(mention, cell_index=index, language=language))

    return ExtractResult(
        mentions=mentions,
        report=ParseReport(
            status=ParseStatus.OK_WITH_ERRORS if failed_cells else ParseStatus.OK,
            language=language,
            bytes_total=len(source),
            detail=f"{failed_cells} unparseable cells" if failed_cells else None,
        ),
    )

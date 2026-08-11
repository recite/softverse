"""Build a machine-readable Stata command -> package index.

R has CRAN and ``renv``; Python has PyPI and ``ast``. Stata has neither a
registry API nor a parser grammar, which is why v1 fell back to a hardcoded list
of twenty command names matched anywhere in the text -- a detector that reported
``is`` (88 deposits) and ``we`` (57) as packages while missing ``reghdfe``.

The index is reconstructed from SSC's own distribution metadata. Every package
on the Boston College RePEc mirror ships a ``.pkg`` manifest listing its files::

    d 'ESTOUT': module to make regression tables
    d Distribution-Date: 20260413
    d Author: Ben Jann, University of Bern
    f estout.ado
    f esttab.ado
    f eststo.ado

That yields the many-commands-per-package mapping a name-only list would miss
(``esttab`` belongs to ``estout``).

**The caveat that governs the design**: an ``f foo.ado`` line says the package
*ships a file*, not that it *exposes a command* named ``foo``. ``reghdfe`` ships
``reghdfe_p.ado`` (a predict helper) and ``estout`` ships ``_eststo.ado`` (an
internal subroutine); neither is a command a user types. So each mapping records
how it was derived -- :data:`Evidence.FILENAME` for an inference from the
manifest, :data:`Evidence.PROGRAM_DEFINE` for one confirmed by parsing the
``.ado`` -- and helpers are flagged rather than silently included. Confirming
every one of ~15,000 shipped ado files would mean 15,000 more fetches, so
confirmation is done lazily for the commands that actually appear in the corpus,
which is the only set whose classification can change a published number.

Historical validity is recorded but not solved here: ``distribution_date`` is
the mirror's current date, so a mapping is a *current* fact. Resolving 2010 code
against a 2026 index can manufacture trends, and the resolver treats
time-inconsistent mappings accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path

import httpx

from softverse.logging_setup import get_logger, stage

logger = get_logger(__name__)

SSC_MIRROR = "http://fmwww.bc.edu/repec/bocode"
LETTERS = "abcdefghijklmnopqrstuvwxyz_"


class Evidence(StrEnum):
    """How a command -> package mapping was established."""

    #: Confirmed by finding `program define <cmd>` in the shipped .ado.
    PROGRAM_DEFINE = "program_define"
    #: Inferred from an `f <cmd>.ado` line in the .pkg manifest.
    FILENAME = "filename"


#: Suffixes marking an ado as an internal helper rather than a user command.
#: `_p` is Stata's predict-helper convention, `_estat` the estat subcommand
#: hook, and so on. These ship with a package but are never typed by a user.
_HELPER_SUFFIXES = (
    "_p",
    "_estat",
    "_footnote",
    "_ll",
    "_lf",
    "_gf",
    "_d2",
    "_predict",
    "_parse",
    "_prog",
    "_sub",
    "_util",
    "_utils",
    "_check",
    "_setup",
    "_header",
    "_display",
    "_post",
    "_common",
    "_internal",
    "_helper",
)

#: `program define` in an .ado, allowing Stata's abbreviations (`pr`, `prog`,
#: `progr`...) and the optional `define`/`def` keyword.
_PROGRAM_DEFINE = re.compile(
    r"^\s*(?:cap(?:t|tu|tur|ture)?\s+)?"
    r"pr(?:o|og|ogr|ogra|ogram)?\s+"
    r"(?:def(?:i|in|ine)?\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


@dataclass
class StataPackage:
    """One SSC or Stata Journal package, as described by its manifest."""

    package: str
    source: str = "ssc"
    title: str | None = None
    author: str | None = None
    distribution_date: date | None = None
    #: Every file the package ships, as written in the manifest.
    files: list[str] = field(default_factory=list)

    @property
    def ado_files(self) -> list[str]:
        """Basenames of shipped ``.ado`` files.

        Manifest paths can be relative (``f ../_/_eststo.ado``), so the
        directory part is discarded.
        """
        out = []
        for entry in self.files:
            name = entry.replace("\\", "/").rsplit("/", 1)[-1]
            if name.lower().endswith(".ado"):
                out.append(name[:-4])
        return out


def is_helper(command: str, package: str, siblings: set[str]) -> bool:
    """Whether ``command`` looks like an internal helper, not a user command.

    Three conventions, in order of reliability:

    1. A leading underscore is Stata's internal-subroutine convention
       (``_eststo``, ``_gcorr``).
    2. A known helper suffix on a stem that is itself a command in the same
       package (``reghdfe_p`` beside ``reghdfe``).
    3. Any name that extends a sibling command with ``_<suffix>`` where the
       suffix is a known helper marker.
    """
    if command.startswith("_"):
        return True
    for suffix in _HELPER_SUFFIXES:
        if command.endswith(suffix):
            stem = command[: -len(suffix)]
            if stem and (stem in siblings or stem == package):
                return True
    return False


def parse_pkg(text: str, package: str, source: str = "ssc") -> StataPackage:
    """Parse the contents of a ``.pkg`` manifest."""
    pkg = StataPackage(package=package, source=source)
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("f ") or line.startswith("F "):
            entry = line[2:].strip()
            if entry:
                pkg.files.append(entry)
        elif line.startswith("d "):
            body = line[2:].strip()
            if body.startswith("Distribution-Date:"):
                # Hand-maintained field: real manifests carry impossible dates
                # (month 00, day 32). One bad date must not abort the crawl.
                raw = body.split(":", 1)[1].strip()
                if len(raw) == 8 and raw.isdigit():
                    try:
                        pkg.distribution_date = date(
                            int(raw[:4]), int(raw[4:6]), int(raw[6:])
                        )
                    except ValueError:
                        logger.debug(
                            "unparseable distribution date",
                            extra={"package": package, "raw": raw},
                        )
            elif body.startswith("Author:") and pkg.author is None:
                pkg.author = body.split(":", 1)[1].strip() or None
            elif pkg.title is None and body.startswith("'"):
                pkg.title = body
    return pkg


def commands_for(pkg: StataPackage) -> list[tuple[str, bool]]:
    """Candidate ``(command, is_helper)`` pairs implied by a manifest."""
    ados = pkg.ado_files
    siblings = set(ados)
    return [(name, is_helper(name, pkg.package, siblings)) for name in ados]


def defined_programs(ado_text: str) -> set[str]:
    """Every program name defined in an ``.ado``.

    Note this includes internal subroutines: ``reghdfe_header.ado`` defines
    ``Ftest``, ``Chi2test`` and ``HeaderDisplay`` alongside its namesake, and
    ``esttab.ado`` defines a dozen ``MakeTeX*`` helpers. So the raw set is *not*
    a list of user commands -- use :func:`confirms_namesake` for that.
    """
    return {m.group(1) for m in _PROGRAM_DEFINE.finditer(ado_text)}


def confirms_namesake(ado_text: str, name: str) -> bool:
    """Whether ``<name>.ado`` actually defines a program called ``name``.

    This is the only confirmation that means anything, and it follows from
    Stata's ado-path semantics: typing ``foo`` executes ``foo.ado``, which must
    define a program ``foo``. Programs defined in the same file under other
    names are internal subroutines a user never invokes.

    Confirming the namesake therefore upgrades a filename inference to
    :data:`Evidence.PROGRAM_DEFINE`; its absence means the file is documentation,
    a renamed shim, or a pure subroutine library.
    """
    return name in defined_programs(ado_text)


class SSCClient:
    """Polite HTTP client for the Boston College RePEc mirror."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "softverse (research; github.com/recite/softverse)"},
        )

    def __enter__(self) -> SSCClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def list_packages(self, letter: str) -> list[str]:
        """Package names with a ``.pkg`` manifest under one letter directory."""
        url = f"{SSC_MIRROR}/{letter}/"
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "letter listing failed", extra={"letter": letter, "err": str(exc)}
            )
            return []
        return sorted(set(re.findall(r'href="([A-Za-z0-9_.\-]+)\.pkg"', response.text)))

    def fetch_pkg(self, letter: str, package: str) -> StataPackage | None:
        url = f"{SSC_MIRROR}/{letter}/{package}.pkg"
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "pkg fetch failed", extra={"package": package, "err": str(exc)}
            )
            return None
        return parse_pkg(response.text, package)

    def fetch_ado(self, letter: str, name: str) -> str | None:
        url = f"{SSC_MIRROR}/{letter}/{name}.ado"
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return response.text


def build_index(
    letters: str = LETTERS,
    client: SSCClient | None = None,
) -> list[dict]:
    """Crawl the mirror and return rows for the ``stata_command_index`` table.

    Args:
        letters: Which letter directories to crawl. Narrow it for testing.
        client: An open :class:`SSCClient`, or ``None`` to create one.

    Returns:
        Rows matching :data:`softverse.model.schemas.STATA_COMMAND_INDEX`.
    """
    today = datetime.now(tz=UTC).date()
    rows: list[dict] = []
    owned = SSCClient() if client is None else client
    try:
        with stage("stata-index", logger) as stats:
            for letter in letters:
                names = owned.list_packages(letter)
                stats.incr("packages_listed", len(names))
                for name in names:
                    pkg = owned.fetch_pkg(letter, name)
                    if pkg is None:
                        stats.incr("packages_failed")
                        continue
                    stats.incr("packages_parsed")
                    commands = commands_for(pkg)
                    if not commands:
                        stats.incr("packages_without_ado")
                    for command, helper in commands:
                        stats.incr("helpers" if helper else "commands")
                        rows.append(
                            {
                                "command": command,
                                "package": pkg.package,
                                "source": pkg.source,
                                "evidence": str(Evidence.FILENAME),
                                "is_helper": helper,
                                "author": pkg.author,
                                "distribution_date": pkg.distribution_date,
                                "first_seen_date": pkg.distribution_date,
                                "snapshot_date": today,
                            }
                        )
                logger.info(
                    "letter done",
                    extra={"letter": letter, "packages": len(names), "rows": len(rows)},
                )
    finally:
        if client is None:
            owned.__exit__()
    return rows


def ambiguous_commands(rows: list[dict]) -> dict[str, list[str]]:
    """Commands claimed by more than one package.

    Returned rather than resolved: forcing a single winner would bury
    classification error that the paper has to report.
    """
    by_command: dict[str, set[str]] = {}
    for row in rows:
        if not row["is_helper"]:
            by_command.setdefault(row["command"], set()).add(row["package"])
    return {c: sorted(p) for c, p in by_command.items() if len(p) > 1}


def write_index(rows: list[dict], directory: Path) -> Path:
    """Validate and write the index as Parquet."""
    from softverse.model.io import write_table

    return write_table(rows, "stata_command_index", directory)

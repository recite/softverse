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

SSC is one of three archives. The *Stata Journal* and its predecessor the
*Stata Technical Bulletin* publish the software that accompanies their articles
in the same ``stata.toc`` / ``.pkg`` format, one directory per issue, and
``net install`` reads all three alike. Indexing SSC alone reported ``renvars``
(STB-60, 105 deposits), ``xtserial`` (SJ 3-2) and ``dropmiss`` as belonging to
no archive, when each has been in a formal, citable one for twenty years.

The journal archives need one rule SSC does not. An article's package bundles
whatever its example needs to run, so ``st0357`` -- a Cox calibration tool --
ships a copy of ``grc1leg.ado``. Crediting ``grc1leg``'s 486 deposits to it
would be the filename caveat at its worst. A journal package is therefore
credited with a command only when it also ships that command's help file,
which a package does for what it publishes and not for what it borrows.

The fourth tier is everything else `net install` reaches: an author's own site
serving the same ``.pkg`` format, which is where ``grc1leg`` (a StataCorp
developer's page) and ``polychoric`` live. No list of such sites exists, so they
are discovered from the corpus -- every ``net install x, from(URL)`` a deposit
contains names one -- and that is a limit worth stating: a site no deposit
names is not found. The documented-command rule applies to them too.

Historical validity is recorded but not solved here: ``distribution_date`` is
the mirror's current date, so a mapping is a *current* fact. Resolving 2010 code
against a 2026 index can manufacture trends, and the resolver treats
time-inconsistent mappings accordingly.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import httpx

from softverse.logging_setup import get_logger, stage

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = get_logger(__name__)

SSC_MIRROR = "http://fmwww.bc.edu/repec/bocode"
LETTERS = "abcdefghijklmnopqrstuvwxyz_"

#: Archives laid out as a top-level ``stata.toc`` of issues, each with its own
#: ``stata.toc`` of packages.
ISSUE_ARCHIVES = {
    "stata_journal": "https://www.stata-journal.com/software",
    "stb": "https://www.stata.com/stb",
}

#: `st0085_2` is the second update of `st0085`. The update replaces the
#: original, so both name one package.
#: Hosts the archive crawls already cover; a `from()` pointing at one is not a
#: net site.
_ARCHIVE_URLS = ("fmwww.bc.edu", "stata-journal.com", "stata.com/stb")

_UPDATE_SUFFIX = re.compile(r"_\d+$")
_HELP_SUFFIXES = (".hlp", ".sthlp", ".ihlp")


class Evidence(StrEnum):
    """How a command -> package mapping was established."""

    #: Confirmed by finding `program define <cmd>` in the shipped .ado.
    PROGRAM_DEFINE = "program_define"
    #: Inferred from an `f <cmd>.ado` line in the .pkg manifest.
    FILENAME = "filename"
    #: No command at all: the package ships Mata, schemes or data and no
    #: `.ado`. Recorded so `ssc install moremata` names something known.
    PACKAGE_ONLY = "package_only"


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
    #: The journal issue a package appeared in; None for SSC.
    issue: str | None = None

    @property
    def _basenames(self) -> list[str]:
        # Manifest paths can be relative (`f ../_/_eststo.ado`) or carry the
        # package directory (`f dm88/renvars.ado`); only the file matters.
        return [e.replace("\\", "/").rsplit("/", 1)[-1] for e in self.files]

    @property
    def ado_files(self) -> list[str]:
        """Basenames of shipped ``.ado`` files, extension removed."""
        return [n[:-4] for n in self._basenames if n.lower().endswith(".ado")]

    @property
    def documented(self) -> frozenset[str]:
        """Lowercased names the package ships a help file for."""
        return frozenset(
            n.rsplit(".", 1)[0].lower()
            for n in self._basenames
            if n.lower().endswith(_HELP_SUFFIXES)
        )

    @property
    def canonical(self) -> str:
        """The package name with a journal update suffix removed."""
        if self.source not in ISSUE_ARCHIVES:
            return self.package
        return _UPDATE_SUFFIX.sub("", self.package)


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
        if line[:2] in {"f ", "F ", "g ", "G "}:
            # `f file`, `f PLATFORM file`, and `g PLATFORM file [installed-as]`.
            # Reading everything after the letter as the filename made
            # `f WIN64 usespss.ado` a file called `WIN64 usespss.ado`, so the
            # package shipped no command anyone could type; platform-specific
            # lines are how packages with compiled plugins list their files.
            parts = line[2:].split()
            if parts:
                pkg.files.append(parts[-1])
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


@dataclass(frozen=True)
class Manifest:
    """One ``.pkg`` manifest as fetched, before anything is inferred from it.

    Kept verbatim in the snapshot so that a change of rule -- which files count
    as commands, which as documentation -- re-derives the index from what the
    archives served on the day, not from what they serve now.
    """

    source: str
    package: str
    url: str
    text: str
    issue: str | None = None


def _toc_entries(toc: str, kind: str) -> list[str]:
    """Names on the ``t`` (directory) or ``p`` (package) lines of a ``stata.toc``."""
    found = re.findall(rf"^{kind}[ \t]+(\S+)", toc, re.MULTILINE)
    # `t ..` links back up, and `p -` continues the previous description.
    return [name for name in found if name not in {"..", "-"}]


class ArchiveClient:
    """Polite HTTP client for the SSC mirror and the two journal archives."""

    def __init__(self, timeout: float = 30.0) -> None:
        """Open a client."""
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "softverse (research; github.com/recite/softverse)"},
        )

    def __enter__(self) -> ArchiveClient:
        """Enter the context manager, returning this client."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection on the way out."""
        self._client.close()

    def get(self, url: str) -> str | None:
        """The body at ``url``, or None when it cannot be had."""
        for attempt in range(3):
            try:
                response = self._client.get(url)
                if response.status_code == httpx.codes.NOT_FOUND:
                    break
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.debug(
                    "retrying", extra={"url": url, "attempt": attempt, "err": str(exc)}
                )
                continue
            return response.text
        logger.warning("fetch failed", extra={"url": url})
        return None

    def ssc_urls(self, letters: str = LETTERS) -> list[tuple[str, None, str, str]]:
        """``(source, issue, package, url)`` for every SSC package."""
        out = []
        for letter in letters:
            listing = self.get(f"{SSC_MIRROR}/{letter}/") or ""
            names = sorted(set(re.findall(r'href="([A-Za-z0-9_.\-]+)\.pkg"', listing)))
            out += [("ssc", None, n, f"{SSC_MIRROR}/{letter}/{n}.pkg") for n in names]
        return out

    def issue_urls(self, source: str) -> list[tuple[str, str, str, str]]:
        """``(source, issue, package, url)`` for every package in a journal archive."""
        root = ISSUE_ARCHIVES[source]
        out = []
        for issue in _toc_entries(self.get(f"{root}/stata.toc") or "", "t"):
            toc = self.get(f"{root}/{issue}/stata.toc") or ""
            out += [
                (source, issue, name, f"{root}/{issue}/{name}.pkg")
                for name in _toc_entries(toc, "p")
            ]
        return out


def fetch_manifests(
    client: ArchiveClient | None = None,
    sources: Iterable[str] = ("ssc", *ISSUE_ARCHIVES),
    letters: str = LETTERS,
) -> list[Manifest]:
    """Every ``.pkg`` manifest the named archives serve.

    Args:
        client: An open client, or None to create one.
        sources: Which archives to crawl.
        letters: Which SSC letter directories. Narrow it for testing.

    Returns:
        The manifests that could be fetched, in archive order.
    """
    owned = ArchiveClient() if client is None else client
    try:
        with stage("stata-index", logger) as stats:
            targets: list[tuple[str, str | None, str, str]] = []
            for source in sources:
                found = (
                    owned.ssc_urls(letters)
                    if source == "ssc"
                    else owned.issue_urls(source)
                )
                stats.incr(f"{source}_listed", len(found))
                targets += found
            with ThreadPoolExecutor(max_workers=4) as pool:
                texts = list(pool.map(lambda t: owned.get(t[3]), targets))
            stats.incr("manifests_failed", sum(t is None for t in texts))
            return [
                Manifest(source, package, url, text, issue)
                for (source, issue, package, url), text in zip(
                    targets, texts, strict=True
                )
                if text is not None
            ]
    finally:
        if client is None:
            owned.__exit__()


def fetch_net_manifests(
    installs: Iterable[tuple[str, str]],
    client: ArchiveClient | None = None,
    sites: Iterable[str] = (),
) -> tuple[list[Manifest], list[str]]:
    """Manifests from the sites the corpus's own `net install` lines name.

    Args:
        installs: ``(package, from-URL)`` pairs, as the tally records them.
        client: An open client, or None to create one.
        sites: Further sites to crawl whole, one level of sub-directories
            deep: the ones found by looking up what the index could not place.

    Returns:
        The manifests fetched, and the URLs that no longer serve one. Sites
        die; a dead one is reported rather than dropped, because "was on a
        personal page that is gone" is itself what the index is measuring.
    """
    owned = ArchiveClient(timeout=15.0) if client is None else client
    named = {
        (package, base.rstrip("/"))
        for package, base in installs
        # A macro in the URL (`.../$version/src`) is not a dead site; it is
        # an address only the author's session could complete.
        if "://" in base
        and not re.search(r"[$`]", base)
        and not any(known in base for known in _ARCHIVE_URLS)
    }
    try:
        # A site the corpus installs one package from usually serves others,
        # listed in its own `stata.toc`. Crawling the site whole is what finds
        # the commands deposits use without an install line beside them --
        # which is most of them, since `net install` is run once, by hand.
        bases = {base for _, base in named}
        for site in (s.rstrip("/") for s in sites):
            toc = owned.get(f"{site}/stata.toc") or ""
            bases |= {site, *(f"{site}/{d}" for d in _toc_entries(toc, "t"))}
        listed = {
            (package, base)
            for base in sorted(bases)
            for package in _toc_entries(owned.get(f"{base}/stata.toc") or "", "p")
        }
        targets = sorted(
            (package, f"{base}/{package}.pkg") for package, base in named | listed
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            texts = list(pool.map(lambda t: owned.get(t[1]), targets))
    finally:
        if client is None:
            owned.__exit__()
    found, dead = [], []
    named_urls = {f"{base}/{package}.pkg" for package, base in named}
    for (package, url), text in zip(targets, texts, strict=True):
        # A site that answers 200 with an HTML error page serves no manifest.
        if text is None or not re.search(r"^[fF] ", text, re.MULTILINE):
            # Only an address a deposit actually names is reported dead; a toc
            # entry with no manifest behind it is the site's own loose end.
            if url in named_urls:
                dead.append(url)
            continue
        host = url.split("://", 1)[1].split("/", 1)[0].removeprefix("www.")
        found.append(Manifest("net", package, url, text, host))
    return found, dead


def index_rows(
    manifests: Iterable[Manifest], snapshot_date: date | None = None
) -> list[dict]:
    """Rows of the ``stata_command_index`` table implied by ``manifests``.

    Args:
        manifests: As returned by :func:`fetch_manifests`.
        snapshot_date: The date to stamp; today by default.

    Returns:
        Rows matching :data:`softverse.model.schemas.STATA_COMMAND_INDEX`. A
        journal package updated across issues yields one row per command, from
        the newest issue -- the archives list newest first.
    """
    stamp = snapshot_date or datetime.now(tz=UTC).date()
    rows: dict[tuple[str, str, str], dict] = {}
    for manifest in manifests:
        pkg = parse_pkg(manifest.text, manifest.package, manifest.source)
        shared = {
            "package": pkg.canonical,
            "source": pkg.source,
            "issue": manifest.issue,
            "author": pkg.author,
            "distribution_date": pkg.distribution_date,
            "first_seen_date": pkg.distribution_date,
            "snapshot_date": stamp,
        }
        commands = commands_for(pkg)
        if not commands:
            key = (pkg.canonical.lower(), pkg.canonical, pkg.source)
            rows.setdefault(
                key,
                {
                    "command": pkg.canonical,
                    "evidence": str(Evidence.PACKAGE_ONLY),
                    "is_helper": True,
                    "is_documented": False,
                    **shared,
                },
            )
        for command, helper in commands:
            rows.setdefault(
                (command.lower(), pkg.canonical, pkg.source),
                {
                    "command": command,
                    "evidence": str(Evidence.FILENAME),
                    "is_helper": helper,
                    "is_documented": command.lower() in pkg.documented,
                    **shared,
                },
            )
    return list(rows.values())


#: The order resolution consults the archives in.
TIERS = ("ssc", *ISSUE_ARCHIVES, "net")


def credited(row: dict) -> bool:
    """Whether resolution may credit ``row``'s package with its command.

    Never a helper; and outside SSC, only a command the package documents.
    """
    return not row["is_helper"] and (row["source"] == "ssc" or row["is_documented"])


def ambiguous_commands(rows: list[dict]) -> dict[str, list[str]]:
    """Commands claimed by more than one package in the archive that wins them.

    Returned rather than resolved: forcing a single winner would bury
    classification error that the paper has to report. A command on SSC and in
    the Journal is not ambiguous -- it is one package mirrored, and SSC is
    consulted first -- so the comparison is within the first archive to list
    it, which is the comparison resolution makes.
    """
    by_command: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        if credited(row):
            tiers = by_command.setdefault(row["command"].lower(), {})
            tiers.setdefault(row["source"], set()).add(row["package"])
    out = {}
    for command, tiers in by_command.items():
        winner = next(tiers[t] for t in TIERS if t in tiers)
        if len(winner) > 1:
            out[command] = sorted(winner)
    return out


def write_snapshot(
    manifests: list[Manifest], root: Path, unreachable: list[str] | None = None
) -> Path:
    """Write a dated, digest-stamped snapshot under ``root``.

    The manifests are written before the index is derived from them, so a
    failure deriving or validating the index cannot cost the crawl that fed it.

    Args:
        manifests: The raw manifests.
        root: ``registries/snapshots/stata_index``.
        unreachable: Net-site URLs the corpus names that served no manifest.

    Returns:
        The snapshot directory.
    """
    from softverse.model.io import write_table

    stamp = datetime.now(tz=UTC)
    directory = root / stamp.date().isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "manifests.jsonl.gz", "wt", encoding="utf-8") as handle:
        for manifest in manifests:
            handle.write(json.dumps(manifest.__dict__) + "\n")
    rows = index_rows(manifests, stamp.date())
    index = write_table(rows, "stata_command_index", directory)
    by_source: dict[str, int] = {}
    for manifest in manifests:
        by_source[manifest.source] = by_source.get(manifest.source, 0) + 1
    (directory / "source.json").write_text(
        json.dumps(
            {
                "registry": "stata_index",
                "url": [SSC_MIRROR, *ISSUE_ARCHIVES.values()],
                "fetched_at": stamp.isoformat(),
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "n_rows": len(rows),
                "n_manifests": by_source,
                "net_sites_unreachable": sorted(unreachable or []),
            },
            indent=2,
        )
        + "\n"
    )
    return directory


def read_manifests(directory: Path) -> list[Manifest]:
    """The raw manifests of a snapshot, to re-derive its index under new rules."""
    with gzip.open(directory / "manifests.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [Manifest(**json.loads(line)) for line in handle]

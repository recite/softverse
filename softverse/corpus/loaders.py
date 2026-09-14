"""Collected files on disk, as one corpus.

Each loader returns `CorpusFile`s for one source. The tally concatenates them
and runs `build()` once, so both halves are hashed, classified, parsed and
resolved by the same code with the same registry pins. They were tallied
separately for a while, which meant the published per-package counts covered
the Zenodo half alone: economics, one seventh of the deposits we hold.

Both sources are now collected by the same rules -- every code, notebook and
knitr file and every manifest, with the deposit's directories -- so the file
types the two halves see no longer differ. `dataverse_legacy_corpus` reads the
January 2024 scrape, which kept three extensions and no directories; it is not
part of `full_corpus()` and stays only so the frame collector can compare
per-journal counts against it.
"""

from __future__ import annotations

import csv
from pathlib import Path

from softverse.build.pipeline import CorpusFile
from softverse.config import PATHS
from softverse.logging_setup import get_logger

logger = get_logger(__name__)

ZENODO_ROOT = PATHS.root / "corpus" / "zenodo"
DATAVERSE_ROOT = PATHS.root / "corpus" / "dataverse_legacy"
DATAVERSE_NEW_ROOT = PATHS.root / "corpus" / "dataverse"

#: Marks every row with the corpus it came from, so a downstream query cannot
#: forget which one it is looking at. A footnote can be missed; a column in a
#: `GROUP BY` cannot.
ZENODO = "zenodo"
DATAVERSE_LEGACY = "dataverse_legacy"
DATAVERSE = "dataverse"


def _zenodo_deposits() -> dict[str, dict]:
    """Community and year per record, from `corpus/zenodo/deposits.csv`.

    Written by `scripts/collect_zenodo.py`, which reads both off every record
    it harvests. Before that file existed the values were computed, printed
    and dropped, and this loader had to hardcode `collection_id="zenodo"`.
    """
    path = ZENODO_ROOT / "deposits.csv"
    if not path.exists():
        logger.warning(
            "no zenodo deposits.csv; run scripts/collect_zenodo.py "
            "--metadata-only to recover community and year"
        )
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["record_id"]: row for row in csv.DictReader(handle)}


def zenodo_corpus(limit: int | None = None) -> list[CorpusFile]:
    """Collected Zenodo files, with their community and year."""
    root = ZENODO_ROOT / "files"
    if not root.exists():
        return []
    deposits = _zenodo_deposits()

    out: list[CorpusFile] = []
    missing: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        record_id = path.relative_to(root).parts[0]
        meta = deposits.get(record_id)
        if meta is None:
            missing.add(record_id)
        year = (meta or {}).get("deposit_year") or ""
        out.append(
            CorpusFile(
                path=path,
                dataset_doi=f"zenodo:{record_id}",
                collection_id=(meta or {}).get("collection_id") or ZENODO,
                source=ZENODO,
                relative_path=str(path.relative_to(root / record_id)),
                deposit_year=int(year) if str(year).isdigit() else None,
            )
        )
        if limit and len(out) >= limit:
            break

    if missing:
        # Reported, never absorbed. A deposit whose metadata we cannot find
        # still counts, it just counts in the `zenodo` bucket with no year.
        logger.warning("deposits with no metadata row", extra={"n": len(missing)})
    return out


def dataverse_metadata() -> dict[tuple[str, str], dict]:
    """(journal, deposit id) -> its Dataverse metadata row.

    Keyed on the trailing segment of `identifier` (`DVN/00IT1L` -> `00IT1L`),
    which is what the scrape used for its directory names. The source is the
    tarball unpacked under `corpus/`, not `data/datasets/`: that directory
    holds 74 of the 150 journal files and is missing `restat`, the largest
    journal in the scrape. Joining against it matched 88.5% of deposits;
    against the tarball, 100%.
    """
    root = DATAVERSE_ROOT / "metadata"
    out: dict[tuple[str, str], dict] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*_datasets.csv")):
        if ".ipynb_checkpoints" in str(path):
            continue
        journal = path.stem.replace("_datasets", "")
        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                identifier = (row.get("identifier") or "").strip()
                if identifier:
                    out[(journal, identifier.rsplit("/", 1)[-1])] = row
    return out


def dataverse_legacy_corpus(limit: int | None = None) -> list[CorpusFile]:
    """The 2024 Harvard Dataverse scrape, unpacked by the ingest script."""
    root = DATAVERSE_ROOT / "files"
    if not root.exists():
        return []
    meta = dataverse_metadata()

    out: list[CorpusFile] = []
    unmatched: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 3 or not parts[0].endswith("_datasets_files"):
            continue
        journal = parts[0].replace("_datasets_files", "")
        deposit = parts[1]
        row = meta.get((journal, deposit))
        if row is None:
            unmatched[journal] = unmatched.get(journal, 0) + 1
            continue
        published = (row.get("publicationDate") or "")[:4]
        out.append(
            CorpusFile(
                path=path,
                # The real DOI, so this corpus is joinable to anything else
                # that speaks Dataverse, and so it cannot collide with a
                # `zenodo:` identifier when the two are pooled.
                dataset_doi=f"doi:10.7910/DVN/{deposit}",
                collection_id=journal,
                source=DATAVERSE_LEGACY,
                # Flattened, and recorded as such: this is the filename with
                # no directory because the scrape kept no directory.
                relative_path="/".join(parts[2:]),
                deposit_year=int(published) if published.isdigit() else None,
            )
        )
        if limit and len(out) >= limit:
            break

    if unmatched:
        logger.warning("files with no deposit metadata", extra=unmatched)
    return out


def _dataverse_frame() -> dict[str, dict]:
    """Deposit id -> its row in `data/frame/dataverse_deposits.csv`.

    Keyed on the identifier's last segment (`DVN/00IT1L` -> `00IT1L`), which is
    the collector's directory name for the deposit.
    """
    path = PATHS.frame / "dataverse_deposits.csv"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = (row.get("identifier") or "").strip()
            if identifier:
                out[identifier.rsplit("/", 1)[-1]] = row
    return out


def dataverse_corpus(limit: int | None = None) -> list[CorpusFile]:
    """The 2026 Harvard Dataverse collection, every file type, paths intact.

    Replaces the 2024 scrape in the tally. That scrape kept three extensions
    and no directories; this one keeps notebooks, knitr documents, `.ado`
    files and manifests, with the deposit's own folders, and code recovered
    from archives sits under `_archives/<archive>_extracted/`. Two things on
    disk are not corpus: an archive kept because it would not extract, which
    sits directly in `_archives/`, and a `.part` left by an interrupted
    download.
    """
    root = DATAVERSE_NEW_ROOT / "files"
    if not root.exists():
        return []
    frame = _dataverse_frame()

    out: list[CorpusFile] = []
    unmatched = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".part":
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 2 or (parts[1] == "_archives" and len(parts) == 3):
            continue
        deposit = parts[0]
        row = frame.get(deposit)
        if row is None:
            unmatched += 1
            continue
        published = (row.get("publication_date") or "")[:4]
        out.append(
            CorpusFile(
                path=path,
                dataset_doi=f"doi:10.7910/DVN/{deposit}",
                collection_id=row["collection_id"],
                source=DATAVERSE,
                relative_path="/".join(parts[1:]),
                deposit_year=int(published) if published.isdigit() else None,
            )
        )
        if limit and len(out) >= limit:
            break

    if unmatched:
        logger.warning("dataverse files with no frame row", extra={"n": unmatched})
    return out


def full_corpus(limit: int | None = None) -> list[CorpusFile]:
    """Every collected file, from every source."""
    files = zenodo_corpus(limit) + dataverse_corpus(limit)
    _assert_disjoint(files)
    return files


def _assert_disjoint(files: list[CorpusFile]) -> None:
    """No deposit may appear under two sources.

    True by construction, since one namespace is `zenodo:NNN` and the other
    `doi:10.7910/DVN/...`, which is exactly why it is worth asserting: a
    collision would double-count a deposit in the pooled numerator while
    leaving the denominator alone, and nothing downstream would notice.
    """
    by_source: dict[str, set[str]] = {}
    for item in files:
        by_source.setdefault(item.source, set()).add(item.dataset_doi)
    sources = sorted(by_source)
    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            shared = by_source[left] & by_source[right]
            if shared:
                raise ValueError(
                    f"{len(shared)} deposits appear in both {left} and "
                    f"{right}, e.g. {sorted(shared)[:3]}"
                )


def deposit_directories(files: list[CorpusFile] | None = None) -> dict[str, Path]:
    """Deposit DOI to the directory holding its files.

    Derived from each file's own `relative_path` rather than from the DOI's
    text. A Zenodo DOI ends in the directory name, so splitting the string
    worked there and produced nonsense for Dataverse, whose DOI is
    `doi:10.7910/DVN/00IT1L` and whose files sit under
    `<journal>_datasets_files/00IT1L/`. Anything that walks a deposit's files
    on disk needs this, and getting it wrong looks like a deposit that is
    simply missing.
    """
    out: dict[str, Path] = {}
    for item in files if files is not None else full_corpus():
        depth = len(Path(item.relative_path).parts)
        out.setdefault(item.dataset_doi, item.path.parents[depth - 1])
    return out


def paths_for(source: str) -> Path:
    """Where a source's files live, for anything that needs to re-read them."""
    return {
        ZENODO: ZENODO_ROOT / "files",
        DATAVERSE: DATAVERSE_NEW_ROOT / "files",
        DATAVERSE_LEGACY: DATAVERSE_ROOT / "files",
    }[source]

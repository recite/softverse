"""Ingest the 2024 Harvard Dataverse scrape.

    uv run python scripts_ingest_dataverse_legacy.py [--limit N]

Reads `data/script_files.tar.gz` (46,735 files, 7,231 deposits, 61 journals)
and `data/datasets_by_dataverse.tar.gz` (DOIs and publication dates), and
writes a tally to `build/tally_dataverse_legacy/`.

**Why this is a separate tally rather than merged into the main one.** The
scrape is *flattened*: its paths are `<journal>_datasets_files/<id>/<file>`
with no in-deposit directories, because the harvester kept only basenames.
That breaks the hygiene rules that depend on paths, and it breaks them in a
way that has now been measured rather than guessed. On the Zenodo corpus,
where the truth is known, the path rules catch 26,776 vendored files; only
**25%** of those would still be caught by the name- and hash-based rules that
survive flattening. Vendored code was separately shown to distort package
counts by 11-22%, so a merged tally would carry roughly twenty thousand
undetectable library files straight into the headline table.

So: ingest it, keep it apart, and use it for the questions it can answer --
which languages appear in which journals, how deposits distribute across
journal-years, and the political-science coverage the Zenodo frame lacks
entirely. Not for package rankings, until in-deposit paths can be recovered.

The paths are not recoverable from anything we hold: the file metadata has
only `doi, fid, fn`, and the OAI records list bare filenames. Harvard's API
returns `directoryLabel` per deposit but answers us with an AWS WAF challenge
(HTTP 202, empty body) on every `/api/` route, authenticated or not.
"""

from __future__ import annotations

import collections
import csv
import sys
import tarfile
from pathlib import Path

from softverse.build.pipeline import CorpusFile, build
from softverse.config import PATHS
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.model.io import write_table

logger = get_logger(__name__)

SCRIPTS_TAR = PATHS.root / "data" / "script_files.tar.gz"
METADATA_TAR = PATHS.root / "data" / "datasets_by_dataverse.tar.gz"
FILES_ROOT = PATHS.root / "corpus" / "dataverse_legacy" / "files"
METADATA_ROOT = PATHS.root / "corpus" / "dataverse_legacy" / "metadata"
OUT = PATHS.root / "build" / "tally_dataverse_legacy"

#: Marks every row as coming from the flattened scrape, so a downstream query
#: cannot forget which corpus it is looking at. A footnote can be missed; a
#: column in a `GROUP BY` cannot.
SOURCE = "dataverse_legacy"


def unpack(archive: Path, destination: Path) -> None:
    """Extract once; skip if it is already there."""
    if destination.exists() and any(destination.iterdir()):
        logger.info("already unpacked", extra={"path": str(destination)})
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # These are our own archives, not a stranger's, but the members still
        # get checked -- the rule is about the code path, not about trust.
        safe = [
            m
            for m in tar.getmembers()
            if m.isfile() and not m.name.startswith(("/", ".."))
        ]
        tar.extractall(destination, members=safe, filter="data")
    logger.info("unpacked", extra={"archive": archive.name, "members": len(safe)})


def deposit_metadata() -> dict[tuple[str, str], dict]:
    """(journal, deposit id) -> its Dataverse metadata row.

    Keyed on the trailing segment of `identifier` (`DVN/00IT1L` -> `00IT1L`),
    which is what the scrape used for its directory names. Note the source is
    the *tarball*, not `data/datasets/`: that directory holds 74 of the 150
    journal files and is missing `restat`, the largest journal in the scrape.
    Joining against it matched 88.5% of deposits; against the tarball, 100%.
    """
    unpack(METADATA_TAR, METADATA_ROOT)
    out: dict[tuple[str, str], dict] = {}
    for path in METADATA_ROOT.rglob("*_datasets.csv"):
        if ".ipynb_checkpoints" in str(path):
            continue
        journal = path.stem.replace("_datasets", "")
        with open(path, encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                identifier = (row.get("identifier") or "").strip()
                if identifier:
                    out[(journal, identifier.rsplit("/", 1)[-1])] = row
    return out


def corpus(limit: int | None = None) -> list[CorpusFile]:
    unpack(SCRIPTS_TAR, FILES_ROOT)
    meta = deposit_metadata()
    out: list[CorpusFile] = []
    unmatched: collections.Counter = collections.Counter()

    for path in FILES_ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(FILES_ROOT).parts
        if len(parts) < 3 or not parts[0].endswith("_datasets_files"):
            continue
        journal = parts[0].replace("_datasets_files", "")
        deposit = parts[1]
        row = meta.get((journal, deposit))
        if row is None:
            unmatched[journal] += 1
            continue
        published = (row.get("publicationDate") or "")[:4]
        out.append(
            CorpusFile(
                path=path,
                # The real DOI, so this corpus is joinable to anything else
                # that speaks Dataverse.
                dataset_doi=f"doi:10.7910/DVN/{deposit}",
                collection_id=journal,
                source=SOURCE,
                # Flattened, and recorded as such: this is the filename with
                # no directory because the scrape kept no directory.
                relative_path="/".join(parts[2:]),
                deposit_year=int(published) if published.isdigit() else None,
            )
        )
        if limit and len(out) >= limit:
            break

    if unmatched:
        # Reported, never absorbed. A deposit we cannot place is a gap in the
        # frame, not a rounding error.
        logger.warning("files with no deposit metadata", extra=dict(unmatched))
    return out


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="dataverse-legacy")
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )

    files = corpus(limit)
    if not files:
        print("nothing to ingest")
        return 1
    print(f"{len(files):,} files from {len({f.dataset_doi for f in files}):,} deposits")

    # Same registries, same pins, same builtin list as the Zenodo tally --
    # the corpora must differ only in what they contain, not in how they were
    # resolved, or nothing can be compared between them.
    from scripts_build_tally import load_registry

    registry, shipped = load_registry()

    with stage("dataverse-legacy-build", logger):
        result = build(
            files, registry, ssc_shipped=shipped, registry_lock_id=registry.lock_id
        )

    OUT.mkdir(parents=True, exist_ok=True)
    write_table(result.files, "files", OUT)
    write_table(result.mentions, "mentions", OUT)

    by_journal: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for row in result.files:
        if row["in_analysis_set"] and row["language"]:
            by_journal[row["collection_id"]][row["language"]] += 1

    print(f"\nfiles     : {len(result.files):,}")
    print(f"mentions  : {len(result.mentions):,}")
    print(f"unresolved: {len(result.unknown):,} distinct names")
    print(f"\nwritten to {OUT}")
    print("\nNOTE: flattened corpus -- package rankings from it are not")
    print("comparable to the Zenodo tally. See this script's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unpack the 2024 Harvard Dataverse scrape into the corpus.

    uv run python scripts_ingest_dataverse_legacy.py

Extracts `data/script_files.tar.gz` (46,735 files, 7,231 deposits, 61
journals) and `data/datasets_by_dataverse.tar.gz` (DOIs and publication
dates) into `corpus/dataverse_legacy/`. That is all it does.

It used to build its own tally into `build/tally_dataverse_legacy/`, which is
why the published per-package counts were Zenodo-only for so long: two
scripts, two output directories, and only one of them aggregating. Corpus
assembly now lives in `softverse/corpus/loaders.py` and aggregation in
`scripts_build_tally.py`, so both halves are hashed, classified and resolved
by one pass over one corpus. The evidence for why the flattened paths are
safe to pool moved to the loader, which is where the question next gets
asked.

Why the paths are unavailable at all, recorded so nobody re-derives it: the
file metadata we hold has only `doi, fid, fn`; the OAI records list bare
filenames; and every route carrying `directoryLabel` sits behind an AWS WAF
challenge (HTTP 202, empty body) that answers this client identically with or
without an API token. The challenge is a JavaScript capability test rather
than a human one, so a real browser passes it, but driving one 7,231 times to
route around a bot control is a decision for the repository's users to make
deliberately, and here there is nothing on the other side of it.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from softverse.config import PATHS
from softverse.corpus.loaders import dataverse_legacy_corpus
from softverse.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

SCRIPTS_TAR = PATHS.root / "data" / "script_files.tar.gz"
METADATA_TAR = PATHS.root / "data" / "datasets_by_dataverse.tar.gz"
FILES_ROOT = PATHS.root / "corpus" / "dataverse_legacy" / "files"
METADATA_ROOT = PATHS.root / "corpus" / "dataverse_legacy" / "metadata"


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


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="dataverse-legacy")
    for archive, destination in (
        (SCRIPTS_TAR, FILES_ROOT),
        (METADATA_TAR, METADATA_ROOT),
    ):
        if not archive.exists():
            print(f"missing {archive}")
            return 1
        unpack(archive, destination)

    files = dataverse_legacy_corpus()
    if not files:
        print("unpacked, but the loader found no placeable files")
        return 1

    deposits = {f.dataset_doi for f in files}
    journals = {f.collection_id for f in files}
    years = sorted({f.deposit_year for f in files if f.deposit_year})
    print(f"{len(files):,} files from {len(deposits):,} deposits")
    print(f"{len(journals)} journals, {years[0]} to {years[-1]}")
    print("\nrun scripts_build_tally.py to tally this alongside Zenodo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

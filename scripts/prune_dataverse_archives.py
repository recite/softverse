"""Delete the archives the Dataverse collector kept after extracting them.

    uv run python scripts/prune_dataverse_archives.py [--dry-run]

Until 615eee2 the collector left every archive on disk once its code was out.
This re-runs extraction on each kept archive, which rewrites the same members
into the same place, and deletes the archive only when that succeeds. An
archive that fails is kept, and its deposit is re-recorded as partial with the
archive counted failed rather than fetched, so the next collection run retries
it -- the state the fixed collector would have written in the first place.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from collect_dataverse import OUT

from softverse.acquire.state import Ledger
from softverse.acquire.unpack import extract
from softverse.logging_setup import setup_logging
from softverse.model.enums import CollectionState
from softverse.sources.dataverse import (
    ARCHIVE_EXTENSIONS,
    MANIFEST_FILENAMES,
    SCRIPT_EXTENSIONS,
    dataset_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    setup_logging("WARNING", stage="prune-dataverse-archives")

    ledger = Ledger(OUT / "ledger.jsonl")
    deleted = kept = 0
    freed = 0
    for record in ledger.records():
        archives_dir = dataset_dir(OUT / "files", record.dataset_doi) / "_archives"
        if not archives_dir.is_dir():
            continue
        failed = []
        for archive in sorted(p for p in archives_dir.iterdir() if p.is_file()):
            size = archive.stat().st_size
            if args.dry_run:
                deleted += 1
                freed += size
                continue
            try:
                error = extract(
                    archive,
                    archives_dir / f"{archive.name}_extracted",
                    frozenset(SCRIPT_EXTENSIONS) | frozenset(ARCHIVE_EXTENSIONS),
                    frozenset(MANIFEST_FILENAMES),
                ).error
            except Exception as exc:  # Deflate64 zips raise rather than report
                error = f"{type(exc).__name__}: {exc}"
            if error:
                failed.append((archive.name, error))
                kept += 1
                continue
            archive.unlink()
            deleted += 1
            freed += size
        if failed:
            ledger.finish(
                replace(
                    record,
                    state=CollectionState.PARTIAL.value,
                    n_fetched=record.n_fetched - len(failed),
                    n_failed=record.n_failed + len(failed),
                    error=f"extract: {failed[0][1]}",
                )
            )

    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {deleted:,} archives, {freed / 1e9:.1f} GB")
    print(f"kept {kept:,} that failed to extract; their deposits are now retryable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

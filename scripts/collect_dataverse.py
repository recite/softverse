"""Collect every deposit in the Harvard Dataverse frame.

    uv run python scripts/collect_dataverse.py [--limit N]

Reads `data/frame/dataverse_deposits.csv` and writes under
`corpus/dataverse/`: the downloaded files, the raw version metadata, the
ledger, and `files.jsonl`, one provenance row per file. `corpus/dataverse` may
be a symlink to another disk; at ~8 s and ~4 MB a deposit the full frame is
about 30 hours and tens of gigabytes.

Resumable. The ledger records each deposit when its work is done, so a rerun
skips what finished and retries what failed.

Two things the pilot did not need and a run this long does:

- **It stops when the host stops answering.** `collect()` records a failure
  and moves on, which is right for one bad deposit and wrong for a WAF block,
  where it would work through the rest of the frame recording a failure per
  deposit while still sending requests to a host that has said no. A streak of
  failures ends the run instead; the ledger keeps them retryable.
- **Provenance rows are written per batch.** `collect()` returns them in
  memory, so a crash at hour twenty would otherwise lose every md5 and file id
  gathered before it.

Deposits are shuffled with a fixed seed, so a partial run is a sample of the
frame rather than its first few journals.
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from collect_dataverse_frame import RATE_PER_S

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, dataverse_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.model.enums import CollectionState
from softverse.sources.dataverse import collect

logger = get_logger(__name__)

FRAME = PATHS.frame / "dataverse_deposits.csv"
OUT = PATHS.root / "corpus" / "dataverse"
BATCH = 10
#: Consecutive failed deposits before the run gives up. The pilot saw none in
#: thirty, so ten in a row is the host, the network or the disk, not the data.
MAX_FAILED_STREAK = 10


def frame_dois() -> list[str]:
    """Every DOI in the frame, in a fixed pseudo-random order."""
    with FRAME.open(encoding="utf-8") as handle:
        dois = [
            f"{row['protocol']}:{row['authority']}/{row['identifier']}"
            for row in csv.DictReader(handle)
        ]
    random.Random(20260913).shuffle(dois)  # noqa: S311
    return dois


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", log_dir=PATHS.logs, stage="collect-dataverse")
    ledger = Ledger(OUT / "ledger.jsonl")
    dois = frame_dois()[: args.limit]
    todo = [d for d in dois if ledger.should_process(d)]
    print(f"{len(dois):,} deposits in frame, {len(todo):,} to do")

    streak = 0
    with (
        stage("collect-dataverse", logger),
        PoliteClient(
            headers=dataverse_headers(), limiter=RateLimiter(rate_per_s=RATE_PER_S)
        ) as client,
        (OUT / "files.jsonl").open("a", encoding="utf-8") as provenance,
    ):
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            rows = collect(batch, OUT / "files", OUT / "raw", ledger, client)
            for row in rows:
                provenance.write(json.dumps(row, default=str) + "\n")
            provenance.flush()

            for doi in batch:
                record = ledger.get(doi)
                failed = record is None or record.state == CollectionState.FAILED
                streak = streak + 1 if failed else 0
            if streak >= MAX_FAILED_STREAK:
                logger.error("stopping: failure streak", extra={"streak": streak})
                print(f"stopped after {streak} consecutive failures; rerun to resume")
                return 1

    print("states:", json.dumps(ledger.summary()))
    print("totals:", json.dumps(ledger.totals()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

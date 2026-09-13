"""Recover the code from Dataverse archives the collector skipped as too large.

    uv run python scripts/recover_oversized.py --zips-only
    uv run python scripts/recover_oversized.py --download-budget-gb 50

Zips are read in place with byte-range requests, so they cost kilobytes to
megabytes each. Other archives are downloaded whole, their code kept and the
archive deleted; `--download-budget-gb` caps what one run spends on those, and
an archive that would cross the cap waits for the next run.

Each deposit's ledger record is rewritten as its archives are recovered, the
code's provenance rows go to `files.jsonl`, and one line per archive -- its
size, the bytes actually transferred, the code kept -- goes to
`oversized.jsonl`. A rerun picks up whatever is still listed as skipped.
"""

from __future__ import annotations

import argparse
import json

from collect_dataverse import OUT, RATE_PER_S

from softverse.acquire.http import RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, dataverse_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources.dataverse_oversized import apply, recover

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zips-only", action="store_true")
    parser.add_argument("--download-budget-gb", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="archives this run")
    args = parser.parse_args()

    setup_logging("INFO", log_dir=PATHS.logs, stage="recover-oversized")
    ledger = Ledger(OUT / "ledger.jsonl")
    limiter = RateLimiter(rate_per_s=RATE_PER_S)
    headers = dataverse_headers()
    budget = args.download_budget_gb * 1e9
    spent = 0
    done = 0

    with (
        stage("recover-oversized", logger),
        (OUT / "files.jsonl").open("a", encoding="utf-8") as provenance,
        (OUT / "oversized.jsonl").open("a", encoding="utf-8") as transfers,
    ):
        for record in ledger.records():
            outcomes = []
            for entry in record.skipped_archives:
                if args.limit is not None and done >= args.limit:
                    break
                is_zip = entry["filename"].lower().endswith(".zip")
                if not is_zip and (
                    args.zips_only or spent + entry["size_bytes"] > budget
                ):
                    continue
                outcome = recover(
                    record.dataset_doi,
                    entry,
                    OUT / "files",
                    headers,
                    lambda: limiter.acquire("dataverse"),
                )
                if not is_zip:
                    spent += outcome.transferred_bytes
                for row in outcome.rows:
                    provenance.write(json.dumps(row, default=str) + "\n")
                transfers.write(json.dumps(outcome.summary()) + "\n")
                provenance.flush()
                transfers.flush()
                outcomes.append(outcome)
                done += 1
                s = outcome.summary()
                print(
                    f"{s['archive_bytes'] / 1e9:7.2f} GB  {s['method']:8} "
                    f"sent {s['transferred_bytes'] / 1e3:>9,.0f} KB  "
                    f"kept {s['n_code']:>4} files {s['code_bytes'] / 1e3:>7,.0f} KB  "
                    f"{s['error'] or ''}  {record.dataset_doi} {s['filename']}"
                )
            if outcomes:
                apply(record, outcomes)
                ledger.finish(record)

    print(f"{done:,} archives; {spent / 1e9:.1f} GB spent on full downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

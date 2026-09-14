"""Recover the code from archives the collectors skipped as too large.

    uv run python scripts/recover_oversized.py --zips-only
    uv run python scripts/recover_oversized.py --download-budget-gb 50
    uv run python scripts/recover_oversized.py --source zenodo --zips-only
    uv run python scripts/recover_oversized.py --codeless-only --download-budget-gb 250

Zips are read in place with byte-range requests, so they cost kilobytes to
megabytes each. Other archives are downloaded whole, their code kept and the
archive deleted; `--download-budget-gb` caps what one run spends on those, and
an archive that would cross the cap waits for the next run. `--codeless-only`
limits downloads to deposits that have yielded no code at all, which is where
an archive is the only way to count the deposit; `--dry-run` prints what
would be fetched and stops.

Each deposit's ledger record is rewritten as its archives are recovered, and
one line per archive -- its size, the bytes actually transferred, the code
kept -- goes to `oversized.jsonl` beside the ledger. For Dataverse the code's
provenance rows also go to `files.jsonl`; the Zenodo loader reads files from
disk. A rerun picks up whatever is still listed as skipped.
"""

from __future__ import annotations

import argparse
import json

from collect_dataverse import OUT as DATAVERSE_OUT
from collect_dataverse import RATE_PER_S as DATAVERSE_RATE

from softverse.acquire.http import RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, dataverse_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources import zenodo
from softverse.sources.dataverse import dataset_dir
from softverse.sources.oversized import (
    apply,
    recover,
    storage_url,
    zenodo_content_url,
)

logger = get_logger(__name__)

ZENODO_OUT = PATHS.root / "corpus" / "zenodo"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["dataverse", "zenodo"], default="dataverse"
    )
    parser.add_argument("--zips-only", action="store_true")
    parser.add_argument("--download-budget-gb", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="archives this run")
    parser.add_argument("--codeless-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    is_zenodo = args.source == "zenodo"
    out = ZENODO_OUT if is_zenodo else DATAVERSE_OUT
    rate = zenodo.DEFAULT_RATE_PER_S if is_zenodo else DATAVERSE_RATE
    setup_logging("INFO", log_dir=PATHS.logs, stage=f"recover-oversized-{args.source}")
    ledger = Ledger(out / "ledger.jsonl")
    limiter = RateLimiter(rate_per_s=rate)
    headers = dataverse_headers()
    budget = args.download_budget_gb * 1e9
    spent = 0
    done = 0

    with (
        stage("recover-oversized", logger),
        (out / "files.jsonl").open("a", encoding="utf-8") as provenance,
        (out / "oversized.jsonl").open("a", encoding="utf-8") as transfers,
    ):
        planned = []
        for record in ledger.records():
            outcomes = []
            doi = record.dataset_doi
            if args.codeless_only and record.n_fetched > 0:
                continue
            for entry in record.skipped_archives:
                if args.limit is not None and done >= args.limit:
                    break
                if entry.get("reason") == "multi-part archive segment":
                    continue  # one part of a split zip has no directory to read
                is_zip = entry["filename"].lower().endswith(".zip")
                if not is_zip and (
                    args.zips_only or spent + entry["size_bytes"] > budget
                ):
                    continue
                if args.dry_run:
                    planned.append(entry["size_bytes"])
                    if not is_zip:
                        spent += entry["size_bytes"]
                    continue
                if is_zenodo:
                    target = out / "files" / doi.rsplit(".", 1)[-1]
                    url = zenodo_content_url(doi, entry["filename"])
                    outcome = recover(
                        doi,
                        entry,
                        target,
                        lambda _session, url=url: url,
                        lambda: limiter.acquire("zenodo"),
                        throttle_every_request=True,
                    )
                else:
                    file_id = entry["file_id"]
                    outcome = recover(
                        doi,
                        entry,
                        dataset_dir(out / "files", doi),
                        lambda session, file_id=file_id: storage_url(
                            session, file_id, headers
                        ),
                        lambda: limiter.acquire("dataverse"),
                    )
                if not is_zip:
                    spent += outcome.transferred_bytes
                if not is_zenodo:
                    for row in outcome.rows:
                        provenance.write(json.dumps(row, default=str) + "\n")
                    provenance.flush()
                transfers.write(json.dumps(outcome.summary()) + "\n")
                transfers.flush()
                outcomes.append(outcome)
                done += 1
                s = outcome.summary()
                print(
                    f"{s['archive_bytes'] / 1e9:7.2f} GB  {s['method']:8} "
                    f"sent {s['transferred_bytes'] / 1e3:>9,.0f} KB  "
                    f"kept {s['n_code']:>4} files {s['code_bytes'] / 1e3:>7,.0f} KB  "
                    f"{s['error'] or ''}  {doi} {s['filename']}"
                )
            if outcomes:
                apply(record, outcomes)
                ledger.finish(record)

    if args.dry_run:
        print(
            f"would recover {len(planned):,} archives "
            f"({sum(planned) / 1e9:.0f} GB of archives, "
            f"{spent / 1e9:.0f} GB of it downloaded whole)"
        )
        return 0
    print(f"{done:,} archives; {spent / 1e9:.1f} GB spent on full downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

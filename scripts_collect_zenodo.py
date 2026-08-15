"""Collect the Zenodo half of the frame.

Reads `data/frame/frame.csv` rather than a hardcoded list. The earlier version
used four keyword-picked communities plus three free-text queries -- a
convenience sample, which cannot carry a journal-year claim however careful the
rest of the pipeline is. Every community is re-verified at run time, because
Zenodo answers an unknown `communities=` filter by ignoring it and returning all
7.1 million records.
"""

from __future__ import annotations

import csv
import sys

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, zenodo_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources import zenodo

logger = get_logger(__name__)
ROOT = PATHS.root / "corpus" / "zenodo"


def frame_communities() -> list[str]:
    """Zenodo collection ids from the verified frame."""
    rows = csv.DictReader(open(PATHS.frame / "frame.csv", encoding="utf-8"))
    return [r["collection_id"] for r in rows if r["source"] == "zenodo"]


def main() -> int:
    # `--fresh` refetches everything; the default is incremental, which fetches
    # only deposits never successfully collected plus anything left partial.
    fresh = "--fresh" in sys.argv
    # `--metadata-only` runs the community search and writes `deposits.csv`,
    # then stops. The search is what carries each record's community and
    # publication date, so this backfills metadata for an already-collected
    # corpus without re-downloading a byte of it.
    metadata_only = "--metadata-only" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    limit = int(positional[0]) if positional else 2000
    workers = zenodo.DEFAULT_WORKERS
    for arg in sys.argv[1:]:
        if arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])
    setup_logging("INFO", log_dir=PATHS.logs, stage="zenodo")
    client = PoliteClient(
        headers=zenodo_headers(),
        limiter=RateLimiter(rate_per_s=zenodo.DEFAULT_RATE_PER_S),
    )
    ledger = Ledger(ROOT / "ledger.jsonl")
    records: dict[str, zenodo.ZenodoRecord] = {}
    declared: dict[str, int] = {}

    with stage("zenodo-frame", logger) as stats:
        for slug in frame_communities():
            try:
                declared[slug] = zenodo.verify_community(client, slug)
            except zenodo.UnknownCommunity as exc:
                logger.error(
                    "skipping community", extra={"slug": slug, "err": str(exc)}
                )
                continue
            stats.incr("communities_verified")
            before = len(records)
            for rec in zenodo.search(client, "", community=slug, authenticated=True):
                records[rec.record_id] = rec
            logger.info(
                "community harvested",
                extra={
                    "slug": slug,
                    "declared": declared[slug],
                    "harvested": len(records) - before,
                },
            )

    # Declared vs harvested is reported, never absorbed: a systematic shortfall
    # means the search is not returning the community, not that the community
    # shrank.
    print(f"\n{'community':<32}{'declared':>10}{'harvested':>11}")
    harvested_by: dict[str, int] = {}
    for rec in records.values():
        for community in rec.communities:
            harvested_by[community] = harvested_by.get(community, 0) + 1
    for slug, n in declared.items():
        got = harvested_by.get(slug, 0)
        flag = "" if got >= n * 0.9 else "   <-- shortfall"
        print(f"  {slug:<30}{n:>10}{got:>11}{flag}")

    # Written before any collection, so the metadata survives a run that is
    # interrupted, and so `--metadata-only` has somewhere to stop.
    n_deposits = zenodo.write_deposits(records.values(), ROOT / "deposits.csv")
    n_communities = len({r.communities[0] for r in records.values() if r.communities})
    n_dated = sum(1 for r in records.values() if r.year)
    print(f"\ndeposits.csv     : {n_deposits:,} rows")
    print(f"  collections    : {n_communities}")
    print(f"  with a year    : {n_dated:,}/{len(records):,}")
    if metadata_only:
        return 0

    todo = list(records.values())[:limit]
    logger.info("collecting", extra={"records": len(todo)})
    rows = zenodo.collect(
        todo, ROOT / "files", ledger, client, fresh=fresh, workers=workers
    )

    print(f"\nrecords in frame : {len(records):,}")
    print(f"collected        : {len(todo):,}")
    print(f"states           : {ledger.summary()}")
    print(f"totals           : {ledger.totals()}")
    checked = [r for r in rows if r["md5_api"]]
    bad = [r for r in checked if not r["md5_verified"]]
    print(f"md5 verified     : {len(checked) - len(bad)}/{len(checked)}")
    print(
        f"reconciling      : "
        f"{len(ledger) - len(ledger.non_reconciling())}/{len(ledger)}"
    )
    skipped = ledger.skipped_archives()
    blocked = [
        r for r in ledger._records.values() if r.n_skipped_over_cap and r.n_fetched == 0
    ]
    print(
        f"archives skipped : {len(skipped)} "
        f"({sum(s['size_bytes'] for s in skipped) / 1e9:.1f} GB)"
    )
    print(f"deposits with NO code obtained (over-cap only): {len(blocked)}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

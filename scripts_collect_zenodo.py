"""Collect the Zenodo frame.

The frame is stated, not implied: named communities that hold journal
replication material, plus targeted searches for replication packages
containing code. Every community is verified to exist before use -- Zenodo
ignores an unknown `communities=` filter and returns the entire 7.1M-record
repository, so a typo would silently swap the archive for a journal.
"""

from __future__ import annotations

import sys

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, zenodo_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources import zenodo

logger = get_logger(__name__)
ROOT = PATHS.root / "corpus" / "zenodo"

#: Communities verified to exist and to hold replication material.
COMMUNITIES = [
    "es-replication-repository",
    "civica-open-social-science",
    "economics",
    "econdesarrollo",
]

#: Targeted searches for journal-linked replication code outside those.
QUERIES = [
    '"replication package" AND (stata OR "do-file")',
    '"replication package" AND (R OR rscript)',
    '"replication materials" AND regression',
]


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    setup_logging("INFO", log_dir=PATHS.logs, stage="zenodo")
    client = PoliteClient(
        headers=zenodo_headers(),
        limiter=RateLimiter(rate_per_s=zenodo.DEFAULT_RATE_PER_S),
    )
    ledger = Ledger(ROOT / "ledger.jsonl")
    records: dict[str, zenodo.ZenodoRecord] = {}

    with stage("zenodo-frame", logger) as stats:
        for slug in COMMUNITIES:
            try:
                n = zenodo.verify_community(client, slug)
            except zenodo.UnknownCommunity as exc:
                logger.error(
                    "skipping community", extra={"slug": slug, "err": str(exc)}
                )
                continue
            stats.incr("communities_verified")
            for rec in zenodo.search(client, "", community=slug, authenticated=True):
                records[rec.record_id] = rec
            logger.info(
                "community harvested",
                extra={"slug": slug, "declared": n, "unique_records": len(records)},
            )
        for query in QUERIES:
            for rec in zenodo.search(
                client, query, max_records=limit, authenticated=True
            ):
                records.setdefault(rec.record_id, rec)
            logger.info(
                "query harvested",
                extra={"q": query[:40], "unique_records": len(records)},
            )

    todo = list(records.values())[:limit]
    logger.info("collecting", extra={"records": len(todo)})
    rows = zenodo.collect(todo, ROOT / "files", ledger, client)

    print(f"\nrecords in frame : {len(records):,}")
    print(f"collected        : {len(todo):,}")
    print(f"states           : {ledger.summary()}")
    print(f"totals           : {ledger.totals()}")
    checked = [r for r in rows if r["md5_api"]]
    bad = [r for r in checked if not r["md5_verified"]]
    print(f"md5 verified     : {len(checked) - len(bad)}/{len(checked)}")
    print(
        f"reconciling      : {len(ledger) - len(ledger.non_reconciling())}/{len(ledger)}"
    )
    skipped = ledger.skipped_archives()
    print(
        f"archives skipped : {len(skipped)} "
        f"({sum(s['size_bytes'] for s in skipped) / 1e9:.2f} GB)"
    )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

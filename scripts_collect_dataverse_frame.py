"""Enumerate the Harvard Dataverse journal collections.

    uv run python scripts_collect_dataverse_frame.py [--limit N] [alias ...]

Writes `data/frame/dataverse_deposits.csv`: every deposit in every journal
dataverse, with its DOI and the collection it belongs to.

This is Stage 1 of the Dataverse arm, and it is small. One request returns a
collection's entire dataset list, unpaginated -- `ajps` answers with 836
datasets in a single call -- so the whole frame is roughly seventy-five
requests, not one per deposit.

Requests go through :mod:`softverse.acquire.browser`, because Harvard puts an
AWS WAF challenge in front of every route and `curl` cannot run the
JavaScript it asks for. See that module for why this is a capability test
rather than an access control, and for how the API Terms of Use are
satisfied: our own user-agent, an authenticated token, and a rate limit.

The counts are printed against the 2024 scrape's per-journal totals, because
a systematic shortfall is the failure worth catching here and an average
would hide it.
"""

from __future__ import annotations

import csv
import sys

from softverse.acquire.browser import BrowserClient
from softverse.acquire.http import RateLimiter
from softverse.config import PATHS, credential
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources.dataverse import (
    collection_missing,
    search_collection,
    walk_collection,
)

logger = get_logger(__name__)

OUT = PATHS.frame / "dataverse_deposits.csv"
LEGACY_METADATA = PATHS.root / "corpus" / "dataverse_legacy" / "metadata"

#: One request per two seconds. The volume does not need it -- seventy-five
#: requests would be unremarkable at any rate -- but the terms speak to
#: bandwidth and impairment, and there is no reason to be the fastest client
#: they see today.
RATE_PER_S = 0.5

FIELDS = [
    "collection_id",
    "source",
    "dataset_id",
    "persistent_id",
    "protocol",
    "authority",
    "identifier",
    "publication_date",
]


def write_frame(rows: list[dict], path, collected: list[str]) -> int:
    """Merge this run's collections into the frame, leaving the rest alone.

    Overwriting is wrong here and the first version did it: any one of ~75
    collections can come back empty, retrying that alias is the obvious
    response, and a one-collection run would have written over the other
    seventy-four. Only the collections named in `collected` are replaced, so
    a retry that fails leaves what was already there rather than erasing a
    journal on the strength of a transient error.
    """
    existing: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            existing = [
                row
                for row in csv.DictReader(handle)
                if row.get("collection_id") not in set(collected)
            ]
    merged = existing + rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(merged)
    return len(merged)


def journal_aliases() -> list[str]:
    """Journal dataverses to enumerate, from the frame and the 2024 scrape.

    The 2024 metadata is the better list: 150 journal files against the 74 in
    `data/datasets/`, and it is what the scrape actually covered, so the
    counts below compare like with like.
    """
    aliases: dict[str, None] = {}
    for path in sorted(LEGACY_METADATA.rglob("*_datasets.csv")):
        if ".ipynb_checkpoints" in str(path):
            continue
        aliases[path.stem.replace("_datasets", "")] = None

    frame_csv = PATHS.frame / "frame.csv"
    if frame_csv.exists():
        with open(frame_csv, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("source") == "dataverse" and row.get("collection_id"):
                    aliases[row["collection_id"]] = None
    return list(aliases)


def legacy_counts() -> dict[str, int]:
    """Deposits per journal in the 2024 scrape, for comparison."""
    out: dict[str, int] = {}
    for path in sorted(LEGACY_METADATA.rglob("*_datasets.csv")):
        if ".ipynb_checkpoints" in str(path):
            continue
        with open(path, encoding="utf-8", errors="replace") as handle:
            out[path.stem.replace("_datasets", "")] = sum(
                1
                for row in csv.DictReader(handle)
                if (row.get("identifier") or "").strip()
            )
    return out


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="dataverse-frame")
    positional = [a for a in sys.argv[1:] if not a.startswith("-")]
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    aliases = positional or journal_aliases()
    if limit:
        aliases = aliases[:limit]
    if not aliases:
        print("no journal aliases found")
        return 1

    client = BrowserClient(
        token=credential("DATAVERSE_API_TOKEN"),
        limiter=RateLimiter(rate_per_s=RATE_PER_S),
    )
    before = legacy_counts()
    rows: list[dict] = []
    failed: list[str] = []
    gone: list[str] = []

    print(f"{'journal':<30}{'now':>8}{'2024':>8}  via")
    with stage("dataverse-frame", logger) as stats:
        for alias in aliases:
            datasets, _nested = walk_collection(client, alias)
            via = "contents"
            if not datasets:
                # `/contents` serves a whole collection in one response and
                # the server gives up on the largest: `jop` returned 466 KB
                # after 87 s, `restat` nothing after 73 s. Paged search
                # handles those, and is the lighter request besides.
                datasets = search_collection(client, alias)
                via = "search"
                if not datasets:
                    # A collection that is gone and one we could not reach
                    # want opposite responses: retrying the first never
                    # works, and lumping them together overstates the gap
                    # while hiding the fact that a journal has vanished.
                    probe = client.get(
                        f"https://dataverse.harvard.edu/api/dataverses/{alias}"
                    )
                    if collection_missing(probe):
                        gone.append(alias)
                        via = "GONE"
                    else:
                        failed.append(alias)
            stats.incr("collections")
            stats.incr("deposits", len(datasets))
            for entry in datasets:
                rows.append(
                    {
                        "collection_id": alias,
                        "source": "dataverse",
                        "dataset_id": entry.get("id"),
                        "persistent_id": entry.get("persistentUrl")
                        or f"doi:{entry.get('authority')}/{entry.get('identifier')}",
                        "protocol": entry.get("protocol"),
                        "authority": entry.get("authority"),
                        "identifier": entry.get("identifier"),
                        "publication_date": entry.get("publicationDate"),
                    }
                )
            was = before.get(alias)
            print(
                f"  {alias:<28}{len(datasets):>8}"
                f"{was if was is not None else '-':>8}  "
                f"{via if datasets else ('GONE' if via == 'GONE' else 'FAILED')}"
            )

    # Only collections that actually returned something are replaced; one
    # that failed keeps whatever the last good run recorded.
    succeeded = sorted({r["collection_id"] for r in rows})
    total = write_frame(rows, OUT, collected=succeeded)

    print(f"\n{len(rows):,} deposits from {len(succeeded)} collections this run")
    print(f"{total:,} deposits in the frame across all runs")
    print(f"written to {OUT}")
    if gone:
        print(f"\n{len(gone)} collections no longer exist at Harvard:")
        for alias in gone:
            print(
                f"  {alias} (had {before.get(alias, '?')} deposits in the 2024 scrape)"
            )

    if failed:
        # Reported, never absorbed -- and named for what was seen rather than
        # diagnosed. An empty response here was once labelled a WAF challenge
        # and turned out to be our own 25-second timeout.
        print(f"\n{len(failed)} collections returned nothing:")
        print("  " + ", ".join(failed[:20]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

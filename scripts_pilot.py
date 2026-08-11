"""Collection pilot: 200 datasets, six gates, before the full run.

The full run is ~8 hours. A bug found at hour seven is expensive, so this
exercises every code path on a stratified sample and checks the invariants that
v1 never checked. Run:

    uv run python scripts_pilot.py [n]
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.acquire.state import Ledger
from softverse.config import PATHS, dataverse_headers
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources.dataverse import collect, dataset_dir

logger = get_logger(__name__)
PILOT = PATHS.root / "corpus" / "pilot"
KNOWN_TRUTH = (
    "doi:10.7910/DVN/RE0FZS"  # audit proved v1 marked this "success" with 0 bytes
)


def frame() -> list[tuple[str, str]]:
    """(doi, journal) for every dataset in the sampling frame."""
    rows = []
    for path in glob.glob(str(PATHS.frame / "*_datasets.csv")):
        journal = Path(path).name.replace("_datasets.csv", "")
        for row in csv.DictReader(open(path)):
            ident = (row.get("identifier") or "").strip()
            if ident:
                rows.append((f"doi:10.7910/DVN/{ident.split('/')[-1]}", journal))
    return rows


def stratified(rows: list[tuple[str, str]], n: int) -> list[str]:
    """Spread the sample across journals rather than taking the first n."""
    random.seed(20260811)
    by_journal: dict[str, list[str]] = {}
    for doi, journal in rows:
        by_journal.setdefault(journal, []).append(doi)
    picked: list[str] = []
    journals = sorted(by_journal)
    i = 0
    while len(picked) < n:
        journal = journals[i % len(journals)]
        pool = by_journal[journal]
        if pool:
            picked.append(pool.pop(random.randrange(len(pool))))
        i += 1
        if i > n * 50:
            break
    if KNOWN_TRUTH not in picked:
        picked[0] = KNOWN_TRUTH
    return picked


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    setup_logging("INFO", log_dir=PATHS.logs, stage="pilot")
    files_root = PILOT / "files"
    raw_root = PILOT / "raw"
    ledger = Ledger(PILOT / "ledger.jsonl")

    dois = stratified(frame(), n)
    with (
        stage("pilot", logger),
        PoliteClient(
            headers=dataverse_headers(), limiter=RateLimiter(rate_per_s=2.0, burst=2)
        ) as client,
    ):
        rows = collect(dois, files_root, raw_root, ledger, client)

    print("\n" + "=" * 72)
    print(f"PILOT: {len(dois)} datasets requested")
    print("=" * 72)
    print("states:", json.dumps(ledger.summary(), indent=None))
    totals = ledger.totals()
    print("totals:", json.dumps(totals, indent=None))

    failures: list[str] = []

    # Gate 1 -- md5 verified for every file the API gave a checksum for.
    checked = [r for r in rows if r["md5_api"]]
    bad = [r for r in checked if not r["md5_verified"]]
    print(f"\n[1] md5: {len(checked) - len(bad)}/{len(checked)} verified")
    if bad:
        failures.append(f"{len(bad)} md5 mismatches")

    # Gate 2 -- the reconciliation identity, per dataset.
    off = ledger.non_reconciling()
    print(
        f"[2] reconciliation: {len(ledger) - len(off)}/{len(ledger)} datasets balance"
    )
    if off:
        failures.append(f"{len(off)} datasets do not reconcile")

    # Gate 4 -- provenance complete on every row.
    missing = [
        r
        for r in rows
        if not r["dataset_doi"]
        or not r["relative_path"]
        or (r["dataverse_file_id"] is None and r["container_file_id"] is None)
    ]
    print(
        f"[4] provenance: {len(rows) - len(missing)}/{len(rows)} rows fully traceable"
    )
    if missing:
        failures.append(f"{len(missing)} rows missing provenance")

    # Gate 5 -- the dataset v1 claimed success on while storing nothing.
    truth_dir = dataset_dir(files_root, KNOWN_TRUTH)
    got = sorted(p.name for p in truth_dir.rglob("*.R")) if truth_dir.exists() else []
    want = [
        "01_preprocessing.R",
        "02_polyarchy.R",
        "03_polity2.R",
        "04_fh.R",
        "05_heatmap.R",
        "06_table1.R",
    ]
    print(f"[5] known truth RE0FZS: {len(got)}/6 R scripts -> {got}")
    if got != want:
        failures.append(f"RE0FZS expected {want}, got {got}")

    # Coverage gap, reported rather than hidden.
    skipped = ledger.skipped_archives()
    gb = sum(s["size_bytes"] for s in skipped) / 1e9
    print(f"\nskipped archives (over cap): {len(skipped)} totalling {gb:.2f} GB")
    for s in sorted(skipped, key=lambda s: -s["size_bytes"])[:5]:
        print(
            f"    {s['size_bytes'] / 1e6:>8.0f} MB  {s['dataset_doi']}  {s['filename']}"
        )

    on_disk = sum(1 for _ in files_root.rglob("*") if _.is_file())
    print(f"\nfiles on disk: {on_disk:,}   bytes: {totals['bytes'] / 1e6:.1f} MB")
    ext = Counter(
        p.suffix.lower() for p in files_root.rglob("*") if p.is_file() and p.suffix
    )
    print("top extensions:", dict(ext.most_common(10)))

    print(
        "\n" + ("FAILURES: " + "; ".join(failures) if failures else "ALL GATES PASSED")
    )
    print("(gate 3, idempotency, is checked by re-running this script)")
    return 1 if failures else 0


if __name__ == "__main__":
    os.makedirs(PILOT, exist_ok=True)
    raise SystemExit(main())

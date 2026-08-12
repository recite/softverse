"""Harvest the Dataverse candidate-file census over OAI-PMH.

The JSON API is WAF-blocked; /oai is not. This gives the frame and manifest
stages for all 13,775 datasets without touching the blocked endpoints, and
yields a real census to replace a 60-dataset extrapolation.

Deliberately slow: Harvard has already challenged this client once, and /oai
being exempt is not a licence to hammer it.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.config import PATHS
from softverse.logging_setup import get_logger, setup_logging, stage
from softverse.sources.dataverse_oai import harvest

logger = get_logger(__name__)
OUT = PATHS.root / "corpus" / "dataverse_oai"


def frame() -> tuple[list[str], dict[str, str]]:
    dois, journal_of = [], {}
    for path in glob.glob(str(PATHS.frame / "*_datasets.csv")):
        journal = Path(path).name.replace("_datasets.csv", "")
        for row in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            ident = (row.get("identifier") or "").strip()
            if ident:
                repo = ident.split("/")[-1]
                dois.append(f"doi:10.7910/DVN/{repo}")
                journal_of[repo] = journal
    return dois, journal_of


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    setup_logging("INFO", log_dir=PATHS.logs, stage="oai")
    dois, journal_of = frame()
    if limit:
        dois = dois[:limit]
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1.0))
    with stage("oai-harvest", logger):
        rows = harvest(dois, client, OUT / "raw", journal_of)
    client.close()

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ext = Counter(r["extension"] for r in rows)
    with_code = len({r["dataset_doi"] for r in rows})
    print(f"\ndatasets queried      : {len(dois):,}")
    print(f"datasets with candidates: {with_code:,} ({with_code / len(dois):.0%})")
    print(f"candidate files       : {len(rows):,}")
    print(f"\ntop extensions: {json.dumps(dict(ext.most_common(12)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

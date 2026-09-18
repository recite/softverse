"""Fetch and pin the download counts validated use is compared against.

    uv run python scripts/fetch_downloads.py

SSC hits for every package, CRAN downloads for every CRAN package over the
year before the tally was built, and PyPI downloads for the most downloaded
projects plus any the corpus uses. About 250 requests to cranlogs and one a
second to pypistats for the stragglers. Follow with
`scripts/pin_registries.py --pin-only`.
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from softverse.config import PATHS
from softverse.registries.downloads import (
    cran_downloads,
    pypi_downloads,
    ssc_hits,
    write,
)
from softverse.registries.lock import SNAPSHOTS, pinned_names, read_lock

USAGE = PATHS.root / "build" / "tally" / "usage_by_package.csv"


def main() -> int:
    usage = pd.read_csv(USAGE)
    built = date.fromisoformat(read_lock()["cran"].date)
    with httpx.Client(
        timeout=120.0,
        follow_redirects=True,
        headers={"User-Agent": "softverse (research; github.com/recite/softverse)"},
    ) as client:
        counts, meta = ssc_hits(client)
        print(write("downloads_ssc", counts, meta, SNAPSHOTS), len(counts))
        # The window ends where the CRAN snapshot was taken, so the names
        # asked about are the names the tally could resolve.
        counts, meta = cran_downloads(client, pinned_names("cran", read_lock()), built)
        print(write("downloads_cran", counts, meta, SNAPSHOTS), len(counts))
        used = usage.loc[usage["ecosystem"].eq("pypi"), "package"]
        counts, meta = pypi_downloads(client, used)
        print(write("downloads_pypi", counts, meta, SNAPSHOTS), len(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

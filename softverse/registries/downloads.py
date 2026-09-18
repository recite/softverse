"""Download counts, pinned like the registries, to set validated use against.

A download count is the number the field already uses for how much a package
is used, and the paper's argument is that it measures something else. That has
to be shown on this corpus rather than borrowed, so each ecosystem's own count
is fetched, dated and digest-pinned beside the registry snapshots:

- **SSC** publishes every package's hits for the latest month in the file
  behind Stata's `ssc hot` (`sschotPPPcur.dta`).
- **CRAN** downloads come from the RStudio mirror's logs through the
  `cranlogs` API, for any window, a hundred packages to a request.
- **PyPI** downloads for the fifteen thousand most downloaded projects over
  thirty days come from `hugovk/top-pypi-packages`, a monthly dump of PyPI's
  BigQuery table; a project the corpus uses that is outside it is asked of
  `pypistats` one at a time.

The windows differ, a month against a year, because the sources do. Every
comparison the paper makes is of ranks within one ecosystem, which a common
multiplier does not move.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import pandas as pd

from softverse.logging_setup import get_logger
from softverse.registries.resolve import normalize_pypi

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = get_logger(__name__)

SSC_HITS = "http://repec.org/docs/sschotPPPcur.dta"
CRANLOGS = "https://cranlogs.r-pkg.org/downloads/total"
TOP_PYPI = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
PYPISTATS = "https://pypistats.org/api/packages"
FILE = "downloads.json"

#: cranlogs accepts a comma-separated list; this many keeps the URL short.
_CRAN_BATCH = 100


def ssc_hits(client: httpx.Client) -> tuple[dict[str, float], dict]:
    """Hits in the latest month for every SSC package."""
    raw = client.get(SSC_HITS).raise_for_status().content
    frame = pd.DataFrame(pd.read_stata(io.BytesIO(raw)))
    # One row per (package, author), the hits repeated on each.
    hits: dict[str, float] = {}
    for package, count in zip(frame["package"], frame["hits_cur"], strict=True):
        hits[str(package).lower()] = max(
            float(count), hits.get(str(package).lower(), 0.0)
        )
    return hits, {
        "url": SSC_HITS,
        "window": "latest month, as served",
    }


def cran_downloads(
    client: httpx.Client, packages: Iterable[str], end: date
) -> tuple[dict[str, float], dict]:
    """Downloads of each CRAN package in the year ending ``end``."""
    start = end - timedelta(days=364)
    window = f"{start.isoformat()}:{end.isoformat()}"
    names = sorted(set(packages))
    out: dict[str, float] = {}
    for i in range(0, len(names), _CRAN_BATCH):
        batch = names[i : i + _CRAN_BATCH]
        response = client.get(f"{CRANLOGS}/{window}/{','.join(batch)}")
        response.raise_for_status()
        for row in response.json():
            if row.get("package") and row.get("downloads") is not None:
                out[row["package"]] = float(row["downloads"])
    return out, {"url": CRANLOGS, "window": window}


def pypi_downloads(
    client: httpx.Client, used: Iterable[str]
) -> tuple[dict[str, float], dict]:
    """Thirty-day downloads: the top projects, plus any the corpus uses."""
    top = client.get(TOP_PYPI).raise_for_status().json()
    out = {
        normalize_pypi(r["project"]): float(r["download_count"]) for r in top["rows"]
    }
    n_top = len(out)
    unanswered = []
    for name in sorted({normalize_pypi(u) for u in used} - set(out)):
        response = client.get(f"{PYPISTATS}/{name}/recent")
        for _attempt in range(5):
            if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                break
            # The service throttles hard and says for how long.
            time.sleep(float(response.headers.get("Retry-After", 60)))
            response = client.get(f"{PYPISTATS}/{name}/recent")
        if response.status_code == httpx.codes.OK:
            out[name] = float(response.json()["data"]["last_month"])
        elif response.status_code != httpx.codes.NOT_FOUND:
            unanswered.append(name)
        time.sleep(2.0)
    return out, {
        "url": [TOP_PYPI, PYPISTATS],
        "window": f"30 days to {top['last_update']}",
        "n_from_top_list": n_top,
        # Asked and not answered, as distinct from asked and not on PyPI.
        "unanswered": unanswered,
    }


def write(registry: str, counts: dict[str, float], meta: dict, root: Path) -> Path:
    """Write a dated snapshot whose digest is of the counts themselves."""
    stamp = datetime.now(tz=UTC)
    directory = root / registry / stamp.date().isoformat()
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(counts, sort_keys=True).encode()
    (directory / FILE).write_bytes(payload)
    (directory / "source.json").write_text(
        json.dumps(
            {
                "registry": registry,
                "fetched_at": stamp.isoformat(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "n_packages": len(counts),
                **meta,
            },
            indent=2,
        )
        + "\n"
    )
    return directory

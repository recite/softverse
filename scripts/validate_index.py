"""Check the SSC tier of the index against a list built another way.

    uv run python scripts/validate_index.py

The AEA Data Editor's `packagesearch` ships a command-to-package list for SSC
that Sergio Correia compiled for it, by a route that shares no code with this
one. Where the two name the same command, do they name the same package? They
cannot both be wrong in the same way by accident, which makes agreement
evidence that a manifest line naming a file does name a command, and
disagreement a list of things to look at.

Writes `build/validation/index_agreement.json`, which the paper reads.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

import duckdb
import httpx

from softverse.config import PATHS
from softverse.registries.load import INDEX_FILE
from softverse.registries.lock import pinned_directory, read_lock

SIGNALS = (
    "https://raw.githubusercontent.com/labordynamicsinstitute/Statapackagesearch/"
    "main/auxiliary/p_signalcommands.txt"
)
OUT = PATHS.root / "build" / "validation" / "index_agreement.json"


def main() -> int:
    text = (
        httpx.get(SIGNALS, timeout=60.0, follow_redirects=True).raise_for_status().text
    )
    theirs: dict[str, set[str]] = {}
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        theirs.setdefault(row["Signals"].lower(), set()).add(row["Package"].lower())

    index = (
        pinned_directory("stata_index", read_lock(), payload=INDEX_FILE) / INDEX_FILE
    )
    ours: dict[str, set[str]] = {}
    everywhere: set[str] = set()
    for command, package, source in duckdb.execute(
        f"SELECT lower(command), lower(package), source FROM '{index}' "
        "WHERE evidence <> 'package_only'"
    ).fetchall():
        everywhere.add(command)
        if source == "ssc":
            ours.setdefault(command, set()).add(package)

    shared = sorted(set(theirs) & set(ours))
    agree = [c for c in shared if theirs[c] & ours[c]]
    report = {
        "source": SIGNALS,
        "checked": datetime.now(tz=UTC).date().isoformat(),
        "n_theirs": len(theirs),
        "n_ours_ssc": len(ours),
        "n_shared": len(shared),
        "n_same_package": len(agree),
        "disagreements": {
            c: [sorted(theirs[c]), sorted(ours[c])] for c in shared if c not in agree
        },
        "n_only_theirs": len(set(theirs) - set(ours)),
        "n_theirs_in_no_tier": len(set(theirs) - everywhere),
        "n_only_ours": len(set(ours) - set(theirs)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=1)
    )
    print("disagreements:", report["disagreements"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

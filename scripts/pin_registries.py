"""Fetch the package registries and pin the lock to what was fetched.

    uv run python scripts/pin_registries.py            # fetch, then pin
    uv run python scripts/pin_registries.py --pin-only # pin what is on disk

A refreshed registry changes resolution for the whole corpus, not only for
new deposits, so this is never run by the weekly update. Run it, then rebuild
the tally, and the lock id stamped on every mention names the new snapshot.
"""

from __future__ import annotations

import sys

from softverse.registries.fetch import fetch_all
from softverse.registries.lock import LOCK, SNAPSHOTS, write_lock


def main() -> int:
    if "--pin-only" not in sys.argv:
        fetch_all(SNAPSHOTS)
    pins = write_lock(SNAPSHOTS, LOCK)
    for name, pin in pins.items():
        print(f"  {name:<17}{pin.date}  {pin.sha256[:12]}")
    print(f"wrote {LOCK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Crawl SSC, the Stata Journal and the STB into a dated index snapshot.

    uv run python scripts/build_stata_index.py              # crawl, then derive
    uv run python scripts/build_stata_index.py --rederive   # newest crawl, new rules
    uv run python scripts/build_stata_index.py --net-sites  # add the corpus's net sites

About six thousand small requests, a few minutes. `--net-sites` keeps the
newest crawl and adds the sites the tally's `net install ..., from(URL)` lines
name, so it needs a tally built by an extractor that records them. Follow any
of these with `scripts/pin_registries.py --pin-only` and a full tally rebuild:
the index decides what every Stata command resolves to.
"""

from __future__ import annotations

import argparse

import duckdb

from softverse.config import PATHS
from softverse.registries.lock import SNAPSHOTS
from softverse.stata.index import (
    fetch_manifests,
    fetch_net_manifests,
    read_manifests,
    write_snapshot,
)

ROOT = SNAPSHOTS / "stata_index"
SITES = PATHS.registries / "stata_net_sites.txt"
MENTIONS = PATHS.root / "build" / "tally" / "mentions.parquet"


def main() -> None:
    """Fetch or reload the manifests and write a snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rederive", action="store_true")
    parser.add_argument("--net-sites", action="store_true")
    parser.add_argument("--remotes", help="CSV of raw_name, remote")
    args = parser.parse_args()
    dead: list[str] = []
    if args.rederive or args.net_sites:
        manifests = read_manifests(max(ROOT.glob("*/manifests.jsonl.gz")).parent)
    else:
        manifests = fetch_manifests()
    if args.net_sites:
        # `--remotes` names a CSV of (raw_name, remote) harvested straight
        # from the corpus. It exists so an extractor change that finds more
        # install lines does not cost a tally rebuild to learn their URLs and
        # then another to resolve against them.
        source = f"'{args.remotes}'" if args.remotes else f"'{MENTIONS}'"
        where = "" if args.remotes else "language = 'stata' AND "
        installs = duckdb.execute(
            f"SELECT DISTINCT raw_name, remote FROM {source} "
            f"WHERE {where}remote IS NOT NULL"
        ).fetchall()
        sites = [
            line.strip()
            for line in SITES.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        found, dead = fetch_net_manifests(installs, sites=sites)
        manifests = [m for m in manifests if m.source != "net"] + found
        print(f"net sites: {len(found)} manifests, {len(dead)} unreachable")
    print(write_snapshot(manifests, ROOT, dead))


if __name__ == "__main__":
    main()

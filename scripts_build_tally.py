"""Build the library-usage tally from whatever has been collected so far.

    uv run python scripts_build_tally.py

Runs against the corpus on disk at the moment you invoke it, so it is safe to
run while collection is still going -- the numbers simply describe less of the
frame. Every output carries the denominators it was computed against, because a
count without one cannot be compared to anything.

Every source goes through here. Zenodo and the 2024 Harvard Dataverse scrape
were tallied by two scripts into two directories for a while, and only one of
them aggregated, so the published per-package counts were the Zenodo half
alone: economics, one seventh of the deposits on disk. `full_corpus()`
concatenates them and `build()` runs once, which is what makes the two
comparable rather than merely adjacent.

Outputs, under `build/tally/`:

    usage_by_package.csv       package -> deposits, files, mentions, years,
                               pooled and split by source
    usage_by_package_year.csv  the trend table
    usage_by_collection.csv    per journal/community
    language_presence.csv      deposits containing each language, per source
    unknown_names.csv          detected names we could not resolve
    mentions.parquet           every mention, with line/col/snippet
    files.parquet              the provenance spine

The headline unit is **deposits**, not mentions: one deposit that calls ggplot2
two hundred times is one user of ggplot2. Mention counts are kept alongside, but
a mention-weighted ranking is a ranking of a few large deposits.
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

import duckdb

from softverse.build.pipeline import (
    build,
    dataset_packages,
    language_presence,
)
from softverse.config import PATHS
from softverse.corpus.loaders import full_corpus
from softverse.logging_setup import get_logger, setup_logging
from softverse.model.io import write_table
from softverse.registries.resolve import Registry
from softverse.stata.builtins import builtins

logger = get_logger(__name__)

#: Commands checked one by one against StataCorp's help server. Widens the
#: curated builtin list by 165 official commands that were otherwise reported
#: as resolving to no registry. Built by scripts_verify_stata_official.py; the
#: tally falls back to the curated list alone if it is absent.
OFFICIAL_SNAPSHOT = (
    PATHS.root / "registries" / "snapshots" / "stata_official" / "official.json"
)
OUT = PATHS.root / "build" / "tally"


def load_registry() -> tuple[Registry, frozenset[str]]:
    """Registries from the pinned snapshots, plus the SSC shipped-file set."""

    def names(registry: str) -> frozenset[str]:
        newest = sorted(glob.glob(f"registries/snapshots/{registry}/*/names.json"))[-1]
        return frozenset(json.load(open(newest)))

    con = duckdb.connect()
    index = "registries/snapshots/ssc/stata_command_index.parquet"
    commands: dict[str, list[str]] = {}
    for command, package in con.execute(
        f"SELECT lower(command), package FROM '{index}' WHERE NOT is_helper"
    ).fetchall():
        commands.setdefault(command, []).append(package)
    shipped = frozenset(
        r[0].lower()
        for r in con.execute(f"SELECT DISTINCT command FROM '{index}'").fetchall()
    )
    # Package names, a different namespace from command names: `ssc install
    # blindschemes` names a package that exposes no command of that name.
    packages = frozenset(
        r[0].lower()
        for r in con.execute(f"SELECT DISTINCT package FROM '{index}'").fetchall()
    )
    return (
        Registry(
            cran=names("cran"),
            cran_archive=names("cran_archive"),
            bioconductor=names("bioconductor"),
            pypi=names("pypi"),
            julia=names("julia_general"),
            stata_commands={k: tuple(v) for k, v in commands.items()},
            stata_builtins=builtins(verified_snapshot=OFFICIAL_SNAPSHOT).forms,
            ssc_packages=packages,
            lock_id=json.load(open("registries/registries.lock.json")).get("cran", "")[
                :12
            ],
        ),
        shipped,
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    setup_logging("WARNING", log_dir=PATHS.logs, stage="tally")
    registry, shipped = load_registry()
    corpus = full_corpus()
    if not corpus:
        print("no corpus collected yet")
        return 1

    n_deposits = len({c.dataset_doi for c in corpus})
    by_source = collections.Counter(c.source for c in corpus)
    print(f"{len(corpus):,} files from {n_deposits:,} deposits")
    for source, n in sorted(by_source.items()):
        print(f"  {source:<20}{n:>9,} files")

    result = build(
        corpus, registry, ssc_shipped=shipped, registry_lock_id=registry.lock_id
    )
    packages = dataset_packages(result.mentions)
    sources = sorted(by_source)

    # -- usage_by_package: the tally ---------------------------------------
    # Pooled counts, with the per-source split beside them rather than in a
    # footnote. The two halves are very different sizes, 7,231 Dataverse
    # deposits against 1,601 from Zenodo, so a pooled number with no
    # breakdown asks the reader to take the composition on trust.
    by_package: dict[tuple, dict] = {}
    for row in packages:
        key = (row["language"], row["package"])
        entry = by_package.setdefault(
            key,
            {
                "package": row["package"],
                "language": row["language"],
                "ecosystem": row["ecosystem"],
                "n_deposits": 0,
                "n_files": 0,
                "n_mentions": 0,
                "first_year": None,
                "last_year": None,
                **{f"n_deposits_{s}": 0 for s in sources},
            },
        )
        entry["n_deposits"] += 1
        entry[f"n_deposits_{row['source']}"] += 1
        entry["n_files"] += row["n_files"]
        entry["n_mentions"] += row["n_mentions"]
        year = row["year"]
        if year:
            entry["first_year"] = min(entry["first_year"] or year, year)
            entry["last_year"] = max(entry["last_year"] or year, year)

    # The denominator travels with the count. Deposits *at risk* of using an R
    # package are the deposits containing analyzable R, not all deposits --
    # a share against the wrong denominator is not comparable to anything.
    #
    # A literate document makes "containing analyzable R" harder than it looks.
    # A notebook's file language is `notebook`, so counting file languages
    # alone put a deposit whose only Python lives in a `.ipynb` into the
    # numerator of every Python package it loads and into the denominator of
    # none. That inflated pandas to 93.9% of Python deposits when the true
    # figure is 82.6%, and it inflated Python alone, which is the language the
    # paper reports as smallest. Counting a deposit at risk when a *mention*
    # in that language came out of it repairs the containers, and for R,
    # Python and Stata it is exact: every deposit with chunks in one of those
    # languages yields at least one mention in it.
    at_risk: dict[str, set[str]] = collections.defaultdict(set)
    at_risk_by_source: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in result.files:
        if row["in_analysis_set"]:
            at_risk[row["language"]].add(row["dataset_doi"])
            at_risk_by_source[(row["source"], row["language"])].add(row["dataset_doi"])
    for mention in result.mentions:
        at_risk[mention["language"]].add(mention["dataset_doi"])
        at_risk_by_source[(mention["source"], mention["language"])].add(
            mention["dataset_doi"]
        )

    tally = sorted(by_package.values(), key=lambda r: -r["n_deposits"])
    for row in tally:
        denom = len(at_risk.get(row["language"], ()))
        row["n_deposits_at_risk"] = denom
        row["share_of_deposits"] = (
            round(row["n_deposits"] / denom, 4) if denom else None
        )
        for source in sources:
            row[f"n_at_risk_{source}"] = len(
                at_risk_by_source.get((source, row["language"]), ())
            )

    write_csv(tally, OUT / "usage_by_package.csv")

    # -- by collection: the per-journal view --------------------------------
    # Promised by this script's docstring since it was written and never
    # produced, because until the Zenodo metadata was persisted there was
    # exactly one collection id on that side. There are now ten communities
    # and sixty-one journals.
    by_collection: dict[tuple, dict] = {}
    for row in packages:
        key = (row["source"], row["collection_id"], row["language"], row["package"])
        entry = by_collection.setdefault(
            key,
            {
                "source": row["source"],
                "collection_id": row["collection_id"],
                "language": row["language"],
                "package": row["package"],
                "n_deposits": 0,
            },
        )
        entry["n_deposits"] += 1
    write_csv(
        sorted(
            by_collection.values(),
            key=lambda r: (r["collection_id"], r["language"], -r["n_deposits"]),
        ),
        OUT / "usage_by_collection.csv",
    )

    # -- by package-year ----------------------------------------------------
    by_year: dict[tuple, dict] = {}
    for row in packages:
        if not row["year"]:
            continue
        key = (row["language"], row["package"], row["year"])
        entry = by_year.setdefault(
            key,
            {
                "package": row["package"],
                "language": row["language"],
                "year": row["year"],
                "n_deposits": 0,
            },
        )
        entry["n_deposits"] += 1
    write_csv(
        sorted(by_year.values(), key=lambda r: (r["package"], r["year"])),
        OUT / "usage_by_package_year.csv",
    )

    # -- language presence and unresolved names -----------------------------
    presence = language_presence(result.files)
    write_csv(
        [
            {"source": source, "language": language, "n_deposits": n}
            for (source, language), n in presence.items()
        ],
        OUT / "language_presence.csv",
    )
    unknown = sorted(result.unknown.items(), key=lambda kv: -kv[1])
    write_csv(
        [
            {"name": name, "language": language, "n_mentions": n}
            for (name, language), n in unknown
        ],
        OUT / "unknown_names.csv",
    )

    write_table(result.mentions, "mentions", OUT)
    write_table(result.files, "files", OUT)

    print(f"deposits collected : {n_deposits:,}")
    print(f"files              : {len(result.files):,}")
    print(f"mentions           : {len(result.mentions):,}")
    print(f"packages tallied   : {len(tally):,}")
    print(f"unresolved names   : {len(unknown):,}")
    print(
        f"\ndeposits with analyzable code, by language: "
        f"{ {k: len(v) for k, v in sorted(at_risk.items())} }"
    )
    print(f"\nwritten to {OUT}/")
    print("\n=== TOP 15 BY DEPOSITS ===")
    split = "  ".join(f"{s.split('_')[0]:>9}" for s in sources)
    print(f"  {'pooled':>9} {'of':<6} {'lang':<7} {'package':<16}{split}")
    for row in tally[:15]:
        counts = "  ".join(f"{row[f'n_deposits_{s}']:>9,}" for s in sources)
        print(
            f"  {row['n_deposits']:>9,} {row['n_deposits_at_risk']:<6,} "
            f"{row['language']:<7} {row['package']:<16}{counts}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

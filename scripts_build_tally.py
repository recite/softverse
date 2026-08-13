"""Build the library-usage tally from whatever has been collected so far.

    uv run python scripts_build_tally.py

Runs against the corpus on disk at the moment you invoke it, so it is safe to
run while collection is still going -- the numbers simply describe less of the
frame. Every output carries the denominators it was computed against, because a
count without one cannot be compared to anything.

Outputs, under `build/tally/`:

    usage_by_package.csv       package -> deposits, files, mentions, years
    usage_by_package_year.csv  the trend table, with denominators
    usage_by_collection.csv    per journal/community
    language_presence.csv      deposits containing each language
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
    CorpusFile,
    build,
    dataset_packages,
    language_presence,
)
from softverse.config import PATHS
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
    return (
        Registry(
            cran=names("cran"),
            cran_archive=names("cran_archive"),
            bioconductor=names("bioconductor"),
            pypi=names("pypi"),
            julia=names("julia_general"),
            stata_commands={k: tuple(v) for k, v in commands.items()},
            stata_builtins=builtins(verified_snapshot=OFFICIAL_SNAPSHOT).forms,
            lock_id=json.load(open("registries/registries.lock.json")).get("cran", "")[
                :12
            ],
        ),
        shipped,
    )


def zenodo_corpus() -> list[CorpusFile]:
    """Collected Zenodo files, with their community and year from the ledger."""
    root = PATHS.root / "corpus" / "zenodo" / "files"
    if not root.exists():
        return []
    ledger_path = PATHS.root / "corpus" / "zenodo" / "ledger.jsonl"
    year_of: dict[str, int] = {}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec["dataset_doi"].rsplit(".", 1)[-1]
                year_of[rid] = rec.get("year") or 0

    out: list[CorpusFile] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        record_id = path.relative_to(root).parts[0]
        out.append(
            CorpusFile(
                path=path,
                dataset_doi=f"zenodo:{record_id}",
                collection_id="zenodo",
                source="zenodo",
                relative_path=str(path.relative_to(root / record_id)),
                deposit_year=year_of.get(record_id) or None,
            )
        )
    return out


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
    corpus = zenodo_corpus()
    if not corpus:
        print("no corpus collected yet")
        return 1

    n_deposits = len({c.dataset_doi for c in corpus})
    result = build(
        corpus, registry, ssc_shipped=shipped, registry_lock_id=registry.lock_id
    )
    packages = dataset_packages(result.mentions)

    # -- usage_by_package: the tally ---------------------------------------
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
            },
        )
        entry["n_deposits"] += 1
        entry["n_files"] += row["n_files"]
        entry["n_mentions"] += row["n_mentions"]
        year = row["year"]
        if year:
            entry["first_year"] = min(entry["first_year"] or year, year)
            entry["last_year"] = max(entry["last_year"] or year, year)

    # The denominator travels with the count. Deposits *at risk* of using an R
    # package are the deposits containing analyzable R, not all deposits --
    # a share against the wrong denominator is not comparable to anything.
    at_risk: dict[str, set[str]] = collections.defaultdict(set)
    for row in result.files:
        if row["in_analysis_set"]:
            at_risk[row["language"]].add(row["dataset_doi"])

    tally = sorted(by_package.values(), key=lambda r: -r["n_deposits"])
    for row in tally:
        denom = len(at_risk.get(row["language"], ()))
        row["n_deposits_at_risk"] = denom
        row["share_of_deposits"] = (
            round(row["n_deposits"] / denom, 4) if denom else None
        )

    write_csv(tally, OUT / "usage_by_package.csv")

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
        [{"language": k, "n_deposits": v} for k, v in presence.items()],
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
    for row in tally[:15]:
        print(
            f"  {row['n_deposits']:>4}/{row['n_deposits_at_risk']:<4} "
            f"{row['language']:<7} {row['package']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the library-usage tally from whatever has been collected so far.

    uv run python scripts/build_tally.py

Runs against the corpus on disk at the moment you invoke it, so it is safe to
run while collection is still going -- the numbers simply describe less of the
frame. Every output carries the denominators it was computed against, because a
count without one cannot be compared to anything.

Every source goes through here. Zenodo and the 2024 Harvard Dataverse scrape
were tallied by two scripts into two directories for a while, and only one of
them aggregated, so the published per-package counts were the Zenodo half
alone: economics, one seventh of the deposits on disk. `full_corpus()`
concatenates them and `build()` runs over all of it, which is what makes the
two comparable rather than merely adjacent.

It runs in batches of deposits. The 2026 corpus is about 30 million mentions,
which held as rows needs more memory than the machine that builds it has, so
each batch's rows are written to Parquet as they come and only the aggregates
are kept. The one piece of corpus-wide state, which file hashes recur across
deposits, is computed once up front; the output is the same as a single pass.

Outputs, under `build/tally/`:

    usage_by_package.csv       package -> deposits, files, mentions, years,
                               pooled and split by source
    usage_by_package_year.csv  the trend table
    usage_by_collection.csv    per journal/community
    usage_by_function.csv      which functions of a package the code calls
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
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from softverse.build.pipeline import (
    build,
    coverage_counts,
    dataset_packages,
    hash_corpus,
)
from softverse.config import PATHS
from softverse.corpus.loaders import full_corpus
from softverse.logging_setup import get_logger, setup_logging
from softverse.model.enums import NON_USE_CONSTRUCTS, Language, Resolution
from softverse.model.io import TableAppender
from softverse.registries.load import load_registry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from softverse.build.pipeline import BuildResult, CorpusFile

logger = get_logger(__name__)

OUT = PATHS.root / "build" / "tally"

#: Deposits per `build` call. Measured on the 2026 corpus: 2,000 deposits
#: peaked at 1.9 GB, which scaled to the whole corpus is about 20 GB against
#: 17 GB of RAM. 250 keeps a batch well under a gigabyte.
BATCH_DEPOSITS = 250

#: The atomic tables, appended one batch at a time.
STREAMED = ("mentions", "files", "declared_dependencies", "environment_signals")

COUNTABLE = frozenset({str(Resolution.KNOWN_CURRENT), str(Resolution.KNOWN_ARCHIVED)})
NON_USE = frozenset(str(c) for c in NON_USE_CONSTRUCTS)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write an aggregate table as CSV, and as Parquet beside it.

    Both, because the two readers want different things and the tables are
    small enough that choosing is a false economy. A person checking whether
    their package is counted correctly opens the CSV, in a browser or a
    spreadsheet, and GitHub renders it as a searchable table; anything loading
    it as data wants the Parquet, where the types are in the file rather than
    in a sidecar.

    The types do currently survive a CSV round trip, but only because no
    column has a null: one null in an integer column and pandas reads the
    whole column as float, so `first_year` comes back 2016.0. That is a
    property of today's data rather than of the format, which is the argument
    for not relying on it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    # `newline=""` stops the module rewriting line endings; `lineterminator`
    # then decides them, and the csv default is CRLF. Published data files got
    # CRLF while git stored them as LF, so a regenerated table never matched
    # its own committed copy and every diff was the whole file.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)


def batches(corpus: list[CorpusFile], size: int) -> Iterator[list[CorpusFile]]:
    """Consecutive groups of ``size`` deposits, each deposit's files together.

    A deposit is never split: `build` resolves Stata programs, siblings and
    within-deposit duplicates across a deposit's files, so half a deposit
    would be judged differently from the whole.

    Args:
        corpus: The files, in corpus order.
        size: Deposits per batch.

    Yields:
        The files of the next ``size`` deposits, in corpus order.
    """
    by_deposit: dict[str, list[CorpusFile]] = {}
    for item in corpus:
        by_deposit.setdefault(item.dataset_doi, []).append(item)
    deposits = list(by_deposit.values())
    for start in range(0, len(deposits), size):
        yield [item for files in deposits[start : start + size] for item in files]


def _sets() -> collections.defaultdict:
    return collections.defaultdict(set)


@dataclass
class Aggregates:
    """What the tally keeps from each batch once its rows are on disk."""

    packages: list[dict] = field(default_factory=list)
    at_risk: dict[str, set[str]] = field(default_factory=_sets)
    at_risk_by_source: dict[tuple[str, str], set[str]] = field(default_factory=_sets)
    by_function: dict[tuple, dict] = field(default_factory=dict)
    presence: dict[tuple[str, str], set[str]] = field(default_factory=_sets)
    unknown: collections.Counter = field(default_factory=collections.Counter)
    carrying: dict[str, set[str]] = field(default_factory=_sets)
    eligible: dict[str, set[str]] = field(default_factory=_sets)

    def add(self, result: BuildResult) -> None:
        """Fold one batch's rows into the running aggregates."""
        self.packages.extend(dataset_packages(result.mentions))

        # The denominator travels with the count. Deposits *at risk* of using
        # an R package are the deposits containing analyzable R, not all
        # deposits -- a share against the wrong denominator is not comparable
        # to anything.
        #
        # A literate document makes "containing analyzable R" harder than it
        # looks. A notebook's file language is `notebook`, so counting file
        # languages alone put a deposit whose only Python lives in a `.ipynb`
        # into the numerator of every Python package it loads and into the
        # denominator of none. That inflated pandas to 93.9% of Python
        # deposits when the true figure is 82.6%, and it inflated Python
        # alone, which is the language the paper reports as smallest. Counting
        # a deposit at risk when a *mention* in that language came out of it
        # repairs the containers, and for R, Python and Stata it is exact:
        # every deposit with chunks in one of those languages yields at least
        # one mention in it.
        for row in result.files:
            if row["in_analysis_set"]:
                self.at_risk[row["language"]].add(row["dataset_doi"])
                self.at_risk_by_source[(row["source"], row["language"])].add(
                    row["dataset_doi"]
                )
            if row["language"] and row["language"] != str(Language.UNKNOWN):
                self.presence[(row["source"], row["language"])].add(row["dataset_doi"])
        for mention in result.mentions:
            self.at_risk[mention["language"]].add(mention["dataset_doi"])
            self.at_risk_by_source[(mention["source"], mention["language"])].add(
                mention["dataset_doi"]
            )
            self._add_function(mention)

        self.unknown.update(result.unknown)
        carrying, eligible = result.coverage_sets()
        for signal, dois in carrying.items():
            self.carrying[signal] |= dois
        for signal, dois in eligible.items():
            self.eligible[signal] |= dois

    def _add_function(self, mention: dict) -> None:
        """Count a call under its package's function, where the source names one.

        Aggregated from the mentions rather than from `packages`, because
        `dataset_packages` collapses to one row per (deposit, package) and the
        function is exactly what that collapse throws away. `library(dplyr)`
        names a package and nothing else, and counting it under a fabricated
        function would put a row in the table that no line of code supports.
        """
        if (
            mention["resolution"] not in COUNTABLE
            or not mention["resolved_package"]
            or mention["construct"] in NON_USE
            or not mention["called_function"]
        ):
            return
        key = (
            mention["source"],
            mention["language"],
            mention["resolved_package"],
            mention["called_function"],
        )
        entry = self.by_function.setdefault(
            key,
            {
                "source": mention["source"],
                "language": mention["language"],
                "package": mention["resolved_package"],
                "function": mention["called_function"],
                "n_calls": 0,
                "_deposits": set(),
            },
        )
        entry["n_calls"] += 1
        entry["_deposits"].add(mention["dataset_doi"])


def run_batches(corpus: list[CorpusFile], out: Path) -> tuple[Aggregates, dict]:
    """Build the corpus a batch at a time, writing the atomic tables as it goes.

    Args:
        corpus: Every file to tally.
        out: Where the Parquet tables are written.

    Returns:
        The aggregates, and the row count of each written table.
    """
    registry, shipped = load_registry()
    # Hashed once over everything: whether a file is vendored depends on its
    # bytes recurring in other deposits, which no batch can see on its own.
    hashes = hash_corpus(corpus)
    writers = {name: TableAppender(name, out) for name in STREAMED}
    agg = Aggregates()
    for batch in batches(corpus, BATCH_DEPOSITS):
        result = build(
            batch,
            registry,
            ssc_shipped=shipped,
            registry_lock_id=registry.lock_id,
            hashes=hashes,
        )
        agg.add(result)
        writers["mentions"].append(result.mentions)
        writers["files"].append(result.files)
        writers["declared_dependencies"].append(result.declarations)
        writers["environment_signals"].append(result.environment)
        print(
            f"  batch of {len(batch):,} files: "
            f"{writers['mentions'].rows:,} mentions so far",
            flush=True,
        )
    for writer in writers.values():
        writer.close()
    return agg, {name: writer.rows for name, writer in writers.items()}


def main(out: Path = OUT, corpus: list[CorpusFile] | None = None) -> int:
    setup_logging("WARNING", log_dir=PATHS.logs, stage="tally")
    corpus = full_corpus() if corpus is None else corpus
    if not corpus:
        print("no corpus collected yet")
        return 1

    n_deposits = len({c.dataset_doi for c in corpus})
    by_source = collections.Counter(c.source for c in corpus)
    print(f"{len(corpus):,} files from {n_deposits:,} deposits")
    for source, n in sorted(by_source.items()):
        print(f"  {source:<20}{n:>9,} files")
    sources = sorted(by_source)

    agg, rows = run_batches(corpus, out)
    packages = agg.packages

    # -- usage_by_package: the tally ---------------------------------------
    # Pooled counts, with the per-source split beside them rather than in a
    # footnote. The two halves are very different sizes, so a pooled number
    # with no breakdown asks the reader to take the composition on trust.
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

    tally = sorted(by_package.values(), key=lambda r: -r["n_deposits"])
    for row in tally:
        denom = len(agg.at_risk.get(row["language"], ()))
        row["n_deposits_at_risk"] = denom
        row["share_of_deposits"] = (
            round(row["n_deposits"] / denom, 4) if denom else None
        )
        for source in sources:
            row[f"n_at_risk_{source}"] = len(
                agg.at_risk_by_source.get((source, row["language"]), ())
            )
    write_csv(tally, out / "usage_by_package.csv")

    # -- by collection: the per-journal view --------------------------------
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
        out / "usage_by_collection.csv",
    )

    # -- by function ---------------------------------------------------------
    functions = sorted(
        (
            {k: v for k, v in row.items() if k != "_deposits"}
            | {"n_deposits": len(row["_deposits"])}
            for row in agg.by_function.values()
        ),
        key=lambda r: (-r["n_deposits"], -r["n_calls"]),
    )
    write_csv(functions, out / "usage_by_function.csv")

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
        out / "usage_by_package_year.csv",
    )

    # -- language presence and unresolved names -----------------------------
    write_csv(
        [
            {"source": source, "language": language, "n_deposits": len(dois)}
            for (source, language), dois in sorted(agg.presence.items())
        ],
        out / "language_presence.csv",
    )
    unknown = sorted(agg.unknown.items(), key=lambda kv: -kv[1])
    write_csv(
        [
            {"name": name, "language": language, "n_mentions": n}
            for (name, language), n in unknown
        ],
        out / "unknown_names.csv",
    )

    # -- the environment layer ----------------------------------------------
    # Sparse by nature, so the denominator ships with it. A count of deposits
    # running Stata 14 means nothing without the number that said anything at
    # all, and a reader who has to compute that themselves will not.
    coverage = coverage_counts(agg.carrying, agg.eligible)
    (out / "environment_coverage.json").write_text(
        json.dumps(coverage, indent=1) + "\n"
    )

    print(f"deposits collected : {n_deposits:,}")
    print(f"files              : {rows['files']:,}")
    print(f"mentions           : {rows['mentions']:,}")
    print(f"packages tallied   : {len(tally):,}")
    print(f"unresolved names   : {len(unknown):,}")
    print(
        f"\ndeposits with analyzable code, by language: "
        f"{ {k: len(v) for k, v in sorted(agg.at_risk.items())} }"
    )
    print(f"\nwritten to {out}/")
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

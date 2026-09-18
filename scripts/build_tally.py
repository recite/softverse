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
from typing import TYPE_CHECKING

from softverse.build.aggregate import aggregate
from softverse.build.downloads import SOURCES, downloads_vs_use
from softverse.build.pipeline import build, hash_corpus
from softverse.config import PATHS
from softverse.corpus.loaders import full_corpus
from softverse.logging_setup import get_logger, setup_logging
from softverse.model.io import TableAppender
from softverse.registries.load import load_registry
from softverse.registries.lock import read_lock

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from softverse.build.pipeline import CorpusFile

logger = get_logger(__name__)

OUT = PATHS.root / "build" / "tally"

#: Deposits per `build` call. Measured on the 2026 corpus: 2,000 deposits
#: peaked at 1.9 GB, which scaled to the whole corpus is about 20 GB against
#: 17 GB of RAM. 250 keeps a batch well under a gigabyte.
BATCH_DEPOSITS = 250

#: The atomic tables, appended one batch at a time.
STREAMED = ("mentions", "files", "declared_dependencies", "environment_signals")


def batches(corpus: list[CorpusFile], size: int) -> Iterator[list[CorpusFile]]:
    """Consecutive groups of ``size`` deposits, each deposit's files together.

    A deposit is never split: `build` resolves Stata programs, siblings and
    within-deposit duplicates across a deposit's files, so half a deposit
    would be judged differently from the whole.

    Args:
        corpus: The files, in corpus order.
        size: Deposits per batch.

    Yields:
        list[CorpusFile]: The files of the next ``size`` deposits, in order.
    """
    by_deposit: dict[str, list[CorpusFile]] = {}
    for item in corpus:
        by_deposit.setdefault(item.dataset_doi, []).append(item)
    deposits = list(by_deposit.values())
    for start in range(0, len(deposits), size):
        yield [item for files in deposits[start : start + size] for item in files]


def run_batches(corpus: list[CorpusFile], out: Path) -> dict[str, int]:
    """Build the corpus a batch at a time, writing the atomic tables as it goes.

    Args:
        corpus: Every file to tally.
        out: Where the Parquet tables are written.

    Returns:
        The row count of each written table.
    """
    registry, shipped = load_registry()
    # Hashed once over everything: whether a file is vendored depends on its
    # bytes recurring in other deposits, which no batch can see on its own.
    hashes = hash_corpus(corpus)
    writers = {name: TableAppender(name, out) for name in STREAMED}
    for batch in batches(corpus, BATCH_DEPOSITS):
        result = build(
            batch,
            registry,
            ssc_shipped=shipped,
            registry_lock_id=registry.lock_id,
            hashes=hashes,
        )
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
    return {name: writer.rows for name, writer in writers.items()}


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

    rows = run_batches(corpus, out)
    # The aggregates are one SQL definition over the tables just written, the
    # same one the weekly update runs over the shards on the Hub.
    counts = aggregate(
        str(out / "files.parquet"),
        str(out / "mentions.parquet"),
        str(out / "environment_signals.parquet"),
        out,
    )

    # Beside the counts, what the registries' own download counts say about
    # the same packages. Skipped until the counts have been fetched and pinned.
    lock = read_lock()
    if all(registry in lock for _, registry, _ in SOURCES.values()):
        downloads_vs_use(out / "usage_by_package.csv", out, lock)

    print(f"deposits collected : {n_deposits:,}")
    print(f"files              : {rows['files']:,}")
    print(f"mentions           : {rows['mentions']:,}")
    print(f"packages tallied   : {counts['usage_by_package']:,}")
    print(f"unresolved names   : {counts['unknown_names']:,}")
    print(f"\nwritten to {out}/")
    print("\n=== TOP 15 BY DEPOSITS ===")
    with (out / "usage_by_package.csv").open(encoding="utf-8") as handle:
        top = [row for _, row in zip(range(15), csv.DictReader(handle), strict=False)]
    sources = sorted(by_source)
    split = "  ".join(f"{s.split('_')[0]:>9}" for s in sources)
    print(f"  {'pooled':>9} {'of':<6} {'lang':<7} {'package':<16}{split}")
    for row in top:
        counts_by_source = "  ".join(
            f"{int(row[f'n_deposits_{s}']):>9,}" for s in sources
        )
        print(
            f"  {int(row['n_deposits']):>9,} {int(row['n_deposits_at_risk']):<6,} "
            f"{row['language']:<7} {row['package']:<16}{counts_by_source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

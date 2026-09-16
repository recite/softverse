"""How much of the tally survives if only code a master script reaches counts.

    uv run python scripts/reachability.py [--limit N]

Writes `build/validation/reachability.json` and `reachable_files.parquet`.

The headline measure counts a package when a deposit's code references it
anywhere. The obvious objection is that deposits carry code nobody runs: an
earlier draft of an analysis, a helper superseded twice, a script for a figure
that did not make the paper. This bounds that objection instead of arguing
with it. Each deposit's scripts are linked by what they run -- `source()`,
`do`, `run`, `include`, a local import -- and the whole tally is recomputed
over the files a master script can reach.

Three definitions, stated because the number means nothing without them:

- a **master** is an analyzable file that runs at least one other file in the
  deposit and that nothing in the deposit runs;
- **reachable** is the closure from the masters, masters included;
- an **orphan** is an analyzable file in a deposit that has a master, which no
  master reaches.

A deposit with no master -- one script, or a set of scripts none of which
calls another -- has nothing to bound, and its files are left out of the
reachable/orphan split rather than counted as either. That is the honest
treatment: the analysis can only speak about deposits that wrote down an entry
point.

The headline is computed on **complete** graphs only: deposits with a master
and with no edge we failed to follow. The reason is visible in the first
deposit I opened, whose master sets `global mypath` from a network drive and
then runs its steps through that macro. Its steps are reached in truth and
unreachable to any static reader, so counting them as orphans would measure
Stata's macro idiom and report it as dead code. Deposits with such an edge are
counted, reported, and kept out of the headline; the wider figure that
includes them is reported beside it as the floor it is.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from softverse.config import PATHS
from softverse.detect.dispatch import decode
from softverse.detect.edges import includes, resolve
from softverse.logging_setup import get_logger, setup_logging
from softverse.model.enums import Language

logger = get_logger(__name__)

TALLY = PATHS.root / "build" / "tally"
OUT = PATHS.root / "build" / "validation"

#: Languages whose files can run another file in the deposit.
GRAPH_LANGUAGES = ("r", "rmarkdown", "stata", "python")

#: Packages per language to compare between the two tallies.
TOP_N = 20


def deposit_graph(files: pd.DataFrame, every_path: list[str]) -> dict:
    """Masters, reachable files and orphans for one deposit.

    Args:
        files: The deposit's analyzable files, with `relative_path`,
            `file_uid`, `language` and `local_path`.
        every_path: Every path in the deposit, analyzable or not, so an edge
            to a file we hold but do not analyze counts as followed rather
            than as a hole in the graph.

    Returns:
        Counts, the reachable `file_uid` set, and how many edges could not be
        followed.
    """
    paths = list(files["relative_path"])
    uid_of = dict(zip(files["relative_path"], files["file_uid"], strict=True))
    out_edges: dict[str, set[str]] = defaultdict(set)
    unresolved = 0
    unfollowed = 0

    for row in files.itertuples():
        try:
            data = Path(row.local_path).read_bytes()
        except OSError:
            continue
        found = includes(decode(data).text, Language(row.language))
        unresolved += found.n_unresolved
        for target in found.targets:
            hit = resolve(target, row.relative_path, every_path)
            if hit is None:
                unfollowed += 1
            elif hit != row.relative_path and hit in uid_of:
                out_edges[row.relative_path].add(hit)
        # An import that matches no file is an installed package, so it is not
        # a hole in the graph and must not be counted as one.
        for name in found.imports:
            hit = resolve(name, row.relative_path, every_path)
            if hit is not None and hit != row.relative_path and hit in uid_of:
                out_edges[row.relative_path].add(hit)

    included = {target for targets in out_edges.values() for target in targets}
    masters = [p for p in out_edges if p not in included]

    reachable: set[str] = set()
    stack = list(masters)
    while stack:
        path = stack.pop()
        if path in reachable:
            continue
        reachable.add(path)
        stack.extend(out_edges.get(path, ()))

    return {
        "n_files": len(paths),
        "n_masters": len(masters),
        "n_edges": sum(len(t) for t in out_edges.values()),
        "n_macro_edges": unresolved,
        "n_unfollowed_edges": unfollowed,
        "n_reachable": len(reachable),
        "reachable_uids": [uid_of[p] for p in reachable],
    }


def usage(connection: duckdb.DuckDBPyConnection, where: str) -> pd.DataFrame:
    """Deposits per package, over the mentions `where` admits."""
    return connection.execute(
        f"""
        SELECT language, resolved_package AS package,
               count(DISTINCT dataset_doi) AS n_deposits
        FROM '{TALLY}/mentions.parquet'
        WHERE resolved_package IS NOT NULL
          AND resolution NOT IN ('builtin', 'base_or_stdlib', 'local_program',
                                 'local_relative', 'dynamic')
          AND {where}
        GROUP BY ALL
        """
    ).df()


def comparison(connection: duckdb.DuckDBPyConnection, reachable: Path) -> dict:
    """The top packages either way, and how far each moved."""
    connection.execute(
        f"CREATE OR REPLACE TABLE reachable AS SELECT * FROM '{reachable}'"
    )
    # Both tallies over the same deposits: the ones with a master. Comparing a
    # reachable-only count against the full corpus would confound the question
    # asked here with which deposits wrote down an entry point.
    scoped = (
        "dataset_doi IN (SELECT DISTINCT dataset_doi FROM reachable WHERE complete) "
        "AND file_uid IN (SELECT file_uid FROM reachable WHERE complete)"
    )
    everything = usage(connection, scoped)
    reached = usage(
        connection,
        f"{scoped} AND file_uid IN "
        "(SELECT file_uid FROM reachable WHERE complete AND is_reachable)",
    )

    out = {}
    for language in ("r", "stata", "python"):
        left = everything[everything["language"] == language].nlargest(
            TOP_N, "n_deposits"
        )
        right = reached[reached["language"] == language].set_index("package")
        rows = []
        for rank, row in enumerate(left.itertuples(), start=1):
            kept = int(right["n_deposits"].get(row.package, 0))
            rows.append(
                {
                    "package": row.package,
                    "rank": rank,
                    "n_deposits": int(row.n_deposits),
                    "n_deposits_reachable_only": kept,
                    "retained": kept / row.n_deposits if row.n_deposits else 1.0,
                }
            )
        ranked = sorted(rows, key=lambda r: -r["n_deposits_reachable_only"])
        for new_rank, row in enumerate(ranked, start=1):
            row["rank_reachable_only"] = new_rank
        out[language] = {
            "top": rows,
            "max_rank_move": max(
                abs(r["rank"] - r["rank_reachable_only"]) for r in rows
            )
            if rows
            else 0,
            "min_retained": min((r["retained"] for r in rows), default=1.0),
            "median_retained": float(pd.Series([r["retained"] for r in rows]).median())
            if rows
            else 1.0,
        }
    return out


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="reachability")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0

    everything = pd.read_parquet(
        TALLY / "files.parquet",
        columns=[
            "file_uid",
            "dataset_doi",
            "relative_path",
            "language",
            "in_analysis_set",
            "local_path",
        ],
    )
    paths_by_deposit = {
        doi: list(group["relative_path"])
        for doi, group in everything.groupby("dataset_doi", observed=True)
    }
    files = everything[everything["in_analysis_set"].astype(bool)]
    files = files[files["language"].astype(str).isin(GRAPH_LANGUAGES)]
    logger.info("graphing", extra={"files": len(files)})

    rows, reachable_rows = [], []
    deposits = list(files.groupby("dataset_doi", observed=True))
    if limit:
        deposits = deposits[:limit]
    for n, (doi, group) in enumerate(deposits, start=1):
        if len(group) < 2:
            continue
        graph = deposit_graph(group, paths_by_deposit[doi])
        if not graph["n_masters"]:
            continue
        reachable = set(graph.pop("reachable_uids"))
        complete = not graph["n_macro_edges"] and not graph["n_unfollowed_edges"]
        rows.append({"dataset_doi": doi, "complete": complete, **graph})
        reachable_rows.extend(
            {
                "dataset_doi": doi,
                "file_uid": uid,
                "is_reachable": uid in reachable,
                "complete": complete,
            }
            for uid in group["file_uid"]
        )
        if n % 500 == 0:
            logger.info("graphed", extra={"deposits": n, "with_master": len(rows)})

    if not rows:
        print("no deposit had a master script; is the corpus present?")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "reachable_files.parquet"
    pq.write_table(pa.Table.from_pylist(reachable_rows), path)

    graphed = pd.DataFrame(rows)
    whole = graphed[graphed["complete"]]
    connection = duckdb.connect()
    report = {
        "definition": (
            "a master runs another file and is run by none; reachable is the "
            "closure from the masters; the headline covers deposits with a "
            "master and no edge we could not follow"
        ),
        "n_deposits_considered": len(deposits),
        "n_deposits_with_master": len(graphed),
        "n_deposits_complete": len(whole),
        "complete": _split(whole),
        # The floor: deposits whose graph has a hole are kept here, and every
        # file an unfollowable edge would have reached counts as an orphan.
        "with_holes_too": _split(graphed),
        "n_edges": int(graphed["n_edges"].sum()),
        "n_macro_edges": int(graphed["n_macro_edges"].sum()),
        "n_unfollowed_edges": int(graphed["n_unfollowed_edges"].sum()),
        "usage": comparison(connection, path),
    }
    (OUT / "reachability.json").write_text(json.dumps(report, indent=1) + "\n")

    print(f"deposits with a master : {len(graphed):,} of {len(deposits):,}")
    print(f"  of those, complete   : {len(whole):,}")
    for name in ("complete", "with_holes_too"):
        stats = report[name]
        print(
            f"  {name:<15} {stats['n_reachable']:,} of {stats['n_files']:,} files "
            f"reachable ({stats['share_reachable']:.1%}), "
            f"median deposit {stats['median_share_reachable']:.1%}"
        )
    print(
        f"edges                  : {report['n_edges']:,} followed, "
        f"{report['n_macro_edges']:,} through a macro, "
        f"{report['n_unfollowed_edges']:,} to no file we hold"
    )
    for language, stats in report["usage"].items():
        print(
            f"  {language:<8} top {TOP_N}: median {stats['median_retained']:.1%} of "
            f"deposits retained, worst {stats['min_retained']:.1%}, "
            f"largest rank move {stats['max_rank_move']}"
        )
    return 0


def _split(graphed: pd.DataFrame) -> dict:
    """Reachable-file counts over a set of deposit graphs."""
    if graphed.empty:
        return {"n_deposits": 0}
    return {
        "n_deposits": len(graphed),
        "n_files": int(graphed["n_files"].sum()),
        "n_reachable": int(graphed["n_reachable"].sum()),
        "share_reachable": float(
            graphed["n_reachable"].sum() / graphed["n_files"].sum()
        ),
        "median_share_reachable": float(
            (graphed["n_reachable"] / graphed["n_files"]).median()
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())

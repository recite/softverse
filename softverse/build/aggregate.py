"""The tally's aggregate tables, computed once, in SQL, from the atomic tables.

There used to be two definitions of every count: a Python fold over batches
in `scripts/build_tally.py`, and DuckDB recomputations in the release checks
that had to agree with it. Two definitions of one quantity is one too many,
and it was also what stood in the way of updating the tally without holding
the whole corpus: the fold needed every row in memory, in order.

This is the one definition. It reads Parquet -- one file or a glob of dated
parts, local or on the Hub -- and writes the six aggregates and the coverage
denominators the site, the paper and the release all read. `build_tally.py`
calls it after streaming the atomic tables; the weekly update calls it over
the shards on the Hub.

Row order is deterministic here (count, then name), where the fold's was the
order deposits happened to be read in. The rows are the same.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from softverse.model.enums import NON_USE_CONSTRUCTS, Language, Resolution

if TYPE_CHECKING:
    from pathlib import Path

COUNTABLE = (str(Resolution.KNOWN_CURRENT), str(Resolution.KNOWN_ARCHIVED))
NON_USE = tuple(str(c) for c in NON_USE_CONSTRUCTS)

#: Which languages put a deposit in a position to declare each environment
#: signal. A deposit with no Python in it was never a candidate to state a
#: Python version, and counting it as a miss understates the coverage.
SIGNAL_LANGUAGES = {
    "r_version": (str(Language.R), str(Language.RMARKDOWN), str(Language.NOTEBOOK)),
    "os": (str(Language.R), str(Language.RMARKDOWN)),
    "python_version": (str(Language.PYTHON), str(Language.NOTEBOOK)),
    "julia_version": (str(Language.JULIA), str(Language.NOTEBOOK)),
    "matlab_version": (str(Language.NOTEBOOK),),
    "stata_version": (str(Language.NOTEBOOK),),
}

AGGREGATES = (
    "usage_by_package",
    "usage_by_package_year",
    "usage_by_collection",
    "usage_by_function",
    "unknown_names",
    "remote_installs",
    "language_presence",
)


def _list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def write_csv(rows: list[dict], path: Path) -> None:
    """Write an aggregate table as CSV, and as Parquet beside it.

    Both, because the two readers want different things and the tables are
    small enough that choosing is a false economy. A person checking whether
    their package is counted correctly opens the CSV; anything loading it as
    data wants the Parquet, where the types are in the file.

    `newline=""` stops the module rewriting line endings and `lineterminator`
    decides them: the csv default is CRLF, which git stored as LF, so a
    regenerated table never matched its committed copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)


def aggregate(files: str, mentions: str, environment: str, out: Path) -> dict[str, int]:
    """Write the six aggregates and `environment_coverage.json` under ``out``.

    Args:
        files: Parquet path or glob for the `files` table.
        mentions: Parquet path or glob for the `mentions` table.
        environment: Parquet path or glob for `environment_signals`.
        out: Where the tables go.

    Returns:
        Rows written per table.
    """
    con = duckdb.connect()
    con.execute(f"CREATE VIEW files AS SELECT * FROM '{files}'")
    con.execute(f"CREATE VIEW mentions AS SELECT * FROM '{mentions}'")
    con.execute(f"CREATE VIEW environment AS SELECT * FROM '{environment}'")
    sources = [
        r[0]
        for r in con.execute("SELECT DISTINCT source FROM files ORDER BY 1").fetchall()
    ]

    # The headline unit: one row per (deposit, language, package), counting
    # only resolutions that name a real third-party package and only
    # constructs that use rather than install. `os` resolves to `os`, so a
    # presence check would let every stdlib mention through.
    con.execute(
        f"""
        CREATE TABLE packages AS
        SELECT dataset_doi, any_value(source) AS source,
               any_value(collection_id) AS collection_id,
               any_value(deposit_year) AS year, language,
               resolved_package AS package, any_value(ecosystem) AS ecosystem,
               count(DISTINCT file_uid) AS n_files, count(*) AS n_mentions
        FROM mentions
        WHERE resolved_package IS NOT NULL
          AND resolution IN ({_list(COUNTABLE)})
          AND construct NOT IN ({_list(NON_USE)})
        GROUP BY dataset_doi, language, resolved_package
        """
    )
    # The denominator travels with the count: deposits at risk of using an R
    # package are those containing analyzable R, or that yielded an R mention
    # (which is what covers notebooks and knitr documents).
    con.execute(
        """
        CREATE TABLE at_risk AS
        SELECT DISTINCT source, language, dataset_doi FROM files WHERE in_analysis_set
        UNION
        SELECT DISTINCT source, language, dataset_doi FROM mentions
        """
    )

    counts: dict[str, int] = {}

    per_source = ", ".join(
        f"count(*) FILTER (WHERE source = '{s}') AS n_deposits_{s}" for s in sources
    )
    usage = con.execute(
        f"""
        SELECT package, language, any_value(ecosystem) AS ecosystem,
               count(*) AS n_deposits, sum(n_files)::BIGINT AS n_files,
               sum(n_mentions)::BIGINT AS n_mentions,
               min(year) AS first_year, max(year) AS last_year, {per_source}
        FROM packages GROUP BY package, language
        ORDER BY n_deposits DESC, language, package
        """
    ).df()
    denominators = {
        (s, lang): n
        for s, lang, n in con.execute(
            "SELECT source, language, count(DISTINCT dataset_doi) FROM at_risk "
            "GROUP BY 1, 2"
        ).fetchall()
    }
    pooled = dict(
        con.execute(
            "SELECT language, count(DISTINCT dataset_doi) FROM at_risk GROUP BY 1"
        ).fetchall()
    )
    rows = []
    for record in usage.to_dict("records"):
        denominator = pooled.get(record["language"], 0)
        record["n_deposits_at_risk"] = denominator
        record["share_of_deposits"] = (
            round(record["n_deposits"] / denominator, 4) if denominator else None
        )
        for s in sources:
            record[f"n_at_risk_{s}"] = denominators.get((s, record["language"]), 0)
        rows.append(_clean(record))
    write_csv(rows, out / "usage_by_package.csv")
    counts["usage_by_package"] = len(rows)

    for name, query in (
        (
            "usage_by_collection",
            """
            SELECT source, collection_id, language, package, count(*) AS n_deposits
            FROM packages GROUP BY ALL
            ORDER BY collection_id, language, n_deposits DESC, package
            """,
        ),
        (
            "usage_by_function",
            f"""
            SELECT source, language, resolved_package AS package,
                   called_function AS function, count(*) AS n_calls,
                   count(DISTINCT dataset_doi) AS n_deposits
            FROM mentions
            WHERE resolution IN ({_list(COUNTABLE)})
              AND resolved_package IS NOT NULL
              AND construct NOT IN ({_list(NON_USE)})
              AND called_function IS NOT NULL
            GROUP BY ALL
            ORDER BY n_deposits DESC, n_calls DESC, package, function
            """,
        ),
        (
            "usage_by_package_year",
            """
            SELECT package, language, year, count(*) AS n_deposits
            FROM packages WHERE year IS NOT NULL
            GROUP BY ALL ORDER BY package, year, language
            """,
        ),
        (
            "language_presence",
            f"""
            SELECT source, language, count(DISTINCT dataset_doi) AS n_deposits
            FROM files
            WHERE language IS NOT NULL AND language != '{Language.UNKNOWN}'
            GROUP BY ALL ORDER BY source, language
            """,
        ),
        (
            "unknown_names",
            # Uses only. `ssc install blindschemes` is a stated dependency,
            # not a call, and counting it here ranked packages among the
            # software "used but in no registry" on the strength of the line
            # that names the registry they are in.
            #
            # `n_deposits_defining` is how many *other* deposits define a
            # program of this name. A name many authors independently give
            # their own helper is more likely one here too, with a `program
            # define` the lexer did not reach, than a package nobody indexed.
            f"""
            WITH defined AS (
                SELECT lower(raw_name) AS name, language,
                       count(DISTINCT dataset_doi) AS n_deposits_defining
                FROM mentions WHERE resolution = '{Resolution.LOCAL_PROGRAM}'
                GROUP BY ALL
            )
            SELECT raw_name AS name, m.language,
                   count(DISTINCT dataset_doi) AS n_deposits,
                   count(*) AS n_mentions,
                   coalesce(max(n_deposits_defining), 0) AS n_deposits_defining
            FROM mentions m
            LEFT JOIN defined d
              ON d.name = lower(m.raw_name) AND d.language = m.language
            WHERE resolution = '{Resolution.UNKNOWN}'
              AND construct NOT IN ({_list(NON_USE)})
            GROUP BY ALL ORDER BY n_deposits DESC, n_mentions DESC, name, language
            """,
        ),
        (
            "remote_installs",
            # What deposits fetch from somewhere other than their language's
            # registry, and whether they go on to use it. `in_registry` is
            # whether the registry has the name anyway -- a development
            # version of a CRAN package, not software CRAN lacks.
            f"""
            WITH installs AS (
                SELECT DISTINCT dataset_doi, language, raw_name AS name,
                       -- A `from()` can be a macro or a folder on the author's
                       -- machine; neither is a host.
                       CASE WHEN regexp_matches(remote, '^([a-z]+://|[a-z0-9.-]+[.][a-z]{{2,}}/)')
                            THEN lower(split_part(
                                regexp_replace(remote, '^[a-z]+://(www[.])?', ''), '/', 1))
                            ELSE '(local path or macro)' END AS host,
                       ecosystem IS NOT NULL
                         AND ecosystem NOT IN ('github', 'gitlab', 'bitbucket')
                         AS in_registry
                FROM mentions WHERE remote IS NOT NULL
            ), loads AS (
                SELECT DISTINCT dataset_doi, language, raw_name AS name
                FROM mentions WHERE construct NOT IN ({_list(NON_USE)})
            )
            SELECT i.name, i.language, i.host, bool_or(i.in_registry) AS in_registry,
                   count(DISTINCT i.dataset_doi) AS n_deposits_installing,
                   count(DISTINCT l.dataset_doi) AS n_deposits_loading
            FROM installs i LEFT JOIN loads l USING (dataset_doi, language, name)
            GROUP BY ALL
            ORDER BY n_deposits_installing DESC, i.name, i.language, i.host
            """,
        ),
    ):
        table = [_clean(r) for r in con.execute(query).df().to_dict("records")]
        write_csv(table, out / f"{name}.csv")
        counts[name] = len(table)

    carrying = dict(
        con.execute(
            "SELECT signal, count(DISTINCT dataset_doi) FROM environment GROUP BY 1"
        ).fetchall()
    )
    eligible = {}
    for signal, languages in SIGNAL_LANGUAGES.items():
        extra = " OR extension = '.do'" if signal == "stata_version" else ""
        row = con.execute(
            f"""
            SELECT count(DISTINCT dataset_doi) FROM files
            WHERE in_analysis_set AND (language IN ({_list(languages)}){extra})
            """
        ).fetchone()
        eligible[signal] = row[0] if row else 0
    coverage = {
        signal: {
            "deposits_carrying": int(carrying.get(signal, 0)),
            "deposits_eligible": int(eligible.get(signal, 0)),
        }
        for signal in sorted(set(carrying) | set(eligible))
    }
    (out / "environment_coverage.json").write_text(
        json.dumps(coverage, indent=1) + "\n"
    )
    return counts


def _clean(record: dict) -> dict:
    """Plain Python values: ints for counts, None for missing years."""
    out = {}
    for key, value in record.items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out

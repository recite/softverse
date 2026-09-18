"""Package the per-package counts for release.

    uv run python scripts/release_tally.py

Writes `data/tally/`: the aggregate tables, the mention rows they
are sums of, a summary of the corpus they were computed on, and a
frictionless datapackage. The Parquet is deliberately untracked; git carries
the 150 KB of CSV that the site and the paper read, and Zenodo carries the
94 MB the recount needs.

This exists to separate two things that were tangled. `build/tally/` holds
57 MB of Parquet derived from 499 GB of downloaded deposits, and none of it
can be tracked or rebuilt anywhere but this machine. The tables a reader
actually wants are 150 KB of CSV. Splitting them means the published site and
the released data can be built from the repository alone, and a number on the
site cannot drift from the number in the paper, because both read this.

The counts pool both repositories, Zenodo and Harvard Dataverse, and every
row carries the per-source split beside the pooled total. Three checks gate
the release: the denominators are recomputed from the raw Parquet by the
documented rule, the pooled counts must reconcile with their own split, and
the ranking must survive being recomputed on the file types both corpora
collected.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from softverse.config import PATHS

if TYPE_CHECKING:
    from pathlib import Path

TALLY = PATHS.root / "build" / "tally"
ZENODO_DEPOSITS = PATHS.root / "corpus" / "zenodo" / "deposits.csv"
OUT = PATHS.root / "data" / "tally"
FRAME = PATHS.root / "data" / "frame"
README = PATHS.root / "README.md"
README_START, README_END = (
    "<!-- release-numbers:start -->",
    "<!-- release-numbers:end -->",
)
LANGUAGE_NAMES = {"stata": "Stata", "r": "R", "python": "Python"}

#: Each aggregate ships as CSV and as Parquet. The CSV is for a person
#: opening it; the Parquet is for anything reading it as data, and it is not
#: redundant. `usage_by_function` records a call to `numpy.NaN`, a real numpy
#: attribute, which pandas reads back out of the CSV as a missing value.
AGGREGATES = (
    "usage_by_package",
    "usage_by_package_year",
    "usage_by_collection",
    "usage_by_function",
    "unknown_names",
    "remote_installs",
    "language_presence",
)

TABLES = (
    *(f"{name}.csv" for name in AGGREGATES),
    *(f"{name}.parquet" for name in AGGREGATES),
    # The atomic record, which the aggregates above are all sums of. Held back
    # until now, which is why the one project consuming this corpus re-parsed
    # 217,573 files to recover what the build had already computed.
    "mentions.parquet",
    "files.parquet",
    "declared_dependencies.parquet",
    "environment_signals.parquet",
)

DESCRIPTOR = """\
# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**{n_packages:,} packages · {n_deposits_analyzable:,} deposits with analyzable
code · {n_deposits:,} deposits collected · built {built}**

A count here is the number of deposits whose code loads the package. Adding
one to it takes a paper published at a journal that checks its authors' code,
which is what makes these counts harder to inflate than download counts. The
[project page](https://recite.github.io/softverse/) makes that case; this
file documents what is in the tables and how to read them.

## Scope

The deposits come from two repositories that hold different disciplines.
Zenodo's verified collections are economics, and Harvard Dataverse's journal
collections are mostly political science.

{composition}

Counts pool the two. `usage_by_package.csv` also carries the split, in
`n_deposits_zenodo` and `n_deposits_dataverse`, because the two are
very different sizes and a pooled figure alone would hide that.

Both repositories were collected with the same rules in 2026: every code,
notebook and knitr file and every dependency manifest, with the deposit's own
directories, and code recovered from archives too large to download. An
archive that could not be read is counted in `summary.json` rather than
dropped. The two collections still differ in discipline, which the per-source
columns keep visible.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | {n_packages:,} | per-package deposit and call counts, pooled and split |
| `usage_by_package_year.csv` | {n_years:,} | the same by deposit year |
| `usage_by_collection.csv` | {n_collection_rows:,} | the same per journal or community |
| `usage_by_function.csv` | {n_functions:,} | package → function, where the source names one |
| `unknown_names.csv` | {n_unknown:,} | names called in code that resolve to no registry |
| `remote_installs.csv` | {n_remote_installs:,} | what deposits install from outside their registry, and from where |
| `language_presence.csv` | {n_languages} | deposits containing each language, per repository |
| `mentions.parquet` | {n_mentions:,} | every mention: package, function, file, line, snippet |
| `files.parquet` | {n_files:,} | the provenance spine every mention joins to |
| `declared_dependencies.parquet` | {n_declarations:,} | what manifests declare: shipped, locked or asked for |
| `environment_signals.parquet` | {n_signals:,} | R, Python and Stata versions, and the OS, where a file says |
| `environment_coverage.json` | | deposits stating each signal, over deposits that could |
| `summary.json` | | corpus counts the tables are shares of |

`mentions.parquet` is the record every count above is a sum of, and it is
here so a reader who disagrees with a decision made upstream can recount
without re-parsing 200,000 files. It is 94 MB; the CSVs are 150 KB.

`declared_dependencies.parquet` and `environment_signals.parquet` are sparse
and answer a different question: not what the code loads but what version of
it the deposit shipped, and what ran it. Most deposits say nothing at all, so
read these next to `environment_coverage.json`, which gives per signal the
deposits that said something over the deposits that were in a position to.

### `usage_by_package.csv`

- `package`, `language`, `ecosystem`: the resolved package and its registry
- `n_deposits`: deposits loading it, counted once per deposit
- `n_files`, `n_mentions`: files, and raw calls
- `n_deposits_at_risk`: the denominator for `share_of_deposits`, meaning deposits
  that hold an analyzable file in that language, or that yielded a reference
  in it. The second clause is what covers literate documents, where the file
  is a notebook and the code inside it is Python. It differs by language, so a
  Stata share and an R share are not shares of the same thing
- `share_of_deposits`: `n_deposits` divided by `n_deposits_at_risk`

### `unknown_names.csv`

Names that code *uses* and that resolve to no registry, unfiltered. Install
and inquiry lines are excluded: `ssc install x` states a dependency and is not
a call. Some rows are false positives, and some are programs a deposit defines
for itself. Pruning the list by hand would put a judgement call inside a file
whose value is that you can check every row of it.

- `name`, `language`
- `n_deposits`, `n_mentions`: deposits using the name, and raw uses. Rank by
  deposits: one deposit calling something six hundred times is one user of it
- `n_deposits_defining`: how many deposits define a Stata program of this
  name for themselves. A name many authors independently give a helper is more
  likely one here too, with a `program define` the lexer did not reach, than
  it is software nobody indexed

### `remote_installs.csv`

What deposits fetch from somewhere other than their language's registry:
`remotes::install_github("user/repo")`, `net install x, from(URL)`,
`pip install git+https://...`. It is the only record a deposit leaves of where
off-registry software lives.

- `name`, `language`, `host`: the package and the host it is fetched from
- `in_registry`: the registry lists the name anyway, so this is a development
  version of a registered package rather than software the registry lacks
- `n_deposits_installing`, `n_deposits_loading`: deposits with the install
  line, and those among them that go on to use the package

## Licence

CC0.

## Regenerating

```bash
uv run python scripts/build_tally.py     # needs the collected corpus
uv run python scripts/release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).
"""


#: Reader-facing names for the sources, and what each one is.
SOURCE_LABEL = {
    "zenodo": "Zenodo (economics)",
    "dataverse": "Harvard Dataverse (political science)",
}


def _composition_table(summary: dict) -> str:
    """The split, as a table, so the pooled totals are never bare."""
    rows = [
        "| repository | deposits | with analyzable code |",
        "|---|---:|---:|",
    ]
    for source, n in summary["deposits_by_source"].items():
        analyzable = summary["deposits_analyzable_by_source"].get(source, 0)
        rows.append(f"| {SOURCE_LABEL.get(source, source)} | {n:,} | {analyzable:,} |")
    rows.append(
        f"| **total** | **{summary['n_deposits']:,}** | "
        f"**{summary['n_deposits_analyzable']:,}** |"
    )
    return "\n".join(rows)


#: pandas dtype kind -> frictionless field type.
_FIELD_TYPE = {"i": "integer", "u": "integer", "f": "number", "b": "boolean"}


def _resource(path: Path) -> dict:
    """A frictionless resource, described from the file rather than by hand.

    The hand-written version listed three of the five tables that shipped and
    named columns that had since been renamed. Reading the header is the only
    version that cannot go stale, which is the same rule the rest of this
    project applies to numbers.
    """
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path).head(0)
    else:
        frame = pd.read_csv(path, nrows=0)
    return {
        # Every aggregate ships twice, and resource names must be unique.
        "name": f"{path.stem}_{path.suffix.lstrip('.')}",
        "path": path.name,
        "format": path.suffix.lstrip("."),
        "schema": {
            "fields": [
                {"name": str(c), "type": _FIELD_TYPE.get(frame[c].dtype.kind, "string")}
                for c in frame.columns
            ]
        },
    }


#: What a package's page and badge are built from, so the site builds from
#: tracked files in CI. Same rule as the headline count: resolved to a known
#: package, and not an install or an inquiry.
_USE = """
    resolution IN ('known_current', 'known_archived')
    AND construct NOT IN ('install', 'shell_install', 'stata_install', 'stata_which')
    AND resolved_package IS NOT NULL
"""


def write_package_tables() -> None:
    """Write the per-package deposit list and version summary.

    `package_deposits.parquet` has one row per (language, package, deposit),
    which is what lets a package page list the papers that use it.
    `package_versions.csv` counts deposits per stated version, from manifests
    and from install calls, so a maintainer can see which releases published
    research pinned.
    """
    con = duckdb.connect()
    mentions = f"'{TALLY / 'mentions.parquet'}'"
    con.execute(
        f"""
        COPY (
            SELECT language, resolved_package AS package, any_value(ecosystem)
                   AS ecosystem, dataset_doi, any_value(source) AS source,
                   any_value(collection_id) AS collection_id,
                   any_value(deposit_year) AS year
            FROM {mentions} WHERE {_USE}
            GROUP BY language, resolved_package, dataset_doi
            ORDER BY language, package, year, dataset_doi
        ) TO '{OUT / "package_deposits.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    rows = con.execute(
        f"""
        SELECT ecosystem, package, version, version_source,
               count(DISTINCT dataset_doi) AS n_deposits
        FROM (
            SELECT ecosystem, package, version_constraint AS version,
                   manifest_kind AS version_source, dataset_doi
            FROM '{TALLY / "declared_dependencies.parquet"}'
            WHERE version_constraint IS NOT NULL
            UNION ALL
            SELECT ecosystem, coalesce(resolved_package, raw_name), pinned_version,
                   construct, dataset_doi
            FROM {mentions} WHERE pinned_version IS NOT NULL
        )
        GROUP BY ALL ORDER BY ecosystem, package, n_deposits DESC
        """
    ).df()
    rows.to_csv(OUT / "package_versions.csv", index=False, lineterminator="\n")


def write_year_denominators() -> None:
    """Deposits with analyzable code per (language, deposit year).

    The denominator a package's share over time is taken against, by the
    tally's own rule: a deposit is at risk for a language when it holds an
    analyzable file in that language or a mention in it came out of it. Years
    are the frame dates the corpus loader stamps on every file, read from the
    same two files, so a package's per-year count and this denominator agree
    on which year a deposit belongs to. A deposit with no date is a row with
    an empty year rather than a dropped one.
    """
    con = duckdb.connect()
    con.execute(
        f"""
        CREATE TEMP TABLE deposit_year AS
        SELECT 'doi:10.7910/DVN/' || split_part(identifier, '/', -1) AS dataset_doi,
               TRY_CAST(left(publication_date, 4) AS INTEGER) AS year
        FROM read_csv_auto('{PATHS.frame / "dataverse_deposits.csv"}', all_varchar = true)
        UNION ALL
        SELECT dataset_doi, TRY_CAST(deposit_year AS INTEGER)
        FROM read_csv_auto('{ZENODO_DEPOSITS}', all_varchar = true)
        """
    )
    con.execute(
        f"""
        COPY (
            WITH at_risk AS (
                SELECT DISTINCT language, dataset_doi
                FROM '{TALLY / "files.parquet"}' WHERE in_analysis_set
                UNION
                SELECT DISTINCT language, dataset_doi FROM '{TALLY / "mentions.parquet"}'
            )
            SELECT a.language, y.year, count(DISTINCT a.dataset_doi) AS n_deposits_at_risk
            FROM at_risk a LEFT JOIN deposit_year y USING (dataset_doi)
            GROUP BY ALL ORDER BY a.language, y.year
        ) TO '{OUT / "language_year_at_risk.csv"}' (HEADER, DELIMITER ',')
        """
    )


def _year_denominators_agree() -> list[str]:
    """Per-year denominators must add up to the tally's per-language one.

    Returns:
        One message per language whose years do not sum to its total.
    """
    (n,) = (
        duckdb.connect()
        .execute(
            f"""
        SELECT count(*) FROM (
            SELECT language, sum(n_deposits_at_risk) AS n
            FROM read_csv_auto('{OUT / "language_year_at_risk.csv"}') GROUP BY 1
        ) y JOIN (
            SELECT DISTINCT language, n_deposits_at_risk
            FROM read_csv_auto('{OUT / "usage_by_package.csv"}')
        ) u USING (language)
        WHERE y.n <> u.n_deposits_at_risk
        """
        )
        .fetchone()
    )
    return [f"year denominators disagree with the tally in {n} languages"] if n else []


def _package_deposits_agree() -> list[str]:
    """The deposit list behind every page must count to the published tally.

    Returns:
        One message per disagreeing (language, package).
    """
    con = duckdb.connect()
    (n,) = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT language, package, count(*) AS n
            FROM '{OUT / "package_deposits.parquet"}' GROUP BY 1, 2
        ) d FULL JOIN read_csv_auto('{OUT / "usage_by_package.csv"}') u
          USING (language, package)
        WHERE d.n IS DISTINCT FROM u.n_deposits
        """
    ).fetchone()
    return (
        [f"package_deposits disagrees with usage_by_package on {n} packages"]
        if n
        else []
    )


def summarize() -> dict:
    """Corpus counts, computed here so nothing downstream opens the Parquet.

    `files.parquet` is 13 MB and cannot be tracked. Every number the site and
    the release descriptor state about the corpus comes out of it once, here.

    The per-language denominators are *read* from the tally rather than
    recomputed. Recomputing them here is what the first version did, and it
    reproduced the exact bug the tally had just been fixed for: counting file
    languages, so a deposit whose only Python lives in a notebook fell out of
    the Python denominator. Two definitions of one quantity is one too many.
    """
    files = pd.read_parquet(TALLY / "files.parquet")
    analyzable = files[files["in_analysis_set"].astype(bool)]
    usage = pd.read_csv(TALLY / "usage_by_package.csv")
    unknown = pd.read_csv(TALLY / "unknown_names.csv")

    by_language = (
        usage.groupby("language")["n_deposits_at_risk"]
        .max()
        .sort_values(ascending=False)
    )

    by_source = (
        analyzable.groupby("source", observed=True)["dataset_doi"]
        .nunique()
        .sort_values(ascending=False)
    )
    deposits_by_source = (
        files.groupby("source", observed=True)["dataset_doi"].nunique().to_dict()
    )

    return {
        "built": datetime.now(tz=UTC).date().isoformat(),
        "sources": sorted(str(s) for s in deposits_by_source),
        "n_deposits": int(files["dataset_doi"].nunique()),
        "n_deposits_analyzable": int(analyzable["dataset_doi"].nunique()),
        "n_files_analyzable": len(analyzable),
        "n_packages": len(usage),
        "n_unresolved_names": len(unknown),
        "n_collections": int(files["collection_id"].nunique()),
        "deposits_by_source": {str(k): int(v) for k, v in deposits_by_source.items()},
        "deposits_analyzable_by_source": {str(k): int(v) for k, v in by_source.items()},
        "deposits_by_language": {
            str(k): int(v) for k, v in by_language.items() if v > 0
        },
        # Deposits stating each environment signal, over deposits that could
        # have. Every figure drawn from `environment_signals.parquet` is a
        # share of the first number, not of the corpus.
        "environment_coverage": environment_coverage(),
        "trawl": trawl(files),
    }


def trawl(files: pd.DataFrame) -> dict:
    """How much was searched, for the README and anyone sharing the release.

    The frame is the deposits the collections list, before any were found to
    hold code, so the funnel reads from what was looked at to what was used.
    """
    dataverse = pd.read_csv(FRAME / "dataverse_deposits.csv")
    frame = pd.read_csv(FRAME / "frame.csv")
    return {
        "collections_by_source": {
            "dataverse": int(dataverse["collection_id"].nunique()),
            "zenodo": int((frame["source"] == "zenodo").sum()),
        },
        "deposits_in_frame_by_source": {
            "dataverse": len(dataverse),
            "zenodo": len(pd.read_csv(ZENODO_DEPOSITS)),
        },
        "n_files": len(files),
        "n_mentions": pq.ParquetFile(TALLY / "mentions.parquet").metadata.num_rows,
    }


def readme_numbers(summary: dict) -> str:
    """The README's release-in-numbers block, written from `summary.json`."""
    trawled = summary["trawl"]
    collections = trawled["collections_by_source"]
    frame = trawled["deposits_in_frame_by_source"]
    rows = "\n".join(
        f"| {LANGUAGE_NAMES.get(language, language)} | {n:,} |"
        for language, n in summary["deposits_by_language"].items()
    )
    return f"""{README_START}
## The {summary["built"][:4]} release in numbers

The frame is every deposit in {sum(collections.values())} journal collections:
{collections["dataverse"]} on Harvard Dataverse and {collections["zenodo"]} on Zenodo,
{sum(frame.values()):,} deposits in all ({frame["dataverse"]:,} and {frame["zenodo"]:,}).

- **{summary["n_deposits"]:,}** deposits held code or a dependency manifest,
  and {summary["n_deposits_analyzable"]:,} held analyzable code.
- **{trawled["n_files"]:,}** files were collected; {summary["n_files_analyzable"]:,} are
  analyzed once vendored libraries and duplicate copies are set aside.
- **{trawled["n_mentions"]:,}** package references were extracted from them,
  resolving to **{summary["n_packages"]:,}** packages.

| Language | Deposits with code |
|---|---:|
{rows}

Not included: code inside tar, 7z and rar archives too large to download,
files a depositor restricted, and the AEA journals, which deposit on openICPSR.

Look up any package at <https://recite.github.io/softverse/lookup/>. The
tables and the code text are at
<https://huggingface.co/datasets/gojiberries/softverse>.
{README_END}"""


def write_readme(summary: dict, readme: Path = README) -> None:
    """Replace the numbers block in the README, which must already have one."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + readme_numbers(summary) + text[end:])


def environment_coverage() -> dict:
    """The coverage block the build computed, or empty if it did not run."""
    path = TALLY / "environment_coverage.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    if not (TALLY / "usage_by_package.csv").exists():
        print("no tally; run scripts/build_tally.py first")
        return 1

    # A sparse table without its denominator is worse than no table: it reads
    # as a census. Refusing to ship one is cheaper than the correction.
    if not environment_coverage():
        print(
            "environment_signals ships without a coverage denominator; "
            "rerun scripts/build_tally.py to write environment_coverage.json"
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copyfile(TALLY / name, OUT / name)
    shutil.copyfile(
        TALLY / "environment_coverage.json", OUT / "environment_coverage.json"
    )

    # The validation artefacts ship with the tables they vouch for. The paper
    # cites precision, recall and Jaccard; a reader who wants to check those
    # rather than take the PDF's word needs the numbers, and they are 28 KB.
    validation = PATHS.root / "build" / "validation"
    for name in ("r_oracle.json", "renv_agreement.json", "reachability.json"):
        if (validation / name).exists():
            shutil.copyfile(validation / name, OUT / name)

    summary = summarize()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_readme(summary)
    write_package_tables()
    write_year_denominators()

    (OUT / "README.md").write_text(
        DESCRIPTOR.format(
            n_packages=summary["n_packages"],
            n_deposits=summary["n_deposits"],
            n_deposits_analyzable=summary["n_deposits_analyzable"],
            n_unknown=summary["n_unresolved_names"],
            n_languages=len(pd.read_csv(OUT / "language_presence.csv")),
            n_years=len(pd.read_csv(OUT / "usage_by_package_year.csv")),
            n_collection_rows=len(pd.read_csv(OUT / "usage_by_collection.csv")),
            n_functions=len(pd.read_csv(OUT / "usage_by_function.csv")),
            n_mentions=len(
                pd.read_parquet(OUT / "mentions.parquet", columns=["file_uid"])
            ),
            n_files=len(pd.read_parquet(OUT / "files.parquet", columns=["file_uid"])),
            n_declarations=len(
                pd.read_parquet(
                    OUT / "declared_dependencies.parquet", columns=["package"]
                )
            ),
            n_signals=len(
                pd.read_parquet(OUT / "environment_signals.parquet", columns=["signal"])
            ),
            n_remote_installs=len(pd.read_csv(OUT / "remote_installs.csv")),
            built=summary["built"],
            composition=_composition_table(summary),
        )
    )

    (OUT / "datapackage.json").write_text(
        json.dumps(
            {
                "name": "softverse-validated-use",
                "title": "Validated use: per-package counts from replication code",
                "licenses": [
                    {
                        "name": "CC0-1.0",
                        "path": "https://creativecommons.org/publicdomain/zero/1.0/",
                    }
                ],
                "created": datetime.now(tz=UTC).isoformat(),
                "resources": [_resource(OUT / name) for name in TABLES],
            },
            indent=1,
        )
        + "\n"
    )

    return report(summary)


def _denominators_recomputed(summary: dict) -> list[str]:
    """Rebuild the denominators from the raw Parquet by the documented rule.

    A second route to the same number, sharing no code with the tally that
    produced it. A deposit is at risk for a language when it holds an
    analyzable file in that language, or when a mention in that language came
    out of it, which is what covers notebooks and knitr documents. If the
    tally ever drifts from the rule the README states, this is what says so.
    """
    files = pd.read_parquet(TALLY / "files.parquet")
    mentions = pd.read_parquet(TALLY / "mentions.parquet")
    analyzable = files[files["in_analysis_set"].astype(bool)]

    problems = []
    for language, shipped in summary["deposits_by_language"].items():
        by_file = set(analyzable.loc[analyzable["language"] == language, "dataset_doi"])
        by_mention = set(mentions.loc[mentions["language"] == language, "dataset_doi"])
        expected = len(by_file | by_mention)
        if expected != shipped:
            problems.append(
                f"{language}: shipped denominator {shipped}, "
                f"recomputed from the Parquet {expected}"
            )
    return problems


def report(summary: dict) -> int:
    """Check the exported files, not the objects they were written from."""
    problems = []
    for name in TABLES:
        # Digests rather than line counts: Parquet is binary, so counting
        # newlines in it raises `UnicodeDecodeError` on the first compressed
        # page. A digest also catches a copy that kept the row count and
        # changed the contents, which a line count never could.
        source = hashlib.sha256((TALLY / name).read_bytes()).hexdigest()
        shipped = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        if source != shipped:
            problems.append(f"{name}: shipped copy differs from the tally")

    with (OUT / "usage_by_package.csv").open(encoding="utf-8") as handle:
        usage = {(r["package"], r["language"]): r for r in csv.DictReader(handle)}

    # Fixed expectations, not spot checks: these are the numbers the paper
    # prints, so a release that disagrees with them is a release that would
    # have quietly contradicted the paper. Moved from the August 2026 release
    # (estout 2,440 of 6,212) to the 2026 collection, and again with extractor
    # 2.3.0, which reads the statements a comment-continuation had been
    # splitting (estout 3,818 to 3,835, reghdfe 1,527 to 1,535). The
    # denominator did not move: it counts deposits, and no deposit changed.
    for key, field, expected in (
        (("estout", "stata"), "n_deposits", "3835"),
        (("estout", "stata"), "n_deposits_zenodo", "706"),
        (("reghdfe", "stata"), "n_deposits", "1535"),
        (("estout", "stata"), "n_deposits_at_risk", "8942"),
    ):
        got = usage.get(key, {}).get(field)
        if got != expected:
            problems.append(f"{key[0]}.{field} is {got}, the paper prints {expected}")

    # `grc1leg` is served from a StataCorp developer's page and nowhere else.
    # A Stata Journal package bundles a copy, so crediting it anywhere but its
    # own site means the documented-command rule has stopped being applied.
    if usage.get(("grc1leg", "stata"), {}).get("ecosystem") != "net_site":
        problems.append("grc1leg should resolve to its author's site, and only there")

    problems.extend(_denominators_recomputed(summary))
    problems.extend(_package_deposits_agree())
    problems.extend(_year_denominators_agree())

    # Pooled counts must reconcile with the split they ship beside them. A
    # pooled table that disagrees with its own breakdown is worse than no
    # pooled table, because it looks checkable and is not.
    sources = summary["sources"]
    for (package, language), row in usage.items():
        parts = sum(
            int(row[f"n_deposits_{s}"]) for s in sources if f"n_deposits_{s}" in row
        )
        if parts != int(row["n_deposits"]):
            problems.append(
                f"{package} ({language}): pooled {row['n_deposits']} but "
                f"the per-source columns sum to {parts}"
            )
            break

    print(f"wrote {OUT}")
    print(
        f"  {summary['n_packages']:,} packages · "
        f"{summary['n_deposits_analyzable']:,} of {summary['n_deposits']:,} deposits "
        f"have analyzable code · {summary['n_unresolved_names']:,} unresolved names"
    )
    for language, n in summary["deposits_by_language"].items():
        print(f"    {language:<10} {n:>5}")
    print("\n  deposits by source:")
    for source, n in summary["deposits_by_source"].items():
        print(f"    {source:<20} {n:>6,}")
    if problems:
        print("\nVERIFICATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "\nverified against the exported files: every shipped file's digest "
        "matches the tally, "
        "grc1leg resolves\nto its author's site, every denominator matches a recomputation from "
        "the Parquet, the\npooled counts reconcile with their per-source split, "
        "and every package's\ndeposit list counts to its tally"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

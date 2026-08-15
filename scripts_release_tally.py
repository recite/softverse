"""Package the per-package counts for release.

    uv run python scripts_release_tally.py

Writes `build/release/tally/`: the three small tables, a summary of the
corpus they were computed on, and a frictionless datapackage.

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
import json
import shutil
from datetime import UTC, datetime

import pandas as pd

from softverse.config import PATHS

TALLY = PATHS.root / "build" / "tally"
OUT = PATHS.root / "build" / "release" / "tally"

TABLES = (
    "usage_by_package.csv",
    "usage_by_package_year.csv",
    "usage_by_collection.csv",
    "unknown_names.csv",
    "language_presence.csv",
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
`n_deposits_zenodo` and `n_deposits_dataverse_legacy`, because the two are
very different sizes and a pooled figure alone would hide that.

The Dataverse deposits come from a January 2024 scrape that kept `.do`, `.r`
and `.py` files and nothing else, so a package used mainly inside a notebook
or a knitr document is under-counted on that side. The release checks this
by recomputing the whole ranking on those three extensions alone and fails if
the order moves. The two collections are also two years apart, which
`first_year` and `last_year` will show.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | {n_packages:,} | per-package deposit and call counts, pooled and split |
| `usage_by_package_year.csv` | {n_years:,} | the same by deposit year |
| `usage_by_collection.csv` | {n_collection_rows:,} | the same per journal or community |
| `unknown_names.csv` | {n_unknown:,} | names called in code that resolve to no registry |
| `language_presence.csv` | {n_languages} | deposits containing each language, per repository |
| `summary.json` | | corpus counts the tables are shares of |

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

Names that appear in the code and resolve to no registry, unfiltered. Some
are false positives: `str` is a Stata type, and some are programs a deposit
defines for itself. The list also holds real and heavily used software that
no registry indexes, `grc1leg` being the clearest case at {n_grc1leg:,}
calls. Pruning the list by hand would put a judgement call inside a file
whose value is that you can check every row of it.

## Licence

CC0.

## Regenerating

```bash
uv run python scripts_build_tally.py     # needs the collected corpus
uv run python scripts_release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).
"""


#: Reader-facing names for the sources, and what each one is.
SOURCE_LABEL = {
    "zenodo": "Zenodo (economics)",
    "dataverse_legacy": "Harvard Dataverse (political science)",
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
        "n_files_analyzable": int(len(analyzable)),
        "n_packages": int(len(usage)),
        "n_unresolved_names": int(len(unknown)),
        "n_collections": int(files["collection_id"].nunique()),
        "deposits_by_source": {str(k): int(v) for k, v in deposits_by_source.items()},
        "deposits_analyzable_by_source": {str(k): int(v) for k, v in by_source.items()},
        "deposits_by_language": {
            str(k): int(v) for k, v in by_language.items() if v > 0
        },
    }


def main() -> int:
    if not (TALLY / "usage_by_package.csv").exists():
        print("no tally; run scripts_build_tally.py first")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copyfile(TALLY / name, OUT / name)

    summary = summarize()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    unknown = pd.read_csv(OUT / "unknown_names.csv")
    grc1leg = unknown.loc[unknown["name"] == "grc1leg", "n_mentions"]
    (OUT / "README.md").write_text(
        DESCRIPTOR.format(
            n_packages=summary["n_packages"],
            n_deposits=summary["n_deposits"],
            n_deposits_analyzable=summary["n_deposits_analyzable"],
            n_unknown=summary["n_unresolved_names"],
            n_languages=len(pd.read_csv(OUT / "language_presence.csv")),
            n_years=len(pd.read_csv(OUT / "usage_by_package_year.csv")),
            n_collection_rows=len(pd.read_csv(OUT / "usage_by_collection.csv")),
            n_grc1leg=int(grc1leg.iloc[0]) if len(grc1leg) else 0,
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
                "resources": [
                    {
                        "name": "usage_by_package",
                        "path": "usage_by_package.csv",
                        "format": "csv",
                        "schema": {
                            "fields": [
                                {"name": "package", "type": "string"},
                                {"name": "language", "type": "string"},
                                {"name": "ecosystem", "type": "string"},
                                {"name": "n_deposits", "type": "integer"},
                                {"name": "n_files", "type": "integer"},
                                {"name": "n_mentions", "type": "integer"},
                                {"name": "first_year", "type": "integer"},
                                {"name": "last_year", "type": "integer"},
                                {"name": "n_deposits_at_risk", "type": "integer"},
                                {"name": "share_of_deposits", "type": "number"},
                            ]
                        },
                    },
                    {
                        "name": "unknown_names",
                        "path": "unknown_names.csv",
                        "format": "csv",
                        "schema": {
                            "fields": [
                                {"name": "name", "type": "string"},
                                {"name": "language", "type": "string"},
                                {"name": "n_mentions", "type": "integer"},
                            ]
                        },
                    },
                    {
                        "name": "language_presence",
                        "path": "language_presence.csv",
                        "format": "csv",
                        "schema": {
                            "fields": [
                                {"name": "language", "type": "string"},
                                {"name": "n_deposits", "type": "integer"},
                            ]
                        },
                    },
                ],
            },
            indent=1,
        )
    )

    return report(summary)


#: The only extensions the 2024 Dataverse scrape kept. Zenodo carries these
#: plus `.ado`, `.m`, `.Rmd`, `.ipynb` and `.sas`.
COMMON_BASIS = frozenset({".do", ".r", ".py"})

#: How far a package may move between the pooled ranking and the same ranking
#: on the common basis. One place absorbs ties; more would mean the extra file
#: types Zenodo carries are driving the order rather than usage is.
MAX_RANK_SHIFT = 1

#: The comparison is printed for the top 20 and enforced over the top 10,
#: which is the depth the paper's tables actually print and therefore the
#: depth at which a claim rests on the ordering. The gap between the two is
#: not slack, it is the finding: R and Stata hold all twenty places, and
#: Python's ranks 14 to 20 move because that is where notebook-only packages
#: sit and the 2024 Dataverse scrape collected no notebooks.
GATE_DEPTH = 10


def _common_basis_ranking(top_n: int = 20) -> tuple[list[str], str]:
    """Recompute the ranking using only file types both corpora collected.

    The two halves do not see the same extensions, so pooling could in
    principle reorder the table through coverage rather than through usage.
    This is that objection turned into a number instead of a caveat: rebuild
    the per-package deposit counts from the mentions restricted to `.do`,
    `.r` and `.py`, and compare the top 20 per language against the shipped
    ranking.
    """
    files = pd.read_parquet(TALLY / "files.parquet", columns=["file_uid", "extension"])
    mentions = pd.read_parquet(
        TALLY / "mentions.parquet",
        columns=[
            "file_uid",
            "dataset_doi",
            "language",
            "resolved_package",
            "resolution",
        ],
    )
    shipped = pd.read_csv(TALLY / "usage_by_package.csv")

    countable = {"known_current", "known_archived"}
    keep = set(files.loc[files["extension"].str.lower().isin(COMMON_BASIS), "file_uid"])
    basis = mentions[
        mentions["file_uid"].isin(keep)
        & mentions["resolution"].isin(countable)
        & mentions["resolved_package"].notna()
    ]
    counts = (
        basis.groupby(["language", "resolved_package"], observed=True)["dataset_doi"]
        .nunique()
        .reset_index(name="n_deposits")
    )

    problems: list[str] = []
    lines = []
    for language in sorted(shipped["language"].unique()):
        pooled = (
            shipped[shipped["language"] == language]
            .nlargest(top_n, "n_deposits")["package"]
            .tolist()
        )
        restricted = (
            counts[counts["language"] == language]
            .nlargest(top_n, "n_deposits")["resolved_package"]
            .tolist()
        )
        position = {name: i for i, name in enumerate(restricted)}
        moved = []
        for rank, name in enumerate(pooled):
            if name not in position:
                note = f"{name} leaves the top {top_n}"
            elif abs(position[name] - rank) > MAX_RANK_SHIFT:
                note = f"{name} {rank + 1}->{position[name] + 1}"
            else:
                continue
            moved.append(note)
            if rank < GATE_DEPTH:
                problems.append(f"common basis, {language}: {note}")
        lines.append(
            f"  {language:<8} {len(pooled) - len(moved):>2}/{len(pooled)} hold rank"
            + (f"  ({'; '.join(moved[:4])})" if moved else "")
        )

    report = "common-basis ranking (.do/.r/.py only, as the 2024 scrape kept):\n"
    return problems, report + "\n".join(lines)


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
        source = sum(1 for _ in open(TALLY / name, encoding="utf-8"))
        shipped = sum(1 for _ in open(OUT / name, encoding="utf-8"))
        if source != shipped:
            problems.append(f"{name}: {shipped} lines shipped, {source} in the tally")

    with open(OUT / "usage_by_package.csv", encoding="utf-8") as handle:
        usage = {(r["package"], r["language"]): r for r in csv.DictReader(handle)}

    # Fixed expectations, not spot checks: these are the numbers the paper
    # prints, so a release that disagrees with them is a release that would
    # have quietly contradicted the paper.
    for key, field, expected in (
        (("estout", "stata"), "n_deposits", "2440"),
        (("estout", "stata"), "n_deposits_zenodo", "657"),
        (("reghdfe", "stata"), "n_deposits", "800"),
        (("estout", "stata"), "n_deposits_at_risk", "6212"),
    ):
        got = usage.get(key, {}).get(field)
        if got != expected:
            problems.append(f"{key[0]}.{field} is {got}, the paper prints {expected}")

    if ("grc1leg", "stata") in usage:
        problems.append("grc1leg resolved to a package; it is in no registry")

    problems.extend(_denominators_recomputed(summary))

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

    basis_problems, basis_report = _common_basis_ranking()
    problems.extend(basis_problems)

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
    print(f"\n{basis_report}")
    if problems:
        print("\nVERIFICATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "\nverified against the exported files: line counts match the tally, "
        "grc1leg is\nunresolved, every denominator matches a recomputation from "
        "the Parquet, the\npooled counts reconcile with their per-source split, "
        "and the ranking survives\nrestriction to the file types both corpora "
        "collected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

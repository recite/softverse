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

Scope worth stating: these counts are the Zenodo half of the frame, which is
economics. The Harvard Dataverse half is political science and is tallied
separately; it contributes to the language comparison and not to these
per-package counts.
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

TABLES = ("usage_by_package.csv", "unknown_names.csv", "language_presence.csv")

DESCRIPTOR = """\
# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**{n_packages:,} packages · {n_deposits_analyzable:,} deposits with analyzable
code · {n_deposits:,} deposits collected · built {built}**

A download says a file was fetched. A mention in prose says an author
remembered to name something. Neither says the software ran. These counts say
a package was loaded by code that shipped with a paper, which is the strongest
of the three claims and the one worth crediting against.

## Scope

The frame is the Zenodo half: {n_deposits:,} deposits from the verified
economics collections. Harvard Dataverse's journal collections are political
science and are tallied separately, so they contribute to the cross-language
comparison in the paper and not to the per-package counts here. A package
heavily used in political science and rarely in economics will look smaller
here than it is.

Counts are static reference in deposited code. A package that is loaded but
never reached at runtime still counts, and a package invoked through a string
that is assembled at runtime does not.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | {n_packages:,} | per-package deposit and call counts |
| `unknown_names.csv` | {n_unknown:,} | names called in code that resolve to no registry |
| `language_presence.csv` | {n_languages} | deposits containing each language |
| `summary.json` | | corpus counts the tables are shares of |

### `usage_by_package.csv`

- `package`, `language`, `ecosystem` — the resolved package and its registry
- `n_deposits` — deposits loading it, counted once per deposit
- `n_files`, `n_mentions` — files and raw calls
- `n_deposits_at_risk` — the denominator `share_of_deposits` uses: deposits
  that hold an analyzable file in that language, or that yielded a reference
  in it. The second clause is what covers literate documents, where the file
  is a notebook and the code inside it is Python. It differs by language, so a
  Stata share and an R share are not shares of the same thing
- `share_of_deposits` — `n_deposits / n_deposits_at_risk`

### `unknown_names.csv`

Deliberately unfiltered. Some entries are false positives: `str` is a Stata
type, and some names are programs defined inside the deposit itself. But the
list also holds real, heavily used software that no registry indexes,
`grc1leg` being the clearest case at {n_grc1leg:,} calls, distributed from
StataCorp's own site and invisible to any credit system built on registries.
Pruning it by hand would bury a judgement call inside a file whose point is
that you can check it.

## Licence

CC0.

## Regenerating

```bash
uv run python scripts_build_tally.py     # needs the collected corpus
uv run python scripts_release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).
"""


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

    return {
        "built": datetime.now(tz=UTC).date().isoformat(),
        "frame": "zenodo",
        "n_deposits": int(files["dataset_doi"].nunique()),
        "n_deposits_analyzable": int(analyzable["dataset_doi"].nunique()),
        "n_files_analyzable": int(len(analyzable)),
        "n_packages": int(len(usage)),
        "n_unresolved_names": int(len(unknown)),
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
            n_grc1leg=int(grc1leg.iloc[0]) if len(grc1leg) else 0,
            built=summary["built"],
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
        (("estout", "stata"), "n_deposits", "657"),
        (("reghdfe", "stata"), "n_deposits", "375"),
        (("estout", "stata"), "n_deposits_at_risk", "1075"),
    ):
        got = usage.get(key, {}).get(field)
        if got != expected:
            problems.append(f"{key[0]}.{field} is {got}, the paper prints {expected}")

    if ("grc1leg", "stata") in usage:
        problems.append("grc1leg resolved to a package; it is in no registry")

    problems.extend(_denominators_recomputed(summary))

    print(f"wrote {OUT}")
    print(
        f"  {summary['n_packages']:,} packages · "
        f"{summary['n_deposits_analyzable']:,} of {summary['n_deposits']:,} deposits "
        f"have analyzable code · {summary['n_unresolved_names']:,} unresolved names"
    )
    for language, n in summary["deposits_by_language"].items():
        print(f"    {language:<10} {n:>5}")
    if problems:
        print("\nVERIFICATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "\nverified against the exported files: line counts match the tally, "
        "estout and\nreghdfe match the paper, grc1leg is unresolved, and every "
        "denominator matches\nthe deposits containing that language"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

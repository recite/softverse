"""List the names code uses that no archive accounts for, for labelling by hand.

    uv run python scripts/label_unregistered.py

Writes `data/validation/unregistered_labelled.csv`: every name in
`unknown_names.csv` used in at least `MIN_DEPOSITS` deposits, with the columns
a person fills in -- `category`, `where_distributed`, `url`, `checked_on`.
Labels already in the file are kept and matched by `(name, language)`, so
rebuilding the tally adds and drops rows without discarding the work. Rows
that fall out of the list are dropped; rows new to it arrive unlabelled.

The threshold is stated rather than hidden: below it the list is singletons,
which are typos, macro fragments and one deposit's own helpers, and labelling
two thousand of them would move no number the paper reports.

One category is filled in mechanically: a name the corpus installs from a
remote is distributed there. `n_deposits_defining` is carried as evidence and
not acted on, because it cuts both ways. A name many deposits define for
themselves is usually their own helper -- and sometimes a third-party `.ado`
passed from author to author by copying the file, which is a distribution
channel, not a local program. `dcdensity` and `ols_spatial_hac` are the latter.
"""

from __future__ import annotations

import csv

import pandas as pd

from softverse.config import PATHS

TALLY = PATHS.root / "build" / "tally"
OUT = PATHS.root / "data" / "validation" / "unregistered_labelled.csv"
MIN_DEPOSITS = 5

#: What a label may be. Anything else in the file fails the paper's check.
CATEGORIES = (
    "formal archive not indexed",
    "code host",
    "author site",
    "commercial",
    "official, undocumented",
    "local program",
    "not software",
    "unidentified",
)
FIELDS = (
    "name",
    "language",
    "n_deposits",
    "n_deposits_defining",
    "category",
    "where_distributed",
    "url",
    "checked_on",
)


def main() -> int:
    unknown = pd.read_csv(TALLY / "unknown_names.csv")
    unknown["key"] = unknown["name"].str.lower()
    rows = (
        unknown.groupby(["key", "language"], as_index=False)
        .agg(
            n_deposits=("n_deposits", "sum"),
            n_deposits_defining=("n_deposits_defining", "max"),
        )
        .rename(columns={"key": "name"})
    )
    rows = rows[rows["n_deposits"] >= MIN_DEPOSITS]

    remote = pd.read_csv(TALLY / "remote_installs.csv")
    remote = remote[~remote["host"].str.startswith("(")]
    hosts = (
        remote.assign(name=remote["name"].str.lower())
        .sort_values("n_deposits_installing", ascending=False)
        .drop_duplicates(["name", "language"])
        .set_index(["name", "language"])["host"]
    )

    kept: dict[tuple[str, str], dict] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8") as handle:
            kept = {(r["name"], r["language"]): r for r in csv.DictReader(handle)}

    out = []
    for row in rows.sort_values(
        ["language", "n_deposits"], ascending=[True, False]
    ).to_dict("records"):
        key = (row["name"], row["language"])
        label = dict.fromkeys(FIELDS, "") | row
        if key in kept and kept[key]["category"]:
            label |= {
                f: kept[key][f]
                for f in ("category", "where_distributed", "url", "checked_on")
            }
        elif key in hosts.index:
            label["category"] = "code host" if "github" in hosts[key] else "author site"
            label["where_distributed"] = hosts[key]
        out.append(label)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    unlabelled = sum(1 for r in out if not r["category"])
    print(f"{len(out)} names in >={MIN_DEPOSITS} deposits; {unlabelled} unlabelled")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

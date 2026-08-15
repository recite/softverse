"""Create, and on request publish, the Zenodo deposit for the per-package counts.

    uv run python scripts_deposit_tally.py             # create/update the draft
    uv run python scripts_deposit_tally.py --show      # print its state
    uv run python scripts_deposit_tally.py --publish   # mint the DOI

The tables are already served at https://recite.github.io/softverse/data/,
which is convenient and not archival: a GitHub Pages URL is whatever the
repository serves today. A paper that cites its own counts needs a version of
them that cannot move, so the same bundle goes to Zenodo under a DOI.

Publishing is a separate act for the reason it is in the Stata index script:
a draft is private and deletable, a DOI is neither.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from softverse.config import PATHS, credential
from softverse.release.zenodo_deposit import Deposit, run

BUNDLE = PATHS.root / "build" / "release" / "tally"

TITLE = (
    "Validated use: per-package counts of software loaded by "
    "social science replication code"
)

#: Published 2026-08-15. Set so a default re-run reports it instead of
#: creating a second deposit of the same tables.
PUBLISHED_RECORD = 21943909

#: The Stata index this corpus was resolved against. Recorded as a related
#: identifier so the two deposits are navigable from each other.
STATA_INDEX_DOI = "10.5281/zenodo.21926099"


def summary() -> dict:
    return json.loads((BUNDLE / "summary.json").read_text())


def description(stats: dict) -> str:
    """Written from the bundle, so the deposit page cannot overstate it."""
    with open(BUNDLE / "usage_by_package.csv", encoding="utf-8") as handle:
        top = max(csv.DictReader(handle), key=lambda r: int(r["n_deposits"]))
    by_source = stats["deposits_by_source"]

    return f"""\
<p>How often each R, Python and Stata package is loaded by the code deposited
with published papers, at economics and political science journals whose
data-and-code policy an editor verifies before publication.</p>

<p><strong>{stats["n_packages"]:,} packages &middot;
{stats["n_deposits_analyzable"]:,} deposits with analyzable code &middot;
{stats["n_collections"]} journal collections.</strong></p>

<p><strong>Why count this way.</strong> A metric used to allocate credit is a
signal, and a signal is informative in proportion to what it costs to produce.
A download is one fetch of a file, which a build server installing on every
commit supplies in quantity and a faster release schedule supplies more of. A
mention in an article's prose is one sentence. Adding one to a count here
takes a paper accepted at a journal that checks its authors' code, with the
package loaded in that code.</p>

<p><strong>Scope.</strong> Two repositories holding two disciplines:
{by_source.get("zenodo", 0):,} deposits from Zenodo's verified economics
collections and {by_source.get("dataverse_legacy", 0):,} from Harvard
Dataverse's political science journals. Counts pool both and
<code>usage_by_package.csv</code> carries the split beside every pooled total,
since the two are very different sizes.</p>

<p><strong>What the counts do not show.</strong> These counts say a package
was loaded by code in the deposit. They do not say the code ran. Authors
often leave older scripts in a deposit, and a script can load a package
inside a branch that never executes. Whatever an author kept out of the
deposit cannot be counted at all.
The Harvard Dataverse material also came from a January 2024 scrape that kept
only <code>.do</code>, <code>.r</code> and <code>.py</code> files, so a
package used mainly inside a notebook is under-counted on that side; the
release checks this by recomputing the whole ranking on those three
extensions and fails if the order moves.</p>

<p><strong>Files.</strong> <code>usage_by_package.csv</code> is the main
table. <code>usage_by_package_year.csv</code> and
<code>usage_by_collection.csv</code> give the same counts by deposit year and
by journal. <code>unknown_names.csv</code> lists names used in code that
resolve to no registry, unfiltered. <code>summary.json</code> holds the
corpus counts the shares are taken against, and <code>README.md</code> is the
data descriptor with column definitions.</p>

<p>Stata resolution uses the command-to-package index deposited separately at
<a href="https://doi.org/{STATA_INDEX_DOI}">{STATA_INDEX_DOI}</a>. The most
loaded package in this corpus is <code>{top["package"]}</code>, in
{int(top["n_deposits"]):,} deposits.</p>
"""


def metadata(stats: dict) -> dict:
    return {
        "metadata": {
            "title": TITLE,
            "upload_type": "dataset",
            "description": description(stats),
            "creators": [{"name": "Sood, Gaurav"}],
            "license": "cc-zero",
            "keywords": [
                "research software",
                "software citation",
                "reproducibility",
                "replication code",
                "metascience",
                "static analysis",
                "Stata",
                "R",
                "Python",
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/recite/softverse",
                    "relation": "isSupplementTo",
                    "scheme": "url",
                },
                {
                    "identifier": STATA_INDEX_DOI,
                    "relation": "isDerivedFrom",
                    "scheme": "doi",
                },
            ],
            "notes": (
                "Produced by softverse (https://github.com/recite/softverse). "
                "The same tables are served at "
                "https://recite.github.io/softverse/data/."
            ),
        }
    }


def main() -> int:
    token = credential("ZENODO_API_TOKEN")
    if not token:
        print("ZENODO_API_TOKEN is not set")
        return 1
    if not (BUNDLE / "summary.json").exists():
        print(f"no bundle at {BUNDLE}; run scripts_release_tally.py first")
        return 1

    stats = summary()
    spec = Deposit(TITLE, Path(BUNDLE), metadata(stats), PUBLISHED_RECORD)
    return run(spec, token, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

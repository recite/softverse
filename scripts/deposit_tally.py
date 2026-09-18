"""Create, and on request publish, the Zenodo deposit for the 2026 release.

    uv run python scripts/deposit_tally.py --new-version --hf-dataset ORG/NAME
    uv run python scripts/deposit_tally.py --show
    uv run python scripts/deposit_tally.py --publish

A new version of the August 2026 record (concept DOI 10.5281/zenodo.21943908),
so citations to the counts stay continuous. The version carries every table
of the release under CC0: the per-package aggregates, and the corpus tables
that say which package each deposit and file uses, at what version, on what
interpreter.

What it does not carry is code. File text and the snippets in `mentions` keep
their authors' licenses -- mostly CC0, but also CC-BY and CC-BY-NC -- and a
Zenodo record has one license, so both are published on Hugging Face instead,
where each row carries its own, and this record links there. `mentions` goes
up with its `snippet` column removed.

Publishing is a separate act for the reason it is in the Stata index script:
a draft is private and deletable, a DOI is neither.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from typing import TYPE_CHECKING

import duckdb
import httpx

from softverse.config import PATHS, credential
from softverse.release.zenodo_deposit import (
    Deposit,
    new_version,
    publish,
    replace_files,
    run,
    show,
)

if TYPE_CHECKING:
    from pathlib import Path

TALLY = PATHS.root / "data" / "tally"
CORPUS = PATHS.root / "build" / "release" / "corpus"
#: The flat directory that is uploaded, rebuilt from the two above each run.
BUNDLE = PATHS.root / "build" / "release" / "zenodo"

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

#: From `data/tally/`: the aggregates and the files that describe them. Not
#: its `mentions.parquet` or `files.parquet`, which are the unredacted build
#: copies; the corpus release's versions of those go up instead.
TALLY_FILES = (
    "usage_by_package",
    "usage_by_package_year",
    "usage_by_collection",
    "usage_by_function",
    "unknown_names",
    "remote_installs",
    "language_presence",
)
TALLY_EXTRAS = (
    "summary.json",
    "environment_coverage.json",
    "r_oracle.json",
    "renv_agreement.json",
    "reachability.json",
    "datapackage.json",
    "package_deposits.parquet",
    "package_versions.csv",
)

#: How many files the mention table is cut into for upload.
MENTION_PARTS = 3

#: From the corpus release: every table except `contents` and `mentions`,
#: which is written without its snippets.
CORPUS_FILES = (
    "deposits.parquet",
    "files.parquet",
    "file_packages.parquet",
    "package_versions.parquet",
    "environment.parquet",
)


def stage_bundle(
    tally: Path = TALLY, corpus: Path = CORPUS, out: Path = BUNDLE
) -> list[Path]:
    """Assemble the upload: CC0 tables only, one flat directory.

    Args:
        tally: The released aggregates.
        corpus: The corpus release.
        out: Where the bundle is assembled, replaced if present.

    Returns:
        The files to upload.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = [f"{n}.{ext}" for n in TALLY_FILES for ext in ("csv", "parquet")]
    for name in (*names, *TALLY_EXTRAS):
        if (tally / name).exists():
            shutil.copyfile(tally / name, out / name)
    shutil.copyfile(tally / "README.md", out / "README.md")
    for name in CORPUS_FILES:
        shutil.copyfile(corpus / name, out / name)
    # In parts. Zenodo takes one stream per file and its gateway answers 502
    # once a request has run about five minutes, which on a slow uplink is
    # some sixty megabytes: measured, 60 MB went up in 301 s and 81 MB never
    # did. `mentions-*.parquet` reads back as one table, and the parts are cut
    # by deposit so no deposit's rows straddle two files.
    con = duckdb.connect()
    source = corpus / "mentions.parquet"
    for part in range(MENTION_PARTS):
        con.execute(
            f"COPY (SELECT * EXCLUDE (snippet) FROM '{source}' "
            f"WHERE hash(dataset_doi) % {MENTION_PARTS} = {part} "
            "ORDER BY dataset_doi, file_uid, line) "
            f"TO '{out / f'mentions-{part + 1}-of-{MENTION_PARTS}.parquet'}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 19)"
        )
    return sorted(p for p in out.iterdir() if p.is_file())


def summary() -> dict:
    return json.loads((TALLY / "summary.json").read_text())


def description(stats: dict, hf_dataset: str) -> str:
    """Written from the bundle, so the deposit page cannot overstate it."""
    with (TALLY / "usage_by_package.csv").open(encoding="utf-8") as handle:
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
collections and {by_source.get("dataverse", 0):,} from Harvard
Dataverse's political science journals. Counts pool both and
<code>usage_by_package.csv</code> carries the split beside every pooled total,
since the two are very different sizes.</p>

<p><strong>What changed in this version.</strong> Stata commands are resolved
against the <em>Stata Journal</em>, the <em>Stata Technical Bulletin</em> and
the authors' sites replication code installs from, as well as SSC, so software
such as <code>grc1leg</code> and <code>renvars</code> is now counted rather
than listed as unresolved. The Stata lexer no longer splits a statement at a
newline inside a comment, which had reported command options as commands.
Install lines record where they fetch from (<code>mentions.remote</code>,
<code>remote_installs.csv</code>), and an R package no registry lists is
credited to its code host when the same deposit installs it from one.</p>

<p><strong>What the counts do not show.</strong> These counts say a package
was loaded by code in the deposit. They do not say the code ran. Authors
often leave older scripts in a deposit, and a script can load a package
inside a branch that never executes. Whatever an author kept out of the
deposit cannot be counted at all. The AEA journals, which deposit on
openICPSR, are not included.</p>

<p><strong>Files.</strong> <code>usage_by_package.csv</code> is the main
table. <code>usage_by_package_year.csv</code> and
<code>usage_by_collection.csv</code> give the same counts by deposit year and
by journal. <code>unknown_names.csv</code> lists names used in code that
resolve to no registry, unfiltered. <code>remote_installs.csv</code> lists
what deposits install from outside their language's registry -- a GitHub
repository, an author's own site -- and from where. <code>summary.json</code> holds the
corpus counts the shares are taken against, and <code>README.md</code> is the
data descriptor with column definitions.</p>

<p><strong>The corpus tables.</strong> <code>deposits.parquet</code> (one row
per deposit, with its journal, year and license), <code>files.parquet</code>
(one row per file, with its sha256 and the packages it loads),
<code>file_packages.parquet</code>, <code>mentions-*.parquet</code> (one table in
three files, cut by deposit, which DuckDB and pandas read as one; one row per
reference in code, without the code snippet), <code>package_versions.parquet</code>
(versions stated in manifests and install calls), <code>environment.parquet</code>
(R, Python, Stata and Julia versions and operating systems a deposit states),
and <code>package_deposits.parquet</code> (which deposits use each package).</p>

<p><strong>The code itself</strong> -- file text and snippets, under each
deposit's own license -- is published at
<a href="https://huggingface.co/datasets/{hf_dataset}">huggingface.co/datasets/{hf_dataset}</a>,
with every file linked to this record's tables by sha256.</p>

<p>Stata resolution uses the command-to-package index deposited separately at
<a href="https://doi.org/{STATA_INDEX_DOI}">{STATA_INDEX_DOI}</a>. The most
loaded package in this corpus is <code>{top["package"]}</code>, in
{int(top["n_deposits"]):,} deposits.</p>
"""


def metadata(stats: dict, hf_dataset: str) -> dict:
    return {
        "metadata": {
            "title": TITLE,
            "upload_type": "dataset",
            "description": description(stats, hf_dataset),
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
    if (
        not (TALLY / "summary.json").exists()
        or not (CORPUS / "deposits.parquet").exists()
    ):
        print("no release; run scripts/release_tally.py and scripts/release_corpus.py")
        return 1
    if "--hf-dataset" not in sys.argv:
        print("pass --hf-dataset ORG/NAME: the record links to the code there")
        return 1
    hf_dataset = sys.argv[sys.argv.index("--hf-dataset") + 1]

    files = stage_bundle()
    spec = Deposit(TITLE, BUNDLE, metadata(summary(), hf_dataset), PUBLISHED_RECORD)
    if "--new-version" in sys.argv:
        with httpx.Client(timeout=3600.0) as client:
            deposit = new_version(client, token, PUBLISHED_RECORD, spec)
            deposit = replace_files(client, token, deposit, files)
            show(deposit)
            if "--publish" not in sys.argv:
                print("\nnothing is published. `--publish` mints the version.")
                return 0
            print()
            return publish(client, token, deposit)
    return run(spec, token, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

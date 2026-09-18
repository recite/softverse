"""Create, and on request publish, the Zenodo deposit for the Stata index.

    uv run python scripts/deposit_stata_index.py             # create/update the draft
    uv run python scripts/deposit_stata_index.py --show      # print its state
    uv run python scripts/deposit_stata_index.py --publish   # mint the DOI

Version 1.0 published with two creators. A published record cannot be edited,
so the correction to sole authorship is a new version:

    uv run python scripts/deposit_stata_index.py --new-version
    uv run python scripts/deposit_stata_index.py --new-version --publish

The concept DOI 10.5281/zenodo.21926099 keeps resolving to the latest, and
version 1.0 stays in the record's history rather than disappearing.

The default never publishes, and `--publish` exists so that the irreversible
step is a deliberate, separate act with a record in the repository rather than
a one-off command in somebody's shell history. Publishing mints a DOI, which
is permanent and public: a draft is private and can be deleted, and the moment
it is published neither is true.

Re-running the default replaces the files on the existing draft rather than
creating a second one, so iterating on the bundle does not litter the account
with half-finished deposits.
"""

from __future__ import annotations

import json
import sys

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

BUNDLE = PATHS.root / "build" / "release" / "stata-index"

#: Version 1.0, already published. New versions attach to this record.
PUBLISHED_RECORD = 21926100

TITLE = (
    "Stata command-to-package index: a machine-readable mapping from the "
    "distribution manifests of SSC, the Stata Journal and the STB"
)

DESCRIPTION = """\
<p>A machine-readable mapping from Stata command names to the packages that
provide them, reconstructed from the distribution manifests of the archives
Stata's <code>net install</code> reads: the Statistical Software Components
(SSC) archive, the <em>Stata Journal</em>, its predecessor the <em>Stata
Technical Bulletin</em>, and the authors' own sites that replication code
installs from.</p>

<p><strong>{n_mappings:,} mappings &middot; {n_packages:,} packages &middot;
{n_commands:,} user commands</strong>, from {by_source}. Packages and commands
both counted excluding internal helper files.</p>

<p><strong>Why this exists.</strong> R has CRAN and Python has PyPI: given an
import, you can look up the package. Stata has no equivalent public mapping
from a command to its package, which makes Stata code effectively unmeasurable
at scale and is a large part of why studies of research software omit it &mdash;
despite Stata being the language social science replication code uses most.
Its software is not informal so much as scattered: several archives, one
manifest format, and no index across them.</p>

<p><strong>Method.</strong> Every package in these archives ships a
<code>.pkg</code> manifest listing the files it distributes. Crawling those
manifests yields the command&rarr;package mapping, including the
many-commands-per-package case a package-name list cannot express:
<code>esttab</code>, <code>eststo</code>, <code>estadd</code> and
<code>estpost</code> all belong to <code>estout</code>. The authors' sites
are those named by <code>net install ..., from(URL)</code> lines in a corpus
of social science replication code, so a site no deposit names is not
here.</p>

<p><strong>Four caveats, which change how you should use this.</strong></p>
<ol>
<li><em>A shipped file is not necessarily a command.</em> An
<code>f foo.ado</code> line says a package distributes a file, not that it
exposes a user command. <code>is_helper</code> flags internal
subroutines.</li>
<li><em>This is a current snapshot, not a history.</em> A command that was
user-written in 2010 and later absorbed into official Stata resolves against
its status today, so time-series use will see packages appear to vanish
exactly when their commands are absorbed.</li>
<li><em>A journal package bundles what its example needs.</em> SJ 14-4
<code>st0357</code>, a Cox calibration tool, ships a copy of
<code>grc1leg.ado</code>. <code>is_documented</code> is true when the package
also ships the command's help file; outside SSC, filter on it.</li>
<li><em>Ambiguity is preserved, not resolved.</em> {n_ambiguous:,} commands
are claimed by more than one package in the archive that lists them first,
and are recorded as such rather than assigned a winner.</li>
</ol>

<p>The accompanying <code>builtins.json</code> lists official Stata commands,
each checked against StataCorp's public help server rather than curated from
memory.</p>

<p>See <code>README.md</code> in the deposit for the full data descriptor,
column definitions and a worked example.</p>
"""

_SOURCE_NAMES = {
    "ssc": "SSC",
    "stata_journal": "the Stata Journal",
    "stb": "the STB",
    "net": "authors' sites",
}


def metadata() -> dict:
    """The record's metadata, its counts read off the bundle being deposited.

    They were typed, and the record went on saying 8,726 mappings from SSC
    after the bundle beside it held half as many again from three archives.
    """
    index = BUNDLE / "stata_command_index.parquet"
    n_mappings, n_packages, n_commands = duckdb.execute(
        f"SELECT count(*), count(DISTINCT package) FILTER (NOT is_helper), "
        f"count(DISTINCT command) FILTER (NOT is_helper) FROM '{index}'"
    ).fetchone()
    by_source = duckdb.execute(
        f"SELECT source, count(DISTINCT package) FROM '{index}' GROUP BY 1"
    ).fetchall()
    ambiguous = json.loads((BUNDLE / "ambiguous.json").read_text())
    description = DESCRIPTION.format(
        n_mappings=n_mappings,
        n_packages=n_packages,
        n_commands=n_commands,
        n_ambiguous=len(ambiguous),
        by_source=", ".join(
            f"{_SOURCE_NAMES[source]} ({n:,} packages)"
            for source, n in sorted(by_source, key=lambda r: -r[1])
        ),
    )
    return {
        "metadata": {
            "title": TITLE,
            "upload_type": "dataset",
            "description": description,
            "creators": [{"name": "Sood, Gaurav"}],
            "version": "2.0",
            "license": "cc-zero",
            "keywords": [
                "Stata",
                "research software",
                "software citation",
                "reproducibility",
                "replication code",
                "SSC",
                "Stata Journal",
                "static analysis",
                "metascience",
            ],
            "related_identifiers": [
                {
                    "identifier": "https://github.com/recite/softverse",
                    "relation": "isSupplementTo",
                    "scheme": "url",
                }
            ],
            "notes": (
                "Produced by softverse (https://github.com/recite/softverse). "
                "The underlying manifests are public metadata from the SSC "
                "archive at Boston College, StataCorp's Stata Journal and STB "
                "software archives, and the authors' sites named in the index."
            ),
        }
    }


def main() -> int:
    token = credential("ZENODO_API_TOKEN")
    if not token:
        print("ZENODO_API_TOKEN is not set")
        return 1
    if not BUNDLE.exists():
        print(f"no bundle at {BUNDLE}; run scripts/release_stata_index.py first")
        return 1

    spec = Deposit(TITLE, BUNDLE, metadata(), PUBLISHED_RECORD)
    if "--new-version" in sys.argv:
        with httpx.Client(timeout=300.0) as client:
            deposit = new_version(client, token, PUBLISHED_RECORD, spec)
            # A version draft opens holding the previous version's files. With
            # only its metadata updated, the new record would describe this
            # index and ship the last one.
            deposit = replace_files(client, token, deposit, sorted(BUNDLE.iterdir()))
            show(deposit)
            if "--publish" not in sys.argv:
                print("\nnothing is published. add `--publish` to mint the version.")
                return 0
            print()
            return publish(client, token, deposit)

    return run(spec, token, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

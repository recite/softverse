"""Create, and on request publish, the Zenodo deposit for the Stata index.

    uv run python scripts_deposit_stata_index.py             # create/update the draft
    uv run python scripts_deposit_stata_index.py --show      # print its state
    uv run python scripts_deposit_stata_index.py --publish   # mint the DOI

Version 1.0 published with two creators. A published record cannot be edited,
so the correction to sole authorship is a new version:

    uv run python scripts_deposit_stata_index.py --new-version
    uv run python scripts_deposit_stata_index.py --new-version --publish

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

import sys

import httpx

from softverse.config import PATHS, credential
from softverse.release.zenodo_deposit import Deposit, new_version, publish, run, show

BUNDLE = PATHS.root / "build" / "release" / "stata-index"

#: Version 1.0, already published. New versions attach to this record.
PUBLISHED_RECORD = 21926100

TITLE = (
    "Stata command-to-package index: a machine-readable mapping "
    "from SSC distribution manifests"
)

DESCRIPTION = """\
<p>A machine-readable mapping from Stata command names to the packages that
provide them, reconstructed from the Statistical Software Components (SSC)
archive's own distribution manifests.</p>

<p><strong>8,726 mappings &middot; 3,967 packages &middot; 7,468 user
commands.</strong> Packages and commands both counted excluding internal
helper files.</p>

<p><strong>Why this exists.</strong> R has CRAN and Python has PyPI: given an
import, you can look up the package. Stata has no equivalent public mapping
from a command to its package, which makes Stata code effectively unmeasurable
at scale and is a large part of why studies of research software omit it &mdash;
despite Stata being the language social science replication code uses most.</p>

<p><strong>Method.</strong> Every package on the SSC mirror ships a
<code>.pkg</code> manifest listing the files it distributes. Crawling those
manifests yields the command&rarr;package mapping, including the
many-commands-per-package case a package-name list cannot express:
<code>esttab</code>, <code>eststo</code>, <code>estadd</code> and
<code>estpost</code> all belong to <code>estout</code>.</p>

<p><strong>Three caveats, which change how you should use this.</strong></p>
<ol>
<li><em>A shipped file is not necessarily a command.</em> An
<code>f foo.ado</code> line says a package distributes a file, not that it
exposes a user command. The <code>evidence</code> column separates filename
inference from <code>program_define</code> confirmation, and
<code>is_helper</code> flags internal subroutines.</li>
<li><em>This is a current snapshot, not a history.</em> A command that was
user-written in 2010 and later absorbed into official Stata resolves against
its status today, so time-series use will see packages appear to vanish
exactly when their commands are absorbed.</li>
<li><em>Ambiguity is preserved, not resolved.</em> 211 commands are claimed by
more than one package and are listed as such rather than assigned a winner.</li>
</ol>

<p>The accompanying <code>builtins.json</code> lists official Stata commands,
each checked against StataCorp's public help server rather than curated from
memory.</p>

<p>See <code>README.md</code> in the deposit for the full data descriptor,
column definitions and a worked example.</p>
"""

METADATA = {
    "metadata": {
        "title": TITLE,
        "upload_type": "dataset",
        "description": DESCRIPTION,
        "creators": [{"name": "Sood, Gaurav"}],
        "version": "1.1",
        "license": "cc-zero",
        "keywords": [
            "Stata",
            "research software",
            "software citation",
            "reproducibility",
            "replication code",
            "SSC",
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
            "archive at Boston College."
        ),
    }
}


def main() -> int:
    token = credential("ZENODO_API_TOKEN")
    if not token:
        print("ZENODO_API_TOKEN is not set")
        return 1
    if not BUNDLE.exists():
        print(f"no bundle at {BUNDLE}; run scripts_release_stata_index.py first")
        return 1

    spec = Deposit(TITLE, BUNDLE, METADATA, PUBLISHED_RECORD)
    if "--new-version" in sys.argv:
        with httpx.Client(timeout=300.0) as client:
            deposit = new_version(client, token, PUBLISHED_RECORD, spec)
            show(deposit)
            if "--publish" not in sys.argv:
                print("\nnothing is published. add `--publish` to mint the version.")
                return 0
            print()
            return publish(client, token, deposit)

    return run(spec, token, sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a Zenodo **draft** deposit for the Stata command→package index.

    uv run python scripts_deposit_stata_index.py          # create or update the draft
    uv run python scripts_deposit_stata_index.py --show   # just print its state

This never publishes. Publishing mints a DOI, which is permanent and public,
and that is the author's decision to make in the web interface -- not
something a script should do on their behalf. The draft is private and can be
deleted; the moment it is published, neither is true.

Re-running replaces the files on the existing draft rather than creating a
second one, so iterating on the bundle does not litter the account with
half-finished deposits.
"""

from __future__ import annotations

import sys

import httpx

from softverse.config import PATHS, credential

API = "https://zenodo.org/api"
BUNDLE = PATHS.root / "build" / "release" / "stata-index"

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
        "creators": [
            {"name": "Sood, Gaurav"},
            {"name": "Weitzel, Daniel"},
        ],
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


def find_draft(client: httpx.Client, token: str) -> dict | None:
    """An existing unpublished deposit with this title, if there is one."""
    response = client.get(
        f"{API}/deposit/depositions", params={"access_token": token, "size": 50}
    )
    response.raise_for_status()
    for deposit in response.json():
        if deposit["title"] == TITLE and not deposit["submitted"]:
            return deposit
    return None


def show(deposit: dict) -> None:
    print(f"  title   {deposit['title'][:70]}")
    print(f"  state   {deposit['state']} (submitted={deposit['submitted']})")
    print(f"  edit    {deposit['links']['html']}")
    print(f"  DOI     {deposit['metadata'].get('prereserve_doi', {}).get('doi', '-')}")


def main() -> int:
    token = credential("ZENODO_API_TOKEN")
    if not token:
        print("ZENODO_API_TOKEN is not set")
        return 1
    if not BUNDLE.exists():
        print(f"no bundle at {BUNDLE}; run scripts_release_stata_index.py first")
        return 1

    files = sorted(p for p in BUNDLE.iterdir() if p.is_file())
    with httpx.Client(timeout=300.0) as client:
        existing = find_draft(client, token)

        if "--show" in sys.argv:
            if existing is None:
                print("no draft exists")
                return 0
            show(existing)
            return 0

        if existing is None:
            created = client.post(
                f"{API}/deposit/depositions",
                params={"access_token": token},
                json=METADATA,
            )
            created.raise_for_status()
            deposit = created.json()
            print(f"created draft {deposit['id']}")
        else:
            deposit = existing
            print(f"reusing draft {deposit['id']}")
            updated = client.put(
                f"{API}/deposit/depositions/{deposit['id']}",
                params={"access_token": token},
                json=METADATA,
            )
            updated.raise_for_status()
            deposit = updated.json()

        bucket = deposit["links"]["bucket"]
        already = {
            f["filename"]
            for f in client.get(
                f"{API}/deposit/depositions/{deposit['id']}/files",
                params={"access_token": token},
            ).json()
        }
        for path in files:
            if path.name in already:
                # Replace rather than skip: the point of re-running is that the
                # bundle changed.
                client.delete(
                    f"{API}/deposit/depositions/{deposit['id']}/files/"
                    + next(
                        f["id"]
                        for f in client.get(
                            f"{API}/deposit/depositions/{deposit['id']}/files",
                            params={"access_token": token},
                        ).json()
                        if f["filename"] == path.name
                    ),
                    params={"access_token": token},
                )
            with path.open("rb") as handle:
                put = client.put(
                    f"{bucket}/{path.name}",
                    params={"access_token": token},
                    content=handle,
                )
            put.raise_for_status()
            print(f"  uploaded {path.name:<32} {path.stat().st_size:>9,} bytes")

        final = client.get(
            f"{API}/deposit/depositions/{deposit['id']}",
            params={"access_token": token},
        ).json()

    print("\nDRAFT ONLY -- nothing has been published.")
    show(final)
    print(
        "\nReview it at the link above. Publishing mints a permanent public "
        "DOI\nand is yours to do; this script will not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

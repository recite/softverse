"""Create, update and publish a Zenodo deposit from a directory of files.

Two release bundles use this, the Stata index and the per-package tally, and
the API handling is identical for both: find an existing draft by title,
create or update it, replace the files, and mint the DOI only when asked.

The shape that matters is that publishing is a separate call. `sync()` never
publishes, so re-running it while iterating on a bundle is safe and leaves no
trail of half-finished deposits. `publish()` mints a DOI, which is permanent
and public: a draft is private and can be deleted, and after publication
neither is true.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

API = "https://zenodo.org/api"


def auth(token: str) -> dict[str, str]:
    """Bearer header rather than `?access_token=`.

    A query parameter travels in every URL, so httpx puts it in the message
    of any `raise_for_status` error, which prints the live token to the
    terminal and into whatever captured that output. A header does not.
    """
    return {"Authorization": f"Bearer {token}"}


@dataclass(frozen=True)
class Deposit:
    """A bundle and the metadata Zenodo should carry for it."""

    title: str
    bundle: Path
    metadata: dict
    #: Set once this bundle has been published, so a later default run
    #: reports the existing record instead of quietly creating a second
    #: deposit of the same thing. Re-running the default is the normal way
    #: to iterate on a draft, and after publication that turns into
    #: duplication with no warning.
    published_record: int | None = None

    def files(self) -> list[Path]:
        return sorted(p for p in self.bundle.iterdir() if p.is_file())


def find_draft(client: httpx.Client, token: str, title: str) -> dict | None:
    """An existing unpublished deposit with this title, if there is one."""
    response = client.get(
        f"{API}/deposit/depositions", params={"size": 50}, headers=auth(token)
    )
    response.raise_for_status()
    for deposit in response.json():
        if deposit["title"] == title and not deposit["submitted"]:
            return deposit
    return None


def show(deposit: dict) -> None:
    print(f"  title   {deposit['title'][:70]}")
    print(f"  state   {deposit['state']} (submitted={deposit['submitted']})")
    print(f"  edit    {deposit['links']['html']}")
    print(f"  DOI     {deposit['metadata'].get('prereserve_doi', {}).get('doi', '-')}")


def _remote_files(client: httpx.Client, token: str, deposit_id: int) -> list[dict]:
    return client.get(
        f"{API}/deposit/depositions/{deposit_id}/files",
        headers=auth(token),
    ).json()


def sync(client: httpx.Client, token: str, spec: Deposit) -> dict:
    """Create or update the draft and put the bundle's current files on it."""
    existing = find_draft(client, token, spec.title)
    if existing is None:
        created = client.post(
            f"{API}/deposit/depositions",
            headers=auth(token),
            json=spec.metadata,
        )
        created.raise_for_status()
        deposit = created.json()
        print(f"created draft {deposit['id']}")
    else:
        updated = client.put(
            f"{API}/deposit/depositions/{existing['id']}",
            headers=auth(token),
            json=spec.metadata,
        )
        updated.raise_for_status()
        deposit = updated.json()
        print(f"reusing draft {deposit['id']}")

    bucket = deposit["links"]["bucket"]
    for path in spec.files():
        # Replace rather than skip: the reason to re-run is that the bundle
        # changed, and a stale file left on the draft would ship with it.
        for remote in _remote_files(client, token, deposit["id"]):
            if remote["filename"] == path.name:
                client.delete(
                    f"{API}/deposit/depositions/{deposit['id']}/files/{remote['id']}",
                    headers=auth(token),
                )
        with path.open("rb") as handle:
            put = client.put(
                f"{bucket}/{path.name}", headers=auth(token), content=handle
            )
        put.raise_for_status()
        print(f"  uploaded {path.name:<34} {path.stat().st_size:>9,} bytes")

    return client.get(
        f"{API}/deposit/depositions/{deposit['id']}", headers=auth(token)
    ).json()


def publish(client: httpx.Client, token: str, deposit: dict) -> int:
    """Mint the DOI. There is no undo."""
    response = client.post(
        f"{API}/deposit/depositions/{deposit['id']}/actions/publish",
        headers=auth(token),
    )
    if response.status_code >= 400:
        print(f"publish failed ({response.status_code}): {response.text[:500]}")
        return 1
    published = response.json()
    print("PUBLISHED")
    print(f"  DOI  {published.get('doi')}")
    print(f"  URL  {published['links'].get('record_html', published['links']['html'])}")
    return 0


def new_version(
    client: httpx.Client, token: str, record_id: int, spec: Deposit
) -> dict:
    """Open a draft for a new version of an already-published record.

    A published record cannot be edited, which is the point of a DOI. What it
    can have is another version: the concept DOI keeps resolving to the
    latest, the old version stays visible in the version history, and the
    correction is on the record rather than in place of it.

    Idempotent, and it finds an open draft by title rather than by following
    the record's `latest_draft` link. On a record whose draft is already
    open, Zenodo returns 400 from `newversion` and then reports
    `latest_draft` pointing back at the published record itself, so following
    that link fetches the wrong deposit and every write to it 404s.
    """
    existing = find_draft(client, token, spec.title)
    if existing is not None:
        deposit = existing
        print(f"reusing open version draft {deposit['id']}")
    else:
        opened = client.post(
            f"{API}/deposit/depositions/{record_id}/actions/newversion",
            headers=auth(token),
        )
        opened.raise_for_status()
        draft = client.get(opened.json()["links"]["latest_draft"], headers=auth(token))
        draft.raise_for_status()
        deposit = draft.json()
        print(f"opened version draft {deposit['id']} of record {record_id}")

    updated = client.put(
        f"{API}/deposit/depositions/{deposit['id']}",
        headers=auth(token),
        json=spec.metadata,
    )
    updated.raise_for_status()
    return updated.json()


def run(spec: Deposit, token: str, argv: list[str]) -> int:
    """The `--show` / `--publish` / default-is-a-draft command shape."""
    if not spec.bundle.exists():
        print(f"no bundle at {spec.bundle}")
        return 1

    with httpx.Client(timeout=300.0) as client:
        if "--show" in argv:
            existing = find_draft(client, token, spec.title)
            if existing is None:
                print("no draft exists")
                return 0
            show(existing)
            return 0

        if "--publish" in argv:
            existing = find_draft(client, token, spec.title)
            if existing is None:
                print("no draft to publish; run without --publish first")
                return 1
            show(existing)
            print()
            return publish(client, token, existing)

        if (
            spec.published_record is not None
            and find_draft(client, token, spec.title) is None
        ):
            print(
                f"already published as record {spec.published_record}.\n"
                "Use `--new-version` to correct or extend it; running the "
                "default would create a second deposit of the same bundle."
            )
            return 0

        deposit = sync(client, token, spec)
        print()
        show(deposit)
        print("\nnothing is published. `--publish` mints the DOI.")
        return 0

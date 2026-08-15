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


@dataclass(frozen=True)
class Deposit:
    """A bundle and the metadata Zenodo should carry for it."""

    title: str
    bundle: Path
    metadata: dict

    def files(self) -> list[Path]:
        return sorted(p for p in self.bundle.iterdir() if p.is_file())


def find_draft(client: httpx.Client, token: str, title: str) -> dict | None:
    """An existing unpublished deposit with this title, if there is one."""
    response = client.get(
        f"{API}/deposit/depositions", params={"access_token": token, "size": 50}
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
        params={"access_token": token},
    ).json()


def sync(client: httpx.Client, token: str, spec: Deposit) -> dict:
    """Create or update the draft and put the bundle's current files on it."""
    existing = find_draft(client, token, spec.title)
    if existing is None:
        created = client.post(
            f"{API}/deposit/depositions",
            params={"access_token": token},
            json=spec.metadata,
        )
        created.raise_for_status()
        deposit = created.json()
        print(f"created draft {deposit['id']}")
    else:
        updated = client.put(
            f"{API}/deposit/depositions/{existing['id']}",
            params={"access_token": token},
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
                    params={"access_token": token},
                )
        with path.open("rb") as handle:
            put = client.put(
                f"{bucket}/{path.name}", params={"access_token": token}, content=handle
            )
        put.raise_for_status()
        print(f"  uploaded {path.name:<34} {path.stat().st_size:>9,} bytes")

    return client.get(
        f"{API}/deposit/depositions/{deposit['id']}", params={"access_token": token}
    ).json()


def publish(client: httpx.Client, token: str, deposit: dict) -> int:
    """Mint the DOI. There is no undo."""
    response = client.post(
        f"{API}/deposit/depositions/{deposit['id']}/actions/publish",
        params={"access_token": token},
    )
    if response.status_code >= 400:
        print(f"publish failed ({response.status_code}): {response.text[:500]}")
        return 1
    published = response.json()
    print("PUBLISHED")
    print(f"  DOI  {published.get('doi')}")
    print(f"  URL  {published['links'].get('record_html', published['links']['html'])}")
    return 0


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

        deposit = sync(client, token, spec)
        print()
        show(deposit)
        print("\nnothing is published. `--publish` mints the DOI.")
        return 0

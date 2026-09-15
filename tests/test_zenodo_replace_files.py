"""A new version's draft must end up with exactly the bundle's files.

A Zenodo version draft starts with every file of the version before it, so a
table dropped from the bundle would ship again unless it is removed.
"""

from __future__ import annotations

import hashlib

import httpx

from softverse.release import zenodo_deposit


def test_stale_files_go_unchanged_files_stay_changed_files_upload(tmp_path):
    same = tmp_path / "usage_by_package.csv"
    same.write_text("package,n\nfixest,2\n")
    changed = tmp_path / "summary.json"
    changed.write_text('{"n": 2}')
    remote = [
        {
            "id": "a",
            "filename": "usage_by_package.csv",
            "checksum": hashlib.md5(
                same.read_bytes(), usedforsecurity=False
            ).hexdigest(),
        },
        {"id": "b", "filename": "summary.json", "checksum": "stale"},
        {"id": "c", "filename": "mentions_old.parquet", "checksum": "x"},
    ]
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "GET" and request.url.path.endswith("/files"):
            return httpx.Response(200, json=remote)
        return httpx.Response(200, json={})

    deposit = {"id": 7, "links": {"bucket": "https://zenodo.org/api/files/bkt"}}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        zenodo_deposit.replace_files(client, "t", deposit, [same, changed])

    deleted = {path.rsplit("/", 1)[-1] for method, path in calls if method == "DELETE"}
    uploaded = {path.rsplit("/", 1)[-1] for method, path in calls if method == "PUT"}
    assert deleted == {"b", "c"}
    assert uploaded == {"summary.json"}

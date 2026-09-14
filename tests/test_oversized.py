"""Code from an archive too big to download, without downloading it.

Served from a local HTTP server that honours byte ranges the way Harvard's S3
storage does, so the test measures what a range read actually transfers rather
than trusting that it is small.
"""

from __future__ import annotations

import io
import re
import tarfile
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from softverse.acquire.state import DatasetRecord
from softverse.sources import dataverse, oversized

DOI = "doi:10.7910/DVN/BIG001"


class _RangeHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    #: Requests left to refuse with a 504, shared across handler instances.
    fail_next = 0

    def do_GET(self):
        if _RangeHandler.fail_next > 0:
            _RangeHandler.fail_next -= 1
            self.send_response(504)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = Path(self.translate_path(self.path)).read_bytes()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if match is None:
            start, end, status = 0, len(data) - 1, 200
        elif match.group(1) == "":
            start = max(0, len(data) - int(match.group(2)))
            end, status = len(data) - 1, 206
        else:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else len(data) - 1
            status = 206
        body = data[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def serve(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_RangeHandler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def publish(name: str, content: bytes) -> dict:
        (root / name).write_bytes(content)
        return {
            "filename": name,
            "size_bytes": len(content),
            "file_id": 42,
            "url": f"{base}/{name}",
        }

    yield publish
    server.shutdown()
    server.server_close()


def _target(tmp_path):
    return dataverse.dataset_dir(tmp_path / "files", DOI)


def _incompressible(n: int) -> bytes:
    import random

    return random.Random(0).randbytes(n)  # noqa: S311


def test_a_zip_gives_up_its_code_for_a_fraction_of_its_size(tmp_path, serve):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("code.do", b"reghdfe y x")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("pkg/analysis.R", b"library(fixest)")
        zf.writestr("pkg/data/panel.dta", _incompressible(20_000_000))
        zf.writestr("pkg/code.zip", inner.getvalue())
        zf.writestr("pkg/big_data.zip", _incompressible(6 * 1024 * 1024))
    entry = serve("replication.zip", buf.getvalue())
    throttled = []

    outcome = oversized.recover(
        DOI,
        entry,
        _target(tmp_path),
        lambda _s: entry["url"],
        lambda: throttled.append(1),
    )

    assert outcome.error is None
    assert len(throttled) == 1, "only the Dataverse request waits on the limiter"
    assert sorted(r["filename"] for r in outcome.rows) == ["analysis.R", "code.do"]
    assert outcome.nested_skipped == 1, "the 6 MB nested archive is not fetched"
    assert outcome.transferred_bytes < 200_000, outcome.transferred_bytes
    target = dataverse.dataset_dir(tmp_path / "files", DOI)
    assert not list(target.rglob("*.dta"))
    assert not list(target.rglob("*.zip")), "the small nested zip is not kept"


def test_many_small_nested_archives_do_not_add_up_to_a_download(
    tmp_path, serve, monkeypatch
):
    monkeypatch.setattr(oversized, "NESTED_BUDGET_BYTES", 2_500_000)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("run.do", b"use panel")
        for i in range(10):
            zf.writestr(f"data/chunk{i}.zip", _incompressible(1_000_000))
    entry = serve("package.zip", buf.getvalue())

    outcome = oversized.recover(
        DOI, entry, _target(tmp_path), lambda _s: entry["url"], lambda: None
    )

    assert outcome.error is None
    assert outcome.nested_skipped == 8, "two 1 MB nested zips fit a 2.5 MB budget"
    assert outcome.transferred_bytes < 3_500_000, outcome.transferred_bytes


def test_a_tarball_is_downloaded_mined_and_deleted(tmp_path, serve):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [("a/main.do", b"use x"), ("a/data.csv", b"1,2\n" * 1000)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    entry = serve("replication.tar.gz", buf.getvalue())

    outcome = oversized.recover(
        DOI, entry, _target(tmp_path), lambda _s: entry["url"], lambda: None
    )

    assert outcome.error is None
    assert [r["filename"] for r in outcome.rows] == ["main.do"]
    target = dataverse.dataset_dir(tmp_path / "files", DOI)
    assert not (target / "_archives" / "replication.tar.gz").exists()
    assert not list(target.rglob("*.csv"))


def test_apply_moves_recovered_archives_out_of_skipped():
    record = DatasetRecord(
        dataset_doi=DOI,
        state="complete",
        n_candidate=3,
        n_fetched=1,
        n_skipped_over_cap=2,
        skipped_archives=[
            {"filename": "ok.zip", "size_bytes": 1, "file_id": 1},
            {"filename": "bad.7z", "size_bytes": 1, "file_id": 2},
        ],
    )
    ok = oversized.Outcome(DOI, "ok.zip", 1, 1, "range")
    bad = oversized.Outcome(DOI, "bad.7z", 2, 1, "download", error="boom")

    oversized.apply(record, [ok, bad])

    assert record.reconciles()
    assert record.n_fetched == 2
    assert record.n_skipped_over_cap == 1
    assert record.skipped_archives == [
        {"filename": "bad.7z", "size_bytes": 1, "file_id": 2, "recovery_error": "boom"}
    ]


def test_zenodo_throttles_every_request(tmp_path, serve):
    """Zenodo serves the bytes itself, so the range reads are its requests too."""
    buf = io.BytesIO()
    # Members bigger than remotezip's 64 KB first read, so each needs its own.
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i in range(3):
            zf.writestr(f"code/s{i}.do", _incompressible(200_000))
    entry = serve("pkg.zip", buf.getvalue())
    throttled = []

    outcome = oversized.recover(
        "10.5281/zenodo.123",
        entry,
        tmp_path / "files" / "123",
        lambda _s: entry["url"],
        lambda: throttled.append(1),
        throttle_every_request=True,
    )

    assert outcome.error is None
    assert len(outcome.rows) == 3
    assert len(throttled) >= 4, "the directory read and each member wait"


def test_zenodo_content_url_is_built_from_the_doi():
    assert oversized.zenodo_content_url("10.5281/zenodo.17387697", "a b.zip") == (
        "https://zenodo.org/api/records/17387697/files/a%20b.zip/content"
    )


def test_a_gateway_timeout_is_retried_not_fatal(tmp_path, serve, monkeypatch):
    """One 504 used to fail the archive; 38 of 41 did during a Zenodo outage."""
    monkeypatch.setattr(oversized.Retry, "DEFAULT_BACKOFF_MAX", 0)
    monkeypatch.setattr(_RangeHandler, "fail_next", 2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("main.do", b"use x")
    entry = serve("pkg.zip", buf.getvalue())

    outcome = oversized.recover(
        DOI, entry, _target(tmp_path), lambda _s: entry["url"], lambda: None
    )

    assert outcome.error is None
    assert [r["filename"] for r in outcome.rows] == ["main.do"]

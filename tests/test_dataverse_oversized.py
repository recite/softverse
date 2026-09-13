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
from softverse.sources import dataverse, dataverse_oversized

DOI = "doi:10.7910/DVN/BIG001"


class _RangeHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        data = Path(self.translate_path(self.path)).read_bytes()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if match is None:
            start, end, status = 0, len(data) - 1, 200
        elif match.group(1) == "":
            start, end, status = len(data) - int(match.group(2)), len(data) - 1, 206
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
def serve(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    root.mkdir()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_RangeHandler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def publish(name: str, content: bytes) -> dict:
        (root / name).write_bytes(content)
        monkeypatch.setattr(
            dataverse_oversized, "storage_url", lambda *a, **k: f"{base}/{name}"
        )
        return {"filename": name, "size_bytes": len(content), "file_id": 42}

    yield publish
    server.shutdown()
    server.server_close()


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

    outcome = dataverse_oversized.recover(
        DOI, entry, tmp_path / "files", {}, lambda: throttled.append(1)
    )

    assert outcome.error is None
    assert len(throttled) == 1, "only the Dataverse request waits on the limiter"
    assert sorted(r["filename"] for r in outcome.rows) == ["analysis.R", "code.do"]
    assert outcome.nested_skipped == 1, "the 6 MB nested archive is not fetched"
    assert outcome.transferred_bytes < 200_000, outcome.transferred_bytes
    target = dataverse.dataset_dir(tmp_path / "files", DOI)
    assert not list(target.rglob("*.dta"))
    assert not list(target.rglob("*.zip")), "the small nested zip is not kept"


def test_a_tarball_is_downloaded_mined_and_deleted(tmp_path, serve):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in [("a/main.do", b"use x"), ("a/data.csv", b"1,2\n" * 1000)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    entry = serve("replication.tar.gz", buf.getvalue())

    outcome = dataverse_oversized.recover(
        DOI, entry, tmp_path / "files", {}, lambda: None
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
    ok = dataverse_oversized.Outcome(DOI, "ok.zip", 1, 1, "range")
    bad = dataverse_oversized.Outcome(DOI, "bad.7z", 2, 1, "download", error="boom")

    dataverse_oversized.apply(record, [ok, bad])

    assert record.reconciles()
    assert record.n_fetched == 2
    assert record.n_skipped_over_cap == 1
    assert record.skipped_archives == [
        {"filename": "bad.7z", "size_bytes": 1, "file_id": 2, "recovery_error": "boom"}
    ]

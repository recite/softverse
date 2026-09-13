"""A Dataverse archive is a container for code, not something to keep.

The Zenodo collector deletes an archive once its code is extracted; the
Dataverse one did not, and 4,400 deposits left 18.9 GB of archives on disk
around 0.67 GB of code. It also counted an archive that failed to extract as
fetched, marking the deposit complete and so unretryable.
"""

from __future__ import annotations

import io
import zipfile

from softverse.sources import dataverse

DOI = "doi:10.7910/DVN/TEST01"


def _payload(filename: str, size: int) -> dict:
    return {
        "versionState": "RELEASED",
        "files": [{"dataFile": {"id": 1, "filename": filename, "filesize": size}}],
    }


def _collect(tmp_path, monkeypatch, filename: str, content: bytes):
    monkeypatch.setattr(
        dataverse,
        "fetch_version_metadata",
        lambda *a, **k: (_payload(filename, len(content)), None, 200),
    )
    monkeypatch.setattr(dataverse, "fetch_single", lambda *a, **k: (content, None))
    return dataverse.collect_dataset(None, DOI, tmp_path / "files", tmp_path / "raw")


def test_the_archive_is_deleted_once_its_code_is_extracted(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("code/analysis.R", b"library(dplyr)")
        zf.writestr("data/big.csv", b"x" * 10_000)
    record, rows = _collect(tmp_path, monkeypatch, "replication.zip", buf.getvalue())

    target = dataverse.dataset_dir(tmp_path / "files", DOI)
    assert record.state == "complete"
    assert [r["filename"] for r in rows] == ["analysis.R"]
    assert not (target / "_archives" / "replication.zip").exists()
    assert not list(target.rglob("*.csv")), "data inside the archive is not kept"


def test_an_archive_that_will_not_open_is_kept_and_retryable(tmp_path, monkeypatch):
    record, rows = _collect(tmp_path, monkeypatch, "broken.zip", b"not a zip")

    target = dataverse.dataset_dir(tmp_path / "files", DOI)
    assert record.state == "partial"
    assert record.n_failed == 1
    assert record.n_fetched == 0
    assert record.needs_retry
    assert rows == []
    assert (target / "_archives" / "broken.zip").exists(), "kept as the evidence"

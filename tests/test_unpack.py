"""Tests for archive extraction.

These are security tests as much as correctness tests: every archive in the
corpus was uploaded by a stranger, and v1 called ``extractall`` on them from six
places with no member validation.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from softverse.acquire.unpack import (
    extract,
    extract_tar,
    extract_zip,
    is_safe_member,
    relative_member_path,
)

KEEP = frozenset({".r", ".py", ".do"})
KEEP_NAMES = frozenset({"renv.lock", "requirements.txt"})


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


# -- path safety ----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "../escape.R",
        "../../etc/passwd",
        "/absolute/evil.R",
        "a/../../../evil.R",
        "C:/windows/evil.R",
        "..\\windows\\evil.R",
    ],
)
def test_traversal_members_are_rejected(name, tmp_path):
    assert not is_safe_member(name, tmp_path)


@pytest.mark.parametrize("name", ["code/analysis.R", "a/b/c/run.do", "top.py"])
def test_ordinary_members_are_accepted(name, tmp_path):
    assert is_safe_member(name, tmp_path)


def test_zip_slip_member_is_not_written(tmp_path):
    """The actual attack, end to end."""
    archive = make_zip(tmp_path / "evil.zip", {"../escaped.R": b"library(x)"})
    dest = tmp_path / "out"
    result = extract_zip(archive, dest, KEEP)
    assert result.files == []
    assert "../escaped.R" in result.skipped_unsafe
    assert not (tmp_path / "escaped.R").exists()


def test_tar_symlink_member_is_rejected(tmp_path):
    """A symlink member turns a later write into an arbitrary-file overwrite."""
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link.R")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    result = extract_tar(archive, tmp_path / "out", KEEP)
    assert result.files == []
    assert "link.R" in result.skipped_unsafe


# -- bombs ----------------------------------------------------------------


def test_declared_member_count_bomb_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("softverse.acquire.unpack.MAX_MEMBERS", 5)
    archive = make_zip(tmp_path / "many.zip", {f"f{i}.R": b"x" for i in range(10)})
    result = extract_zip(archive, tmp_path / "out", KEEP)
    assert result.error is not None and "members" in result.error


def test_uncompressed_size_bomb_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("softverse.acquire.unpack.MAX_UNCOMPRESSED_BYTES", 100)
    archive = make_zip(tmp_path / "bomb.zip", {"big.R": b"x" * 5000})
    result = extract_zip(archive, tmp_path / "out", KEEP)
    assert result.error is not None and "uncompressed" in result.error


# -- what gets kept -------------------------------------------------------


def test_directory_structure_is_preserved(tmp_path):
    """v1 flattened to basenames, which is why vendored libraries could not be
    told from research code: only 5 subdirectories survived across 12,054
    deposits."""
    archive = make_zip(
        tmp_path / "a.zip",
        {"code/01_clean.R": b"library(dplyr)", "renv/activate.R": b"# vendored"},
    )
    dest = tmp_path / "out"
    result = extract_zip(archive, dest, KEEP)
    paths = sorted(relative_member_path(p, dest) for p in result.files)
    assert paths == ["code/01_clean.R", "renv/activate.R"]


def test_same_basename_in_two_directories_both_survive(tmp_path):
    """v1 renamed collisions to _1/_2, so re-runs manufactured duplicates."""
    archive = make_zip(tmp_path / "a.zip", {"a/run.R": b"1", "b/run.R": b"2"})
    dest = tmp_path / "out"
    result = extract_zip(archive, dest, KEEP)
    assert len(result.files) == 2
    assert (dest / "a/run.R").read_bytes() == b"1"
    assert (dest / "b/run.R").read_bytes() == b"2"


def test_unwanted_files_are_skipped_not_deleted_from_disk(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip", {"data.csv": b"a,b", "code.R": b"library(x)"}
    )
    result = extract_zip(archive, tmp_path / "out", KEEP)
    assert [p.name for p in result.files] == ["code.R"]
    assert "data.csv" in result.skipped_other


def test_manifests_are_kept(tmp_path):
    """renv.lock is free recall ground truth; v1 deleted every one of them."""
    archive = make_zip(tmp_path / "a.zip", {"renv.lock": b"{}", "notes.txt": b"hi"})
    result = extract_zip(archive, tmp_path / "out", KEEP, KEEP_NAMES)
    assert [p.name for p in result.files] == ["renv.lock"]


def test_appledouble_forks_are_excluded(tmp_path):
    """__MACOSX/._x.py are resource forks, not source. v1 parsed 275 of them."""
    archive = make_zip(
        tmp_path / "a.zip",
        {"__MACOSX/._main.py": b"\x00\x05\x16\x07", "main.py": b"import os"},
    )
    result = extract_zip(archive, tmp_path / "out", KEEP)
    assert [p.name for p in result.files] == ["main.py"]


# -- nesting --------------------------------------------------------------


def test_nested_archive_is_extracted(tmp_path):
    """code.zip inside replication.zip -- invisible to v1's Dataverse path."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep/analysis.R", b"library(brms)")
    archive = make_zip(
        tmp_path / "outer.zip", {"code.zip": inner.getvalue(), "top.R": b"library(x)"}
    )
    result = extract(archive, tmp_path / "out", KEEP | {".zip"})
    names = {p.name for p in result.files}
    assert "analysis.R" in names
    assert "top.R" in names


def test_nesting_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr("softverse.acquire.unpack.MAX_NESTING", 0)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("analysis.R", b"library(brms)")
    archive = make_zip(tmp_path / "outer.zip", {"code.zip": inner.getvalue()})
    result = extract(archive, tmp_path / "out", KEEP | {".zip"})
    assert "analysis.R" not in {p.name for p in result.files}


# -- failure is reported, never silent ------------------------------------


def test_corrupt_archive_reports_an_error(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip at all")
    result = extract_zip(bad, tmp_path / "out", KEEP)
    assert result.files == []
    assert result.error is not None


def test_unsupported_type_is_reported(tmp_path):
    odd = tmp_path / "thing.rar"
    odd.write_bytes(b"Rar!")
    result = extract(odd, tmp_path / "out", KEEP)
    assert result.error is not None and "unsupported" in result.error


def test_tar_is_handled(tmp_path):
    """v1 marked .tar as an archive but had no branch for it, then deleted the
    file regardless -- downloaded, never extracted, gone."""
    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tf:
        data = b"library(x)"
        info = tarfile.TarInfo("code/run.R")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    result = extract(archive, tmp_path / "out", KEEP)
    assert [p.name for p in result.files] == ["run.R"]
    assert result.error is None


def test_limits_admit_real_research_archives():
    """Calibrated against the corpus, not an imagined attacker.

    The first limits (2 GB, 50k members) rejected 6 of the first 100 Zenodo
    deposits -- economics packages declaring 3.0-6.6 GB uncompressed and one
    with 140,219 members. None was an attack. A guard that drops 6% of the
    corpus is not protecting the analysis, it is biasing it toward
    less-data-heavy work.
    """
    from softverse.acquire.unpack import MAX_MEMBERS, MAX_UNCOMPRESSED_BYTES

    assert MAX_UNCOMPRESSED_BYTES >= 7 * 1024**3, "6.6 GB deposits must pass"
    assert MAX_MEMBERS >= 200_000, "a 140,219-member deposit must pass"
    # ...but a real bomb is still refused.
    assert MAX_UNCOMPRESSED_BYTES < 1024**4


def test_7z_is_supported(tmp_path):
    """A .7z deposit was downloaded and lost to a missing branch -- the same
    way v1 lost every .tar."""
    py7zr = pytest.importorskip("py7zr")
    archive = tmp_path / "a.7z"
    with py7zr.SevenZipFile(archive, "w") as sz:
        src = tmp_path / "analysis.R"
        src.write_text("library(dplyr)")
        sz.write(src, "code/analysis.R")
    result = extract(archive, tmp_path / "out", KEEP)
    assert result.error is None, result.error
    assert [p.name for p in result.files] == ["analysis.R"]

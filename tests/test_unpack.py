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
    archive_format,
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


def test_size_bomb_of_wanted_files_is_refused(tmp_path, monkeypatch):
    """The bomb that matters is one made of files we would actually write."""
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 100)
    archive = make_zip(tmp_path / "bomb.zip", {"big.R": b"x" * 5000})
    result = extract_zip(archive, tmp_path / "out", KEEP)
    assert result.error is not None and "expanded" in result.error


def test_an_archive_of_mostly_data_is_extracted_not_refused(tmp_path, monkeypatch):
    """A huge archive holding a little code must yield that code.

    Twelve deposits were lost to a guard that read the archive's *declared*
    uncompressed size: they declared 432 GB between them and held 26.5 MB of
    `.do` and `.R`, because extraction keeps only source and drops the panel
    datasets that are the other 99.994%. Rejecting on a number three orders of
    magnitude larger than what we write is not a safety guard, it is a filter
    -- and it selects against the largest empirical replication packages,
    which is a bias correlated with the outcome being measured.
    """
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 10_000)
    archive = make_zip(
        tmp_path / "replication.zip",
        {"panel.dta": b"\0" * 500_000, "analysis.do": b"reg y x"},
    )
    result = extract_zip(archive, tmp_path / "out", frozenset({".do"}))
    assert result.error is None, result.error
    assert [p.name for p in result.files] == ["analysis.do"]


def test_the_budget_stops_extraction_rather_than_reporting_it_afterwards(
    tmp_path, monkeypatch
):
    """Checked after the loop, a budget bounds the report, not the disk.

    Two 5 KB members against a 6 KB budget: if the check only runs at the end,
    both are on disk before anything notices. The point of the limit is that
    the second one never gets written.
    """
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 6_000)
    dest = tmp_path / "out"
    archive = make_zip(
        tmp_path / "runaway.zip", {"a.R": b"x" * 5000, "b.R": b"y" * 5000}
    )
    result = extract_zip(archive, dest, KEEP)
    assert result.error is not None and "expanded" in result.error
    on_disk = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    assert on_disk <= 6_000, f"{on_disk} bytes written against a 6,000 budget"


def test_one_oversized_member_cannot_outrun_the_budget(tmp_path, monkeypatch):
    """A single member larger than the whole budget must not be written whole.

    zipfile's read() pulls the entire member into memory first, so a 4 GB
    member inside a compliant archive is both an unbounded allocation and an
    unbounded write. The copy has to be chunked for the budget to mean
    anything at member granularity.
    """
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 1_000)
    dest = tmp_path / "out"
    archive = make_zip(tmp_path / "one.zip", {"huge.R": b"x" * 200_000})
    result = extract_zip(archive, dest, KEEP)
    assert result.error is not None
    on_disk = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    assert on_disk <= 1_000 + (1 << 20), f"{on_disk} bytes written"


def test_the_budget_spans_nesting_levels(tmp_path, monkeypatch):
    """An archive of archives must not get a fresh allowance per level."""
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 8_000)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep/b.R", b"y" * 5000)
    archive = make_zip(
        tmp_path / "outer.zip", {"a.R": b"x" * 5000, "data.zip": inner.getvalue()}
    )
    result = extract(archive, tmp_path / "out", KEEP | {".zip"})
    assert result.error is not None and "expanded" in result.error


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
    """`.rar` used to be the example here, and is now supported."""
    odd = tmp_path / "thing.sit"
    odd.write_bytes(b"StuffIt!")
    result = extract(odd, tmp_path / "out", KEEP)
    assert result.error is not None and "unsupported" in result.error


def test_rar_is_routed_to_the_rar_reader(tmp_path):
    """Five deposits ship one, and each was recorded as an unsupported type.

    Asserting on the routing rather than on a round trip, because RAR3+ has no
    pure-Python encoder either: there is no way to build a fixture here. What
    is testable is that a RAR header no longer falls off the end of the
    dispatch, and the error it now produces comes from the rar reader.
    """
    pytest.importorskip("rarfile")
    odd = tmp_path / "package.rar"
    odd.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"\0" * 32)
    assert archive_format(odd) == "rar"
    result = extract(odd, tmp_path / "out", KEEP)
    assert "unsupported" not in (result.error or "")


def test_7z_symlink_member_is_rejected(tmp_path):
    """The one extractor that did not check, because it does not do the writing.

    zip and tar are written member by member here, so a link never gets
    created. py7zr writes the members itself, and a member named `code/x.R`
    pointing at `/etc/passwd` clears `is_safe_member` -- the name stays inside
    the root -- leaving a link that a later write goes through.
    """
    py7zr = pytest.importorskip("py7zr")
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.R").write_text("library(x)")
    (src / "evil.R").symlink_to("/etc/passwd")
    archive = tmp_path / "a.7z"
    with py7zr.SevenZipFile(archive, "w") as sz:
        sz.write(src / "real.R", "real.R")
        sz.write(src / "evil.R", "evil.R")

    dest = tmp_path / "out"
    result = extract(archive, dest, KEEP)
    assert [p.name for p in result.files] == ["real.R"]
    assert "evil.R" in result.skipped_unsafe
    assert not (dest / "evil.R").exists()


def test_a_rar_misnamed_zip_is_still_routed_to_the_rar_reader(tmp_path):
    odd = tmp_path / "replication package.zip"
    odd.write_bytes(b"Rar!\x1a\x07\x01\x00" + b"\0" * 32)
    assert archive_format(odd) == "rar"


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
    with 140,219 members. None was an attack. Raising the declared-size
    ceiling to 20 GB bought 1,300 deposits before twelve more hit it, which is
    the tell that the quantity was wrong rather than the number: it is now the
    *written* bytes that are bounded, and the largest package observed writes
    12.9 MB.
    """
    from softverse.acquire.unpack import MAX_EXTRACTED_BYTES, MAX_MEMBERS

    assert MAX_MEMBERS >= 200_000, "a 140,219-member deposit must pass"
    assert MAX_EXTRACTED_BYTES >= 1024**3, "12.9 MB of code must pass with room"
    # ...but a real bomb is still refused.
    assert MAX_EXTRACTED_BYTES < 1024**4


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


def test_archive_type_is_read_from_the_header_not_the_extension(tmp_path):
    """A deposit shipped a 7z named `.zip`, and dispatching on the suffix lost it.

    `zipfile` said "File is not a zip file", which the collector recorded as a
    transport failure -- indistinguishable in the ledger from a timeout, so it
    would have been retried forever and never worked. Eight bytes of header
    settle it.
    """
    py7zr = pytest.importorskip("py7zr")
    archive = tmp_path / "replication package.zip"
    with py7zr.SevenZipFile(archive, "w") as sz:
        src = tmp_path / "analysis.do"
        src.write_text("reg y x")
        sz.write(src, "code/analysis.do")
    result = extract(archive, tmp_path / "out", frozenset({".do"}))
    assert result.error is None, result.error
    assert [p.name for p in result.files] == ["analysis.do"]


def test_a_zip_named_tar_is_still_read_as_a_zip(tmp_path):
    """The same rule in the other direction."""
    archive = make_zip(tmp_path / "bundle.tar.gz", {"run.R": b"library(x)"})
    result = extract(archive, tmp_path / "out", KEEP)
    assert result.error is None, result.error
    assert [p.name for p in result.files] == ["run.R"]


def test_an_unrecognised_header_falls_back_to_the_extension(tmp_path):
    """Self-extracting zips carry a stub before the `PK` signature.

    One deposit's zip starts `d2 75 6f 7f` and opens fine, because a zip's
    directory lives at the end of the file. Header sniffing must not turn a
    working case into a failure.
    """
    payload = make_zip(tmp_path / "inner.zip", {"run.R": b"library(x)"}).read_bytes()
    archive = tmp_path / "selfextract.zip"
    archive.write_bytes(b"\xd2\x75\x6f\x7f" * 64 + payload)
    result = extract(archive, tmp_path / "out", KEEP)
    assert result.error is None, result.error
    assert [p.name for p in result.files] == ["run.R"]


def test_nested_archive_is_deleted_but_its_contents_survive(tmp_path):
    """The leak, stated as a test.

    extract() recurses into a data.zip inside a replication.zip, writes it to
    disk, extracts it -- and used to keep the compressed copy forever. Measured
    in production: 572 nested archives holding 2.2 GB, still growing mid-run,
    on a disk at 99%. The top-level delete had always existed, which is exactly
    why "we delete archives after extracting" was true and insufficient.
    """
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("deep/analysis.R", b"library(brms)")
    archive = make_zip(
        tmp_path / "outer.zip", {"data.zip": inner.getvalue(), "top.R": b"library(x)"}
    )
    dest = tmp_path / "out"
    result = extract(archive, dest, KEEP | {".zip"})

    assert "analysis.R" in {p.name for p in result.files}, "contents must survive"
    assert not (dest / "data.zip").exists(), "the nested archive should be gone"
    assert not any(
        p.suffix == ".zip" for p in result.files
    ), "a deleted archive must not remain in the file list"


def test_a_nested_archive_that_fails_to_extract_is_kept(tmp_path):
    """Same rule as top-level: a failure keeps its evidence."""
    archive = make_zip(
        tmp_path / "outer.zip", {"broken.zip": b"not a zip at all", "top.R": b"x"}
    )
    dest = tmp_path / "out"
    extract(archive, dest, KEEP | {".zip"})
    assert (dest / "broken.zip").exists()


def test_expansion_budget_stops_a_runaway_deposit(tmp_path, monkeypatch):
    """The download cap bounds transfer; this bounds expansion.

    Without it a small archive of nested archives can still fill the disk,
    whatever the download cap says.
    """
    monkeypatch.setattr("softverse.acquire.unpack.MAX_EXTRACTED_BYTES", 100)
    archive = make_zip(tmp_path / "big.zip", {"a.R": b"x" * 5000})
    result = extract(archive, tmp_path / "out", KEEP)
    assert result.error is not None and "expanded" in result.error

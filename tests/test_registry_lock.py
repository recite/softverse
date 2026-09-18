"""The lock decides which snapshot resolves names, and says so when it cannot."""

from __future__ import annotations

import hashlib
import json

import pytest

from softverse.registries.lock import (
    LockError,
    lock_id,
    pinned_directory,
    pinned_file,
    pinned_names,
    read_lock,
    write_lock,
)


def _snapshot(root, registry, date, names):
    directory = root / registry / date
    directory.mkdir(parents=True)
    raw = json.dumps(names).encode()
    (directory / "names.json").write_text(json.dumps(names))
    (directory / "raw.bin").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (directory / "source.json").write_text(json.dumps({"sha256": digest}))
    return digest


def test_the_lock_pins_the_newest_snapshot_and_the_loader_honours_it(tmp_path):
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    old = _snapshot(snapshots, "cran", "2026-08-11", ["old"])
    write_lock(snapshots, lock_path)
    _snapshot(snapshots, "cran", "2026-09-01", ["new"])

    lock = read_lock(lock_path)
    assert lock["cran"].date == "2026-08-11"
    assert lock["cran"].sha256 == old
    # A newer snapshot on disk changes nothing until the lock is rewritten.
    assert pinned_names("cran", lock, snapshots) == {"old"}

    write_lock(snapshots, lock_path)
    assert pinned_names("cran", read_lock(lock_path), snapshots) == {"new"}


def test_a_snapshot_whose_digest_moved_is_refused(tmp_path):
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _snapshot(snapshots, "pypi", "2026-08-11", ["a"])
    write_lock(snapshots, lock_path)
    source = snapshots / "pypi" / "2026-08-11" / "source.json"
    source.write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(LockError, match="digest"):
        pinned_names("pypi", read_lock(lock_path), snapshots)


def test_an_unpinned_or_missing_snapshot_is_named_in_the_error(tmp_path):
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _snapshot(snapshots, "cran", "2026-08-11", ["a"])
    write_lock(snapshots, lock_path)
    lock = read_lock(lock_path)
    with pytest.raises(LockError, match="not in the lock"):
        pinned_names("bioconductor", lock, snapshots)
    lock_path.write_text(json.dumps({"cran": {"date": "2020-01-01", "sha256": "x"}}))
    with pytest.raises(LockError, match="not on disk"):
        pinned_names("cran", read_lock(lock_path), snapshots)


def test_a_payload_is_hashed_not_taken_on_the_snapshots_word(tmp_path):
    """The Stata index is pinned by its bytes, so an edited parquet is refused."""
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    directory = snapshots / "stata_index" / "2026-09-17"
    directory.mkdir(parents=True)
    (directory / "index.parquet").write_bytes(b"index")
    digest = hashlib.sha256(b"index").hexdigest()
    (directory / "source.json").write_text(json.dumps({"sha256": digest}))
    write_lock(snapshots, lock_path)
    lock = read_lock(lock_path)
    assert (
        pinned_directory("stata_index", lock, snapshots, "index.parquet") == directory
    )
    (directory / "index.parquet").write_bytes(b"edited")
    with pytest.raises(LockError, match="digest"):
        pinned_directory("stata_index", lock, snapshots, "index.parquet")


def test_the_official_command_cache_is_pinned_by_its_bytes(tmp_path):
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _snapshot(snapshots, "cran", "2026-08-11", ["a"])
    official = snapshots / "stata_official" / "official.json"
    official.parent.mkdir(parents=True)
    official.write_text(json.dumps({"snapshot_date": "2026-08-14", "commands": {}}))
    write_lock(snapshots, lock_path)
    lock = read_lock(lock_path)
    assert lock["stata_official"].date == "2026-08-14"
    assert pinned_file("stata_official", lock, snapshots) == official
    official.write_text(
        json.dumps({"snapshot_date": "2026-08-14", "commands": {"x": True}})
    )
    with pytest.raises(LockError, match="re-pin"):
        pinned_file("stata_official", lock, snapshots)


def test_the_lock_id_moves_when_any_pin_does(tmp_path):
    """It was CRAN's digest alone, so refreshing PyPI left the stamp unchanged."""
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _snapshot(snapshots, "cran", "2026-08-11", ["a"])
    _snapshot(snapshots, "pypi", "2026-08-11", ["b"])
    write_lock(snapshots, lock_path)
    before = lock_id(read_lock(lock_path))
    _snapshot(snapshots, "pypi", "2026-09-01", ["c"])
    write_lock(snapshots, lock_path)
    assert lock_id(read_lock(lock_path)) != before

"""The registry lock: which dated snapshot each registry resolves against.

`registries.lock.json` used to hold a digest per registry and nothing read it
but a paper table. Resolution picked whichever dated snapshot sorted last, so
refetching a registry changed every resolution in the corpus while the lock
still named the old digests. The lock now names a date as well as a digest,
the loader reads the snapshot the lock names, and the digest is checked
against the snapshot's own `source.json` before a single name is resolved.

Refreshing a registry is a deliberate act: fetch, then pin, then rebuild the
whole tally, because a package archived from CRAN between two snapshots moves
from `known_current` to `known_archived` for every deposit that loads it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from softverse.config import PATHS

if TYPE_CHECKING:
    from pathlib import Path

LOCK = PATHS.registries / "registries.lock.json"
SNAPSHOTS = PATHS.registries / "snapshots"

#: Snapshots that are one growing file rather than a dated directory, by path
#: under the snapshot root. The help-server answers are a cache that is
#: extended, never refetched, so there is no second copy to date -- but they
#: decide what is a Stata builtin, and an unpinned input to resolution is what
#: the lock exists to rule out.
PINNED_FILES = {"stata_official": "stata_official/official.json"}


class LockError(RuntimeError):
    """The lock and the snapshots on disk disagree."""


@dataclass(frozen=True)
class Pin:
    """One registry's pinned snapshot."""

    date: str
    sha256: str


def read_lock(path: Path = LOCK) -> dict[str, Pin]:
    """The lock as written by `write_lock`.

    Args:
        path: The lock file.

    Returns:
        Registry name to its pin.
    """
    raw = json.loads(path.read_text())
    return {name: Pin(entry["date"], entry["sha256"]) for name, entry in raw.items()}


def write_lock(
    snapshots: Path = SNAPSHOTS,
    path: Path = LOCK,
    registries: list[str] | None = None,
) -> dict[str, Pin]:
    """Pin every dated registry to its newest snapshot on disk.

    Args:
        snapshots: The snapshot root, `<registry>/<date>/source.json` under it.
        path: Where to write the lock.
        registries: Which registries to pin; every dated one by default.

    Returns:
        What was written.
    """
    names = registries or sorted(
        d.name for d in snapshots.iterdir() if any(d.glob("*/source.json"))
    )
    pins = {}
    for name in names:
        newest = max((snapshots / name).glob("*/source.json"))
        source = json.loads(newest.read_text())
        pins[name] = Pin(newest.parent.name, source["sha256"])
    if registries is None:
        for name, relative in PINNED_FILES.items():
            if (file := snapshots / relative).exists():
                pins[name] = Pin(
                    json.loads(file.read_text())["snapshot_date"],
                    hashlib.sha256(file.read_bytes()).hexdigest(),
                )
    pins = dict(sorted(pins.items()))
    path.write_text(
        json.dumps(
            {
                name: {"date": pin.date, "sha256": pin.sha256}
                for name, pin in pins.items()
            },
            indent=2,
        )
        + "\n"
    )
    return pins


def pinned_directory(
    registry: str,
    lock: dict[str, Pin],
    snapshots: Path = SNAPSHOTS,
    payload: str | None = None,
) -> Path:
    """The snapshot directory the lock pins, after checking its digest.

    Args:
        registry: Which registry.
        lock: The lock, from `read_lock`.
        snapshots: The snapshot root.
        payload: A file in the snapshot to hash and compare as well. Without
            it the digest the snapshot *recorded* is checked, not its bytes.

    Returns:
        The directory.

    Raises:
        LockError: if the registry is unpinned, the snapshot is missing, or
            its digest is not the one the lock names.
    """
    if registry not in lock:
        raise LockError(f"{registry} is not in the lock; run scripts/pin_registries.py")
    pin = lock[registry]
    directory = snapshots / registry / pin.date
    source = directory / "source.json"
    if not source.exists():
        raise LockError(f"{registry}: the lock names {pin.date}, which is not on disk")
    found = json.loads(source.read_text())["sha256"]
    if payload is not None:
        found = hashlib.sha256((directory / payload).read_bytes()).hexdigest()
    if found != pin.sha256:
        raise LockError(
            f"{registry}: snapshot {pin.date} has digest {found[:12]}, "
            f"the lock names {pin.sha256[:12]}"
        )
    return directory


def pinned_file(
    registry: str, lock: dict[str, Pin], snapshots: Path = SNAPSHOTS
) -> Path:
    """One of `PINNED_FILES`, after checking its bytes against the lock.

    Args:
        registry: A key of `PINNED_FILES`.
        lock: The lock, from `read_lock`.
        snapshots: The snapshot root.

    Returns:
        The file.

    Raises:
        LockError: if it is unpinned or its digest is not the one the lock names.
    """
    if registry not in lock:
        raise LockError(f"{registry} is not in the lock; run scripts/pin_registries.py")
    path = snapshots / PINNED_FILES[registry]
    found = hashlib.sha256(path.read_bytes()).hexdigest()
    if found != lock[registry].sha256:
        raise LockError(
            f"{registry}: has digest {found[:12]}, the lock names "
            f"{lock[registry].sha256[:12]}; re-pin and rebuild the tally"
        )
    return path


def pinned_names(
    registry: str, lock: dict[str, Pin], snapshots: Path = SNAPSHOTS
) -> frozenset[str]:
    """The names in the snapshot the lock pins, after checking its digest.

    Args:
        registry: Which registry.
        lock: The lock, from `read_lock`.
        snapshots: The snapshot root.

    Returns:
        The package names.
    """
    directory = pinned_directory(registry, lock, snapshots)
    return frozenset(json.loads((directory / "names.json").read_text()))


def lock_id(lock: dict[str, Pin]) -> str:
    """One short digest over every pin, stamped on each mention.

    It used to be CRAN's digest alone, so refreshing PyPI or the Stata index
    changed resolutions under a stamp that did not move.
    """
    joined = "\n".join(f"{name}:{pin.sha256}" for name, pin in sorted(lock.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]

"""Assemble a :class:`~softverse.registries.resolve.Registry` from pinned snapshots.

The snapshots are written by :func:`softverse.registries.fetch.fetch_all` under
``registries/snapshots/``. Paths are anchored to the project root rather than
the working directory, so the registry loads the same from a script, a test or
a notebook.
"""

from __future__ import annotations

import json

import duckdb

from softverse.config import PATHS
from softverse.registries.resolve import Registry
from softverse.stata.builtins import builtins

#: Commands checked one by one against StataCorp's help server. Widens the
#: curated builtin list by 165 official commands that were otherwise reported
#: as resolving to no registry. Built by scripts/verify_stata_official.py; the
#: tally falls back to the curated list alone if it is absent.
OFFICIAL_SNAPSHOT = PATHS.registries / "snapshots" / "stata_official" / "official.json"


def load_registry() -> tuple[Registry, frozenset[str]]:
    """Registries from the pinned snapshots, plus the SSC shipped-file set.

    Returns:
        The registry, and every command name any SSC package ships a file for.
    """
    snapshots = PATHS.registries / "snapshots"

    def names(registry: str) -> frozenset[str]:
        newest = max((snapshots / registry).glob("*/names.json"))
        return frozenset(json.loads(newest.read_text()))

    con = duckdb.connect()
    # A `FROM` clause takes no bound parameters, and the path is ours.
    index = snapshots / "ssc" / "stata_command_index.parquet"
    commands: dict[str, list[str]] = {}
    for command, package in con.execute(
        f"SELECT lower(command), package FROM '{index}' WHERE NOT is_helper"  # noqa: S608
    ).fetchall():
        commands.setdefault(command, []).append(package)
    shipped = frozenset(
        r[0].lower()
        for r in con.execute(
            f"SELECT DISTINCT command FROM '{index}'"  # noqa: S608
        ).fetchall()
    )
    # Package names, a different namespace from command names: `ssc install
    # blindschemes` names a package that exposes no command of that name.
    packages = frozenset(
        r[0].lower()
        for r in con.execute(
            f"SELECT DISTINCT package FROM '{index}'"  # noqa: S608
        ).fetchall()
    )
    lock = json.loads((PATHS.registries / "registries.lock.json").read_text())
    return (
        Registry(
            cran=names("cran"),
            cran_archive=names("cran_archive"),
            bioconductor=names("bioconductor"),
            pypi=names("pypi"),
            julia=names("julia_general"),
            stata_commands={k: tuple(v) for k, v in commands.items()},
            stata_builtins=builtins(verified_snapshot=OFFICIAL_SNAPSHOT).forms,
            ssc_packages=packages,
            lock_id=lock.get("cran", "")[:12],
        ),
        shipped,
    )

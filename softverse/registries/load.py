"""Assemble a :class:`~softverse.registries.resolve.Registry` from pinned snapshots.

The snapshots are written by :func:`softverse.registries.fetch.fetch_all` under
``registries/snapshots/``. Paths are anchored to the project root rather than
the working directory, so the registry loads the same from a script, a test or
a notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from softverse.model.enums import Ecosystem
from softverse.registries.lock import (
    LOCK,
    SNAPSHOTS,
    lock_id,
    pinned_directory,
    pinned_file,
    pinned_names,
    read_lock,
)
from softverse.registries.resolve import Registry
from softverse.stata.builtins import builtins

if TYPE_CHECKING:
    from pathlib import Path

INDEX_FILE = "stata_command_index.parquet"


#: Archive tiers of the Stata index, in the order resolution consults them.
_JOURNALS = {
    "stata_journal": Ecosystem.STATA_JOURNAL,
    "stb": Ecosystem.STB,
    "net": Ecosystem.NET_SITE,
}


def _commands(con: duckdb.DuckDBPyConnection, index: Path, where: str) -> dict:
    found: dict[str, list[str]] = {}
    # A `FROM` clause takes no bound parameters, and the path is ours.
    for command, package in con.execute(
        f"SELECT DISTINCT lower(command), package FROM '{index}' WHERE {where}"  # noqa: S608
    ).fetchall():
        found.setdefault(command, []).append(package)
    return {k: tuple(sorted(v)) for k, v in found.items()}


def _import_map(entries: frozenset[str]) -> dict[str, str]:
    # pipreqs lists some import names under several distributions (`google`
    # under a dozen). Those are left to the other rules: picking one would be
    # a guess recorded as a lookup.
    found: dict[str, set[str]] = {}
    for entry in entries:
        name, distribution = entry.split(":")
        found.setdefault(name, set()).add(distribution)
    return {name: next(iter(d)) for name, d in found.items() if len(d) == 1}


def load_registry() -> tuple[Registry, frozenset[str]]:
    """Registries from the pinned snapshots, plus the SSC shipped-file set.

    Returns:
        The registry, and every command name any SSC package ships a file for.
    """
    snapshots = SNAPSHOTS
    lock = read_lock(LOCK)

    def names(registry: str) -> frozenset[str]:
        return pinned_names(registry, lock, snapshots)

    con = duckdb.connect()
    index = (
        pinned_directory("stata_index", lock, snapshots, payload=INDEX_FILE)
        / INDEX_FILE
    )
    commands = _commands(con, index, "source = 'ssc' AND NOT is_helper")
    # A helper is not a command anyone types, but a script that calls
    # `_eststo` is using `estout` all the same. It resolves to its package
    # where no real command claims the name; discarding helpers outright
    # reported them as software in no archive.
    helpers = _commands(
        con, index, "source = 'ssc' AND is_helper AND evidence <> 'package_only'"
    )
    commands |= {k: v for k, v in helpers.items() if k not in commands}
    shipped = frozenset(
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT lower(command) FROM '{index}' "  # noqa: S608
            "WHERE source = 'ssc' AND evidence <> 'package_only'"
        ).fetchall()
    )
    # Package names, a different namespace from command names: `ssc install
    # blindschemes` names a package that exposes no command of that name.
    packages = frozenset(
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT lower(package) FROM '{index}' WHERE source = 'ssc'"  # noqa: S608
        ).fetchall()
    )
    journal_packages = {}
    for source, ecosystem in _JOURNALS.items():
        for (package,) in con.execute(
            f"SELECT DISTINCT package FROM '{index}' WHERE source = ?",  # noqa: S608
            [source],
        ).fetchall():
            journal_packages[package.lower()] = (package, ecosystem)
    return (
        Registry(
            cran=names("cran"),
            cran_archive=names("cran_archive"),
            bioconductor=names("bioconductor"),
            pypi=names("pypi"),
            julia=names("julia_general"),
            stata_commands=commands,
            # Curated and help-server-verified names, plus every `.ado` in
            # Stata's own base directory: the second finds the undocumented
            # commands the first two cannot.
            stata_builtins=builtins(
                verified_snapshot=pinned_file("stata_official", lock, snapshots)
            ).forms
            | names("stata_base_ado"),
            ssc_packages=packages,
            # Only what a journal package documents: see softverse.stata.index.
            stata_journal_commands=_commands(
                con, index, "source = 'stata_journal' AND is_documented"
            ),
            stb_commands=_commands(con, index, "source = 'stb' AND is_documented"),
            net_commands=_commands(con, index, "source = 'net' AND is_documented"),
            journal_packages=journal_packages,
            pypi_import_map=_import_map(names("pypi_import_map")),
            lock_id=lock_id(lock),
        ),
        shipped,
    )

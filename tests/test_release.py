"""Tests for the released Stata index bundle.

These run against the *exported* artifacts, not against in-memory objects, so
what ships is what was checked. A released dataset is the one artifact you
cannot quietly correct after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RELEASE = Path("build/release/stata-index")
pytestmark = pytest.mark.skipif(
    not (RELEASE / "stata_command_index.parquet").exists(),
    reason="release bundle not built; run scripts_release_stata_index.py",
)


@pytest.fixture(scope="module")
def index():
    import duckdb

    return duckdb.connect()


def packages_for(index, command: str) -> list[str]:
    q = (
        f"SELECT package FROM '{RELEASE / 'stata_command_index.parquet'}' "
        "WHERE lower(command)=? AND NOT is_helper"
    )
    return [r[0] for r in index.execute(q, [command]).fetchall()]


def test_the_readme_worked_example_actually_works(index):
    """A descriptor whose example does not run is worse than none."""
    assert packages_for(index, "esttab") == ["estout"]
    assert packages_for(index, "reghdfe") == ["reghdfe"]
    assert packages_for(index, "winsor2") == ["winsor2"]


def test_official_stata_is_absent_from_the_index():
    """`regress` must resolve as builtin, never as a package."""
    builtins = set(json.loads((RELEASE / "builtins.json").read_text())["forms"])
    assert "regress" in builtins
    assert "reghdfe" not in builtins, "a user package must not be marked builtin"


def test_helpers_are_flagged_not_exposed(index):
    """`f foo.ado` means a package ships a file, not that it exposes `foo`."""
    assert packages_for(index, "_eststo") == []
    assert packages_for(index, "reghdfe_p") == []


def test_ambiguity_is_preserved_not_resolved(index):
    """Forcing a winner would bury classification error."""
    ambiguous = json.loads((RELEASE / "ambiguous.json").read_text())
    assert len(ambiguous) > 100
    example = next(iter(ambiguous))
    assert len(ambiguous[example]) > 1
    assert len(packages_for(index, example)) > 1


def test_csv_and_parquet_agree(index):
    csv_rows = sum(1 for _ in open(RELEASE / "stata_command_index.csv")) - 1
    parquet_rows = index.execute(
        f"SELECT count(*) FROM '{RELEASE / 'stata_command_index.parquet'}'"
    ).fetchone()[0]
    assert csv_rows == parquet_rows


def test_the_caveats_are_in_the_descriptor():
    """The three things that change how the data should be used must be in the
    descriptor, not left for a reader to discover."""
    readme = (RELEASE / "README.md").read_text()
    assert "not necessarily a command" in readme
    assert "current snapshot, not a history" in readme
    assert "Ambiguity is preserved" in readme

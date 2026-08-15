"""What a person gets when they install this.

The entry point named `softverse.cli:cli` for months. No such module has ever
existed, so `pip install softverse` installed a `softverse` command that
raised `ModuleNotFoundError` on any invocation. Nothing caught it, because
every test imported the package directly and never looked at what the
distribution declares about itself.

These read the metadata rather than the source.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
PROJECT = PYPROJECT["project"]


def test_every_declared_entry_point_imports():
    """A console script that does not import is a crash on first use."""
    import importlib

    for script, target in PROJECT.get("scripts", {}).items():
        module, _, attr = target.partition(":")
        try:
            loaded = importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - the failure being guarded
            pytest.fail(
                f"entry point {script} names {module}, which does not import: {exc}"
            )
        assert hasattr(loaded, attr), f"{module} has no attribute {attr!r}"


def test_project_urls_are_declared_where_the_standard_looks():
    """`homepage = ...` under `[project]` is not PEP 621.

    The backend dropped those keys silently, so the built wheel carried no
    Project-URL at all and the PyPI page linked to nothing.
    """
    urls = PROJECT.get("urls", {})
    assert urls, "no [project.urls] table"
    assert "Repository" in urls and "Documentation" in urls

    stray = {"homepage", "repository", "documentation"} & set(PROJECT)
    assert not stray, f"non-standard keys under [project], silently dropped: {stray}"


def test_the_licence_is_declared_and_present():
    assert PROJECT.get("license") == "MIT"
    assert (ROOT / "LICENSE").exists(), "MIT is declared with no licence text"


def test_no_dependency_is_declared_without_being_imported():
    """Eight were, so every installer paid to download packages nothing used.

    Import names differ from distribution names often enough that this maps
    the ones that differ rather than guessing.
    """
    import ast

    alias = {
        "pyyaml": "yaml",
        "python-dotenv": "dotenv",
        "tomli-w": "tomli_w",
        "zenodo-client": "zenodo_client",
        "tree-sitter-language-pack": "tree_sitter_language_pack",
    }
    sources = list((ROOT / "softverse").rglob("*.py")) + list(ROOT.glob("scripts_*.py"))
    imported: set[str] = set()
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])

    unused = []
    for spec in PROJECT["dependencies"]:
        name = spec.split(">")[0].split("=")[0].split("[")[0].strip()
        if alias.get(name, name.replace("-", "_")) not in imported:
            unused.append(name)
    assert not unused, f"declared but never imported: {unused}"


def test_the_version_is_not_typed_in_two_places():
    """`__init__.py` carried its own copy, free to drift from pyproject.

    Checks for the release number specifically, not for any assignment to
    `__version__`: the fallback used when the package is not installed has to
    assign something, and an earlier version of this test failed on it.
    """
    source = (ROOT / "softverse" / "__init__.py").read_text()
    declared = PROJECT["version"]
    assert declared not in source, (
        f"{declared!r} is typed into __init__.py as well as pyproject.toml"
    )

    import softverse

    assert softverse.__version__ == declared


def test_the_extractor_version_matches_the_released_data():
    """It stamps every mention row, so code and released data must agree.

    Bumping it without rebuilding the tally leaves published counts claiming
    an extractor that never produced them.
    """
    import softverse

    mentions = ROOT / "build" / "tally" / "mentions.parquet"
    if not mentions.exists():
        pytest.skip("no tally built")

    import duckdb

    stamped = {
        row[0]
        for row in duckdb.connect()
        .execute(f"SELECT DISTINCT extractor_version FROM '{mentions}'")
        .fetchall()
    }
    assert stamped == {softverse.EXTRACTOR_VERSION}, (
        f"data says {stamped}, code says {softverse.EXTRACTOR_VERSION}"
    )


def test_entry_points_installed_in_this_environment_resolve():
    """Belt and braces: the metadata actually installed, not just the source."""
    import importlib

    for entry in entry_points(group="console_scripts"):
        if entry.module.split(".")[0] == "softverse":
            importlib.import_module(entry.module)

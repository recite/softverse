"""A registered R package's source shipped in a deposit is not the author's code.

147 deposits in the 2026 collection unpack a package tarball -- `Zelig_3.5.4`
among them -- whose `R/` files the V2 marker rule missed, because it looks for
`DESCRIPTION` beside the file and the code sits one level down. 8,648 files and
19,962 mentions were counted as the author's.

The rule is gated on the registry so a research compendium is not caught: an
author who packages their own analysis has a DESCRIPTION too, but its name is
not on CRAN. And a DESCRIPTION at the deposit root is never vendoring, even for
a CRAN name, because that is an author depositing their own package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from softverse.build.pipeline import CorpusFile, build
from softverse.registries.resolve import Registry

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def registry() -> Registry:
    return Registry(
        cran=frozenset({"MASS", "sandwich"}),
        cran_archive=frozenset({"Zelig"}),
        bioconductor=frozenset(),
        pypi=frozenset(),
        julia=frozenset(),
        stata_commands={},
        stata_builtins=frozenset(),
        lock_id="test",
    )


def _deposit(tmp_path: Path, files: dict[str, str]) -> list[CorpusFile]:
    out = []
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        out.append(
            CorpusFile(
                path=path,
                dataset_doi="doi:a",
                collection_id="ajps",
                source="dataverse",
                relative_path=relative,
                deposit_year=2012,
            )
        )
    return out


def _in_analysis(result) -> set[str]:
    return {r["relative_path"] for r in result.files if r["in_analysis_set"]}


def test_a_shipped_cran_package_tree_is_vendored(tmp_path, registry):
    corpus = _deposit(
        tmp_path,
        {
            "_archives/Zelig_3.5.4.tar.gz_extracted/Zelig/DESCRIPTION": (
                "Package: Zelig\nVersion: 3.5.4\n"
            ),
            "_archives/Zelig_3.5.4.tar.gz_extracted/Zelig/R/describe.R": (
                "library(MASS)\n"
            ),
            "analysis.R": "library(sandwich)\n",
        },
    )
    result = build(corpus, registry)
    assert _in_analysis(result) == {"analysis.R"}
    rules = {r["relative_path"]: r["vendor_rule"] for r in result.files}
    assert rules["_archives/Zelig_3.5.4.tar.gz_extracted/Zelig/R/describe.R"] == (
        "v2_r_package_tree"
    )
    assert {m["resolved_package"] for m in result.mentions} >= {"sandwich"}


def test_a_compendium_under_an_unregistered_name_is_the_authors(tmp_path, registry):
    corpus = _deposit(
        tmp_path,
        {
            "paper/DESCRIPTION": "Package: myreplication\nVersion: 0.1\n",
            "paper/R/models.R": "library(sandwich)\n",
        },
    )
    assert "paper/R/models.R" in _in_analysis(build(corpus, registry))


def test_a_cran_package_at_the_deposit_root_is_the_authors(tmp_path, registry):
    corpus = _deposit(
        tmp_path,
        {"DESCRIPTION": "Package: sandwich\nVersion: 3.0\n", "R/vcov.R": "x <- 1\n"},
    )
    assert "R/vcov.R" in _in_analysis(build(corpus, registry))

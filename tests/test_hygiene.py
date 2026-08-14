"""Tests for the vendored-code rules.

This module decides what counts as research code, and it had no tests. Its own
docstring says an exclusion you cannot explain is indistinguishable from a
bug; so is an exclusion nothing checks.

The rules were also written to R and Python conventions -- `renv/`,
`packrat/`, `site-packages/` -- and knew nothing of Stata's, which is the same
omission the paper this corpus feeds is about.
"""

from __future__ import annotations

import pytest

from softverse.corpus.hygiene import (
    classify,
    vendored_by_marker,
    vendored_by_name,
    vendored_by_path,
)
from softverse.model.enums import VendorRule


@pytest.mark.parametrize(
    "path",
    [
        "renv/library/dplyr/R/dplyr.R",
        "packrat/lib/x86_64/4.0/MASS/R/MASS.R",
        ".checkpoint/2020-04-22/lib/LearnBayes/demo/Chapter.1.2.R",
        "code/venv/lib/site-packages/numpy/core.py",
        "node_modules/thing/index.js",
    ],
)
def test_r_and_python_library_trees_are_vendored(path):
    assert vendored_by_path(path) is not None


@pytest.mark.parametrize(
    "path",
    [
        # Stata's own installation layout, shipped inside a deposit.
        "Replication/Codes/ado/base/_/_stubstar2names.ado",
        "Replication/Codes/ado/plus/e/estout.ado",
        "Replication/Codes/ado/personal/mycmd.ado",
        # And the bare form, which replication packages use so their code runs
        # without installing anything. Ten randomly sampled files under one
        # were ten third-party packages.
        "REPLICATION/code/ado/reghdfe.ado",
        "SSG/code/stata/ado/g/gegen.ado",
        "CA_r/ado/p/parallel_map.ado",
    ],
)
def test_stata_library_trees_are_vendored(path):
    """The gap: the markers covered R and Python and not Stata.

    12,231 of 124,828 corpus files sit under an `ado/` tree, contributing 14%
    of all mentions and 45% of the unresolved ones.
    """
    assert vendored_by_path(path) is VendorRule.V1_STATA_ADO


@pytest.mark.parametrize(
    "path",
    [
        "code/analysis.do",
        "Codes/01_clean.R",
        "src/model.py",
        # `ado` has to be a path component, not a substring: this is a
        # directory whose name merely contains the letters.
        "code/adopted_methods/run.do",
        "avocado/analysis.do",
    ],
)
def test_research_code_is_not_vendored(path):
    assert vendored_by_path(path) is None


def test_an_ssc_shipped_basename_is_vendored():
    """A `.ado` whose basename a published package ships is a copy of it."""
    assert (
        vendored_by_name("code/ado/reghdfe.ado", frozenset({"reghdfe"}))
        is VendorRule.V4_NAME_SHAPE
    )
    assert vendored_by_name("code/myhelper.ado", frozenset({"reghdfe"})) is None


def test_a_package_source_tree_is_vendored_without_a_conventional_name():
    siblings = {"vendored/thing/DESCRIPTION", "vendored/thing/R/code.R"}
    assert (
        vendored_by_marker("vendored/thing/R/code.R", siblings) is None
    ), "DESCRIPTION is a sibling of R/, not of R/code.R"
    assert vendored_by_marker("vendored/thing/code.R", siblings) is VendorRule.V2_MARKER


def test_the_rule_that_fired_is_recorded():
    """Auditability is the point: every exclusion must name its reason."""
    verdict = classify("renv/library/dplyr/R/dplyr.R", "abc")
    assert verdict.is_vendored
    assert verdict.rule is VendorRule.V1_PATH

    stata = classify("code/ado/reghdfe.ado", "abc")
    assert stata.is_vendored
    assert stata.rule is VendorRule.V1_STATA_ADO


def test_a_file_seen_in_five_deposits_is_not_bespoke():
    verdict = classify(
        "code/helper.do", "shared-hash", shared_hashes=frozenset({"shared-hash"})
    )
    assert verdict.is_vendored
    assert verdict.rule is VendorRule.V3_CROSS_DATASET


def test_ordinary_research_code_survives_every_rule():
    verdict = classify("code/01_analysis.do", "unique-hash")
    assert not verdict.is_vendored
    assert verdict.in_analysis_set

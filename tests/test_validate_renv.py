"""The renv comparison sets aside only what it says it does, and counts it."""

from __future__ import annotations

from validate_renv import BASE, IMPLICIT, OUTSIDE, align, jaccard

ANALYSIS = {"/d/main.R", "/d/report.Rmd"}


def test_aligned_names_match_once_the_by_design_differences_are_out():
    renv = [
        ("fixest", "/d/main.R"),
        ("grid", "/d/main.R"),
        ("rmarkdown", "/d/report.Rmd"),
        ("AER", "/d/.checkpoint/lib/AER/demo/a.R"),
        ("ggplot2", "/d/report.Rmd"),
    ]
    mine, theirs, set_aside = align(
        ours_all={"fixest", "ggplot2", "AER"},
        ours_analysis={"fixest", "ggplot2"},
        renv=renv,
        analysis_files=ANALYSIS,
    )
    assert jaccard(mine, theirs) == 1.0
    assert set_aside == {
        OUTSIDE: 1,
        BASE: 1,
        IMPLICIT: 1,
        "ours_" + OUTSIDE: 1,
    }


def test_rmarkdown_a_script_names_is_still_compared():
    # Found by us too, so it is a reference, not renv's implicit addition.
    mine, theirs, set_aside = align(
        {"rmarkdown"}, {"rmarkdown"}, [("rmarkdown", "/d/report.Rmd")], ANALYSIS
    )
    assert theirs == {"rmarkdown"}
    assert not set_aside[IMPLICIT]
    # Named in a plain script, where renv adds nothing implicitly: a real miss.
    mine, theirs, _ = align(set(), set(), [("rmarkdown", "/d/main.R")], ANALYSIS)
    assert theirs == {"rmarkdown"}
    assert jaccard(mine, theirs) == 0.0


def test_a_package_renv_finds_in_analysis_files_and_we_miss_is_a_disagreement():
    mine, theirs, _ = align(set(), set(), [("sandwich", "/d/main.R")], ANALYSIS)
    assert jaccard(mine, theirs) == 0.0

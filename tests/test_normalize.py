"""`normalized_name` has to mean normalized.

It was `raw_name.lower()` for every language. Lowercasing is the correct
normalization for exactly one of the four -- Stata -- and for the other three
it is either not enough or not right.
"""

from __future__ import annotations

import pytest

from softverse.model.enums import Language
from softverse.registries.resolve import normalize


@pytest.mark.parametrize(
    "raw,expected",
    [
        # PEP 503: lowercase *and* collapse runs of -_. to a single hyphen.
        # Lowercasing alone left `scikit_learn` and `scikit-learn` as two
        # different keys for one distribution, so anything grouping by this
        # column counted the same package twice.
        ("scikit_learn", "scikit-learn"),
        ("scikit-learn", "scikit-learn"),
        ("Scikit.Learn", "scikit-learn"),
        ("zope..interface", "zope-interface"),
        ("NumPy", "numpy"),
    ],
)
def test_python_names_get_pep503_normalization(raw, expected):
    assert normalize(raw, Language.PYTHON) == expected


@pytest.mark.parametrize("name", ["MASS", "Hmisc", "RColorBrewer", "ggplot2", "Rcpp"])
def test_r_names_keep_their_case(name):
    """R package names are case-sensitive and have no normal form.

    `MASS` is the package; `mass` is not. Lowercasing was safe only by luck --
    no two of CRAN's 24,719 names collide under it -- but it made the column
    useless for R: you cannot look a value up on CRAN without first undoing
    it, and `resolved_package` already carries the canonical spelling.
    """
    assert normalize(name, Language.R) == name


def test_stata_commands_are_lowercased():
    """The one language the old rule was right about."""
    assert normalize("REGHDFE", Language.STATA) == "reghdfe"


def test_julia_names_keep_their_case():
    assert normalize("DataFrames", Language.JULIA) == "DataFrames"


def test_normalizing_is_idempotent():
    """A second pass must not change the answer, or the key is not a key."""
    for language, name in (
        (Language.PYTHON, "Scikit_Learn"),
        (Language.R, "MASS"),
        (Language.STATA, "REGHDFE"),
        (Language.JULIA, "DataFrames"),
    ):
        once = normalize(name, language)
        assert normalize(once, language) == once

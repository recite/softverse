"""Tests for the aggregation, which defines the headline metric.

v1 had no tests here at all, and its aggregation dropped 71,069 of 71,069 rows.
"""

from __future__ import annotations

from softverse.build.pipeline import dataset_packages, language_presence
from softverse.model.enums import Construct, Resolution


def mention(**kw):
    base = {
        "dataset_doi": "doi:a",
        "source": "dataverse_legacy",
        "collection_id": "ajps",
        "deposit_year": 2020,
        "language": "python",
        "construct": str(Construct.IMPORT),
        "resolved_package": "pandas",
        "resolution": str(Resolution.KNOWN_CURRENT),
        "ecosystem": "pypi",
        "file_uid": "f1",
    }
    return {**base, **kw}


def test_stdlib_is_excluded_from_the_headline_tally():
    """The v1 defect, reproduced here on the first run and now fixed.

    The resolver gives stdlib mentions a package name -- `os` resolves to `os`
    -- so filtering on "has a resolved package" lets every one through. v1
    published os 101, sys 66, grid 268, parallel 115 for exactly this reason,
    and this pipeline reproduced it (os 52, sys 42) until the filter keyed on
    the resolution rather than the name.
    """
    rows = dataset_packages(
        [
            mention(resolved_package="os", resolution=str(Resolution.BASE_OR_STDLIB)),
            mention(resolved_package="pandas"),
        ]
    )
    assert [r["package"] for r in rows] == ["pandas"]


def test_stata_builtins_are_excluded():
    rows = dataset_packages(
        [
            mention(
                language="stata",
                resolved_package=None,
                resolution=str(Resolution.BUILTIN),
            ),
            mention(language="stata", resolved_package="reghdfe"),
        ]
    )
    assert [r["package"] for r in rows] == ["reghdfe"]


def test_archived_packages_still_count():
    """rgdal and Zelig were real when the code was written."""
    rows = dataset_packages(
        [mention(resolved_package="rgdal", resolution=str(Resolution.KNOWN_ARCHIVED))]
    )
    assert [r["package"] for r in rows] == ["rgdal"]


def test_provisioning_is_not_use():
    """`install.packages` is intent, not evidence the analysis ran on it."""
    installs = [mention(construct=str(Construct.INSTALL), resolved_package="brms")]
    assert dataset_packages(installs, use_only=True) == []
    assert len(dataset_packages(installs, use_only=False)) == 1


def test_a_package_counts_once_per_deposit():
    """The headline unit. Ten mentions in one deposit is still one deposit."""
    rows = dataset_packages([mention(file_uid=f"f{i}") for i in range(10)])
    assert len(rows) == 1
    assert rows[0]["n_mentions"] == 10
    assert rows[0]["n_files"] == 10


def test_the_same_package_in_two_deposits_is_two_rows():
    rows = dataset_packages([mention(), mention(dataset_doi="doi:b")])
    assert len(rows) == 2


def test_unresolved_names_never_enter_the_tally():
    assert (
        dataset_packages(
            [mention(resolved_package=None, resolution=str(Resolution.UNKNOWN))]
        )
        == []
    )


def test_language_presence_is_claimable_even_without_package_attribution():
    """MATLAB has no resolvable registry, but 'N deposits contain MATLAB' is a
    verifiable claim and worth reporting."""
    presence = language_presence(
        [
            {"dataset_doi": "doi:a", "source": "zenodo", "language": "matlab"},
            {"dataset_doi": "doi:b", "source": "zenodo", "language": "matlab"},
            {"dataset_doi": "doi:a", "source": "zenodo", "language": "r"},
        ]
    )
    assert presence == {("zenodo", "matlab"): 2, ("zenodo", "r"): 1}

"""Package pages and badges, on a tally small enough to read."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_packages
from build_lookup import slug


@pytest.fixture
def tally(tmp_path, monkeypatch):
    tally = tmp_path / "tally"
    tally.mkdir()
    pd.DataFrame(
        [
            {
                "package": "fixest",
                "language": "r",
                "ecosystem": "cran",
                "n_deposits": 2,
                "n_files": 3,
                "n_mentions": 9,
                "first_year": 2020,
                "last_year": 2021,
                "n_deposits_dataverse": 1,
                "n_deposits_zenodo": 1,
                "n_deposits_at_risk": 10,
                "share_of_deposits": 0.2,
            },
            {
                "package": "arrow",
                "language": "r",
                "ecosystem": "cran",
                "n_deposits": 1,
                "n_files": 1,
                "n_mentions": 1,
                "first_year": 2021,
                "last_year": 2021,
                "n_deposits_dataverse": 1,
                "n_deposits_zenodo": 0,
                "n_deposits_at_risk": 10,
                "share_of_deposits": 0.1,
            },
            {
                "package": "arrow",
                "language": "python",
                "ecosystem": "pypi",
                "n_deposits": 1,
                "n_files": 1,
                "n_mentions": 2,
                "first_year": 2022,
                "last_year": 2022,
                "n_deposits_dataverse": 0,
                "n_deposits_zenodo": 1,
                "n_deposits_at_risk": 4,
                "share_of_deposits": 0.25,
            },
        ]
    ).to_csv(tally / "usage_by_package.csv", index=False)
    pd.DataFrame(
        [
            {
                "language": "r",
                "package": "fixest",
                "ecosystem": "cran",
                "dataset_doi": "doi:10.7910/DVN/AAA",
                "source": "dataverse",
                "collection_id": "ajps",
                "year": 2020,
            },
            {
                "language": "r",
                "package": "fixest",
                "ecosystem": "cran",
                "dataset_doi": "zenodo:123",
                "source": "zenodo",
                "collection_id": "restud-replication",
                "year": 2021,
            },
            {
                "language": "r",
                "package": "arrow",
                "ecosystem": "cran",
                "dataset_doi": "doi:10.7910/DVN/AAA",
                "source": "dataverse",
                "collection_id": "ajps",
                "year": 2021,
            },
            {
                "language": "python",
                "package": "arrow",
                "ecosystem": "pypi",
                "dataset_doi": "zenodo:9",
                "source": "zenodo",
                "collection_id": "restud-replication",
                "year": 2022,
            },
        ]
    ).to_parquet(tally / "package_deposits.parquet")
    pd.DataFrame(
        [
            {
                "source": "zenodo",
                "language": "r",
                "package": "fixest",
                "function": "feols",
                "n_calls": 5,
                "n_deposits": 1,
            }
        ]
    ).to_parquet(tally / "usage_by_function.parquet")
    pd.DataFrame(
        [
            {
                "ecosystem": "cran",
                "package": "fixest",
                "version": "0.11.1",
                "version_source": "renv_lock",
                "n_deposits": 1,
            }
        ]
    ).to_csv(tally / "package_versions.csv", index=False)
    (tally / "summary.json").write_text(json.dumps({"built": "2026-09-15"}))
    frame = tmp_path / "frame.csv"
    frame.write_text(
        "collection_id,journal_name\najps,American Journal of Political Science\n"
        "restud-replication,Review of Economic Studies\n"
    )
    monkeypatch.setattr(build_packages, "TALLY", tally)
    monkeypatch.setattr(build_packages, "FRAME", frame)
    return tmp_path / "site"


def test_every_package_gets_a_page_badge_and_record(tally):
    assert build_packages.main(out=tally) == 0
    page = (tally / "p" / "cran" / "fixest" / "index.html").read_text()
    assert "2 papers" in page
    assert "https://doi.org/10.7910/DVN/AAA" in page
    assert "https://zenodo.org/records/123" in page
    assert "feols" in page
    assert "0.11.1" in page
    assert (tally / "badges" / "cran" / "fixest.svg").read_text().startswith("<?xml")


def test_badge_is_a_valid_shields_endpoint(tally):
    build_packages.main(out=tally)
    badge = json.loads((tally / "badges" / "cran" / "fixest.json").read_text())
    assert badge["schemaVersion"] == 1
    assert badge["label"] == "replication code"
    assert badge["message"] == "2 papers"


def test_same_name_in_two_ecosystems_gets_two_pages(tally):
    build_packages.main(out=tally)
    r = json.loads((tally / "api" / "v1" / "cran" / "arrow.json").read_text())
    py = json.loads((tally / "api" / "v1" / "pypi" / "arrow.json").read_text())
    assert (r["language"], r["n_deposits"]) == ("r", 1)
    assert (py["language"], py["deposits"][0]["doi"]) == ("python", "zenodo:9")
    one = json.loads((tally / "badges" / "pypi" / "arrow.json").read_text())
    assert one["message"] == "1 paper"


def test_pypi_names_are_normalized_cran_names_keep_case():
    assert slug("pypi", "Scikit_Learn") == ("pypi", "scikit-learn")
    assert slug("cran", "Matrix") == ("cran", "Matrix")
    assert slug("cran_archive", "Zelig") == ("cran", "Zelig")


def test_a_case_only_collision_fails_the_build():
    with pytest.raises(ValueError, match="share a page path"):
        build_packages._refuse_collisions(
            [
                {"path": "cran/Rcpp", "ecosystem": "cran", "package": "Rcpp"},
                {"path": "cran/rcpp", "ecosystem": "cran", "package": "rcpp"},
            ]
        )

"""The SQL aggregation on a world small enough to count by hand.

Two deposits. The first loads fixest twice in one file and once in another,
installs it once (not a use), and declares an R version. The second is a
Python deposit whose only Python lives in a notebook cell, which is the case
that once put pandas in the numerator of every Python package and in the
denominator of none.
"""

from __future__ import annotations

import csv
import json

import pytest

from softverse.build.aggregate import aggregate
from softverse.model.io import write_table

A, B = "doi:10.7910/DVN/AAAAAA", "zenodo:1"


def _file(uid, doi, source, language, extension, analyzable=True):
    return {
        "file_uid": uid,
        "dataset_version_uid": doi,
        "dataset_doi": doi,
        "source": source,
        "collection_id": "c",
        "relative_path": f"{uid}{extension}",
        "filename": f"{uid}{extension}",
        "extension": extension,
        "language": language,
        "is_vendored": False,
        "parse_status": "ok",
        "n_mentions": 0,
        "in_analysis_set": analyzable,
    }


def _mention(uid, file_uid, doi, source, year, language, name, **over):
    row = {
        "mention_uid": uid,
        "file_uid": file_uid,
        "dataset_doi": doi,
        "source": source,
        "collection_id": "c",
        "deposit_year": year,
        "language": language,
        "construct": "library",
        "raw_name": name,
        "resolved_package": name,
        "ecosystem": "cran",
        "resolution": "known_current",
        "line": 1,
        "snippet": name,
        "is_dynamic": False,
        "extractor_version": "t",
        "registry_lock_id": "t",
    }
    row.update(over)
    return row


@pytest.fixture
def tally(tmp_path):
    files = [
        _file("a1", A, "dataverse", "r", ".R"),
        _file("a2", A, "dataverse", "r", ".R"),
        _file("a3", A, "dataverse", "stata", ".do", analyzable=False),
        _file("b1", B, "zenodo", "notebook", ".ipynb"),
    ]
    mentions = [
        _mention(
            "m1", "a1", A, "dataverse", 2021, "r", "fixest", called_function="feols"
        ),
        _mention(
            "m2", "a1", A, "dataverse", 2021, "r", "fixest", called_function="feols"
        ),
        _mention("m3", "a2", A, "dataverse", 2021, "r", "fixest"),
        _mention("m4", "a2", A, "dataverse", 2021, "r", "fixest", construct="install"),
        _mention(
            "m5",
            "a2",
            A,
            "dataverse",
            2021,
            "r",
            "mystery",
            resolved_package=None,
            ecosystem=None,
            resolution="unknown",
        ),
        _mention("m6", "b1", B, "zenodo", 2024, "python", "pandas", ecosystem="pypi"),
        # An install line for something no registry lists is not a *use* of
        # unregistered software, and must stay out of `unknown_names`.
        _mention(
            "m7",
            "a2",
            A,
            "dataverse",
            2021,
            "r",
            "ghpkg",
            construct="install",
            remote="github.com/someone/ghpkg",
            resolved_package=None,
            ecosystem=None,
            resolution="unknown",
        ),
        _mention("m8", "a2", A, "dataverse", 2021, "r", "ghpkg", ecosystem="github"),
    ]
    write_table(files, "files", tmp_path)
    write_table(mentions, "mentions", tmp_path)
    write_table(
        [
            {
                "dataset_doi": A,
                "source_file_uid": "a1",
                "manifest_kind": "renv_lock",
                "signal": "r_version",
                "value": "4.2",
            }
        ],
        "environment_signals",
        tmp_path,
    )
    return tmp_path


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_counts_deposits_not_mentions_and_carries_the_denominator(tally):
    out = tally / "out"
    aggregate(
        str(tally / "files.parquet"),
        str(tally / "mentions.parquet"),
        str(tally / "environment_signals.parquet"),
        out,
    )
    usage = {r["package"]: r for r in _rows(out / "usage_by_package.csv")}
    fixest = usage["fixest"]
    assert fixest["n_deposits"] == "1", "three loads in one deposit is one user"
    assert fixest["n_files"] == "2"
    assert fixest["n_mentions"] == "3", "the install is intent, not use"
    assert fixest["n_deposits_at_risk"] == "1"
    assert fixest["share_of_deposits"] == "1.0"
    assert fixest["n_at_risk_dataverse"] == "1"
    assert fixest["n_at_risk_zenodo"] == "0"
    # The notebook deposit is in Python's denominator because it yielded a
    # Python mention, even though no file of language `python` exists.
    assert usage["pandas"]["n_deposits_at_risk"] == "1"
    assert "mystery" not in usage

    functions = _rows(out / "usage_by_function.csv")
    assert functions == [
        {
            "source": "dataverse",
            "language": "r",
            "package": "fixest",
            "function": "feols",
            "n_calls": "2",
            "n_deposits": "1",
        }
    ]
    unknown = _rows(out / "unknown_names.csv")
    assert unknown == [
        {
            "name": "mystery",
            "language": "r",
            "n_deposits": "1",
            "n_mentions": "1",
            "n_deposits_defining": "0",
        }
    ]
    assert _rows(out / "remote_installs.csv") == [
        {
            "name": "ghpkg",
            "language": "r",
            "host": "github.com",
            "in_registry": "False",
            "n_deposits_installing": "1",
            "n_deposits_loading": "1",
        }
    ]
    presence = {
        (r["source"], r["language"]): r["n_deposits"]
        for r in _rows(out / "language_presence.csv")
    }
    assert presence[("dataverse", "stata")] == "1", "presence counts every file"
    coverage = json.loads((out / "environment_coverage.json").read_text())
    assert coverage["r_version"] == {"deposits_carrying": 1, "deposits_eligible": 2}
    assert coverage["stata_version"]["deposits_eligible"] == 1, "notebooks count"
    assert coverage["python_version"]["deposits_eligible"] == 1

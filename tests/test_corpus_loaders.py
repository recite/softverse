"""Tests for corpus assembly across sources.

The tally pools two repositories into one set of counts. What makes that
sound is that every file goes through one `build()`, and what makes it
checkable is that each row remembers which corpus it came from. These test
the two ways that can silently go wrong: a deposit counted under two sources,
and a source column that does not survive the trip to the aggregates.
"""

from __future__ import annotations

import csv

import pytest

from softverse.build.pipeline import CorpusFile, dataset_packages, language_presence
from softverse.corpus import loaders


def corpus_file(doi: str, source: str, path_name: str = "a.do") -> CorpusFile:
    return CorpusFile(
        path=loaders.PATHS.root / path_name,
        dataset_doi=doi,
        collection_id="c",
        source=source,
        relative_path=path_name,
    )


def test_disjointness_passes_for_distinct_namespaces():
    files = [
        corpus_file("zenodo:1", "zenodo"),
        corpus_file("doi:10.7910/DVN/AAA", "dataverse_legacy"),
    ]
    loaders._assert_disjoint(files)


def test_a_deposit_in_two_sources_is_refused():
    """It would inflate the pooled numerator and leave the denominator alone,
    and no downstream check looks at deposit identity across sources."""
    files = [
        corpus_file("zenodo:1", "zenodo"),
        corpus_file("zenodo:1", "dataverse_legacy"),
    ]
    with pytest.raises(ValueError, match="both"):
        loaders._assert_disjoint(files)


def mention(doi: str, source: str, package: str, language: str = "stata") -> dict:
    return {
        "dataset_doi": doi,
        "source": source,
        "collection_id": "c",
        "deposit_year": 2020,
        "language": language,
        "resolved_package": package,
        "ecosystem": "ssc",
        "resolution": "known_current",
        "construct": "stata_command",
        "file_uid": f"{doi}-{package}",
    }


def test_dataset_packages_carries_the_source():
    """Without it the tally cannot split a pooled count, and the split is the
    only thing that lets a reader see the composition."""
    rows = dataset_packages(
        [
            mention("zenodo:1", "zenodo", "estout"),
            mention("doi:10.7910/DVN/A", "dataverse_legacy", "estout"),
        ]
    )
    assert {r["source"] for r in rows} == {"zenodo", "dataverse_legacy"}


def test_language_presence_is_reported_per_source():
    """The two corpora do not carry the same file types, so one pooled
    language share would read as a fact about social science when part of it
    is a fact about what a 2024 scrape happened to collect."""
    files = [
        {"dataset_doi": "zenodo:1", "source": "zenodo", "language": "stata"},
        {"dataset_doi": "zenodo:2", "source": "zenodo", "language": "r"},
        {"dataset_doi": "doi:A", "source": "dataverse_legacy", "language": "stata"},
    ]
    assert language_presence(files) == {
        ("dataverse_legacy", "stata"): 1,
        ("zenodo", "r"): 1,
        ("zenodo", "stata"): 1,
    }


@pytest.mark.skipif(
    not (loaders.ZENODO_ROOT / "deposits.csv").exists(),
    reason="needs a collected Zenodo corpus",
)
def test_zenodo_deposits_carry_a_community_and_a_year():
    """Both were computed at harvest and thrown away for months, leaving one
    collection id across the whole corpus and a null year on every mention.
    This fails against the old ledger-only path and passes against the file.
    """
    with open(loaders.ZENODO_ROOT / "deposits.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len({r["collection_id"] for r in rows}) > 1
    dated = sum(1 for r in rows if r["deposit_year"].isdigit())
    assert dated == len(rows)

"""Validated use set beside download counts, from pinned snapshots."""

from __future__ import annotations

import csv
import hashlib
import json

import pandas as pd

from softverse.build.downloads import (
    agreement,
    deciles,
    downloads_vs_use,
    read,
    reverse_dependencies,
    spearman,
)
from softverse.registries.fetch import parse_cran_packages
from softverse.registries.lock import read_lock, write_lock


def _counts(root, registry, counts):
    directory = root / registry / "2026-09-17"
    directory.mkdir(parents=True)
    payload = json.dumps(counts, sort_keys=True).encode()
    (directory / "downloads.json").write_bytes(payload)
    (directory / "source.json").write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()})
    )


def _names(root, registry, names):
    directory = root / registry / "2026-09-17"
    directory.mkdir(parents=True)
    (directory / "names.json").write_text(json.dumps(names))
    (directory / "source.json").write_text(json.dumps({"sha256": registry}))


def _world(tmp_path):
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _counts(
        snapshots, "downloads_ssc", {"estout": 900.0, "reghdfe": 500.0, "unused": 5.0}
    )
    _counts(snapshots, "downloads_cran", {"vctrs": 9e6, "stargazer": 1e5, "AER": 2e5})
    _counts(snapshots, "downloads_pypi", {"scikit-learn": 5e7})
    _names(
        snapshots,
        "cran_dependencies",
        ["stargazer:", "AER:vctrs", "vctrs:", "dplyr:vctrs"],
    )
    _names(snapshots, "cran_task_views", ["Econometrics:AER"])
    write_lock(snapshots, lock_path)
    usage = tmp_path / "usage_by_package.csv"
    pd.DataFrame(
        [
            ("estout", "stata", "ssc", 30),
            ("REGHDFE", "stata", "ssc", 10),
            ("grc1leg", "stata", "net_site", 5),
            ("stargazer", "r", "cran", 20),
            ("AER", "r", "cran", 3),
            ("scikit_learn", "python", "pypi", 8),
            ("pyblp", "python", "pypi", 2),
        ],
        columns=["package", "language", "ecosystem", "n_deposits"],
    ).to_csv(usage, index=False)
    n = downloads_vs_use(usage, tmp_path, read_lock(lock_path), snapshots)
    with (tmp_path / "downloads_vs_use.csv").open(encoding="utf-8") as handle:
        return n, {r["package"]: r for r in csv.DictReader(handle)}, tmp_path


def test_both_sides_of_the_join_are_kept(tmp_path):
    """Used packages only was a table silently conditional on use."""
    n, rows, _ = _world(tmp_path)
    assert n == len(rows) == 8, "grc1leg is on no primary registry, so it has no row"
    # Counted by the registry and never used: present, at zero.
    assert rows["unused"]["n_deposits"] == "0"
    assert rows["vctrs"]["n_deposits"] == "0"
    # Used and not counted: present, with no downloads.
    assert rows["pyblp"]["downloads"] == ""
    # Names are matched as each registry spells them.
    assert rows["REGHDFE"]["downloads"] == "500.0"
    assert rows["scikit_learn"]["downloads"] == "50000000.0"


def test_cran_rows_carry_what_separates_the_two_reasons_to_disagree(tmp_path):
    _, rows, _ = _world(tmp_path)
    assert rows["vctrs"]["reverse_dependencies"] == "2", "AER and dplyr install it"
    assert rows["stargazer"]["reverse_dependencies"] == "0"
    assert rows["AER"]["in_task_view"] == "True"
    assert rows["stargazer"]["in_task_view"] == "False"
    assert rows["estout"]["reverse_dependencies"] == ""


def test_agreement_names_its_population(tmp_path):
    _, _, out = _world(tmp_path)
    found = agreement(read(out / "downloads_vs_use.csv"))
    assert found["ssc"]["used"]["n"] == 2
    assert found["ssc"]["registry"]["n"] == 3
    assert round(found["ssc"]["registry"]["ever_used"], 3) == 0.667
    assert found["cran"]["registry"]["top100_used"] == 2


def test_reverse_dependencies_are_transitive_and_survive_a_cycle():
    graph = {"a": ["b"], "b": ["c"], "c": ["a"], "d": ["c"], "e": []}
    counts = reverse_dependencies(graph)
    # Installing a, b or d pulls in c; the cycle is walked once.
    assert counts["c"] == 3
    assert counts["a"] == 3, "b needs c needs a; d reaches a through c"
    assert counts["e"] == 0
    assert counts["d"] == 0


def test_deciles_are_shares_used_from_the_least_downloaded_up():
    rows = [{"downloads": float(i), "n_deposits": int(i >= 15)} for i in range(20)]
    assert deciles(rows) == [0, 0, 0, 0, 0, 0, 0, 0.5, 1, 1]


def test_the_cran_index_is_read_across_continuation_lines():
    text = (
        "Package: AER\nDepends: R (>= 3.0.0), car (>= 2.0-19), lmtest,\n"
        "        sandwich (>= 2.4-0)\nImports: stats, Formula (>= 0.2-0)\n"
        "Suggests: boot\n\nPackage: vctrs\nImports: cli\nLinkingTo: cpp11\n"
    )
    assert parse_cran_packages(text) == {
        "AER": ["Formula", "car", "lmtest", "sandwich", "stats"],
        "vctrs": ["cli", "cpp11"],
    }


def test_spearman_is_pearson_on_ranks_with_ties_averaged():
    assert spearman([(1, 10), (2, 20), (3, 30)]) == 1.0
    assert spearman([(1, 30), (2, 20), (3, 10)]) == -1.0
    # By hand: x ranks 1.5, 1.5, 3 and y ranks 1, 2, 3 give 0.866.
    assert round(spearman([(5, 1), (5, 2), (9, 3)]), 3) == 0.866


def test_a_count_fetched_because_we_use_the_package_is_not_the_registry(tmp_path):
    """PyPI's bottom decile looked well used: the extras were used by construction."""
    snapshots, lock_path = tmp_path / "snapshots", tmp_path / "lock.json"
    _counts(snapshots, "downloads_ssc", {})
    _counts(snapshots, "downloads_cran", {})
    _counts(snapshots, "downloads_pypi", {"boto3": 9e9, "pandas": 5e8, "pyblp": 40.0})
    source = snapshots / "downloads_pypi" / "2026-09-17" / "source.json"
    source.write_text(
        json.dumps({**json.loads(source.read_text()), "n_from_top_list": 2})
    )
    _names(snapshots, "cran_dependencies", [])
    _names(snapshots, "cran_task_views", [])
    write_lock(snapshots, lock_path)
    usage = tmp_path / "usage_by_package.csv"
    pd.DataFrame(
        [("pandas", "python", "pypi", 9), ("pyblp", "python", "pypi", 2)],
        columns=["package", "language", "ecosystem", "n_deposits"],
    ).to_csv(usage, index=False)
    downloads_vs_use(usage, tmp_path, read_lock(lock_path), snapshots)
    rows = {r["package"]: r for r in read(tmp_path / "downloads_vs_use.csv")}
    assert rows["boto3"]["in_registry_counts"]
    assert rows["pandas"]["in_registry_counts"]
    assert not rows["pyblp"]["in_registry_counts"]
    found = agreement(list(rows.values()))["pypi"]
    assert found["used"]["n"] == 2, "pyblp is still a used package with a count"
    assert found["registry"]["n"] == 2, "and is not part of the registry's population"

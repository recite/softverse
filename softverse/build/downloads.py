"""Set each package's validated use beside its registry's download count.

One row per package that a primary registry counts downloads for *or* that
the corpus uses, carrying both counts. Both sides, because the comparison
means different things from each. Starting from the packages the corpus uses
asks how well downloads order software social science has chosen; starting
from everything the registry counts asks whether downloads say which software
it chooses at all. The first version kept only the used side, and every
number drawn from it was silently conditional on use.

Ranks are what get compared, because the registries count over different
windows and a rank within one registry does not care.

For CRAN two more facts ride along, because a download count and a use count
can part for two reasons and only one of them is about downloads being cheap.
A package may be downloaded by people who are not social scientists, which
the task views hold roughly fixed; or it may be downloaded by other packages
and by build servers, which its reverse dependencies measure.
"""

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING

from softverse.build.aggregate import write_csv
from softverse.registries.downloads import FILE
from softverse.registries.lock import SNAPSHOTS, Pin, pinned_directory, pinned_names
from softverse.registries.resolve import normalize_pypi

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Ecosystem -> its language, the snapshot holding its counts, and how it
#: spells a name.
SOURCES: dict[str, tuple[str, str, Callable[[str], str]]] = {
    "ssc": ("stata", "downloads_ssc", str.lower),
    "cran": ("r", "downloads_cran", str),
    "pypi": ("python", "downloads_pypi", normalize_pypi),
}


def reverse_dependencies(graph: dict[str, list[str]]) -> dict[str, int]:
    """How many packages install each package, directly or through others.

    Args:
        graph: Package -> the packages it depends on.

    Returns:
        Package -> the number of distinct packages whose installation pulls it
        in. Walked from each package outward over the reversed graph; a cycle
        is visited once, so it cannot loop.
    """
    dependents: dict[str, set[str]] = {}
    for package, needs in graph.items():
        for need in needs:
            dependents.setdefault(need, set()).add(package)
    counts = {}
    for package in set(graph) | set(dependents):
        seen: set[str] = set()
        frontier = [package]
        while frontier:
            for dependent in dependents.get(frontier.pop(), ()):
                if dependent not in seen and dependent != package:
                    seen.add(dependent)
                    frontier.append(dependent)
        counts[package] = len(seen)
    return counts


def downloads_vs_use(
    usage: Path, out: Path, lock: dict[str, Pin], snapshots: Path = SNAPSHOTS
) -> int:
    """Write `downloads_vs_use.csv` (and Parquet) into ``out``.

    Args:
        usage: `usage_by_package.csv`.
        out: The tally directory.
        lock: The registry lock, from `read_lock`.
        snapshots: The snapshot root.

    Returns:
        The number of rows written.
    """
    with usage.open(encoding="utf-8") as handle:
        used_rows = list(csv.DictReader(handle))

    graph = {
        package: [d for d in needs.split(",") if d]
        for package, _, needs in (
            entry.partition(":") for entry in pinned_names("cran_dependencies", lock, snapshots)
        )
    }
    depended_on = reverse_dependencies(graph)
    in_view = {
        entry.partition(":")[2] for entry in pinned_names("cran_task_views", lock, snapshots)
    }

    rows = []
    for ecosystem, (language, registry, spell) in SOURCES.items():
        directory = pinned_directory(registry, lock, snapshots, payload=FILE)
        counts: dict[str, float] = json.loads((directory / FILE).read_text())
        # PyPI's counts are its most downloaded projects plus the packages the
        # corpus uses, looked up one at a time. Those extras are used by
        # construction, so counting them as part of "the registry" put a bump
        # of used packages at the bottom of PyPI's download distribution.
        listed = json.loads((directory / "source.json").read_text()).get("n_from_top_list")
        by_count = sorted(counts, key=counts.__getitem__, reverse=True)
        registry_side = set(by_count[:listed] if listed else by_count)
        used: dict[str, dict] = {}
        for row in used_rows:
            if row["ecosystem"] == ecosystem:
                entry = used.setdefault(spell(row["package"]), {"package": row["package"], "n": 0})
                entry["n"] += int(row["n_deposits"])
        for key in sorted(set(counts) | set(used)):
            package = used[key]["package"] if key in used else key
            rows.append(
                {
                    "package": package,
                    "language": language,
                    "ecosystem": ecosystem,
                    "n_deposits": used[key]["n"] if key in used else 0,
                    "downloads": counts.get(key),
                    # Part of the registry's own population of counted
                    # packages, as against a count fetched because we use it.
                    "in_registry_counts": key in registry_side,
                    "reverse_dependencies": depended_on.get(package, 0)
                    if ecosystem == "cran"
                    else None,
                    "in_task_view": (package in in_view) if ecosystem == "cran" else None,
                }
            )
    write_csv(rows, out / "downloads_vs_use.csv")
    return len(rows)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, ties sharing the mean of the places they occupy."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(pairs: list[tuple[float, float]]) -> float:
    """Spearman's rank correlation: Pearson's, computed on the ranks.

    Written out because it is twenty lines and the alternative is scipy, a
    dependency the package would carry for one number.
    """
    x, y = _ranks([a for a, _ in pairs]), _ranks([b for _, b in pairs])
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    spread = sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    return covariance / spread**0.5


def _correlate(pairs: list[tuple[float, float]]) -> float | None:
    """`spearman`, or None where it is undefined: under three points, or no
    variation in one of the two."""
    if len(pairs) < 3 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return None
    return spearman(pairs)


def _rho(rows: list[dict]) -> float | None:
    """Rank agreement between use and downloads."""
    return _correlate([(r["n_deposits"], r["downloads"]) for r in rows])


def deciles(rows: list[dict]) -> list[float]:
    """Share of packages used at all, by download decile, lowest first."""
    if len(rows) < 10:
        return []
    ordered = sorted(rows, key=lambda r: r["downloads"])
    shares = []
    for k in range(10):
        part = ordered[len(ordered) * k // 10 : len(ordered) * (k + 1) // 10]
        shares.append(sum(r["n_deposits"] > 0 for r in part) / len(part))
    return shares


#: Reverse-dependency cut-offs for the mechanism check, fixed before looking.
DEPENDENCY_CUTS = (10, 100)


def agreement(rows: list[dict]) -> dict[str, dict]:
    """How far the two counts agree, per ecosystem and per population.

    Args:
        rows: `downloads_vs_use.csv`, one dict per row, numbers parsed.

    Returns:
        Per ecosystem, for packages with a download count:

        - ``used``: those the corpus uses. ``rho``, and ``rho_5`` and
          ``rho_20`` over packages in at least that many deposits.
        - ``registry``: every package the registry counts, unused ones at
          zero. ``rho``, the share ``ever_used``, that share by download
          ``deciles``, and of the hundred most downloaded how many are used at
          all and in twenty deposits, with their share of downloads and use.
        - for CRAN, ``task_view``: the same ``rho`` among packages a field
          task view lists, which holds the audience roughly fixed; and
          ``dependencies``: the rank correlation between a package's reverse
          dependencies and how far downloads over-rank it (``rho_gap``), of
          each count with reverse dependencies, the partial correlation of
          use with downloads given them, and ``rho`` recomputed after dropping
          packages above each of `DEPENDENCY_CUTS`.
    """
    out = {}
    for ecosystem in SOURCES:
        mine = [r for r in rows if r["ecosystem"] == ecosystem]
        with_count = [r for r in mine if r["downloads"] is not None and r["downloads"] > 0]
        used = [r for r in with_count if r["n_deposits"] > 0]
        counted = [r for r in with_count if r["in_registry_counts"]]
        uncounted = [r for r in mine if r["n_deposits"] > 0 and not r["downloads"]]
        if not counted:
            # No counts pinned for this registry: nothing to compare.
            continue
        top = sorted(counted, key=lambda r: r["downloads"], reverse=True)[:100]
        result = {
            "n_used_all": sum(r["n_deposits"] > 0 for r in mine),
            "used": {
                "n": len(used),
                "rho": _rho(used),
                "rho_5": _rho([r for r in used if r["n_deposits"] >= 5]),
                "rho_20": _rho([r for r in used if r["n_deposits"] >= 20]),
                "n_20": sum(r["n_deposits"] >= 20 for r in used),
                # A used package with no count is not missing at random: it
                # fell below whatever the registry's list reaches, so its
                # downloads are known to be lower than every listed one. Tied
                # at the bottom, it can be ranked with the rest.
                "n_censored": len(uncounted),
                "rho_censored": _correlate(
                    [(r["n_deposits"], r["downloads"] or 0.0) for r in (*used, *uncounted)]
                ),
            },
            "registry": {
                "n": len(counted),
                "rho": _rho(counted),
                "ever_used": sum(r["n_deposits"] > 0 for r in counted) / len(counted),
                "deciles": deciles(counted),
                "top100_used": sum(r["n_deposits"] > 0 for r in top),
                "top100_used_20": sum(r["n_deposits"] >= 20 for r in top),
                "top100_share_of_downloads": sum(r["downloads"] for r in top)
                / sum(r["downloads"] for r in counted),
                "top100_share_of_use": sum(r["n_deposits"] for r in top)
                / sum(r["n_deposits"] for r in mine),
            },
        }
        if ecosystem == "cran":
            viewed = [r for r in used if r["in_task_view"]]
            result["task_view"] = {
                "n_listed": sum(bool(r["in_task_view"]) for r in counted),
                "n": len(viewed),
                "rho": _rho(viewed),
                "rho_5": _rho([r for r in viewed if r["n_deposits"] >= 5]),
            }
            # How far downloads over-rank a package: its use rank minus its
            # download rank, positive when downloads place it higher.
            by_use = _ranks([-r["n_deposits"] for r in used])
            by_downloads = _ranks([-r["downloads"] for r in used])
            gaps = [u - d for u, d in zip(by_use, by_downloads, strict=True)]
            use_deps = _correlate([(r["n_deposits"], r["reverse_dependencies"]) for r in used])
            downloads_deps = _correlate(
                [(r["downloads"], r["reverse_dependencies"]) for r in used]
            )
            use_downloads = _rho(used)
            partial = None
            if use_deps is not None and downloads_deps is not None and use_downloads is not None:
                # What downloads still say about use once reverse dependencies
                # are known: the first-order partial rank correlation.
                partial = (use_downloads - use_deps * downloads_deps) / math.sqrt(
                    (1 - use_deps**2) * (1 - downloads_deps**2)
                )
            result["dependencies"] = {
                "rho_downloads": downloads_deps,
                "rho_use": use_deps,
                "rho_partial": partial,
                "rho_gap": _correlate(
                    [
                        (math.log1p(r["reverse_dependencies"]), gap)
                        for r, gap in zip(used, gaps, strict=True)
                    ]
                ),
                **{
                    f"rho_at_most_{cut}": _rho(
                        [r for r in used if r["reverse_dependencies"] <= cut]
                    )
                    for cut in DEPENDENCY_CUTS
                },
                **{
                    f"n_at_most_{cut}": sum(r["reverse_dependencies"] <= cut for r in used)
                    for cut in DEPENDENCY_CUTS
                },
            }
        out[ecosystem] = result
    return out


def read(path: Path) -> list[dict]:
    """`downloads_vs_use.csv` with its numbers parsed, for `agreement`."""
    with path.open(encoding="utf-8") as handle:
        return [
            {
                **row,
                "n_deposits": int(row["n_deposits"]),
                "downloads": float(row["downloads"]) if row["downloads"] else None,
                "reverse_dependencies": int(row["reverse_dependencies"] or 0),
                "in_task_view": row["in_task_view"] == "True",
                "in_registry_counts": row["in_registry_counts"] == "True",
            }
            for row in csv.DictReader(handle)
        ]

"""Compare our R extraction against `renv::dependencies()`.

    uv run python scripts/validate_renv.py [--limit N]

Writes `build/validation/renv_agreement.json`.

The paper reported a Jaccard of 0.998 for this comparison. Nothing in the
repository computed it: no script, no artifact, no recorded run -- it existed
only as a literal in the prose, which is the one kind of number the paper's
whole arrangement is supposed to make impossible. This computes it.

**What the comparison is worth.** `renv` performs the same kind of static scan
we do, over the same files, so it shares our blind spots: a package named only
in a string, or loaded through a variable, is invisible to both. Agreement
therefore measures whether two implementations of the same idea agree, not
whether either is right. It is convergent validity, and reporting it as
accuracy would be a mistake. What it *can* catch is an implementation bug on
our side large enough to show up against an independent codebase -- which is
worth knowing, and is why the per-deposit disagreements are written out rather
than reduced to the headline number.

**Comparing like with like.** Run over a whole deposit directory, renv reports
three kinds of name our comparison leaves out by design, and on the 2026 corpus
they made up most of the disagreement:

- names from files outside the analysis set -- a shipped package library or a
  `.checkpoint/` tree, which our side treats as vendored;
- base R (`grid`, `stats`, `parallel`), which our resolver files as
  `base_or_stdlib` rather than as a package;
- `rmarkdown`, which renv 1.1.8 adds for every `.Rmd` file even when the file
  loads nothing (checked on a one-line `.Rmd`; a `.qmd` adds nothing).

The headline figures compare in-analysis files only, with those names set
aside, and each set-aside name is counted by category. The whole-directory
comparison is kept under `raw`, so the adjustment can be seen rather than
trusted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import duckdb

from softverse.config import PATHS
from softverse.corpus.loaders import deposit_directories
from softverse.logging_setup import get_logger, setup_logging
from softverse.registries.fetch import BASE_R

logger = get_logger(__name__)

OUT = PATHS.root / "build" / "validation"
TALLY = PATHS.root / "build" / "tally"

#: `renv::dependencies()` on one directory, as `[package, source file]` pairs
#: in JSON on stdout. `quiet` and the error handler matter: renv warns on
#: unparseable files and would otherwise abort the whole directory over one bad
#: script, which would silently turn a parse difference into a coverage
#: difference.
R_SCRIPT = """
args <- commandArgs(trailingOnly = TRUE)
deps <- tryCatch(
  renv::dependencies(args[1], quiet = TRUE, errors = "ignored"),
  error = function(e) NULL
)
pairs <- if (is.null(deps) || nrow(deps) == 0) {
  matrix(character(0), ncol = 2)
} else {
  unique(as.matrix(deps[c("Package", "Source")]))
}
cat(jsonlite::toJSON(unname(pairs)))
"""

#: Why a name was left out of the aligned comparison, in the order tested.
OUTSIDE, BASE, IMPLICIT = "outside_analysis_files", "base_r", "implicit_rmarkdown"


def renv_packages(script: Path, directory: Path) -> list[tuple[str, str]] | None:
    """`(package, source file)` pairs renv finds under ``directory``.

    Args:
        script: The R script that runs `renv::dependencies`.
        directory: The deposit's directory.

    Returns:
        The pairs, or None if renv failed.
    """
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", str(script), str(directory)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        found = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    # renv reports a dependency it could not name as NA, which arrives as
    # null. It names no package, so it can be neither agreement nor
    # disagreement; left in, it crashed the comparison on the 2026 corpus.
    return [
        (name, os.path.realpath(source))
        for name, source in found
        if isinstance(name, str) and isinstance(source, str)
    ]


def jaccard(a: set[str], b: set[str]) -> float:
    """1.0 when both are empty: two scanners agreeing there is nothing here
    agree, and scoring that as 0 would punish the easiest case.
    """
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def align(
    ours_all: set[str],
    ours_analysis: set[str],
    renv: list[tuple[str, str]],
    analysis_files: set[str],
) -> tuple[set[str], set[str], Counter]:
    """Both sides restricted to the analysis set, with renv's extra names out.

    Args:
        ours_all: Names we found in any of the deposit's files.
        ours_analysis: Names we found in its in-analysis files.
        renv: renv's `(package, source)` pairs for the whole directory.
        analysis_files: Real paths of the deposit's in-analysis R files.

    Returns:
        Our names, renv's names, and a count per reason of the renv names set
        aside -- plus `ours_outside_analysis_files` for our names that only
        came from files outside the analysis set.
    """
    sources: dict[str, set[str]] = {}
    for name, source in renv:
        sources.setdefault(name, set()).add(source)
    theirs: set[str] = set()
    set_aside: Counter = Counter()
    for name, found_in in sources.items():
        inside = found_in & analysis_files
        if not inside:
            set_aside[OUTSIDE] += 1
        elif name in BASE_R:
            set_aside[BASE] += 1
        elif (
            name == "rmarkdown"
            and name not in ours_analysis
            and all(path.lower().endswith(".rmd") for path in inside)
        ):
            set_aside[IMPLICIT] += 1
        else:
            theirs.add(name)
    set_aside["ours_" + OUTSIDE] = len(ours_all - ours_analysis)
    return ours_analysis, theirs, set_aside


def summarize(rows: list[dict]) -> dict:
    """Mean and pooled Jaccard over comparison rows.

    Two summaries, because they answer different questions. The mean of
    per-deposit Jaccards weights every deposit equally; the pooled figure
    weights every package-deposit pair equally, so one enormous deposit cannot
    be outvoted by a hundred trivial ones. Reporting only the first is how a
    headline agreement number gets flattered by deposits with one package that
    both tools find.
    """
    union = sum(r["n_union"] for r in rows)
    return {
        "n_deposits_compared": len(rows),
        "mean_jaccard": sum(r["jaccard"] for r in rows) / len(rows) if rows else 1.0,
        "pooled_jaccard": sum(r["n_shared"] for r in rows) / union if union else 1.0,
        "n_perfect_agreement": sum(1 for r in rows if r["jaccard"] == 1.0),
        "n_ours_only": sum(len(r["ours_only"]) for r in rows),
        "n_renv_only": sum(len(r["renv_only"]) for r in rows),
    }


def row(doi: str, mine: set[str], theirs: set[str]) -> dict:
    """One deposit's comparison."""
    return {
        "dataset_doi": doi,
        "jaccard": jaccard(mine, theirs),
        "ours_only": sorted(mine - theirs),
        "renv_only": sorted(theirs - mine),
        "n_ours": len(mine),
        "n_renv": len(theirs),
        "n_shared": len(mine & theirs),
        "n_union": len(mine | theirs),
        "source": "dataverse" if doi.startswith("doi:") else "zenodo",
    }


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="validate-renv")
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    con = duckdb.connect()
    # Deposits with R code, and what we found in them, split by whether the
    # file is in the analysis set. Restricted to mentions that could name a
    # third-party package, because that is the set `renv` reports.
    ours = con.execute(
        f"""
        SELECT m.dataset_doi,
               list(DISTINCT m.raw_name) AS everywhere,
               list(DISTINCT m.raw_name) FILTER (WHERE f.in_analysis_set)
                   AS in_analysis
        FROM '{TALLY}/mentions.parquet' m
        JOIN '{TALLY}/files.parquet' f USING (file_uid)
        WHERE m.language = 'r'
          AND m.resolution NOT IN ('builtin', 'local_program', 'local_relative',
                                   'dynamic', 'base_or_stdlib')
          -- `library(p, character.only = TRUE)` where `p` is a loop variable
          -- names no package, and neither tool can say which one. Comparing
          -- it would score our honesty as a disagreement.
          AND m.construct NOT IN ('dynamic_unresolved', 'local_relative')
          AND m.raw_name IS NOT NULL AND m.raw_name != ''
        GROUP BY m.dataset_doi
        """
    ).fetchall()
    if limit:
        ours = ours[:limit]
    analysis: dict[str, set[str]] = {}
    for doi, local_path in con.execute(
        f"""
        SELECT dataset_doi, local_path FROM '{TALLY}/files.parquet'
        WHERE in_analysis_set AND language IN ('r', 'rmarkdown')
        """
    ).fetchall():
        analysis.setdefault(doi, set()).add(os.path.realpath(local_path))
    logger.info("comparing", extra={"deposits": len(ours)})

    # Every source. This was `corpus/zenodo/files` with the deposit directory
    # taken from the tail of the DOI, which is a Zenodo record id and is not a
    # path for anything else, so the comparison silently covered 377 of the
    # corpus's 3,462 R deposits and none of Harvard Dataverse.
    directories = deposit_directories()
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(R_SCRIPT)
        script = Path(handle.name)

    raw_rows, rows = [], []
    unreadable: list[dict] = []
    set_aside: Counter = Counter()
    skipped = no_analysis_r = 0
    for doi, everywhere, in_analysis in ours:
        directory = directories.get(doi)
        if directory is None or not directory.is_dir():
            skipped += 1
            continue
        pairs = renv_packages(script, directory)
        if pairs is None:
            skipped += 1
            continue
        # Compared case-sensitively: R package names are case-sensitive, and
        # folding case here would hide a real class of disagreement rather
        # than measure it.
        mine_all = set(everywhere)
        theirs_all = {name for name, _ in pairs}
        # renv returning nothing for a deposit we found packages in is not a
        # disagreement, it is renv declining to read the files. It runs with
        # `errors = "ignored"`, so a file it cannot decode is dropped in
        # silence. Spot-checked: a `polbehavior` deposit whose only script is
        # ISO-8859 with CRLF contains `require(MASS)` at byte 43,566, and renv
        # reports nothing for the directory. Scoring that as our false
        # positive would credit the comparison for a file one side never saw,
        # which is the treatment the R oracle already refuses for files R
        # cannot parse.
        if not theirs_all and mine_all:
            unreadable.append({"dataset_doi": doi, "n_ours": len(mine_all)})
            continue
        raw_rows.append(row(doi, mine_all, theirs_all))
        mine, theirs, reasons = align(
            mine_all, set(in_analysis or ()), pairs, analysis.get(doi, set())
        )
        set_aside.update(reasons)
        # A deposit whose R lives entirely outside the analysis set (a shipped
        # library and nothing else) has nothing left to compare. Scoring two
        # empty sets as perfect agreement would pad the headline.
        if not mine and not theirs:
            no_analysis_r += 1
            continue
        rows.append(row(doi, mine, theirs))
    script.unlink(missing_ok=True)

    if not rows:
        print("no deposits compared; is the corpus present?")
        return 1

    headline = summarize(rows)
    disagreements = sorted(
        (r for r in rows if r["jaccard"] < 1.0), key=lambda r: r["jaccard"]
    )
    # Per source, so nobody has to read this file to learn what was covered.
    # The previous version reported one pooled figure over a slice that
    # happened to be entirely Zenodo, and nothing in the output said so.
    by_source = {
        source: summarize([r for r in rows if r["source"] == source])
        for source in sorted({r["source"] for r in rows})
    }

    # Both lists are samples, capped at 50, and the counts beside them are
    # the totals. Worth saying out loud: diffing two runs' `disagreements` or
    # `renv_read_nothing` compares the samples, not the populations, so
    # deposits appear to arrive and leave when only the ordering moved. Only
    # the aggregate fields are comparable across runs.
    report = {
        **headline,
        "comparison": "in-analysis R files; base R and renv's implicit "
        "rmarkdown set aside",
        "by_source": by_source,
        "set_aside": dict(sorted(set_aside.items())),
        "n_no_analysis_r": no_analysis_r,
        "raw": {
            "comparison": "whole deposit directory, every name renv reports",
            **summarize(raw_rows),
            "top_renv_only": Counter(
                p for r in raw_rows for p in r["renv_only"]
            ).most_common(20),
        },
        "n_renv_read_nothing": len(unreadable),
        "renv_read_nothing_sample": unreadable[:50],
        "n_skipped": skipped,
        "n_disagreements": len(disagreements),
        # Which packages, not just how many. A residual that is one package
        # repeated across hundreds of deposits is a rule difference; a
        # residual spread thinly over hundreds of packages is a recall
        # problem, and the totals alone cannot tell those apart.
        "top_renv_only": Counter(
            p for r in disagreements for p in r["renv_only"]
        ).most_common(20),
        "top_ours_only": Counter(
            p for r in disagreements for p in r["ours_only"]
        ).most_common(20),
        "renv_version": _renv_version(),
        "disagreements_sample": disagreements[:50],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "renv_agreement.json").write_text(json.dumps(report, indent=1) + "\n")

    raw = report["raw"]
    print(f"compared      : {len(rows):,} deposits ({skipped} skipped)")
    print(f"renv read none: {len(unreadable):,} deposits excluded, see the JSON")
    print(f"no analysis R : {no_analysis_r:,} deposits")
    for source, stats in by_source.items():
        print(
            f"  {source:<20}{stats['n_deposits_compared']:>6,} deposits  "
            f"mean {stats['mean_jaccard']:.3f}  pooled {stats['pooled_jaccard']:.3f}"
        )
    print(
        f"aligned Jaccard: mean {headline['mean_jaccard']:.3f}  "
        f"pooled {headline['pooled_jaccard']:.3f}"
    )
    print(
        f"raw Jaccard    : mean {raw['mean_jaccard']:.3f}  "
        f"pooled {raw['pooled_jaccard']:.3f}"
    )
    print(f"set aside     : {report['set_aside']}")
    print(f"perfect       : {headline['n_perfect_agreement']:,}/{len(rows):,}")
    if disagreements:
        print("\nworst disagreements:")
        for r in disagreements[:10]:
            print(
                f"  {r['dataset_doi']:<24} {r['jaccard']:.2f}  "
                f"ours-only {r['ours_only'][:4]}  renv-only {r['renv_only'][:4]}"
            )
    return 0


def _renv_version() -> str:
    result = subprocess.run(
        ["Rscript", "-e", 'cat(as.character(packageVersion("renv")))'],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())

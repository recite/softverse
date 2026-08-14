"""Compare our R extraction against `renv::dependencies()`.

    uv run python scripts_validate_renv.py [--limit N]

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
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

from softverse.config import PATHS
from softverse.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

OUT = PATHS.root / "build" / "validation"
TALLY = PATHS.root / "build" / "tally"

#: `renv::dependencies()` on one directory, as JSON on stdout. `quiet` and the
#: error handler matter: renv warns on unparseable files and would otherwise
#: abort the whole directory over one bad script, which would silently turn a
#: parse difference into a coverage difference.
R_SCRIPT = """
args <- commandArgs(trailingOnly = TRUE)
deps <- tryCatch(
  renv::dependencies(args[1], quiet = TRUE, errors = "ignored"),
  error = function(e) NULL
)
pkgs <- if (is.null(deps) || nrow(deps) == 0) character(0) else unique(deps$Package)
cat(jsonlite::toJSON(pkgs, auto_unbox = FALSE))
"""


def renv_packages(script: Path, directory: Path) -> set[str] | None:
    """Packages `renv` finds under ``directory``, or None if it failed."""
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
        return set(json.loads(result.stdout))
    except json.JSONDecodeError:
        return None


def jaccard(a: set[str], b: set[str]) -> float:
    """1.0 when both are empty: two scanners agreeing there is nothing here
    agree, and scoring that as 0 would punish the easiest case."""
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="validate-renv")
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    con = duckdb.connect()
    # Deposits with R code, and what we found in them. Restricted to mentions
    # that could name a third-party package, because that is the set `renv`
    # reports: it does not list base R.
    ours = con.execute(
        f"""
        SELECT dataset_doi, list(DISTINCT raw_name) AS packages
        FROM '{TALLY}/mentions.parquet'
        WHERE language = 'r'
          AND resolution NOT IN ('builtin', 'local_program', 'local_relative',
                                 'dynamic', 'base_or_stdlib')
          -- `library(p, character.only = TRUE)` where `p` is a loop variable
          -- names no package, and neither tool can say which one. Comparing
          -- it would score our honesty as a disagreement.
          AND construct NOT IN ('dynamic_unresolved', 'local_relative')
          AND raw_name IS NOT NULL AND raw_name != ''
        GROUP BY dataset_doi
        """
    ).fetchall()
    if limit:
        ours = ours[:limit]
    logger.info("comparing", extra={"deposits": len(ours)})

    files_root = PATHS.root / "corpus" / "zenodo" / "files"
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as handle:
        handle.write(R_SCRIPT)
        script = Path(handle.name)

    rows = []
    skipped = 0
    for doi, packages in ours:
        directory = files_root / doi.split(":")[-1]
        if not directory.is_dir():
            skipped += 1
            continue
        theirs = renv_packages(script, directory)
        if theirs is None:
            skipped += 1
            continue
        # Compared case-sensitively: R package names are case-sensitive, and
        # folding case here would hide a real class of disagreement rather
        # than measure it.
        mine = set(packages)
        rows.append(
            {
                "dataset_doi": doi,
                "jaccard": jaccard(mine, theirs),
                "ours_only": sorted(mine - theirs),
                "renv_only": sorted(theirs - mine),
                "n_ours": len(mine),
                "n_renv": len(theirs),
                "n_shared": len(mine & theirs),
                "n_union": len(mine | theirs),
            }
        )
    script.unlink(missing_ok=True)

    if not rows:
        print("no deposits compared; is the corpus present?")
        return 1

    # Two summaries, because they answer different questions. The mean of
    # per-deposit Jaccards weights every deposit equally; the pooled figure
    # weights every package-deposit pair equally, so one enormous deposit
    # cannot be outvoted by a hundred trivial ones. Reporting only the first
    # is how a headline agreement number gets flattered by deposits with one
    # package that both tools find.
    mean_jaccard = sum(r["jaccard"] for r in rows) / len(rows)
    union = sum(r["n_union"] for r in rows)
    pooled = sum(r["n_shared"] for r in rows) / union if union else 1.0

    disagreements = sorted(
        (r for r in rows if r["jaccard"] < 1.0), key=lambda r: r["jaccard"]
    )
    report = {
        "n_deposits_compared": len(rows),
        "n_skipped": skipped,
        "mean_jaccard": mean_jaccard,
        "pooled_jaccard": pooled,
        "n_perfect_agreement": sum(1 for r in rows if r["jaccard"] == 1.0),
        "renv_version": _renv_version(),
        "disagreements": disagreements[:50],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "renv_agreement.json").write_text(json.dumps(report, indent=1))

    print(f"compared      : {len(rows):,} deposits ({skipped} skipped)")
    print(f"mean Jaccard  : {mean_jaccard:.3f}")
    print(f"pooled Jaccard: {pooled:.3f}")
    print(f"perfect       : {report['n_perfect_agreement']:,}/{len(rows):,}")
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

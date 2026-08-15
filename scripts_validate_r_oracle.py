"""Measure the R extractor against R's own parser.

    uv run python scripts_validate_r_oracle.py [--limit N]

Writes `build/validation/r_oracle.json`.

**Why this is different from the `renv` comparison.** `renv` is a peer: it
performs the same kind of static scan we do and shares our blind spots, so
agreement is biased upward by correlated error and cannot establish accuracy.
R's parser is not a peer. For the syntactic question -- "does this file
contain a call to `library` whose first argument is the symbol or string
`x`" -- R's parser *defines* the answer. Disagreement is our error, full stop,
and precision and recall are exact rather than indicative.

The paper already argued this was possible and reported no measurement. This
is the measurement.

**What it does not establish.** That a file calls `library(dplyr)` is not that
the analysis used dplyr, and the oracle says nothing about Stata, about
resolution against CRAN, or about the semantic question of research use. It
bounds one thing exactly: whether our tree-sitter walk sees what R sees.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

from softverse.config import PATHS
from softverse.corpus.loaders import full_corpus
from softverse.detect.r import extract
from softverse.logging_setup import get_logger, setup_logging
from softverse.model.enums import Construct

logger = get_logger(__name__)

OUT = PATHS.root / "build" / "validation"

#: The oracle. Recursion over the expression tree R itself produced, using R's
#: own notion of what a call is -- no regex, no second grammar. `library` and
#: `require` take a symbol or a string; `requireNamespace` and friends take a
#: string; `::` and `:::` take a symbol on the left.
#:
#: Files that do not parse are reported as such rather than as empty: an
#: unparseable file is not a file with no dependencies, and scoring it as one
#: would credit us for finding nothing in it.
R_ORACLE = r"""
suppressWarnings(suppressMessages(library(jsonlite)))
args <- commandArgs(trailingOnly = TRUE)

# The asymmetry is measured, not assumed. `library(pkg)` with pkg <- "stats"
# errors with "there is no package called 'pkg'": it is non-standard
# evaluation and the symbol *is* the name. `requireNamespace(pkg)` with the
# same binding returns TRUE, because its body starts
# `package <- as.character(package)[[1L]]`, which forces the promise and uses
# the variable's value. So a bare symbol means opposite things in the two
# groups, and only the first can be read statically.
NSE_LOADERS <- c("library", "require")
STRING_LOADERS <- c("requireNamespace", "loadNamespace",
                    "attachNamespace", "isNamespaceLoaded")
LOADERS <- c(NSE_LOADERS, STRING_LOADERS)

found <- NULL
add <- function(x) if (nzchar(x)) assign(x, TRUE, envir = found)

as_name <- function(a) {
  if (is.character(a) && length(a) == 1) return(a)
  if (is.name(a)) return(as.character(a))
  ""
}

#: The callee's bare name, seeing through `base::library` to `library`.
#: Without this the oracle missed every namespace-qualified loader, because
#: such a call's head is itself a call rather than a symbol. renv's own test
#: fixture contains `base::library(c)` and `base::requireNamespace("f")`
#: precisely to catch that, and it caught it here.
callee_name <- function(head) {
  if (is.name(head)) return(as.character(head))
  if (is.call(head) && is.name(head[[1]]) &&
      as.character(head[[1]]) %in% c("::", ":::") && length(head) >= 3) {
    return(as_name(head[[3]]))
  }
  ""
}

walk <- function(e) {
  if (!is.call(e)) return(invisible(NULL))
  head <- e[[1]]
  fname <- callee_name(head)
  if (nzchar(fname)) {
    if (fname %in% c("::", ":::") && length(e) >= 2) {
      # `"m"::baz()` is legal: the left operand may be a string.
      add(as_name(e[[2]]))
    } else if (fname %in% LOADERS && length(e) >= 2) {
      # A string literal is always the package name, `character.only` or not:
      # that flag only says "do not evaluate this as a symbol", and a literal
      # was never a symbol. It is a *variable* that cannot be resolved, and R
      # can no more do it statically than we can -- so the oracle declines
      # only there, rather than scoring our honesty as a miss.
      #
      # Blanket-skipping every `character.only = TRUE` call was this script's
      # own bug, and the two disagreements it produced were both the oracle
      # being wrong: `require("foo", ..., character.only = TRUE)` names foo.
      nms <- names(e)
      co <- if (!is.null(nms) && "character.only" %in% nms) e[["character.only"]] else FALSE
      # `T` as well as `TRUE`. `character.only = T` is the common spelling in
      # the wild, and `T` is a *symbol* rather than the literal, so testing
      # only for `quote(TRUE)` reads the flag as absent, takes the loop
      # variable for a package name, and reports `library(package,
      # character.only = T)` as a dependency called "package". Every
      # disagreement in the Dataverse half was this. `T` is rebindable, so
      # this is a reading rather than a certainty, and it is the same reading
      # the extractor makes.
      character_only <- isTRUE(co) || identical(co, quote(TRUE)) ||
                        identical(co, quote(T))
      arg <- e[[2]]
      if (is.character(arg)) {
        add(as_name(arg))
      } else if (is.name(arg) && !character_only && fname %in% NSE_LOADERS) {
        add(as_name(arg))
      }
    }
  }
  # A function definition's formals are a *pairlist*, not a call, so plain
  # recursion stops at it and never sees a default argument. That hid every
  # `pkg::fun()` used as a default -- `height = grid::unit(0.2, "null")` in
  # cowplot, and the same shape in latticeExtra, mapview, sf and plyr.
  for (part in as.list(e)) {
    if (missing(part)) next
    if (is.pairlist(part)) {
      for (default in as.list(part)) if (!missing(default)) walk(default)
    } else {
      walk(part)
    }
  }
  invisible(NULL)
}

# One process for the whole batch. Rscript startup is ~0.4s, which over
# 18,000 files is two and a half hours of doing nothing; batching makes a
# full-corpus run a few minutes and removes the temptation to validate on a
# convenience sample. The first attempt did exactly that -- 120 files taken
# alphabetically, all of them inside one deposit's vendored checkpoint
# library, which is not a sample of anything.
paths <- readLines(args[1], warn = FALSE)
for (path in paths) {
  # `all.names = TRUE` below: `ls()` hides dot-prefixed names by default, so
  # the oracle was concealing `..notexisting..` and `.` from itself and
  # scoring them as our false positives.
  found <- new.env(parent = emptyenv())
  out <- tryCatch({
    exprs <- parse(path, keep.source = FALSE, encoding = "UTF-8")
    for (e in exprs) walk(e)
    list(path = path, ok = TRUE, packages = I(ls(found, all.names = TRUE)))
  }, error = function(e) list(path = path, ok = FALSE, packages = I(character(0))))
  # `I()` keeps `packages` an array. Without it `auto_unbox` collapses a
  # one-element vector to a bare string, and a Python `set()` over that string
  # is a set of its letters -- which scored `LearnBayes` as eight missing
  # packages named L, e, a, r, n, B, y and s.
  cat(toJSON(out, auto_unbox = TRUE), "\n", sep = "")
}
"""

#: Constructs the oracle also looks for. Ours finds more than R's parser does
#: -- `p_load`, install calls, YAML front matter -- and comparing those would
#: measure a difference in scope rather than an error in extraction.
COMPARABLE = frozenset(
    {
        Construct.LIBRARY,
        Construct.REQUIRE,
        Construct.NAMESPACE_CHECK,
        Construct.NAMESPACE_OP,
        Construct.NAMESPACE_INTERNAL,
    }
)


def ours(path: Path) -> set[str] | None:
    try:
        source = path.read_bytes()
    except OSError:
        return None
    result = extract(source)
    return {
        m.raw_name
        for m in result.mentions
        if m.construct in COMPARABLE and not m.is_dynamic
    }


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="validate-r-oracle")
    limit = (
        int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    )

    # Every source, not one directory. This globbed `corpus/zenodo/files`,
    # so the check covered 18,120 of the corpus's 28,527 R files and none of
    # Harvard Dataverse at all, while the paper described the result as
    # holding over "corpus .R files". A validation that cannot see half the
    # data is not a weaker claim about the whole, it is a claim about a
    # different corpus.
    source_of = {
        item.path: item.source
        for item in full_corpus()
        if item.path.suffix.lower() == ".r"
    }
    files = sorted(source_of)
    if limit:
        files = files[:limit]
    by_source = collections.Counter(source_of[p] for p in files)
    logger.info("comparing", extra={"files": len(files), **by_source})
    print(f"comparing {len(files):,} R files")
    for source, n in sorted(by_source.items()):
        print(f"  {source:<20}{n:>8,}")

    script = OUT / "_r_oracle.R"
    OUT.mkdir(parents=True, exist_ok=True)
    script.write_text(R_ORACLE)

    tp = fp = fn = 0
    unparseable = 0
    compared = 0
    # Per source, so a future reader can see coverage without reading this
    # file, and so a regression confined to one corpus cannot hide inside a
    # pooled precision of 1.0000.
    per_source: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    misses: dict[str, int] = {}
    spurious: dict[str, int] = {}
    examples: list[dict] = []

    listing = OUT / "_r_oracle_files.txt"
    listing.write_text("\n".join(str(p) for p in files))
    result = subprocess.run(
        ["Rscript", "--vanilla", str(script), str(listing)],
        capture_output=True,
        text=True,
    )
    listing.unlink(missing_ok=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        return 1

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = Path(payload["path"])
        if not payload.get("ok"):
            # R could not parse it. Whatever we found there is unscoreable:
            # the oracle has no opinion, not an empty one.
            unparseable += 1
            per_source[source_of.get(path, "unknown")]["unparseable"] += 1
            continue
        mine = ours(path)
        if mine is None:
            continue
        found_names = payload.get("packages") or []
        if isinstance(found_names, str):  # belt and braces against auto_unbox
            found_names = [found_names]
        theirs = set(found_names)
        source = source_of.get(path, "unknown")
        compared += 1
        counts = per_source[source]
        counts["compared"] += 1
        tp += len(mine & theirs)
        counts["tp"] += len(mine & theirs)
        for name in theirs - mine:
            fn += 1
            counts["fn"] += 1
            misses[name] = misses.get(name, 0) + 1
        for name in mine - theirs:
            fp += 1
            counts["fp"] += 1
            spurious[name] = spurious.get(name, 0) + 1
        if (mine != theirs) and len(examples) < 40:
            examples.append(
                {
                    "path": str(path.relative_to(PATHS.root)),
                    "missed": sorted(theirs - mine),
                    "spurious": sorted(mine - theirs),
                }
            )

    script.unlink(missing_ok=True)
    if not compared:
        print("nothing compared; is the corpus present?")
        return 1

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    def rates(c: collections.Counter) -> dict:
        return {
            "n_files_compared": c["compared"],
            "n_files_unparseable_by_r": c["unparseable"],
            "true_positives": c["tp"],
            "false_positives": c["fp"],
            "false_negatives": c["fn"],
            "precision": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0,
            "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0,
        }

    report = {
        "by_source": {s: rates(c) for s, c in sorted(per_source.items())},
        "n_files_compared": compared,
        "n_files_unparseable_by_r": unparseable,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "top_missed": sorted(misses.items(), key=lambda kv: -kv[1])[:25],
        "top_spurious": sorted(spurious.items(), key=lambda kv: -kv[1])[:25],
        "examples": examples,
    }
    (OUT / "r_oracle.json").write_text(json.dumps(report, indent=1) + "\n")

    print(f"files compared      : {compared:,} ({unparseable} unparseable by R)")
    print(f"name-file pairs     : tp {tp:,}  fp {fp:,}  fn {fn:,}")
    print(f"precision           : {precision:.4f}")
    print(f"recall              : {recall:.4f}")
    if report["top_missed"]:
        print(f"\nmost missed  : {report['top_missed'][:8]}")
    if report["top_spurious"]:
        print(f"most spurious: {report['top_spurious'][:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

### Softverse: measuring software use in social science replication code

[![CI](https://github.com/recite/softverse/actions/workflows/ci.yml/badge.svg)](https://github.com/recite/softverse/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/softverse.svg)](https://pypi.python.org/pypi/softverse)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://recite.github.io/softverse/)
[![PePy Downloads](https://static.pepy.tech/badge/softverse)](https://www.pepy.tech/projects/softverse)

Which libraries does social science research actually run on? Softverse answers
that by statically analyzing the code inside replication packages deposited to
journal collections, across **R, Python and Stata**.

see: https://gojiberries.io/2023/07/02/hard-problems-about-research-software/

## What is measured

Precisely: **software referenced in deposited replication code.**

> Among replication deposits in collection *J* containing at least one
> analyzable script in language *L*, the share that statically reference
> package *P*.

Static reference is not runtime use — a deposit can load a package in a
superseded script or a dead branch, and code kept out of the deposit is
invisible. The distance between the two is the main limitation and is reported
rather than glossed.

The unit is the **deposit**, not the mention. A deposit calling `ggplot2` two
hundred times is one user of `ggplot2`; a mention-weighted ranking would rank
whichever deposits happen to be largest.

Every count carries its denominator. "273 deposits use `estout`" is reported as
**273/469** — of the deposits containing analyzable Stata, not of all deposits.

## Sampling frame

The inclusion rule is not ours to invent. We use the journal list maintained by
the [Social Science Data Editors](https://github.com/social-science-data-editors/DCAS)
— journals whose data-and-code policy the editors *actively verify* — mapped to
the repositories that actually hold the material.

`data/frame/frame.csv` is deliberately small and readable: someone who knows
these journals should be able to read all of it and say one is misplaced.
Journals we could not locate are rows in that file, not omissions.

## Quick start

```bash
uv sync --all-extras

# Build registry snapshots (CRAN, CRAN Archive, Bioconductor, PyPI, Julia)
uv run python -c "from softverse.registries.fetch import fetch_all; from pathlib import Path; fetch_all(Path('registries/snapshots'))"

# Collect. Incremental by default: only deposits never successfully fetched.
uv run python scripts_collect_zenodo.py
uv run python scripts_collect_zenodo.py --fresh   # full re-scrape

# Tally. Safe to run mid-collection; it describes whatever is on disk.
uv run python scripts_build_tally.py
```

Outputs land in `build/tally/`:

| file | contents |
|---|---|
| `usage_by_package.csv` | the tally: package → deposits, files, mentions, share |
| `usage_by_package_year.csv` | the same over time |
| `language_presence.csv` | deposits containing each language |
| `unknown_names.csv` | detected names resolving to no registry |
| `mentions.parquet` | every mention, with line, column and snippet |
| `files.parquet` | the provenance spine |

## How detection works

| language | mechanism |
|---|---|
| R | `tree-sitter` syntax walk |
| Python | stdlib `ast`, with a `tokenize` fallback for Python 2 |
| Stata | purpose-built lexer + an SSC command→package index |
| `.Rmd` / `.qmd` / `.Rnw` | knitr chunk splitter, routed per engine |
| `.ipynb` | per cell, kernel language read from metadata |

**Strings and comments are excluded structurally, not by pre-stripping.** In a
syntax tree, `cat("run library(dplyr)")` is a call whose argument is a string
node — those bytes never form a call, so a walk over call nodes cannot see them.

**Stata needed an artifact that did not exist.** R has CRAN and Python has PyPI;
Stata has no machine-readable registry. Softverse reconstructs a command→package
index from SSC distribution manifests — 3,992 packages, 7,468 commands — and
releases it. Without it Stata is unmeasurable, which is why it tends to be left
out of work like this, despite being the language this corpus uses most.

## Design commitments

These exist because the previous version of this project got each one wrong, in
ways that silently invalidated its published numbers.

- **Join keys are never null.** Enforced at the write boundary, not merely
  declared: pyarrow treats `nullable=False` as documentation.
- **Every stage reconciles.** `files_total == analyzed + vendored + duplicate +
  skipped + unparseable`, asserted in code.
- **"No packages here" is never the same value as "could not read this."**
  Parse status is recorded per file.
- **Provenance on every row.** Extractor version, grammar version, registry lock,
  plus source line and snippet, so any published number traces back to text.
- **Registries are pinned**, so resolution does not depend on when it ran.
- **Coverage gaps are rows, not silences.** Skipped archives and unlocated
  journals are recorded with their sizes and reasons.

## Repository layout

```
softverse/
  frame.py          the sampling frame, verified
  sources/          zenodo, osf, dataverse, dataverse_oai
  acquire/          http (rate limits, retries), state (ledger), unpack
  detect/           r, python_, stata, notebooks, dispatch
  registries/       snapshot fetch + resolution
  stata/            command index, builtins, lexer
  corpus/           vendored-library and duplicate rules
  build/            pipeline and aggregation
paper/              manuscript (Quarto) + validate_bib.py
data/frame/         the frame, human-readable
```

## Authors

Gaurav Sood and Daniel Weitzel

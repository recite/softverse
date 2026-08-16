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

A reference is not a run. A deposit can load a package in an older script or
inside a branch that never executes, and code the author kept out of the
deposit cannot be seen at all. How far apart those two things are is the main
limitation, and the paper measures it.

The unit is the **deposit**, not the mention. A deposit calling `ggplot2` two
hundred times is one user of `ggplot2`; a mention-weighted ranking would rank
whichever deposits happen to be largest.

Every count carries its denominator. "273 deposits use `estout`" is reported
as 273 of 469, where 469 counts the deposits containing analyzable Stata and
not the deposits overall.

## Sampling frame

The inclusion rule comes from outside this project. It is the journal list
kept by the [Social Science Data Editors](https://github.com/social-science-data-editors/DCAS),
covering journals whose data-and-code policy the editors actively verify,
mapped to the repositories that hold the material.

`data/frame/frame.csv` is deliberately small and readable: someone who knows
these journals should be able to read all of it and say one is misplaced.
Journals that could not be located are rows in that file, so a gap in the
frame is visible rather than absent.

The two repositories hold two disciplines. Zenodo's verified collections are
economics and Harvard Dataverse's journal collections are mostly political
science. One pass tallies both, and every row of `usage_by_package.csv`
carries the pooled count next to the per-repository split, since the two are
very different sizes.

The Dataverse material is a January 2024 scrape that collected only `.do`,
`.r` and `.py` files, so a package used mainly inside notebooks or knitr
documents is under-counted on that side. That is checked rather than
asserted: `scripts_release_tally.py` recomputes the whole ranking restricted
to the file types both halves collected, and fails if the ordering moves.

## Using it as a library

`pip install softverse` gives you the two pieces worth reusing: an extractor
that reads a file and reports what it loads, and a resolver that maps a name
to the package providing it.

```python
from pathlib import Path
from softverse.detect.dispatch import extract_file

result, _decoded, language = extract_file(Path("analysis.do"))
print(language)                    # Language.STATA
for m in result.mentions:
    print(m.raw_name, m.construct, m.line)
# use      stata_command 1
# reghdfe  stata_command 2
# esttab   stata_command 3
print(result.report.status)        # ParseStatus.OK
```

Resolution needs the registry snapshots, which the quick start below builds:

```python
from scripts_build_tally import load_registry
from softverse.model.enums import Language

registry, _shipped = load_registry()
registry.resolve("esttab", Language.STATA).package    # 'estout'
registry.resolve("regress", Language.STATA).resolution  # builtin, so no package
registry.resolve("grc1leg", Language.STATA).resolution  # unknown: in no registry
```

`esttab` resolving to `estout` is the case a package-name list cannot handle,
and the reason the Stata index exists: one package ships many commands.

## Reproducing the corpus

```bash
uv sync --all-extras

# Build registry snapshots (CRAN, CRAN Archive, Bioconductor, PyPI, Julia)
uv run python -c "from softverse.registries.fetch import fetch_all; from pathlib import Path; fetch_all(Path('registries/snapshots'))"

# Collect. Incremental by default: only deposits never successfully fetched.
uv run python scripts_collect_zenodo.py
uv run python scripts_collect_zenodo.py --fresh          # full re-scrape
uv run python scripts_collect_zenodo.py --metadata-only  # community and year only

# Unpack the 2024 Harvard Dataverse scrape into the corpus.
uv run python scripts_ingest_dataverse_legacy.py

# Tally every source in one pass. Safe to run mid-collection; it describes
# whatever is on disk.
uv run python scripts_build_tally.py
```

Softverse produces four things, and they are worth naming separately because
only two of them are published.

The **corpus** is the code itself: 217,573 files from 22,346 deposits, kept on
disk under `corpus/` and mirrored by `scripts_corpus_view.py` into the layout
other projects read. It is far too large to publish and is rebuilt by the
ingest scripts above.

The **atomic record** is one row per mention: the package, the function where
the source names one, the file, the line, the column and the snippet. This is
what everything else is a sum of.

The **aggregates** are those sums, cut by package, year, journal and function.

The **registry** is the instrument rather than a result: the CRAN, PyPI and
SSC name tables, plus the Stata command index, that turn a token into a
package. It is what makes the counts reproducible, and it changes under you if
you do not pin it.

Outputs land in `build/tally/`:

| file | layer | contents |
|---|---|---|
| `usage_by_package.csv` | aggregate | the tally: package → deposits, files, mentions, share, pooled and split by repository |
| `usage_by_package_year.csv` | aggregate | the same over time |
| `usage_by_collection.csv` | aggregate | the same per journal or community |
| `usage_by_function.csv` | aggregate | package → function, where the source names one |
| `language_presence.csv` | aggregate | deposits containing each language, per repository |
| `unknown_names.csv` | diagnostic | detected names resolving to no registry |
| `mentions.parquet` | atomic | every mention, with line, column and snippet |
| `files.parquet` | atomic | the provenance spine |
| `declared_dependencies.parquet` | atomic | what manifests declare: shipped, locked or asked for |
| `environment_signals.parquet` | atomic | R, Python and Stata versions, and the OS, where a file says |
| `environment_coverage.json` | denominator | deposits stating each signal, over deposits that could |

The last three are sparse on purpose. Most deposits record nothing about the
environment they ran in, so read them with `environment_coverage.json`
alongside: it gives, per signal, the deposits that said something over the
deposits that were in a position to.

Then `make release-tables` cuts the small tracked copy the site and the paper
both read, and `make site` builds the published pages from it.

## How detection works

| language | mechanism |
|---|---|
| R | `tree-sitter` syntax walk |
| Python | stdlib `ast`, with a `tokenize` fallback for Python 2 |
| Stata | purpose-built lexer + an SSC command→package index |
| `.Rmd` / `.qmd` / `.Rnw` | knitr chunk splitter, routed per engine |
| `.ipynb` | per cell, kernel language read from metadata |

Strings and comments are excluded by structure instead of being stripped out
first. In a syntax tree, `cat("run library(dplyr)")` is a call whose argument
is a string node, so those bytes never form a call and a walk over call nodes
cannot reach them.

Stata needed an artifact that did not exist. R has CRAN and Python has PyPI;
Stata has no machine-readable registry. Softverse reconstructs a
command-to-package index from SSC distribution manifests, 3,967 packages and
7,468 commands, both counted excluding internal helper files, and releases it.
Without that index Stata cannot be measured at all, which is why work like
this leaves out the language this corpus uses most.

## Design commitments

These exist because the previous version of this project got each one wrong, in
ways that silently invalidated its published numbers.

- Join keys are checked for nulls at the write boundary rather than declared
  in a schema, because pyarrow treats `nullable=False` as documentation.
- Every stage reconciles: `files_total == analyzed + vendored + duplicate +
  skipped + unparseable`, asserted in code.
- Parse status is recorded per file, so "no packages here" and "could not read
  this" are never stored as the same value.
- Every row carries its provenance: extractor version, grammar version,
  registry lock, source line and snippet, so a published number traces back
  to the text it came from.
- Registries are pinned, so a resolution does not depend on the day it ran.
- Skipped archives and journals that could not be located are recorded as
  rows, with their sizes and reasons, rather than dropped.

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

## Author

Gaurav Sood

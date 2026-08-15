# Changelog

## 2.0.0 (unreleased)

A rewrite. Version 1 published counts that its own pipeline could not account
for: between the source archive and the published CSV, 26,681 files and 4,189
repositories disappeared with nothing to notice, and the final `groupby`
dropped 71,069 of 71,069 rows on a null key. Version 2 is built so that each
stage has to reconcile with the last, and it fails loudly when it does not.

### What it measures

Per-package counts of validated use: how often each R, Python and Stata
package is loaded by the code deposited with a published paper, at journals
whose data-and-code policy an editor verifies. 3,223 packages across 8,349
deposits from 69 journal collections, pooled over Zenodo's economics
collections and Harvard Dataverse's political science journals, with the
per-repository split on every row.

### Added

- A Stata command-to-package index reconstructed from SSC distribution
  manifests, 3,967 packages and 7,468 commands, released separately under CC0
  at [10.5281/zenodo.21926100](https://doi.org/10.5281/zenodo.21926100).
  Nothing equivalent existed, and without it Stata cannot be measured.
- Extraction for literate documents: knitr chunks routed per engine, and
  Jupyter cells parsed against the kernel language from the notebook metadata.
- A published lookup interface and the released tables at
  <https://recite.github.io/softverse/>.
- Validation against R's own parser as an oracle, precision and recall of
  1.0000 over 25,969 name-file pairs, plus cross-tool agreement with `renv`.
- Every mention carries the extractor version, grammar version, registry lock,
  source line and snippet, so a published number traces back to its text.

### Fixed

Defects found and corrected during the rebuild, each of which had moved a
published number:

- Notebooks with a kernel we cannot parse were handed to the Python
  extractor. `import Ipopt` is valid Python, so a Julia notebook parsed
  cleanly and put a Julia package into a corpus in which no `.jl` file is
  ever read.
- The at-risk denominator counted file languages, so a deposit whose only
  Python lived in a notebook entered the numerator of every Python package it
  loaded and the denominator of none. `pandas` read 93.9% against a true
  82.6%.
- Vendored `.ado` files shipped inside deposits were read as locally defined
  programs, so a deposit that bundled `reghdfe.ado` credited its own calls to
  itself rather than to the package. 21,004 mentions across 279 packages.
- Stata builtins were curated from memory. Checking every command in the
  corpus against StataCorp's help server moved 165 commands out of
  "resolves to nothing" and stopped 6,742 mentions being credited to an SSC
  package that merely ships a file of the same name.
- The archive size cap rejected deposits on their declared uncompressed size,
  losing 12 deposits and 31 GB of archives over 26.5 MB of code.
- Relative Python imports (`from .models import ...`) resolved against PyPI,
  crediting 289 of them to real distributions that share a name with somebody's
  own file.

### Changed

- Zenodo and Harvard Dataverse are tallied in one pass rather than two, so the
  per-package counts cover both disciplines. They previously covered economics
  alone.
- The console entry point is gone. It named `softverse.cli:cli`, which has
  never existed, so `pip install softverse` installed a command that raised
  `ModuleNotFoundError` on any invocation. Softverse is a library.
- Eight declared dependencies that nothing imports are removed: `tenacity`,
  `nbformat`, `pyyaml`, `click`, `tomli-w`, `zenodo-client`, `pydataverse`
  and `tabulate`.
- Project URLs are declared in a `[project.urls]` table. The bare
  `homepage`/`repository` keys used before are not PEP 621, so the wheel
  carried no `Project-URL` metadata and the PyPI page linked nowhere.
- `__version__` is read from installed package metadata instead of being
  typed into `__init__.py` alongside the copy in `pyproject.toml`.
- `EXTRACTOR_VERSION` is documented as independent of the release version,
  because it stamps data rows and must move only when extraction changes.

## 0.1.0

The first release, from the version 1 pipeline. Superseded.

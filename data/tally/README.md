# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**4,565 packages · 13,245 deposits with analyzable
code · 13,985 deposits collected · built 2026-09-18**

A count here is the number of deposits whose code loads the package. Adding
one to it takes a paper published at a journal that checks its authors' code,
which is what makes these counts harder to inflate than download counts. The
[project page](https://recite.github.io/softverse/) makes that case; this
file documents what is in the tables and how to read them.

## Scope

The deposits come from two repositories that hold different disciplines.
Zenodo's verified collections are economics, and Harvard Dataverse's journal
collections are mostly political science.

| repository | deposits | with analyzable code |
|---|---:|---:|
| Zenodo (economics) | 1,543 | 1,358 |
| Harvard Dataverse (political science) | 12,442 | 11,887 |
| **total** | **13,985** | **13,245** |

Counts pool the two. `usage_by_package.csv` also carries the split, in
`n_deposits_zenodo` and `n_deposits_dataverse`, because the two are
very different sizes and a pooled figure alone would hide that.

Both repositories were collected with the same rules in 2026: every code,
notebook and knitr file and every dependency manifest, with the deposit's own
directories, and code recovered from archives too large to download. An
archive that could not be read is counted in `summary.json` rather than
dropped. The two collections still differ in discipline, which the per-source
columns keep visible.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | 4,565 | per-package deposit and call counts, pooled and split |
| `usage_by_package_year.csv` | 14,670 | the same by deposit year |
| `usage_by_collection.csv` | 22,838 | the same per journal or community |
| `usage_by_function.csv` | 20,942 | package → function, where the source names one |
| `unknown_names.csv` | 2,787 | names called in code that resolve to no registry |
| `remote_installs.csv` | 473 | what deposits install from outside their registry, and from where |
| `downloads_vs_use.csv` | 43,918 | each package's validated use beside its registry's download count |
| `language_presence.csv` | 24 | deposits containing each language, per repository |
| `mentions.parquet` | 14,853,082 | every mention: package, function, file, line, snippet |
| `files.parquet` | 447,289 | the provenance spine every mention joins to |
| `declared_dependencies.parquet` | 33,874 | what manifests declare: shipped, locked or asked for |
| `environment_signals.parquet` | 19,983 | R, Python and Stata versions, and the OS, where a file says |
| `environment_coverage.json` | | deposits stating each signal, over deposits that could |
| `summary.json` | | corpus counts the tables are shares of |

`mentions.parquet` is the record every count above is a sum of, and it is
here so a reader who disagrees with a decision made upstream can recount
without re-parsing 200,000 files. It is 94 MB; the CSVs are 150 KB.

`declared_dependencies.parquet` and `environment_signals.parquet` are sparse
and answer a different question: not what the code loads but what version of
it the deposit shipped, and what ran it. Most deposits say nothing at all, so
read these next to `environment_coverage.json`, which gives per signal the
deposits that said something over the deposits that were in a position to.

### `usage_by_package.csv`

- `package`, `language`, `ecosystem`: the resolved package and its registry
- `n_deposits`: deposits loading it, counted once per deposit
- `n_files`, `n_mentions`: files, and raw calls
- `n_deposits_at_risk`: the denominator for `share_of_deposits`, meaning deposits
  that hold an analyzable file in that language, or that yielded a reference
  in it. The second clause is what covers literate documents, where the file
  is a notebook and the code inside it is Python. It differs by language, so a
  Stata share and an R share are not shares of the same thing
- `share_of_deposits`: `n_deposits` divided by `n_deposits_at_risk`

### `unknown_names.csv`

Names that code *uses* and that resolve to no registry, unfiltered. Install
and inquiry lines are excluded: `ssc install x` states a dependency and is not
a call. Some rows are false positives, and some are programs a deposit defines
for itself. Pruning the list by hand would put a judgement call inside a file
whose value is that you can check every row of it.

- `name`, `language`
- `n_deposits`, `n_mentions`: deposits using the name, and raw uses. Rank by
  deposits: one deposit calling something six hundred times is one user of it
- `n_deposits_defining`: how many deposits define a Stata program of this
  name for themselves. A name many authors independently give a helper is more
  likely one here too, with a `program define` the lexer did not reach, than
  it is software nobody indexed

### `remote_installs.csv`

What deposits fetch from somewhere other than their language's registry:
`remotes::install_github("user/repo")`, `net install x, from(URL)`,
`pip install git+https://...`. It is the only record a deposit leaves of where
off-registry software lives.

- `name`, `language`, `host`: the package and the host it is fetched from
- `in_registry`: the registry lists the name anyway, so this is a development
  version of a registered package rather than software the registry lacks
- `n_deposits_installing`, `n_deposits_loading`: deposits with the install
  line, and those among them that go on to use the package

### `downloads_vs_use.csv`

Validated use beside download counts, for every package that SSC, CRAN or
PyPI has a count for *or* that the corpus uses. The download counts are not
ours: SSC's monthly hits behind `ssc hot`, CRAN's from one mirror through
`cranlogs`, and PyPI's for its fifteen thousand most downloaded projects from
`hugovk/top-pypi-packages`, each pinned by date and digest. The windows
differ, so compare ranks within one registry, not counts across them.

- `package`, `language`, `ecosystem`
- `n_deposits`: deposits using the package; 0 for a counted package this
  corpus never uses
- `downloads`: the registry's count; empty for a used package it has none for
- `in_registry_counts`: the row belongs to the registry's own list of counted
  packages, as against a count fetched because the corpus uses the package
- `reverse_dependencies`, `in_task_view`: CRAN only. How many packages
  install this one as a dependency, directly or through others, and whether
  CRAN's *Econometrics* or *Causal Inference* task view lists it

## Licence

CC0.

## Regenerating

```bash
uv run python scripts/build_tally.py     # needs the collected corpus
uv run python scripts/release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).

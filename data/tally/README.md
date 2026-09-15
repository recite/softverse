# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**4,447 packages · 13,245 deposits with analyzable
code · 13,985 deposits collected · built 2026-09-15**

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
| `usage_by_package.csv` | 4,447 | per-package deposit and call counts, pooled and split |
| `usage_by_package_year.csv` | 14,154 | the same by deposit year |
| `usage_by_collection.csv` | 22,203 | the same per journal or community |
| `usage_by_function.csv` | 21,849 | package → function, where the source names one |
| `unknown_names.csv` | 12,359 | names called in code that resolve to no registry |
| `language_presence.csv` | 24 | deposits containing each language, per repository |
| `mentions.parquet` | 14,381,788 | every mention: package, function, file, line, snippet |
| `files.parquet` | 447,289 | the provenance spine every mention joins to |
| `declared_dependencies.parquet` | 33,848 | what manifests declare: shipped, locked or asked for |
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

Names that appear in the code and resolve to no registry, unfiltered. Some
are false positives: `str` is a Stata type, and some are programs a deposit
defines for itself. The list also holds real and heavily used software that
no registry indexes, `grc1leg` being the clearest case at 1,971
calls. Pruning the list by hand would put a judgement call inside a file
whose value is that you can check every row of it.

## Licence

CC0.

## Regenerating

```bash
uv run python scripts/build_tally.py     # needs the collected corpus
uv run python scripts/release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).

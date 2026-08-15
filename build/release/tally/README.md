# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**3,223 packages · 8,349 deposits with analyzable
code · 8,685 deposits collected · built 2026-08-15**

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
| Zenodo (economics) | 1,454 | 1,272 |
| Harvard Dataverse (political science) | 7,231 | 7,077 |
| **total** | **8,685** | **8,349** |

Counts pool the two. `usage_by_package.csv` also carries the split, in
`n_deposits_zenodo` and `n_deposits_dataverse_legacy`, because the two are
very different sizes and a pooled figure alone would hide that.

The Dataverse deposits come from a January 2024 scrape that kept `.do`, `.r`
and `.py` files and nothing else, so a package used mainly inside a notebook
or a knitr document is under-counted on that side. The release checks this
by recomputing the whole ranking on those three extensions alone and fails if
the order moves. The two collections are also two years apart, which
`first_year` and `last_year` will show.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | 3,223 | per-package deposit and call counts, pooled and split |
| `usage_by_package_year.csv` | 9,738 | the same by deposit year |
| `usage_by_collection.csv` | 13,880 | the same per journal or community |
| `unknown_names.csv` | 7,686 | names called in code that resolve to no registry |
| `language_presence.csv` | 15 | deposits containing each language, per repository |
| `summary.json` | | corpus counts the tables are shares of |

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
no registry indexes, `grc1leg` being the clearest case at 1,224
calls. Pruning the list by hand would put a judgement call inside a file
whose value is that you can check every row of it.

## Licence

CC0.

## Regenerating

```bash
uv run python scripts_build_tally.py     # needs the collected corpus
uv run python scripts_release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).

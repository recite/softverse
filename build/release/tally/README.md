# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**3,223 packages · 8,349 deposits with analyzable
code · 8,685 deposits collected · built 2026-08-15**

A download says a file was fetched. A mention in prose says an author
remembered to name something. Neither says the software ran. These counts say
a package was loaded by code that shipped with a paper, which is the strongest
of the three claims and the one worth crediting against.

## Scope

Two repositories, two disciplines, pooled. Zenodo's verified collections are
economics; Harvard Dataverse's journal collections are mostly political
science. Every count here is the pooled total, and `usage_by_package.csv`
carries the split beside it in `n_deposits_zenodo` and
`n_deposits_dataverse_legacy`, because the halves are very different sizes
and a pooled number with no breakdown asks you to take the composition on
trust.

| repository | deposits | with analyzable code |
|---|---:|---:|
| Zenodo (economics) | 1,454 | 1,272 |
| Harvard Dataverse (political science) | 7,231 | 7,077 |
| **total** | **8,685** | **8,349** |

Two asymmetries worth knowing before you use the split. The Dataverse half is
a January 2024 scrape that kept only `.do`, `.r` and `.py`, so a package used
mainly inside notebooks or knitr documents is under-counted there; the
ranking is checked against a recomputation restricted to those three
extensions, and the release fails if it moves. And the halves are different
vintages, 2024 against 2026, which `first_year` and `last_year` will show.

Counts are static reference in deposited code. A package that is loaded but
never reached at runtime still counts, and a package invoked through a string
that is assembled at runtime does not.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | 3,223 | per-package deposit and call counts |
| `unknown_names.csv` | 7,686 | names called in code that resolve to no registry |
| `language_presence.csv` | 15 | deposits containing each language |
| `summary.json` | | corpus counts the tables are shares of |

### `usage_by_package.csv`

- `package`, `language`, `ecosystem` — the resolved package and its registry
- `n_deposits` — deposits loading it, counted once per deposit
- `n_files`, `n_mentions` — files and raw calls
- `n_deposits_at_risk` — the denominator `share_of_deposits` uses: deposits
  that hold an analyzable file in that language, or that yielded a reference
  in it. The second clause is what covers literate documents, where the file
  is a notebook and the code inside it is Python. It differs by language, so a
  Stata share and an R share are not shares of the same thing
- `share_of_deposits` — `n_deposits / n_deposits_at_risk`

### `unknown_names.csv`

Deliberately unfiltered. Some entries are false positives: `str` is a Stata
type, and some names are programs defined inside the deposit itself. But the
list also holds real, heavily used software that no registry indexes,
`grc1leg` being the clearest case at 1,224 calls, distributed from
StataCorp's own site and invisible to any credit system built on registries.
Pruning it by hand would bury a judgement call inside a file whose point is
that you can check it.

## Licence

CC0.

## Regenerating

```bash
uv run python scripts_build_tally.py     # needs the collected corpus
uv run python scripts_release_tally.py
```

Produced by [softverse](https://github.com/recite/softverse).

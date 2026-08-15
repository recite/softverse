# Validated use: per-package counts

How often each R, Python and Stata package is loaded by the code deposited
with published papers, at journals whose data-and-code policy the Social
Science Data Editors record as *actively verified*.

**1,806 packages · 1,272 deposits with analyzable
code · 1,454 deposits collected · built 2026-08-15**

A download says a file was fetched. A mention in prose says an author
remembered to name something. Neither says the software ran. These counts say
a package was loaded by code that shipped with a paper, which is the strongest
of the three claims and the one worth crediting against.

## Scope

The frame is the Zenodo half: 1,454 deposits from the verified
economics collections. Harvard Dataverse's journal collections are political
science and are tallied separately, so they contribute to the cross-language
comparison in the paper and not to the per-package counts here. A package
heavily used in political science and rarely in economics will look smaller
here than it is.

Counts are static reference in deposited code. A package that is loaded but
never reached at runtime still counts, and a package invoked through a string
that is assembled at runtime does not.

## Files

| file | rows | contents |
|---|---:|---|
| `usage_by_package.csv` | 1,806 | per-package deposit and call counts |
| `unknown_names.csv` | 3,590 | names called in code that resolve to no registry |
| `language_presence.csv` | 12 | deposits containing each language |
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
`grc1leg` being the clearest case at 530 calls, distributed from
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

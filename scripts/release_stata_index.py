"""Package the Stata command→package index for release.

    uv run python scripts/release_stata_index.py

Writes `build/release/stata-index/`: Parquet plus a CSV mirror, the ambiguous
commands, the curated builtin list, a data descriptor, and a frictionless
datapackage.

The index is released on its own because its value does not depend on our
corpus. R has CRAN and Python has PyPI; Stata has no machine-readable mapping
from a command to the package providing it, which is a large part of why work
like this omits Stata -- the language social science replication code uses most.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb

from softverse.config import PATHS
from softverse.registries.load import INDEX_FILE
from softverse.registries.lock import pinned_directory, pinned_names, read_lock
from softverse.stata.builtins import builtins
from softverse.stata.index import TIERS, ambiguous_commands, credited

OUT = PATHS.root / "build" / "release" / "stata-index"

_FRICTIONLESS = {"VARCHAR": "string", "BOOLEAN": "boolean", "DATE": "date"}
SOURCE = pinned_directory("stata_index", read_lock(), payload=INDEX_FILE) / INDEX_FILE

DESCRIPTOR = """\
# Stata command → package index

A machine-readable mapping from Stata command names to the packages that provide
them, reconstructed from the distribution manifests of the three archives Stata's
`net install` reads: the Statistical Software Components (SSC) archive, the
*Stata Journal*, and its predecessor the *Stata Technical Bulletin* (STB).

**{n_mappings:,} mappings · {n_packages:,} packages · {n_commands:,} user commands · snapshot {snapshot}**

Packages and commands are both counted excluding helper files (`is_helper`).

## Why this exists

R has CRAN and Python has PyPI: given an import, you can look up the package. For
Stata there is no equivalent public mapping from a command to its package, which
makes Stata code effectively unmeasurable at scale and is a large part of why
studies of research software omit it — despite Stata being the language social
science replication code uses most.

## Method

Every package on the SSC mirror ships a `.pkg` manifest listing the files it
distributes:

```
d 'ESTOUT': module to make regression tables
d Distribution-Date: 20260413
f estout.ado
f esttab.ado
f eststo.ado
```

Crawling all `{{a-z,_}}` directories and parsing those manifests yields the
command→package mapping, including the many-commands-per-package case that a
package-name list cannot express: `esttab`, `eststo`, `estadd` and `estpost` all
belong to `estout`.

The *Stata Journal* and the STB publish their software in the same format, one
directory per issue, each with a `stata.toc` naming its packages. `renvars` is
STB-60 `dm88`; `xtserial` is SJ 3-2 `st0039`. Neither is on SSC, and an index of
SSC alone reports both as belonging to no archive.

## Four caveats, which change how you should use this

**1. A shipped file is not necessarily a command.** An `f foo.ado` line says a
package distributes a file, not that it exposes a user command called `foo`.
`reghdfe` ships `reghdfe_p.ado` (a predict helper), `reghdfe_estat.ado` and
`reghdfe_footnote.ado`; `estout` ships `_eststo.ado`, an internal subroutine.
The `evidence` column distinguishes `filename` (inferred from the manifest) from
`program_define` (confirmed by parsing the `.ado`), and `is_helper` flags names
matching internal-subroutine conventions. **Filter `is_helper = false` unless you
want the internals.**

**2. This is a current snapshot, not a history.** A command that was
user-written in 2010 and later became official Stata resolves here against its
status today. If you are studying change over time, this will manufacture
trends: a package appears to vanish exactly when its command is absorbed into
official Stata. Use `distribution_date` as a partial guard and treat
time-inconsistent mappings as unresolved.

**3. A journal package bundles what its example needs.** SJ 14-4 `st0357`, a Cox
calibration tool, ships a copy of `grc1leg.ado`. `is_documented` is true when
the package also ships a help file of the same name, which a package does for
what it publishes and not for what it borrows. **For `source` other than `ssc`,
filter `is_documented = true`**, or `grc1leg` becomes a survival-analysis command.

**4. Ambiguity is preserved, not resolved.** {n_ambiguous} commands are claimed
by more than one package; they are listed in `ambiguous.json` rather than
assigned to a winner. Forcing a choice would bury classification error.

## Files

| file | contents |
|---|---|
| `stata_command_index.parquet` | the index (authoritative) |
| `stata_command_index.csv` | identical, as text |
| `ambiguous.json` | commands claimed by more than one package |
| `builtins.json` | official Stata commands, each verified against StataCorp's help server |
| `datapackage.json` | frictionless schema |

### Columns

- `command` — the command as typed
- `package` — the package providing it; a journal update (`st0085_2`) is filed
  under the package it updates (`st0085`)
- `source` — `ssc`, `stata_journal` or `stb`
- `issue` — the journal issue (`sj14-2`, `stb60`); empty for SSC
- `evidence` — `filename`, or `package_only` for a package that ships no `.ado`
  (Mata libraries, graph schemes), listed so `ssc install moremata` names
  something known
- `is_helper` — internal subroutine rather than a user command
- `is_documented` — the package ships a help file of the same name
- `author`, `distribution_date` — from the manifest
- `snapshot_date` — when this index was built

## Worked example

Classifying a Stata command is a three-step decision, and the order matters:

```python
import duckdb, json
idx = duckdb.connect()
q = ("SELECT DISTINCT source, package FROM 'stata_command_index.parquet' "
     "WHERE lower(command)=? AND NOT is_helper AND (source='ssc' OR is_documented)")
TIERS = ["ssc", "stata_journal", "stb", "net"]   # the order `classify` consults

builtins = set(json.load(open("builtins.json"))["forms"])

def classify(command, local_programs=()):
    c = command.lower()
    if c in local_programs:            # defined in the deposit itself
        return "local"
    if c in builtins:                  # official Stata
        return "builtin"
    hits = idx.execute(q, [c]).fetchall()
    for tier in TIERS:                 # journal authors mirror to SSC: first tier wins
        packages = sorted(p for s, p in hits if s == tier)
        if packages:
            return packages[0] if len(packages) == 1 else "ambiguous"
    return "unknown"

classify("esttab")    # -> 'estout'
classify("reghdfe")   # -> 'reghdfe'
classify("regress")   # -> 'builtin'
classify("renvars")   # -> 'dm88', from the Stata Technical Bulletin
```

The names in `builtins.json` were checked against StataCorp's public help
server rather than curated from memory, which is the difference between "we
believe these are official" and "we asked". The server answers only for
commands with a help page, so the list also holds every `.ado` file name in a
Stata installation's base directory, which is where `_dots` and `tset` are. Pages from the `[U]` and `[FN]`
manuals are excluded: they document system variables and functions such as
`_n` and `e()`, which are not commands.

The list is still incomplete, and that costs *recall*, never *precision*,
because resolution is inclusive: a command counts as a package only if the
index has it. A missing builtin lands in `unknown`, never in a package.

## Licence

CC0. The underlying manifests are public metadata from the SSC archive at
Boston College and from StataCorp's Stata Journal and STB software archives.

## Regenerating

```bash
uv run python scripts/build_stata_index.py
```

Produced by [softverse](https://github.com/recite/softverse).
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    rows = con.execute(f"SELECT * FROM '{SOURCE}'").fetchall()
    columns = [d[0] for d in con.description]
    con.execute(
        f"COPY (SELECT * FROM '{SOURCE}') TO '{OUT / 'stata_command_index.parquet'}' (FORMAT PARQUET)"
    )
    con.execute(
        f"COPY (SELECT * FROM '{SOURCE}') TO '{OUT / 'stata_command_index.csv'}' (HEADER, DELIMITER ',')"
    )

    dict_rows = [dict(zip(columns, r, strict=True)) for r in rows]
    ambiguous = ambiguous_commands(dict_rows)
    (OUT / "ambiguous.json").write_text(json.dumps(ambiguous, indent=1, sort_keys=True))

    # Ship the verified list, not the curated one. Every name in it was
    # checked against StataCorp's help server, which is the difference between
    # "we believe these are official" and "we asked".
    official_snapshot = (
        PATHS.root / "registries" / "snapshots" / "stata_official" / "official.json"
    )
    builtin_set = builtins(verified_snapshot=official_snapshot)
    base_ado = pinned_names("stata_base_ado", read_lock())
    (OUT / "builtins.json").write_text(
        json.dumps(
            {
                "source": builtin_set.source,
                "note": (
                    "Each name was checked against StataCorp's help server "
                    "(help.cgi), not curated from memory, and the file names "
                    "in a Stata installation's base ado directory are added "
                    "for the commands that ship without a help page. Still "
                    "incomplete, "
                    "but resolution is inclusive, so a missing builtin costs "
                    "recall and never precision: it lands in `unknown`, never "
                    "in a package. Pages from the [U] and [FN] manuals are "
                    "excluded -- they document system variables and functions "
                    "such as `_n` and `e()`, which are not commands."
                ),
                "canonical": sorted(builtin_set.canonical),
                # The list the tally resolved against: the names above, plus
                # every `.ado` in Stata's base directory, which is where the
                # commands with no help page are.
                "forms": sorted(builtin_set.forms | base_ado),
                "n_from_base_ado_listing": len(base_ado - builtin_set.forms),
            },
            indent=1,
        )
    )

    n_mappings = len(dict_rows)
    # Both counts apply the same helper filter. They did not, which made the
    # headline read "3,992 packages · 7,468 user commands" -- packages counted
    # including 25 whose only contribution is internal subroutines, commands
    # counted excluding them. Two filters in one sentence is the kind of
    # inconsistency a reader is right to distrust the rest of the file over.
    n_packages = len({r["package"] for r in dict_rows if not r["is_helper"]})
    n_commands = len({r["command"] for r in dict_rows if not r["is_helper"]})
    snapshot = max(str(r["snapshot_date"]) for r in dict_rows)

    (OUT / "README.md").write_text(
        DESCRIPTOR.format(
            n_mappings=n_mappings,
            n_packages=n_packages,
            n_commands=n_commands,
            n_ambiguous=len(ambiguous),
            snapshot=snapshot,
        )
    )

    (OUT / "datapackage.json").write_text(
        json.dumps(
            {
                "name": "stata-command-package-index",
                "title": "Stata command to package index",
                "licenses": [
                    {
                        "name": "CC0-1.0",
                        "path": "https://creativecommons.org/publicdomain/zero/1.0/",
                    }
                ],
                "created": datetime.now(tz=UTC).isoformat(),
                "resources": [
                    {
                        "name": "stata_command_index",
                        "path": "stata_command_index.csv",
                        "format": "csv",
                        # Read off the table, so a new column cannot be
                        # shipped and left out of its own description.
                        "schema": {
                            "fields": [
                                {
                                    "name": name,
                                    "type": _FRICTIONLESS.get(kind, "string"),
                                }
                                for name, kind in con.execute(
                                    f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM '{SOURCE}')"
                                ).fetchall()
                            ]
                        },
                    }
                ],
            },
            indent=1,
        )
    )

    # Verify against the *exported* files, not the in-memory objects, so what
    # ships is what was checked.
    check = duckdb.connect()
    exported = OUT / "stata_command_index.parquet"
    rows_out = check.execute(f"SELECT * FROM '{exported}'").fetchall()
    shipped = [dict(zip(columns, r, strict=True)) for r in rows_out]

    def winners(command: str) -> list[str]:
        hits = [r for r in shipped if r["command"].lower() == command and credited(r)]
        for tier in TIERS:
            if found := sorted({r["package"] for r in hits if r["source"] == tier}):
                return found
        return []

    problems = []
    for command, expected in (("esttab", "estout"), ("reghdfe", "reghdfe")):
        if winners(command) != [expected]:
            problems.append(f"{command} -> {winners(command)}, expected {expected}")
    helper = check.execute(
        f"SELECT is_helper FROM '{exported}' WHERE command='_eststo'"
    ).fetchone()
    if not (helper and helper[0]):
        problems.append("_eststo should be flagged as a helper")
    if winners("regress"):
        problems.append("regress is official Stata and should not be in the index")
    # The two rules the journal tiers exist for, checked on what ships.
    if winners("renvars") != ["dm88"]:
        problems.append("renvars should resolve to STB/SJ dm88")
    if "st0357" in winners("grc1leg"):
        problems.append("grc1leg credited to st0357, which only bundles a copy")

    csv_rows = sum(1 for _ in (OUT / "stata_command_index.csv").open()) - 1
    if csv_rows != n_mappings:
        problems.append(f"CSV has {csv_rows} rows, Parquet has {n_mappings}")

    print(f"wrote {OUT}")
    print(
        f"  {n_mappings:,} mappings · {n_packages:,} packages · {n_commands:,} commands"
    )
    print(f"  {len(ambiguous)} ambiguous · snapshot {snapshot}")
    print(f"  builtins: {len(builtin_set.forms)} forms ({builtin_set.source})")
    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nverified against the exported files: esttab->estout, reghdfe->reghdfe,")
    print("_eststo flagged helper, regress absent, CSV and Parquet agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

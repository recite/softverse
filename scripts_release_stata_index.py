"""Package the Stata command→package index for release.

    uv run python scripts_release_stata_index.py

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
from softverse.stata.builtins import builtins
from softverse.stata.index import ambiguous_commands

OUT = PATHS.root / "build" / "release" / "stata-index"
SOURCE = PATHS.root / "registries" / "snapshots" / "ssc" / "stata_command_index.parquet"

DESCRIPTOR = """\
# Stata command → package index

A machine-readable mapping from Stata command names to the packages that provide
them, reconstructed from the Statistical Software Components (SSC) archive's own
distribution manifests.

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

## Three caveats, which change how you should use this

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

**3. Ambiguity is preserved, not resolved.** {n_ambiguous} commands are claimed
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
- `package` — the SSC package providing it
- `source` — `ssc`
- `evidence` — `filename` or `program_define`
- `is_helper` — internal subroutine rather than a user command
- `author`, `distribution_date` — from the manifest
- `snapshot_date` — when this index was built

## Worked example

Classifying a Stata command is a three-step decision, and the order matters:

```python
import duckdb, json
idx = duckdb.connect()
q = "SELECT package FROM 'stata_command_index.parquet' WHERE lower(command)=? AND NOT is_helper"

builtins = set(json.load(open("builtins.json"))["forms"])

def classify(command, local_programs=()):
    c = command.lower()
    if c in local_programs:            # defined in the deposit itself
        return "local"
    if c in builtins:                  # official Stata
        return "builtin"
    hits = [r[0] for r in idx.execute(q, [c]).fetchall()]
    return hits[0] if len(hits) == 1 else ("ambiguous" if hits else "unknown")

classify("esttab")    # -> 'estout'
classify("reghdfe")   # -> 'reghdfe'
classify("regress")   # -> 'builtin'
```

Every name in `builtins.json` was checked against StataCorp's public help
server rather than curated from memory, which is the difference between "we
believe these are official" and "we asked". Pages from the `[U]` and `[FN]`
manuals are excluded: they document system variables and functions such as
`_n` and `e()`, which are not commands.

The list is still incomplete, and that costs *recall*, never *precision*,
because resolution is inclusive: a command counts as a package only if the
index has it. A missing builtin lands in `unknown`, never in a package.

## Licence

CC0. The underlying manifests are public metadata from the SSC archive at
Boston College.

## Regenerating

```bash
uv run python -c "from softverse.stata.index import build_index, write_index; \\
  from pathlib import Path; write_index(build_index(), Path('out'))"
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
    (OUT / "builtins.json").write_text(
        json.dumps(
            {
                "source": builtin_set.source,
                "note": (
                    "Each name was checked against StataCorp's help server "
                    "(help.cgi), not curated from memory. Still incomplete, "
                    "but resolution is inclusive, so a missing builtin costs "
                    "recall and never precision: it lands in `unknown`, never "
                    "in a package. Pages from the [U] and [FN] manuals are "
                    "excluded -- they document system variables and functions "
                    "such as `_n` and `e()`, which are not commands."
                ),
                "canonical": sorted(builtin_set.canonical),
                "forms": sorted(builtin_set.forms),
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
                        "schema": {
                            "fields": [
                                {"name": "command", "type": "string"},
                                {"name": "package", "type": "string"},
                                {"name": "source", "type": "string"},
                                {"name": "evidence", "type": "string"},
                                {"name": "is_helper", "type": "boolean"},
                                {"name": "author", "type": "string"},
                                {"name": "distribution_date", "type": "date"},
                                {"name": "first_seen_date", "type": "date"},
                                {"name": "snapshot_date", "type": "date"},
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
    q = f"SELECT package FROM '{exported}' WHERE lower(command)=? AND NOT is_helper"
    problems = []
    for command, expected in (("esttab", "estout"), ("reghdfe", "reghdfe")):
        got = [r[0] for r in check.execute(q, [command]).fetchall()]
        if expected not in got:
            problems.append(f"{command} -> {got}, expected {expected}")
    helper = check.execute(
        f"SELECT is_helper FROM '{exported}' WHERE command='_eststo'"
    ).fetchone()
    if not (helper and helper[0]):
        problems.append("_eststo should be flagged as a helper")
    if check.execute(q, ["regress"]).fetchall():
        problems.append("regress is official Stata and should not be in the index")

    csv_rows = sum(1 for _ in open(OUT / "stata_command_index.csv")) - 1
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

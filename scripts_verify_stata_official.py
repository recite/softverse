"""Check corpus Stata command names against StataCorp's help server.

    uv run python scripts_verify_stata_official.py

Two questions, and the first is the one that can change published numbers.

**Precision.** Every corpus command currently attributed to an SSC package is
checked for also being official Stata. A name that is both is a collision: when
a script says `pca`, it almost certainly means StataCorp's `pca`, not an SSC
package that happens to ship a file of that name. Collisions are over-
attribution, and they inflate exactly the package counts the paper reports.

**Recall.** Every command that resolved to nothing is checked for being
official. These cost no precision -- inclusive resolution means an unmatched
name is never attributed to a package -- but they inflate the "resolves to no
registry" rate, which the paper reports as a coverage statistic. An unknown
share that is mostly `ds` and `summarize` means something very different from
one that is mostly unrecognised third-party packages.

Answers are cached in `registries/snapshots/stata_official/official.json`, so
re-running is cheap and does not re-crawl a vendor's documentation server.
"""

from __future__ import annotations

import json

import duckdb

from softverse.config import PATHS
from softverse.stata.official import verify

SNAPSHOT = PATHS.root / "registries" / "snapshots" / "stata_official" / "official.json"
MENTIONS = PATHS.root / "build" / "tally" / "mentions.parquet"

#: Names seen in only one deposit are a long tail of typos, macro fragments and
#: one-off local programs. Checking all 1,951 of them would quadruple the crawl
#: to move the reported rate by a fraction of a point, so the cut is stated
#: rather than hidden -- the unchecked remainder is reported below.
MIN_DEPOSITS = 2


def main() -> int:
    if not MENTIONS.exists():
        print(f"no tally at {MENTIONS}; run scripts_build_tally.py first")
        return 1

    con = duckdb.connect()
    attributed = {
        r[0]
        for r in con.execute(
            f"""SELECT DISTINCT normalized_name FROM '{MENTIONS}'
                WHERE language='stata' AND resolution IN ('known_current','ambiguous')"""
        ).fetchall()
    }
    unknown = {
        r[0]: r[1]
        for r in con.execute(
            f"""SELECT normalized_name, count(DISTINCT dataset_doi) FROM '{MENTIONS}'
                WHERE language='stata' AND resolution='unknown'
                GROUP BY 1 HAVING count(DISTINCT dataset_doi) >= {MIN_DEPOSITS}"""
        ).fetchall()
    }
    n_unknown_total = con.execute(
        f"""SELECT count(DISTINCT normalized_name) FROM '{MENTIONS}'
            WHERE language='stata' AND resolution='unknown'"""
    ).fetchone()[0]

    targets = attributed | set(unknown)
    print(f"checking {len(targets)} names against StataCorp's help server")
    print(f"  {len(attributed)} attributed to an SSC package (precision)")
    print(f"  {len(unknown)} unresolved in >={MIN_DEPOSITS} deposits (recall)")
    print(f"  skipping {n_unknown_total - len(unknown)} unresolved singletons")
    print("at one request per two seconds this takes a while\n")

    snapshot = verify(targets, SNAPSHOT)

    collisions = sorted(n for n in attributed if n in snapshot)
    recoverable = sorted(
        (n for n in unknown if n in snapshot), key=lambda n: -unknown[n]
    )
    unanswered = sorted(targets - set(snapshot.checked))

    print(f"\n== precision: {len(collisions)} attributed names are also official ==")
    if collisions:
        rows = con.execute(
            f"""SELECT normalized_name, any_value(resolved_package),
                       count(DISTINCT dataset_doi) FROM '{MENTIONS}'
                WHERE language='stata' AND normalized_name IN ({
                ",".join("?" * len(collisions))
            })
                GROUP BY 1 ORDER BY 3 DESC""",
            collisions,
        ).fetchall()
        for name, package, deposits in rows:
            # `ambiguous` rows carry no single package, so this is None rather
            # than a string.
            print(f"  {name:<18} -> {package or '(ambiguous)':<18} {deposits:>4} dep")

    print(f"\n== recall: {len(recoverable)} unresolved names are official Stata ==")
    for name in recoverable[:30]:
        print(f"  {name:<18} {unknown[name]:>4} deposits")
    if len(recoverable) > 30:
        print(f"  ... and {len(recoverable) - 30} more")

    still_unknown = sorted(
        (n for n in unknown if n not in snapshot), key=lambda n: -unknown[n]
    )
    print(f"\n== genuinely unresolved: {len(still_unknown)} names ==")
    for name in still_unknown[:20]:
        print(f"  {name:<18} {unknown[name]:>4} deposits")

    if unanswered:
        print(f"\n{len(unanswered)} names could not be checked (unaskable or failed)")

    SNAPSHOT.with_name("report.json").write_text(
        json.dumps(
            {
                "snapshot_date": snapshot.snapshot_date,
                "min_deposits": MIN_DEPOSITS,
                "n_checked": len(snapshot.checked),
                "collisions": collisions,
                "recoverable_builtins": recoverable,
                "still_unknown": still_unknown,
                "unchecked_singletons": n_unknown_total - len(unknown),
            },
            indent=1,
        )
    )
    print(f"\nwrote {SNAPSHOT} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

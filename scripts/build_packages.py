"""Build a page, a badge and a JSON record for every counted package.

    uv run python scripts/build_packages.py

Writes, under `docs/_extra/`:

    p/<ecosystem>/<package>/index.html   who uses the package, and how
    badges/<ecosystem>/<package>.json    a shields.io endpoint badge
    badges/<ecosystem>/<package>.svg     the same badge, pre-rendered
    api/v1/<ecosystem>/<package>.json    the page's numbers, for reuse

Everything is static, so GitHub Pages serves it for free and nothing sleeps: a
badge in a README has to answer every time someone opens that README. The
shields.io JSON lets a maintainer style the badge; the SVG works with no third
party at all.

A package is its (ecosystem, name), never its name alone. The site this
replaces keyed on the name and so gave R's and Python's same-named packages one
page and one count between them.

Reads only `data/tally/` and `data/frame/`, which are tracked, so it builds in
CI.
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING
from urllib.parse import quote

import anybadge
import pandas as pd
from build_lookup import TEMPLATE as LOOKUP_TEMPLATE
from build_lookup import slug

from softverse.config import PATHS

if TYPE_CHECKING:
    from pathlib import Path

TALLY = PATHS.root / "data" / "tally"
FRAME = PATHS.frame / "frame.csv"
OUT = PATHS.root / "docs" / "_extra"
SITE = "https://recite.github.io/softverse"

ECOSYSTEM_LABEL = {
    "cran": "CRAN",
    "cran_archive": "CRAN (archived)",
    "bioconductor": "Bioconductor",
    "ssc": "SSC",
    "pypi": "PyPI",
    "julia_general": "Julia General",
}

LANGUAGE_LABEL = {"r": "R", "python": "Python", "stata": "Stata", "julia": "Julia"}

BADGE_LABEL = "replication code"
BADGE_COLOR = "#007ec6"

#: The lookup page's own stylesheet, so a package page looks like part of it.
STYLE = re.search(r"<style>.*?</style>", LOOKUP_TEMPLATE, re.DOTALL).group(0)  # type: ignore[union-attr]


def badge_message(n_deposits: int) -> str:
    """The badge's right-hand text: the count, in papers.

    Args:
        n_deposits: Deposits whose code uses the package.

    Returns:
        ``"3,259 papers"``, or ``"1 paper"``.
    """
    return f"{n_deposits:,} paper" + ("" if n_deposits == 1 else "s")


def deposit_url(doi: str) -> str:
    """Where a reader finds a deposit, from its identifier.

    Args:
        doi: The deposit's identifier as the tally records it.

    Returns:
        A doi.org link for Dataverse, the record page for Zenodo.
    """
    if doi.startswith("zenodo:"):
        return f"https://zenodo.org/records/{doi.split(':', 1)[1]}"
    return f"https://doi.org/{doi.removeprefix('doi:')}"


def records() -> list[dict]:
    """One record per counted package, with everything its page shows.

    Returns:
        The records, most used first.
    """
    journals: dict[str, str] = {}
    with FRAME.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["collection_id"]:
                journals[row["collection_id"]] = row["journal_name"]

    usage = pd.read_csv(TALLY / "usage_by_package.csv")
    deposits = pd.read_parquet(TALLY / "package_deposits.parquet")
    functions = (
        pd.read_parquet(TALLY / "usage_by_function.parquet")
        .groupby(["language", "package", "function"], as_index=False)[
            ["n_deposits", "n_calls"]
        ]
        .sum()
    )
    versions = pd.read_csv(TALLY / "package_versions.csv", dtype={"version": str})

    by_package = dict(list(deposits.groupby(["language", "package"])))
    fn_by_package = dict(list(functions.groupby(["language", "package"])))
    ver_by_package = dict(list(versions.groupby(["ecosystem", "package"])))
    source_columns = [
        c
        for c in usage.columns
        if c.startswith("n_deposits_") and c != "n_deposits_at_risk"
    ]

    out = []
    for row in usage.itertuples(index=False):
        key = (row.language, row.package)
        rows = by_package.get(key, pd.DataFrame(columns=deposits.columns))
        fns = fn_by_package.get(key)
        vers = ver_by_package.get((row.ecosystem, row.package))
        years = Counter(int(y) for y in rows["year"].dropna())
        journal_counts = Counter(rows["collection_id"])
        family, name = slug(row.ecosystem, row.package)
        out.append(
            {
                "package": row.package,
                "language": row.language,
                "ecosystem": row.ecosystem,
                "path": f"{family}/{name}",
                "n_deposits": int(row.n_deposits),
                "n_deposits_at_risk": int(row.n_deposits_at_risk),
                "share_of_deposits": float(row.share_of_deposits),
                "n_mentions": int(row.n_mentions),
                "by_source": {
                    c.removeprefix("n_deposits_"): int(getattr(row, c))
                    for c in source_columns
                },
                "by_year": dict(sorted(years.items())),
                "journals": [
                    {"id": j, "name": journals.get(j, j), "n_deposits": n}
                    for j, n in journal_counts.most_common()
                ],
                "functions": []
                if fns is None
                else [
                    {"function": f, "n_deposits": int(d), "n_calls": int(c)}
                    for f, d, c in fns.sort_values(
                        ["n_deposits", "n_calls"], ascending=False
                    )[["function", "n_deposits", "n_calls"]]
                    .head(50)
                    .itertuples(index=False)
                ],
                "versions": []
                if vers is None
                else [
                    {"version": v, "source": s, "n_deposits": int(n)}
                    for v, s, n in vers[["version", "version_source", "n_deposits"]]
                    .head(50)
                    .itertuples(index=False)
                ],
                "deposits": [
                    {
                        "doi": d,
                        "url": deposit_url(d),
                        "journal": journals.get(c, c),
                        "year": None if pd.isna(y) else int(y),
                    }
                    for d, c, y in rows.sort_values(
                        ["year", "dataset_doi"], ascending=[False, True]
                    )[["dataset_doi", "collection_id", "year"]].itertuples(index=False)
                ],
            }
        )
    _refuse_collisions(out)
    out.sort(key=lambda r: -r["n_deposits"])
    return out


def _refuse_collisions(items: list[dict]) -> None:
    """Fail rather than let two packages share a page.

    GitHub Pages is case-sensitive but the disks the site is built on often
    are not, so two CRAN names differing only in case would silently
    overwrite each other here and publish one page for both.

    Args:
        items: The package records about to be written.

    Raises:
        ValueError: naming the colliding packages.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for item in items:
        seen[item["path"].lower()].append(f"{item['ecosystem']}:{item['package']}")
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    if clashes:
        raise ValueError(f"packages share a page path: {clashes}")


def page(record: dict, built: str) -> str:
    """The package's HTML page.

    Args:
        record: The package's record from :func:`records`.
        built: When the tally was built.

    Returns:
        A self-contained HTML document.
    """
    e = html.escape
    lang = LANGUAGE_LABEL.get(record["language"], record["language"])
    eco = ECOSYSTEM_LABEL.get(record["ecosystem"], record["ecosystem"])
    endpoint = f"{SITE}/badges/{record['path']}.json"
    shields = f"https://img.shields.io/endpoint?url={quote(endpoint, safe='')}"
    page_url = f"{SITE}/p/{record['path']}/"
    markdown = f"[![{BADGE_LABEL}]({shields})]({page_url})"
    static = f"[![{BADGE_LABEL}]({SITE}/badges/{record['path']}.svg)]({page_url})"

    def rows(items: list[dict], cells) -> str:
        return "".join(f"<tr>{cells(i)}</tr>" for i in items)

    years = rows(
        [{"y": y, "n": n} for y, n in record["by_year"].items()],
        lambda i: f"<td>{i['y']}</td><td class='num'>{i['n']:,}</td>",
    )
    journals = rows(
        record["journals"],
        lambda i: f"<td>{e(i['name'])}</td><td class='num'>{i['n_deposits']:,}</td>",
    )
    functions = rows(
        record["functions"],
        lambda i: f"<td class='pkg'>{e(i['function'])}</td>"
        f"<td class='num'>{i['n_deposits']:,}</td><td class='num'>{i['n_calls']:,}</td>",
    )
    versions = rows(
        record["versions"],
        lambda i: f"<td class='pkg'>{e(i['version'])}</td><td>{e(i['source'])}</td>"
        f"<td class='num'>{i['n_deposits']:,}</td>",
    )
    deposits = rows(
        record["deposits"],
        lambda i: f"<td><a href='{e(i['url'])}'>{e(i['doi'])}</a></td>"
        f"<td>{e(i['journal'])}</td><td class='num'>{i['year'] or ''}</td>",
    )

    def section(title: str, head: str, body: str) -> str:
        if not body:
            return ""
        return (
            f"<h2>{title}</h2><div class='tablewrap'><table><thead><tr>{head}"
            f"</tr></thead><tbody>{body}</tbody></table></div>"
        )

    split = " · ".join(f"{s}: {n:,}" for s, n in record["by_source"].items())
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{e(record["package"])} ({eco}) · softverse</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{STYLE}
<style>
h2 {{ font-size: 1.05rem; margin: 2rem 0 0.6rem; }}
.stat {{ font-size: 2.2rem; font-weight: 650; letter-spacing: -0.02em; }}
pre {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
       padding: 0.7rem; overflow-x: auto; font-size: 0.8rem; white-space: pre-wrap; }}
button.copy {{ font: inherit; font-size: 0.8rem; cursor: pointer; }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow"><a href="../../../lookup/">softverse lookup</a> · {eco} · {lang}</div>
  <h1 class="pkg">{e(record["package"])}</h1>
  <p class="lede">Referenced in the replication code of</p>
  <div class="stat">{badge_message(record["n_deposits"])}</div>
  <p class="lede">{record["share_of_deposits"]:.1%} of the {record["n_deposits_at_risk"]:,}
  deposits with analyzable {lang} code. {split}.</p>
</header>
<h2>Badge</h2>
<p><img alt="{BADGE_LABEL}: {badge_message(record["n_deposits"])}"
  src="../../../badges/{record["path"]}.svg"></p>
<pre id="md">{e(markdown)}</pre>
<button class="copy" onclick="navigator.clipboard.writeText(document.getElementById('md').textContent)">Copy Markdown</button>
<details><summary>Without shields.io</summary><pre>{e(static)}</pre></details>
{section("Deposits by year", "<th>Year</th><th class='num'>Deposits</th>", years)}
{section("Journals", "<th>Journal</th><th class='num'>Deposits</th>", journals)}
{section("Most-used functions", "<th>Function</th><th class='num'>Deposits</th><th class='num'>Calls</th>", functions)}
{section("Versions researchers stated", "<th>Version</th><th>Where stated</th><th class='num'>Deposits</th>", versions)}
{section(f"The {record['n_deposits']:,} deposits", "<th>Deposit</th><th>Journal</th><th class='num'>Year</th>", deposits)}
<footer>
  A reference is not a run: these deposits' code names {e(record["package"])},
  which does not show the code executed. Counts cover deposits at economics
  and political science journals that verify replication packages, built
  {e(built)}. <a href="../../../api/v1/{record["path"]}.json">This page as JSON</a> ·
  <a href="../../../data/">the tables</a>.
</footer>
</div>
"""


def write(record: dict, out: Path, built: str) -> None:
    """Write one package's page, badges and JSON."""
    family, name = record["path"].split("/", 1)
    message = badge_message(record["n_deposits"])

    page_dir = out / "p" / family / name
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "index.html").write_text(page(record, built), encoding="utf-8")

    badges = out / "badges" / family
    badges.mkdir(parents=True, exist_ok=True)
    (badges / f"{name}.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "label": BADGE_LABEL,
                "message": message,
                "color": BADGE_COLOR,
                "cacheSeconds": 86400,
            }
        )
    )
    (badges / f"{name}.svg").write_text(
        anybadge.Badge(
            label=BADGE_LABEL, value=message, default_color=BADGE_COLOR
        ).badge_svg_text
    )

    api = out / "api" / "v1" / family
    api.mkdir(parents=True, exist_ok=True)
    (api / f"{name}.json").write_text(json.dumps({**record, "built": built}))


def main(out: Path = OUT) -> int:
    summary = json.loads((TALLY / "summary.json").read_text())
    items = records()
    for record in items:
        write(record, out, summary["built"])
    print(f"wrote {len(items):,} package pages, badges and API records under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

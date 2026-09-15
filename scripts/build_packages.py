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
from collections import Counter, defaultdict
from typing import TYPE_CHECKING
from urllib.parse import quote

import anybadge
import pandas as pd
from build_lookup import slug
from site_style import STYLE

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

#: Years whose denominator is smaller than this are left off the trend. A
#: share of 3 out of 11 deposits moves 9 points on one paper, and a line
#: through such years draws noise as if it were a change in practice.
MIN_AT_RISK = 30


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
    at_risk = {
        (row.language, int(row.year)): int(row.n_deposits_at_risk)
        for row in pd.read_csv(TALLY / "language_year_at_risk.csv")
        .dropna(subset=["year"])
        .itertuples(index=False)
    }

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
                # Every year the language has deposits in, zeros included: a
                # year a package went unused is a point on its line, and
                # leaving it out draws a straight segment across the gap.
                "by_year": [
                    {
                        "year": year,
                        "n_deposits": years.get(year, 0),
                        "n_deposits_at_risk": n_at_risk,
                    }
                    for (language, year), n_at_risk in sorted(at_risk.items())
                    if language == row.language
                ],
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


def trend(record: dict) -> str:
    """The package's share of each year's deposits, as a plain line.

    Only years with at least :data:`MIN_AT_RISK` deposits in the language. The
    y-axis starts at zero, so a small share looks small.

    Args:
        record: The package's record from :func:`records`.

    Returns:
        An inline SVG, or an empty string with fewer than two usable years.
    """
    points = [
        (y["year"], y["n_deposits"] / y["n_deposits_at_risk"], y)
        for y in record["by_year"]
        if y["n_deposits_at_risk"] >= MIN_AT_RISK
    ]
    if len(points) < 2:
        return ""
    width, height, left, bottom, top = 600, 180, 34, 22, 8
    first, last = points[0][0], points[-1][0]
    peak = max(share for _, share, _ in points)
    ceiling = max(0.05, round(peak * 1.15 + 0.005, 2))

    def x(year: int) -> float:
        return left + (width - left - 8) * (year - first) / max(1, last - first)

    def y(share: float) -> float:
        return top + (height - top - bottom) * (1 - share / ceiling)

    line = " ".join(f"{x(yr):.1f},{y(sh):.1f}" for yr, sh, _ in points)
    dots = "".join(
        f'<circle class="pt" cx="{x(yr):.1f}" cy="{y(sh):.1f}" r="2.5">'
        f"<title>{yr}: {d['n_deposits']:,} of {d['n_deposits_at_risk']:,} "
        f"deposits, {sh:.1%}</title></circle>"
        for yr, sh, d in points
    )
    ticks = "".join(
        f'<text x="{x(yr):.1f}" y="{height - 6}" text-anchor="middle">{yr}</text>'
        for yr, _, _ in points
        if yr in (first, last) or (yr - first) % max(1, (last - first) // 4) == 0
    )
    labels = "".join(
        f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>'
        for v in (0, ceiling / 2, ceiling)
    )
    base = height - bottom
    return (
        f'<svg class="trend" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Share of deposits using {html.escape(record["package"])}, by year">'
        f'<line class="axis" x1="{left}" y1="{base}" x2="{width - 8}" y2="{base}"/>'
        f'{labels}{ticks}<polyline class="line" points="{line}"/>{dots}</svg>'
    )


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

    def table(head: list[tuple[str, str]], rows: list[list[str]]) -> str:
        if not rows:
            return ""
        th = "".join(f"<th class='{c}'>{t}</th>" for t, c in head)
        body = "".join(
            "<tr>"
            + "".join(f"<td class='{head[i][1]}'>{v}</td>" for i, v in enumerate(r))
            + "</tr>"
            for r in rows
        )
        return f"<div class='scroll'><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"

    chart = trend(record)
    journals = table(
        [("Journal", ""), ("Deposits", "num")],
        [[e(j["name"]), f"{j['n_deposits']:,}"] for j in record["journals"]],
    )
    functions = table(
        [("Function", ""), ("Deposits", "num"), ("Calls", "num")],
        [
            [
                f"<span class='pkg'>{e(f['function'])}</span>",
                f"{f['n_deposits']:,}",
                f"{f['n_calls']:,}",
            ]
            for f in record["functions"]
        ],
    )
    versions = table(
        [("Version", ""), ("Stated in", ""), ("Deposits", "num")],
        [
            [
                f"<span class='pkg'>{e(v['version'])}</span>",
                e(v["source"]),
                f"{v['n_deposits']:,}",
            ]
            for v in record["versions"]
        ],
    )
    deposits = table(
        [("Deposit", ""), ("Journal", ""), ("Year", "num")],
        [
            [
                f"<a href='{e(d['url'])}'>{e(d['doi'])}</a>",
                e(d["journal"]),
                str(d["year"] or ""),
            ]
            for d in record["deposits"]
        ],
    )
    split = ", ".join(
        f"{n:,} from {s.capitalize()}" for s, n in record["by_source"].items() if n
    )

    sections = []
    if chart:
        sections.append(
            f"<h2>Share of {lang} deposits, by year</h2>{chart}"
            f"<p class='note'>Years with fewer than {MIN_AT_RISK} deposits containing "
            f"{lang} code are left out.</p>"
        )
    if journals:
        sections.append(f"<h2>Journals</h2>{journals}")
    if functions:
        sections.append(
            f"<h2>Functions called</h2>{functions}<p class='note'>A call is counted "
            f"only where the code names the package, as in "
            f"<code>{e(record['package'])}::f()</code> or through an import alias, so "
            "these undercount calls made after a plain <code>library()</code>.</p>"
        )
    if versions:
        sections.append(f"<h2>Versions researchers stated</h2>{versions}")
    sections.append(f"<h2>The {record['n_deposits']:,} deposits</h2>{deposits}")

    return f"""<!doctype html>
<meta charset="utf-8">
<title>{e(record["package"])} · softverse</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{STYLE}
<div class="wrap">
<header>
  <p class="crumb"><a href="../../../lookup/">softverse</a> · {eco} · {lang}</p>
  <h1 class="pkg">{e(record["package"])}</h1>
  <p class="stat">Loaded by the replication code of {badge_message(record["n_deposits"])}</p>
  <p class="lede">{record["share_of_deposits"]:.1%} of the {record["n_deposits_at_risk"]:,}
  deposits with {lang} code; {split}.</p>
</header>
{"".join(sections)}
<h2>Badge</h2>
<p><img alt="{BADGE_LABEL}: {badge_message(record["n_deposits"])}" src="../../../badges/{record["path"]}.svg"></p>
<pre id="md">{e(markdown)}</pre>
<p class="note"><a href="#" onclick="navigator.clipboard.writeText(document.getElementById('md').textContent); this.textContent='copied'; return false">copy</a>
 · without shields.io: <code>{e(static)}</code></p>
<footer>
  A reference is not a run: these deposits' code names {e(record["package"])},
  which does not show the code executed. Deposits at economics and political
  science journals that verify replication packages; built {e(built)}.
  <a href="../../../api/v1/{record["path"]}.json">JSON</a> ·
  <a href="../../../data/">tables</a>
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

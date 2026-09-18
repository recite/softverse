"""Build the package lookup page.

    uv run python scripts/build_lookup.py

Writes `build/lookup/index.html`, a single self-contained file with every
package's validated-use count embedded.

The paper argues that a credit metric nobody can query is an assertion. This
is the part that makes that not true of ours. A software author, or somebody
evaluating one, should be able to type a package name and get the number
rather than take a table's word for the top ten.

Counts come from `data/tally/`, which is the tracked copy of the
tables the paper's own exhibits are built from, so the page and the paper
cannot disagree, and the page can be built anywhere the repository is.

`unknown_names.csv` is carried too. A search for `btscs` that returns
nothing reads as "nobody uses it", when the truth is that it is used and no
archive this project indexes lists it. The page has to be able to say that
rather than go quiet.
"""

from __future__ import annotations

import csv
import html
import json
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from site_style import STYLE

from softverse.config import PATHS
from softverse.logging_setup import get_logger, setup_logging

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

TALLY = PATHS.root / "data" / "tally"
OUT = PATHS.root / "build" / "lookup"

#: Names shown to a reader rather than column names from the pipeline.
LANGUAGE_LABEL = {
    "stata": "Stata",
    "r": "R",
    "python": "Python",
    "julia": "Julia",
}


#: Registry names as a reader would recognize them.
ECOSYSTEM_LABEL = {
    "ssc": "SSC",
    "cran": "CRAN",
    "cran_archive": "CRAN archive",
    "bioconductor": "Bioconductor",
    "pypi": "PyPI",
    "julia_general": "Julia General",
}


#: Registry families as they appear in a package page's URL. An archived CRAN
#: package keeps its CRAN address: it is the same package, removed.
ECOSYSTEM_SLUG = {
    "cran": "cran",
    "cran_archive": "cran",
    "bioconductor": "bioconductor",
    "ssc": "ssc",
    "pypi": "pypi",
    "julia_general": "julia",
}


def slug(ecosystem: str, package: str) -> tuple[str, str]:
    """The (ecosystem, name) path segments of a package's page and badge.

    PyPI names are normalized as PEP 503 does, because `scikit_learn` and
    `Scikit-Learn` are the same distribution. CRAN names keep their case,
    because CRAN treats `Matrix` and `matrix` as different packages.

    Args:
        ecosystem: The registry the package resolved to.
        package: The package name as counted.

    Returns:
        URL-safe path segments.
    """
    family = ECOSYSTEM_SLUG.get(ecosystem, ecosystem)
    name = re.sub(r"[-_.]+", "-", package).lower() if family == "pypi" else package
    return family, quote(name, safe="._-")


#: Reader-facing names for the two repositories.
SOURCE_LABEL = {
    "zenodo": "Zenodo",
    "dataverse": "Dataverse",
}


def rows() -> list[dict]:
    """Per-package counts, smallest useful payload rather than the whole CSV.

    Each row carries the pooled count and the per-source split, because the
    two repositories are different disciplines and very different sizes. A
    reader looking up a package they wrote should be able to see which of the
    two is crediting them.
    """
    with (TALLY / "usage_by_package.csv").open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))

    sources = [
        key[len("n_deposits_") :]
        for key in (raw[0] if raw else {})
        if key.startswith("n_deposits_") and key != "n_deposits_at_risk"
    ]

    out = []
    for row in raw:
        deposits = int(row["n_deposits"])
        at_risk = int(row["n_deposits_at_risk"] or 0)
        out.append(
            {
                "p": row["package"],
                "l": LANGUAGE_LABEL.get(row["language"], row["language"]),
                "e": ECOSYSTEM_LABEL.get(row["ecosystem"], row["ecosystem"] or ""),
                "d": deposits,
                "a": at_risk,
                # Rounded here rather than in the browser so the page and the
                # paper round the same way. `at_risk` is zero only for a
                # language with no analyzable deposits, where a share would be
                # a division by nothing rather than a zero.
                "s": round(100 * deposits / at_risk, 1) if at_risk else None,
                "m": int(row["n_mentions"]),
                "u": "/".join(slug(row["ecosystem"], row["package"])),
                "src": {
                    SOURCE_LABEL.get(s, s): int(row.get(f"n_deposits_{s}") or 0)
                    for s in sources
                },
            }
        )
    out.sort(key=lambda r: -r["d"])
    return out


def unresolved() -> list[dict]:
    """Names called in deposited code that resolve to no registry.

    Deliberately unfiltered. Some are false positives, `str` in Stata being
    the obvious one, and hand-pruning the list would hide a judgment call
    inside a page whose whole point is that you can check the number.
    """
    with (TALLY / "unknown_names.csv").open(encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    out = [
        {
            "p": row["name"],
            "l": LANGUAGE_LABEL.get(row["language"], row["language"]),
            "m": int(row["n_mentions"]),
        }
        for row in raw
    ]
    out.sort(key=lambda r: -r["m"])
    return out


def analyzable_deposits() -> int:
    """Deposits holding at least one parseable analysis file.

    From the summary rather than the Parquet it was computed from: that file
    is 13 MB and cannot be tracked, and this page has to build in CI.
    """
    return int(
        json.loads((TALLY / "summary.json").read_text())["n_deposits_analyzable"]
    )


def build(data: list[dict], misses: list[dict], n_deposits: int) -> str:
    languages = sorted({r["l"] for r in data})
    options = "".join(f'<option value="{lang}">{lang}</option>' for lang in languages)
    sources = sorted({s for r in data for s in r["src"]})
    source_options = "".join(f'<option value="{s}">{s}</option>' for s in sources)

    return (
        TEMPLATE.replace("__PAYLOAD__", json.dumps(data, separators=(",", ":")))
        .replace("__MISSES__", json.dumps(misses, separators=(",", ":")))
        .replace("__OPTIONS__", options)
        .replace("__SOURCE_OPTIONS__", source_options)
        .replace("__SOURCES__", json.dumps(sources))
        .replace("__NPACKAGES__", f"{len(data):,}")
        .replace("__NDEPOSITS__", f"{n_deposits:,}")
        .replace("__BOARDS__", leaderboards(data))
        .replace("__STYLE__", STYLE)
    )


#: Languages given a leaderboard, in the order they are shown, and its length.
BOARD_LANGUAGES = ("R", "Stata", "Python")
BOARD_LENGTH = 15


def leaderboards(data: list[dict]) -> str:
    """The most used packages per language, as plain ranked lists.

    Args:
        data: The rows the lookup table is built from, one per package.

    Returns:
        One ``<ol>`` per language, each package linked to its page.
    """
    out = []
    for lang in BOARD_LANGUAGES:
        top = sorted((r for r in data if r["l"] == lang), key=lambda r: -r["d"])
        items = "".join(
            f'<li><a class="pkg" href="../p/{r["u"]}/">{html.escape(r["p"])}</a>'
            f'<span class="n">{r["d"]:,}'
            + (f" · {r['s']:.0f}%" if r["s"] is not None else "")
            + "</span></li>"
            for r in top[:BOARD_LENGTH]
        )
        out.append(
            f'<div><p><strong>{lang}</strong></p><ol class="board">{items}</ol></div>'
        )
    return "".join(out)


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Package lookup · softverse</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
__STYLE__
<style>
.wrap { max-width: 54rem; }
.count, .empty { color: var(--soft); font-size: 0.85rem; margin: 0.6rem 0; }
.lang { color: var(--soft); }
.controls { margin: 2.4rem 0 0.4rem; }
.miss { color: var(--soft); font-size: 0.85rem; margin-top: 1rem; }
</style>
<div class="wrap">
<header>
  <p class="crumb"><a href="../">softverse</a></p>
  <h1>Package lookup</h1>
  <p class="lede">
    How often each package is loaded by the code behind published papers.
    __NPACKAGES__ packages across __NDEPOSITS__ deposits at economics and
    political science journals that check an author's code before
    publication. A download count rises when a build server installs a
    package; a count here rises when somebody publishes a paper that uses
    it.
  </p>
</header>

<h2>Most used, by language</h2>
<div class="boards">__BOARDS__</div>

<div class="controls">
  <input type="search" id="q" placeholder="Search a package, e.g. reghdfe" autocomplete="off" aria-label="Search packages">
  <select id="lang" aria-label="Filter by language"><option value="">All languages</option>__OPTIONS__</select>
  <select id="src" aria-label="Filter by repository"><option value="">Both repositories</option>__SOURCE_OPTIONS__</select>
  <select id="min" aria-label="Minimum deposits">
    <option value="1">1+ deposits</option>
    <option value="5">5+ deposits</option>
    <option value="25">25+ deposits</option>
    <option value="100">100+ deposits</option>
  </select>
</div>
<div class="count" id="count"></div>

<div class="scroll">
<table>
  <thead><tr>
    <th><button data-k="p">Package</button></th>
    <th><button data-k="l">Language</button></th>
    <th class="num"><button data-k="d">Deposits</button></th>
    <th class="num"><button data-k="s">Share</button></th>
    <th class="num"><button data-k="m">Calls</button></th>
    <th><button data-k="e" title="The registry the package is published on">Listed on</button></th>
    <th><button data-k="d">Repositories</button></th>
  </tr></thead>
  <tbody id="body"></tbody>
</table>
</div>
<div class="empty" id="empty" hidden>No package matches that.</div>
<div class="miss" id="miss" hidden></div>

<footer>
  <strong>Deposits</strong> counts each deposit once, however many times its
  code calls the package. <strong>Calls</strong> is the raw total.
  <strong>Share</strong> divides by the deposits containing code in that
  language, so a Stata share and an R share have different denominators and
  should not be read against each other. <strong>Repositories</strong> splits
  the count between the two collections, and the filter narrows to one.
  <br>Two things to know before relying on a number. A package can be loaded
  by code that never runs, so these are loads and not executions. And both
  collections are replication deposits only, so code an author kept out of
  the deposit is not counted.
  <br><a href="../paper/softverse.pdf">The paper</a> gives the method and
  <a href="../data/">the tables</a> are CC0.
</footer>
</div>

<script>
const DATA = __PAYLOAD__;
const MISSES = __MISSES__;
const SOURCES = __SOURCES__;
const body = document.getElementById("body");
const count = document.getElementById("count");
const empty = document.getElementById("empty");
const miss = document.getElementById("miss");
let sortKey = "d", sortDesc = true;

function render() {
  const q = document.getElementById("q").value.trim().toLowerCase();
  const lang = document.getElementById("lang").value;
  const src = document.getElementById("src").value;
  const min = +document.getElementById("min").value;

  // Filtering by repository counts that repository's deposits, not the
  // pooled total. Showing a pooled number under a "Zenodo only" filter would
  // be the same mistake as reporting one repository as if it were the field.
  let rows = DATA.filter(r =>
    (!q || r.p.toLowerCase().includes(q)) &&
    (!lang || r.l === lang) &&
    (!src || (r.src[src] || 0) > 0) &&
    (src ? r.src[src] : r.d) >= min)
    .map(r => src ? {...r, d: r.src[src], s: null} : r);

  rows.sort((a, b) => {
    const x = a[sortKey] ?? -1, y = b[sortKey] ?? -1;
    const cmp = typeof x === "string" ? x.localeCompare(y) : x - y;
    return sortDesc ? -cmp : cmp;
  });

  count.textContent = rows.length.toLocaleString() + " packages";
  empty.hidden = rows.length > 0;

  const shown = rows.slice(0, 500);
  body.innerHTML = shown.map(r => `
    <tr>
      <td><a class="pkg" href="../p/${r.u}/">${esc(r.p)}</a></td>
      <td class="lang">${esc(r.l)}</td>
      <td class="num">${r.d.toLocaleString()} <span class="lang">of ${r.a.toLocaleString()}</span></td>
      <td class="num">${share(r)}</td>
      <td class="num">${r.m.toLocaleString()}</td>
      <td class="lang">${esc(r.e)}</td>
      <td class="lang">${split(r)}</td>
    </tr>`).join("");

  if (rows.length > shown.length) {
    count.textContent += ` (showing first ${shown.length})`;
  }
  showMisses(q, lang);
}

function share(r) {
  if (r.s === null || r.s === undefined) { return '<span class="lang">&mdash;</span>'; }
  return `${r.s.toFixed(1)}%`;
}

// Which repository is doing the crediting. A package can be near the top of
// the pooled table on the strength of one of the two.
function split(r) {
  return SOURCES
    .filter(s => (r.src[s] || 0) > 0)
    .map(s => `${esc(s)}&nbsp;${r.src[s].toLocaleString()}`)
    .join(" &middot; ") || "&mdash;";
}

// A search that goes quiet reads as "nobody uses this". Often the name is
// used heavily and simply belongs to no registry, which is worth saying.
function showMisses(q, lang) {
  if (q.length < 2) { miss.hidden = true; return; }
  const hits = MISSES
    .filter(r => r.p.toLowerCase().includes(q) && (!lang || r.l === lang))
    .slice(0, 6);
  miss.hidden = hits.length === 0;
  if (!hits.length) { return; }
  miss.innerHTML =
    "<p>Used in deposited code, but listed in no registry, so no " +
    "registry-based credit system can see it:</p><ul>" +
    hits.map(r => `<li><strong>${esc(r.p)}</strong> (${esc(r.l)}) &middot; ` +
      `${r.m.toLocaleString()} calls</li>`).join("") +
    "</ul><p>These are raw unresolved names. Some are false positives: a " +
    "language keyword, or a program the deposit defines for itself.</p>";
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

document.querySelectorAll("th button").forEach(b =>
  b.addEventListener("click", () => {
    const k = b.dataset.k;
    if (k === sortKey) { sortDesc = !sortDesc; }
    else { sortKey = k; sortDesc = k !== "p" && k !== "l"; }
    render();
  }));

["q", "lang", "src", "min"].forEach(id =>
  document.getElementById(id).addEventListener("input", render));

render();
</script>
"""


def main(out: Path = OUT) -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="lookup")
    if not (TALLY / "usage_by_package.csv").exists():
        print("no released tables; run scripts/release_tally.py first")
        return 1

    data = rows()
    misses = unresolved()
    out.mkdir(parents=True, exist_ok=True)
    page = build(data, misses, analyzable_deposits())
    (out / "index.html").write_text(page, encoding="utf-8")

    print(f"{len(data):,} packages, {len(misses):,} unresolved names")
    for lang in sorted({r["l"] for r in data}):
        subset = [r for r in data if r["l"] == lang]
        top = max(subset, key=lambda r: r["d"])
        print(
            f"  {lang:<8} {len(subset):>5} packages, most used {top['p']} ({top['d']})"
        )
    print(f"\nwrote {out / 'index.html'} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

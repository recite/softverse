"""Build the package lookup page.

    uv run python scripts_build_lookup.py

Writes `build/lookup/index.html`, a single self-contained file with every
package's validated-use count embedded.

The paper argues that a credit metric nobody can query is an assertion. This
is the part that makes that not true of ours. A software author, or somebody
evaluating one, should be able to type a package name and get the number
rather than take a table's word for the top ten.

Counts come from `build/release/tally/`, which is the tracked copy of the
tables the paper's own exhibits are built from, so the page and the paper
cannot disagree, and the page can be built anywhere the repository is.

`unknown_names.csv` is carried too. A search for `grc1leg` that returns
nothing reads as "nobody uses it", when the truth is that it is used heavily
and is in no registry, which is one of the paper's findings. The page has to
be able to say that rather than go quiet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from softverse.config import PATHS
from softverse.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

TALLY = PATHS.root / "build" / "release" / "tally"
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


#: Reader-facing names for the two repositories.
SOURCE_LABEL = {
    "zenodo": "Zenodo",
    "dataverse_legacy": "Dataverse",
}


def rows() -> list[dict]:
    """Per-package counts, smallest useful payload rather than the whole CSV.

    Each row carries the pooled count and the per-source split, because the
    two repositories are different disciplines and very different sizes. A
    reader looking up a package they wrote should be able to see which of the
    two is crediting them.
    """
    with open(TALLY / "usage_by_package.csv", encoding="utf-8") as handle:
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
    with open(TALLY / "unknown_names.csv", encoding="utf-8") as handle:
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
    )


TEMPLATE = r"""<title>Validated Use Lookup</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --ink: #16181d; --soft: #5a626e; --faint: #8d95a1;
  --ground: #fcfcfb; --panel: #f1f2ef; --rule: #dfe1dc;
  --accent: #1f4e7a; --accent-soft: #e9eff5;
}
:root:not([data-theme="light"]) {
  @media (prefers-color-scheme: dark) {
    --ink: #e7e9ec; --soft: #a6aeb9; --faint: #79818d;
    --ground: #15171b; --panel: #1d2026; --rule: #2c3038;
    --accent: #82b2dd; --accent-soft: #1a2733;
  }
}
:root[data-theme="dark"] {
  --ink: #e7e9ec; --soft: #a6aeb9; --faint: #79818d;
  --ground: #15171b; --panel: #1d2026; --rule: #2c3038;
  --accent: #82b2dd; --accent-soft: #1a2733;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px; line-height: 1.5;
}
.wrap { max-width: 60rem; margin: 0 auto; padding: 2.6rem 1.2rem 5rem; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 1.2rem; margin-bottom: 1.4rem; }
.eyebrow {
  font-size: 0.66rem; font-weight: 650; letter-spacing: 0.15em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 0.5rem;
}
h1 { font-size: 1.6rem; font-weight: 650; margin: 0 0 0.4rem; letter-spacing: -0.01em; }
.lede { color: var(--soft); max-width: 46rem; margin: 0; }
.controls { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 1.4rem 0 0.9rem; }
input[type="search"], select {
  font: inherit; color: var(--ink); background: var(--ground);
  border: 1px solid var(--rule); border-radius: 5px; padding: 0.5rem 0.7rem;
}
input[type="search"] { flex: 1 1 18rem; }
input:focus-visible, select:focus-visible, th button:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 1px;
}
.count { color: var(--faint); font-size: 0.85rem; margin-bottom: 0.5rem; }
.tablewrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th { background: var(--panel); text-align: left; white-space: nowrap; }
th button {
  all: unset; cursor: pointer; display: block; width: 100%;
  padding: 0.55rem 0.8rem; font-weight: 640; font-size: 0.82rem; color: var(--soft);
}
th button:hover { color: var(--accent); }
th.num button, td.num { text-align: right; }
td { padding: 0.45rem 0.8rem; border-top: 1px solid var(--rule); font-size: 0.88rem; }
tbody tr:hover { background: var(--accent-soft); }
.pkg { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-weight: 600; }
.lang { color: var(--soft); font-size: 0.8rem; }
.bar { display: block; height: 3px; background: var(--accent); border-radius: 2px; margin: 3px 0 0 auto; }
th:last-child button, td:last-child { padding-left: 1.8rem; }
.empty { padding: 2.5rem 1rem 1rem; text-align: center; color: var(--faint); }
.miss {
  margin-top: 0.9rem; padding: 0.9rem 1.1rem; border-radius: 6px;
  background: var(--accent-soft); border: 1px solid var(--rule);
  font-size: 0.88rem; color: var(--soft);
}
.miss strong { color: var(--ink); }
footer { margin-top: 2rem; color: var(--faint); font-size: 0.8rem; line-height: 1.6; }
a { color: var(--accent); }
</style>
<div class="wrap">
<header>
  <div class="eyebrow">softverse</div>
  <h1>Validated use lookup</h1>
  <p class="lede">
    How often each package is loaded by the code behind published papers, at
    journals whose data-and-code policies are actively verified.
    __NPACKAGES__ packages across __NDEPOSITS__ deposits holding analyzable
    code, from Zenodo's economics collections and Harvard Dataverse's
    political science journals. Not downloads, not mentions in prose: code
    that shipped with a paper.
  </p>
</header>

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

<div class="tablewrap">
<table>
  <thead><tr>
    <th><button data-k="p">Package</button></th>
    <th><button data-k="l">Language</button></th>
    <th class="num"><button data-k="d">Deposits</button></th>
    <th class="num"><button data-k="s">Share</button></th>
    <th class="num"><button data-k="m">Calls</button></th>
    <th><button data-k="e">Registry</button></th>
    <th><button data-k="d">Repositories</button></th>
  </tr></thead>
  <tbody id="body"></tbody>
</table>
</div>
<div class="empty" id="empty" hidden>No package matches that.</div>
<div class="miss" id="miss" hidden></div>

<footer>
  A deposit counts once however many times it calls the package;
  <em>calls</em> is the raw total. <em>Share</em> is of deposits containing
  analyzable code in that language, which is the only denominator these are
  comparable against, so a Stata share and an R share have different
  denominators by design. Counts are static reference in deposited code, which
  is not the same as execution.
  <br>Counts pool both repositories; the last column says which one is doing
  the crediting, and the filter narrows to one. The Dataverse half is a
  January 2024 scrape that collected only <code>.do</code>, <code>.r</code>
  and <code>.py</code>, so a package used mainly inside notebooks or knitr
  documents is under-counted there.
  <br>Built from the released tables at
  <a href="https://github.com/recite/softverse">recite/softverse</a>. Full
  method in <a href="../paper/softverse.pdf">the paper</a>.
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

const maxShare = Math.max(...DATA.map(r => r.s || 0), 1);

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
      <td><span class="pkg">${esc(r.p)}</span></td>
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
  return `${r.s.toFixed(1)}%<span class="bar" style="width:${(100 * r.s / maxShare).toFixed(1)}%"></span>`;
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
    "<p>Called in deposited code but in no registry we index, so it earns no " +
    "credit anywhere:</p><ul>" +
    hits.map(r => `<li><strong>${esc(r.p)}</strong> (${esc(r.l)}) &middot; ` +
      `${r.m.toLocaleString()} calls</li>`).join("") +
    "</ul><p>These are raw unresolved names, so some are false positives: a " +
    "language keyword or a program defined in the deposit itself.</p>";
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
        print("no released tables; run scripts_release_tally.py first")
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

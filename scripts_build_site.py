"""Stage everything the documentation site serves.

    uv run python scripts_build_site.py

Reads only `build/release/tally/`, which is tracked, so this runs in CI where
the 57 GB of collected deposits and the 57 MB of Parquet derived from them do
not exist. Writes:

    docs/index.md          the landing page, its numbers filled in
    docs/_extra/lookup/    the package lookup page
    docs/_extra/data/      the released tables
    docs/_extra/paper/     the paper

`html_extra_path` copies `_extra/` into the site root untouched, so the lookup
page keeps its own design instead of being wrapped in the theme.

The landing page is a template for the same reason the paper is: a number
typed into prose is a number that goes stale silently. Every figure it states
comes out of the released tables at build time, and an unresolved placeholder
fails the build.
"""

from __future__ import annotations

import csv
import json
import re
import shutil

from softverse.config import PATHS

RELEASE = PATHS.root / "build" / "release" / "tally"
DOCS = PATHS.root / "docs"
EXTRA = DOCS / "_extra"
PDF = PATHS.root / "paper" / "softverse.pdf"

#: Stata packages whose job is formatting output rather than estimating.
#: The same named list the paper and its figures use, so the landing page
#: cannot claim a different count than the paper does.
TABLE_MAKERS = {
    "estout",
    "outreg2",
    "coefplot",
    "tabout",
    "asdoc",
    "outtable",
    "esttab",
    "mktab",
    "xml_tab",
    "logout",
    "putexcel",
    "outreg",
}

NUMBER_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "all four",
}


def load() -> tuple[dict, list[dict], list[dict]]:
    summary = json.loads((RELEASE / "summary.json").read_text())
    with open(RELEASE / "usage_by_package.csv", encoding="utf-8") as handle:
        usage = list(csv.DictReader(handle))
    with open(RELEASE / "unknown_names.csv", encoding="utf-8") as handle:
        unknown = list(csv.DictReader(handle))
    return summary, usage, unknown


def landing(summary: dict, usage: list[dict], unknown: list[dict]) -> str:
    stata = sorted(
        (r for r in usage if r["language"] == "stata"),
        key=lambda r: -int(r["n_deposits"]),
    )
    by_language = summary["deposits_by_language"]
    grc1leg = next((int(r["n_mentions"]) for r in unknown if r["name"] == "grc1leg"), 0)
    n_formatters = sum(1 for r in stata[:4] if r["package"] in TABLE_MAKERS)

    values = {
        "__N_ANALYZABLE__": f"{summary['n_deposits_analyzable']:,}",
        "__N_STATA__": f"{by_language.get('stata', 0):,}",
        "__N_R__": f"{by_language.get('r', 0):,}",
        "__N_PACKAGES__": f"{summary['n_packages']:,}",
        "__TOP_STATA__": stata[0]["package"],
        "__TOP_STATA_N__": f"{int(stata[0]['n_deposits']):,}",
        "__N_FORMATTERS__": NUMBER_WORD.get(n_formatters, str(n_formatters)),
        "__N_GRC1LEG__": f"{grc1leg:,}",
    }

    page = (DOCS / "index.md.in").read_text()
    for token, value in values.items():
        page = page.replace(token, value)

    left = re.findall(r"__[A-Z0-9_]+__", page)
    if left:
        raise SystemExit(f"unresolved placeholders in index.md.in: {sorted(set(left))}")
    return page


def stage() -> None:
    """Copy what the site serves. Rebuilt from scratch so a removed file goes."""
    if EXTRA.exists():
        shutil.rmtree(EXTRA)
    (EXTRA / "data").mkdir(parents=True)
    (EXTRA / "paper").mkdir(parents=True)

    for path in sorted(RELEASE.iterdir()):
        if path.is_file():
            shutil.copyfile(path, EXTRA / "data" / path.name)

    # A directory of CSVs with no index is a 404 to anyone who clicks it.
    (EXTRA / "data" / "index.html").write_text(_data_index())

    # The release descriptor becomes a real page rather than a `.md` the
    # browser offers to download.
    shutil.copyfile(RELEASE / "README.md", DOCS / "data-dictionary.md")

    if PDF.exists():
        shutil.copyfile(PDF, EXTRA / "paper" / PDF.name)
    else:
        print(f"warning: {PDF} is absent, so the site will not carry the paper")


def _data_index() -> str:
    """A plain listing, deliberately unstyled: it is a directory, not a page."""
    rows = "\n".join(
        f'<li><a href="{p.name}">{p.name}</a></li>'
        for p in sorted(RELEASE.iterdir())
        if p.is_file()
    )
    return (
        "<title>Softverse data</title>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<h1>Released tables</h1>\n"
        "<p>Per-package counts of validated use, CC0. What each column means "
        'is in the <a href="../data-dictionary.html">data dictionary</a>.</p>\n'
        f"<ul>\n{rows}\n</ul>\n"
        '<p><a href="../">Back to softverse</a></p>\n'
    )


def main() -> int:
    if not (RELEASE / "summary.json").exists():
        print("no released tables; run scripts_release_tally.py first")
        return 1

    summary, usage, unknown = load()
    (DOCS / "index.md").write_text(landing(summary, usage, unknown))
    stage()

    from scripts_build_lookup import main as build_lookup

    if build_lookup(out=EXTRA / "lookup") != 0:
        return 1

    print(f"staged {EXTRA.relative_to(PATHS.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

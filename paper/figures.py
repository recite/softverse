"""Build the paper's figures from the released tables.

    uv run python paper/figures.py

Writes PDFs to `paper/figures/`. Three exhibits, each carrying an argument the
prose currently has to make in words:

`downloads.pdf` sets validated use against each registry's download count.
The claim is that the two part company, and where: a scatter shows the cloud
and names the packages furthest from it in each direction.

`languages.pdf` shows the disciplinary split. Economics deposits are Stata,
political science deposits are mixed, and the methods journal inverts to R.
That is the case for why a study of R and Python alone describes a field
while claiming to describe social science, and it is far more legible as
bars than as a paragraph of ratios.

`credit.pdf` shows what the counts credit. The most used Stata packages are
sorted by deposit share and split by what they do, and the tools that format
output sit at the top. The claim is the ordering, so the figure is the claim.

Deliberately plain: no gridlines competing with the bars, no color carrying
information that the labels do not, and a greyscale-safe palette, because a
reviewer prints things.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
TALLY = HERE.parent / "build" / "tally"

#: Stata packages whose job is formatting output rather than estimating.
#: Named rather than inferred, so a reader can disagree with the membership
#: and recompute. Same list the prose uses.
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

INK = "#1a1a1a"
FILL = "#4a4a4a"
HIGHLIGHT = "#111111"
LIGHT = "#bdbdbd"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9, length=3)
    ax.set_axisbelow(True)


#: Reader-facing names, and what each repository is.
SOURCE_LABEL = {
    "zenodo": "Zenodo (economics)",
    "dataverse": "Dataverse (political science)",
}


def languages(files: pd.DataFrame) -> Path:
    """Deposits containing each language, by repository.

    Two repositories, two disciplines: Zenodo's verified collections are
    economics and Harvard Dataverse's journal collections are mostly
    political science.

    One frame, grouped by `source`. It used to read a second tally directory
    and silently fall back to a single bar if that file was absent, so the
    figure could quietly stop making the comparison it exists to make.
    """
    analyzable_all = files[files["in_analysis_set"].astype(bool)]
    frames = {
        SOURCE_LABEL.get(source, source): group
        for source, group in analyzable_all.groupby("source", observed=True)
    }

    langs = ["stata", "r", "python"]
    labels = {"stata": "Stata", "r": "R", "python": "Python"}

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    width = 0.38
    for offset, (name, analyzable) in zip((-0.5, 0.5), frames.items(), strict=False):
        counts = analyzable.groupby("language", observed=True)["dataset_doi"].nunique()
        total = analyzable["dataset_doi"].nunique()
        shares = [100 * counts.get(lang, 0) / max(1, total) for lang in langs]
        positions = [i + offset * width for i in range(len(langs))]
        ax.bar(
            positions,
            shares,
            width=width,
            color=HIGHLIGHT if offset < 0 else LIGHT,
            edgecolor=INK,
            linewidth=0.6,
            label=name,
        )
        for x, share in zip(positions, shares, strict=True):
            ax.text(
                x, share + 1.5, f"{share:.0f}%", ha="center", fontsize=8.5, color=INK
            )

    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([labels[x] for x in langs])
    ax.set_ylabel("deposits containing the language (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax)
    fig.tight_layout()
    path = OUT / "languages.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def credit(usage: pd.DataFrame) -> Path:
    """The most used Stata packages, split by what they do."""
    top = usage[usage["language"] == "stata"].nlargest(12, "n_deposits").iloc[::-1]
    colors = [HIGHLIGHT if p in TABLE_MAKERS else LIGHT for p in top["package"]]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(
        top["package"],
        100 * top["share_of_deposits"],
        color=colors,
        edgecolor=INK,
        linewidth=0.6,
    )
    for y, (share, n) in enumerate(
        zip(top["share_of_deposits"], top["n_deposits"], strict=True)
    ):
        ax.text(100 * share + 0.8, y, f"{n:,}", va="center", fontsize=8, color=INK)

    ax.set_xlabel("share of deposits containing Stata (%)", fontsize=9, color=INK)
    ax.set_xlim(0, 100 * top["share_of_deposits"].max() * 1.18)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=HIGHLIGHT, edgecolor=INK, linewidth=0.6),
        plt.Rectangle((0, 0), 1, 1, facecolor=LIGHT, edgecolor=INK, linewidth=0.6),
    ]
    ax.legend(
        handles,
        ["formats output", "everything else"],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    _style(ax)
    fig.tight_layout()
    path = OUT / "credit.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


#: Panels, in the order agreement falls: by how much of a registry's traffic
#: is one package installing another.
PANELS = (("ssc", "Stata (SSC)"), ("cran", "R (CRAN)"), ("pypi", "Python (PyPI)"))


def downloads(versus: pd.DataFrame, floor: int = 5, n_labels: int = 3) -> Path:
    """Validated use against downloads, one panel per registry.

    Each point is a package the corpus uses in at least ``floor`` deposits. Both
    axes are logarithmic, because both counts span five orders of magnitude and
    a linear axis would show three packages and a smear. The labeled points are
    the ones furthest from agreement in each direction, chosen by rank
    difference and not by eye.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    for ax, (ecosystem, title) in zip(axes, PANELS, strict=True):
        d = versus[
            versus["ecosystem"].eq(ecosystem)
            & versus["downloads"].gt(0)
            & versus["n_deposits"].ge(floor)
        ].copy()
        d["gap"] = d["downloads"].rank(ascending=False) - d["n_deposits"].rank(
            ascending=False
        )
        ax.scatter(d["downloads"], d["n_deposits"], s=5, color=LIGHT, linewidths=0)
        groups = (d.nlargest(n_labels, "gap"), d.nsmallest(n_labels, "gap"))
        for group, align in zip(groups, ("right", "left"), strict=True):
            ax.scatter(group["downloads"], group["n_deposits"], s=11, color=HIGHLIGHT)
            # The overstated packages all sit on the floor of the plot, so
            # their labels are stacked upward with a leader, not overprinted.
            # Stacked in the order of the points themselves, so no leader
            # crosses another.
            key = "n_deposits" if align == "right" else "downloads"
            ordered = group.sort_values(key, ascending=align == "right")
            for step, row in enumerate(ordered.itertuples()):
                ax.annotate(
                    row.package,
                    (row.downloads, row.n_deposits),
                    xytext=(-8 if align == "right" else 6, 4 + 11 * step),
                    textcoords="offset points",
                    ha=align,
                    fontsize=7.5,
                    color=INK,
                    arrowprops={"arrowstyle": "-", "color": LIGHT, "lw": 0.5},
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.margins(x=0.16)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("downloads", fontsize=9, color=INK)
        _style(ax)
    axes[0].set_ylabel("deposits using the package", fontsize=9, color=INK)
    fig.tight_layout()
    path = OUT / "downloads.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def reach(versus_path: Path) -> Path:
    """Share of a registry's packages the corpus ever uses, by download decile.

    The other direction of the comparison. Downloads order the packages a
    field uses only loosely, but they say a good deal about whether it uses a
    package at all, and nearly all of that is in the top tenth.
    """
    from softverse.build.downloads import agreement, read

    found = agreement(read(versus_path))
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    styles = {"ssc": ("-", HIGHLIGHT), "cran": ("--", FILL), "pypi": (":", FILL)}
    for ecosystem, title in PANELS:
        shares = found[ecosystem]["registry"]["deciles"]
        linestyle, color = styles[ecosystem]
        ax.plot(range(1, 11), shares, linestyle=linestyle, color=color, lw=1.4)
        ax.annotate(
            title,
            (10, shares[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_xticks(range(1, 11))
    ax.set_xlim(1, 12.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("download decile within the registry, lowest to highest", fontsize=9)
    ax.set_ylabel("share used in any deposit", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    path = OUT / "reach.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    os.chdir(HERE)
    OUT.mkdir(exist_ok=True)
    files = pd.read_parquet(TALLY / "files.parquet")
    usage = pd.read_csv(TALLY / "usage_by_package.csv")

    versus = pd.read_csv(TALLY / "downloads_vs_use.csv")
    for path in (
        languages(files),
        credit(usage),
        downloads(versus),
        reach(TALLY / "downloads_vs_use.csv"),
    ):
        print(f"wrote {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

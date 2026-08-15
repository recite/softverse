"""Build the paper's figures from the released tables.

    uv run python paper/figures.py

Writes PDFs to `paper/figures/`. Two exhibits, each carrying an argument the
prose currently has to make in words:

`languages.pdf` shows the disciplinary split. Economics deposits are Stata,
political science deposits are mixed, and the methods journal inverts to R.
That is the case for why a study of R and Python alone describes a field
while claiming to describe social science, and it is far more legible as
bars than as a paragraph of ratios.

`credit.pdf` shows what the counts credit. The most used Stata packages are
sorted by deposit share and split by what they do, and the tools that format
output sit at the top. The claim is the ordering, so the figure is the claim.

Deliberately plain: no gridlines competing with the bars, no colour carrying
information that the labels do not, and a greyscale-safe palette, because a
reviewer prints things.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

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


def languages(presence: pd.DataFrame, files: pd.DataFrame) -> Path:
    """Deposits containing each language, by repository.

    Two repositories, two disciplines: Zenodo's verified collections are
    economics and Harvard Dataverse's journal collections are mostly
    political science.
    """
    legacy = TALLY.parent / "tally_dataverse_legacy" / "files.parquet"
    frames = {"Zenodo (economics)": files}
    if legacy.exists():
        frames["Dataverse (political science)"] = pd.read_parquet(legacy)

    langs = ["stata", "r", "python"]
    labels = {"stata": "Stata", "r": "R", "python": "Python"}

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    width = 0.38
    for offset, (name, frame) in zip((-0.5, 0.5), frames.items(), strict=False):
        analyzable = frame[frame["in_analysis_set"]]
        counts = analyzable.groupby("language")["dataset_doi"].nunique()
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


def main() -> int:
    os.chdir(HERE)
    OUT.mkdir(exist_ok=True)
    files = pd.read_parquet(TALLY / "files.parquet")
    usage = pd.read_csv(TALLY / "usage_by_package.csv")
    presence = pd.read_csv(TALLY / "language_presence.csv")

    for path in (languages(presence, files), credit(usage)):
        print(f"wrote {path.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

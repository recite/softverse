"""Numbered, captioned LaTeX floats for the paper's tables and figures.

Markdown pipe tables render as unnumbered blocks a reader cannot refer to,
which is why the first draft of this paper had six exhibits and no way to
cite any of them. These emit proper floats instead: booktabs rules, a caption
above, a label to reference, and a note beneath.

The note is not decoration. A reader who meets a table out of order should be
able to read it without hunting the prose for what a column means, and a
table that cannot survive that is a table that will be misread.
"""

from __future__ import annotations

import pandas as pd

#: Wrapped in a raw-LaTeX fence so pandoc passes it through untouched rather
#: than trying to parse the tabular as markdown.
_RAW_OPEN = "```{=latex}\n"
_RAW_CLOSE = "\n```\n"


def _tabular(df: pd.DataFrame, align: str, index: bool) -> str:
    """Just the `tabular` environment, with booktabs rules, from pandas."""
    body = df.to_latex(index=index, escape=True, column_format=align, na_rep="")
    inner = body.split(r"\begin{tabular}", 1)[1]
    inner = inner.rsplit(r"\end{tabular}", 1)[0]
    return r"\begin{tabular}" + inner + r"\end{tabular}"


def table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    note: str | None = None,
    align: str | None = None,
    index: bool = False,
) -> str:
    """A dataframe as a numbered table float."""
    ncols = len(df.columns) + (1 if index else 0)
    align = align or ("l" + "r" * (ncols - 1))
    parts = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        _tabular(df, align, index).strip(),
    ]
    if note:
        parts.append(rf"\caption*{{\footnotesize \textit{{Note:}} {note}}}")
    parts.append(r"\end{table}")
    return _RAW_OPEN + "\n".join(parts) + _RAW_CLOSE


def figure(
    path: str,
    caption: str,
    label: str,
    note: str | None = None,
    width: str = "0.92",
) -> str:
    """An image built by `paper/figures.py` as a numbered figure float."""
    parts = [
        r"\begin{figure}[htbp]",
        r"\centering",
        rf"\includegraphics[width={width}\linewidth]{{{path}}}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if note:
        parts.append(rf"\caption*{{\footnotesize \textit{{Note:}} {note}}}")
    parts.append(r"\end{figure}")
    return _RAW_OPEN + "\n".join(parts) + _RAW_CLOSE

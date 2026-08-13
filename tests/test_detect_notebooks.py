"""Tests for literate-document extraction.

v1 collected .Rmd and .ipynb from every source and parsed neither, so 1,159
.Rmd files across 31 deposits and 295 notebooks across 46 contributed nothing
while looking like deposits that simply had no code.
"""

from __future__ import annotations

import json

from softverse.detect.notebooks import (
    extract_notebook,
    extract_rmarkdown,
    split_chunks,
)
from softverse.model.enums import Construct, Language, ParseStatus

RMD = """\
---
title: "Analysis"
output: html_document
---

Some prose that mentions library(notapackage) in passing.

```{r setup, include=FALSE}
library(dplyr)
library(ggplot2)
```

More prose.

```{python}
import pandas as pd
```

The estimate is `r round(mean(x), 2)` and `r scales::percent(p)`.
"""


def names(result) -> set[str]:
    return {m.raw_name for m in result.mentions}


def test_chunks_are_split_by_engine():
    chunks = split_chunks(RMD)
    assert [c[0] for c in chunks] == [Language.R, Language.PYTHON]


def test_prose_is_not_code():
    """`library(notapackage)` in a paragraph is text, not a dependency."""
    assert "notapackage" not in names(extract_rmarkdown(RMD))


def test_r_and_python_chunks_both_resolve_to_their_own_language():
    """A .Rmd holds several languages; a mention must carry its own.

    Resolving a Python chunk against `rmarkdown` would send every one of its
    mentions to no registry and return UNKNOWN.
    """
    result = extract_rmarkdown(RMD)
    by_name = {m.raw_name: m.language for m in result.mentions}
    assert by_name["dplyr"] is Language.R
    assert by_name["pandas"] is Language.PYTHON


def test_inline_r_spans_are_read():
    assert "scales" in names(extract_rmarkdown(RMD))


def test_line_numbers_refer_to_the_document_not_the_chunk():
    """Otherwise every snippet points at the wrong place and the audit trail
    the schema promises is fiction."""
    result = extract_rmarkdown(RMD)
    dplyr = next(m for m in result.mentions if m.raw_name == "dplyr")
    assert RMD.splitlines()[dplyr.line - 1].strip() == "library(dplyr)"


def test_sweave_rnw_chunks():
    rnw = "Text.\n<<setup>>=\nlibrary(xtable)\n@\nMore text.\n"
    assert names(extract_rmarkdown(rnw)) == {"xtable"}


# -- notebooks ------------------------------------------------------------


def notebook(cells, language="python", nbformat=4) -> str:
    payload = {
        "metadata": {"kernelspec": {"language": language}},
        "nbformat": nbformat,
    }
    if nbformat >= 4:
        payload["cells"] = cells
    else:
        payload["worksheets"] = [{"cells": cells}]
    return json.dumps(payload)


def code(source: str) -> dict:
    return {"cell_type": "code", "source": source.splitlines(keepends=True)}


def test_notebook_imports():
    result = extract_notebook(notebook([code("import numpy as np\n")]))
    assert names(result) == {"numpy"}


def test_markdown_cells_are_ignored():
    nb = notebook(
        [{"cell_type": "markdown", "source": "import requests"}, code("import numpy")]
    )
    assert names(extract_notebook(nb)) == {"numpy"}


def test_kernel_language_is_read_not_assumed():
    """R and Stata kernels exist; assuming Python sends them to the wrong
    registry."""
    result = extract_notebook(notebook([code("library(dplyr)\n")], language="R"))
    assert names(result) == {"dplyr"}
    assert result.mentions[0].language is Language.R


def test_nbformat_3_worksheets_are_read():
    """v1 read only `cells`, so older notebooks silently yielded nothing."""
    nb = notebook([code("import scipy\n")], nbformat=3)
    assert names(extract_notebook(nb)) == {"scipy"}


def test_magics_do_not_break_the_cell():
    """`%matplotlib inline` is not valid Python; dropping the line must not
    cost the whole cell."""
    result = extract_notebook(
        notebook([code("%matplotlib inline\nimport seaborn as sns\n")])
    )
    assert names(result) == {"seaborn"}


def test_pip_install_is_recorded_as_installation_not_use():
    result = extract_notebook(notebook([code("!pip install torch\nimport numpy\n")]))
    constructs = {m.raw_name: m.construct for m in result.mentions}
    assert constructs["torch"] is Construct.SHELL_INSTALL
    assert constructs["numpy"] is Construct.IMPORT


def test_one_bad_cell_does_not_void_the_notebook():
    nb = notebook(
        [code("import numpy\n"), code("def broken(:\n"), code("import scipy\n")]
    )
    result = extract_notebook(nb)
    assert names(result) == {"numpy", "scipy"}
    assert result.report.status is ParseStatus.OK_WITH_ERRORS


def test_invalid_json_is_reported_not_silently_empty():
    result = extract_notebook("{ not json")
    assert result.mentions == []
    assert result.report.status is ParseStatus.SYNTAX_ERROR
    assert result.report.detail

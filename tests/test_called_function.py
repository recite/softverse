"""The function a mention refers to, where the source says.

`dplyr::select(x)` was recorded as a mention of `dplyr` with `select` parsed
and thrown away, so the corpus could say which packages published code loads
and never which parts of them it uses. Stata never had the problem, because
there the command is the call; these tests pin all three languages to the
same shape so one query answers the question for each.
"""

from __future__ import annotations

import json

from softverse.detect import python_, r, stata
from softverse.detect.notebooks import extract_notebook, extract_rmarkdown


def by_name(result) -> dict[str, str | None]:
    return {m.raw_name: m.called_function for m in result.mentions}


def test_r_namespace_call_keeps_the_function():
    assert by_name(r.extract(b"dplyr::select(-areas)")) == {"dplyr": "select"}


def test_r_internal_namespace_call_keeps_the_function():
    assert by_name(r.extract(b"MASS:::internal(x)")) == {"MASS": "internal"}


def test_r_string_left_operand_still_works():
    """`"m"::baz()` is legal R and was the one name the oracle caught us on."""
    assert by_name(r.extract(b'"m"::baz()')) == {"m": "baz"}


def test_r_library_names_a_package_and_no_function():
    """A loader names a package. Inventing a function for it would be a lie."""
    assert by_name(r.extract(b"library(dplyr)")) == {"dplyr": None}


def test_python_from_import_keeps_every_name():
    """One mention per imported name: `read_csv` and `DataFrame` are two
    things the code uses, and a fifth of these statements import several."""
    result = python_.extract("from pandas import read_csv, DataFrame")
    assert {m.called_function for m in result.mentions} == {"read_csv", "DataFrame"}
    assert {m.raw_name for m in result.mentions} == {"pandas"}


def test_python_star_import_names_no_function():
    result = python_.extract("from pandas import *")
    assert [m.called_function for m in result.mentions] == [None]


def test_python_plain_import_names_no_function():
    assert by_name(python_.extract("import pandas as pd")) == {"pandas": None}


def test_stata_command_is_its_own_called_function():
    """Deliberately redundant, so the same query works across languages."""
    result = stata.extract("reghdfe y x, absorb(id)\n")
    assert by_name(result) == {"reghdfe": "reghdfe"}


def test_stata_extracts_commands_at_all():
    """There was no test for this extractor, only for the lexer under it."""
    result = stata.extract("use data.dta, clear\nesttab using t.tex\n")
    assert {m.raw_name for m in result.mentions} == {"use", "esttab"}


def test_the_field_survives_a_knitr_document():
    """`_shift` rebuilds every mention. When it listed fields by hand, a new
    one was dropped for every `.Rmd` in the corpus and nothing errored."""
    rmd = "---\ntitle: t\n---\n\n```{r}\ndplyr::select(x)\n```\n"
    assert by_name(extract_rmarkdown(rmd)) == {"dplyr": "select"}


def test_the_field_survives_a_notebook():
    """The notebook cell walk rebuilds mentions too, and had the same trap."""
    notebook = json.dumps(
        {
            "metadata": {"kernelspec": {"language": "python"}},
            "nbformat": 4,
            "cells": [
                {"cell_type": "code", "source": ["from pandas import read_csv\n"]}
            ],
        }
    )
    assert by_name(extract_notebook(notebook)) == {"pandas": "read_csv"}


def calls(source: str) -> list[tuple[str, str | None]]:
    """Attribute-call mentions only: (package, function)."""
    return [
        (m.raw_name, m.called_function)
        for m in python_.extract(source).mentions
        if str(m.construct).endswith("attribute_call")
    ]


def test_an_alias_resolves_to_its_package():
    assert calls("import pandas as pd\npd.read_csv('f')\n") == [("pandas", "read_csv")]


def test_a_plain_module_name_resolves_too():
    assert calls("import pandas\npandas.read_csv('f')\n") == [("pandas", "read_csv")]


def test_a_method_call_on_an_object_stays_unresolved():
    """`df.head()` and `self.fit()` have no root in the alias map. If they
    did, this would be a list of every method the file calls."""
    assert calls(
        "import pandas as pd\ndf = pd.read_csv('f')\ndf.head()\nself.fit(x)\n"
    ) == [("pandas", "read_csv")]


def test_a_shadowed_alias_is_dropped_entirely():
    """`np = load()` means `np.mean` is no longer numpy, and the source does
    not say from which line. Losing the real calls beats inventing fake ones."""
    assert calls("import numpy as np\nnp = load()\nnp.mean(x)\n") == []


def test_a_nested_attribute_keeps_the_whole_path():
    assert calls("import numpy as np\nnp.linalg.norm(v)\n") == [
        ("numpy", "linalg.norm")
    ]


def test_a_submodule_alias_resolves_to_the_top_level_package():
    source = "import sklearn.model_selection as ms\nms.train_test_split(X)\n"
    assert calls(source) == [("sklearn", "train_test_split")]


def test_a_from_import_call_is_not_counted_twice():
    """`read_csv(f)` after `from pandas import read_csv` is already a mention
    at the import. Counting the call site as well counts one use twice."""
    assert calls("from pandas import read_csv\nread_csv('f')\n") == []


def test_two_names_in_one_import_get_distinct_uids():
    """They share a file, a byte offset and a package, so the mention uid has
    to include the function or the table has duplicate primary keys."""
    import uuid

    result = python_.extract("from pandas import read_csv, DataFrame")
    uids = {
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"f/{m.byte_start}/{m.raw_name}/{m.called_function or ''}",
        ).hex
        for m in result.mentions
    }
    assert len(uids) == len(result.mentions)


def install_names(source: str) -> list[str]:
    return [
        m.raw_name
        for m in r.extract(source.encode()).mentions
        if str(m.construct).endswith("install")
    ]


def test_install_github_accepts_a_pasted_url():
    """`split("/")[1]` on a URL is the empty string between the scheme's own
    slashes, so three deposits contributed a package named "" to the
    published unresolved-names table."""
    assert install_names('install_github("https://github.com/cran/ivpack")') == [
        "ivpack"
    ]


def test_install_github_still_reads_user_repo():
    assert install_names('install_github("hadley/dplyr")') == ["dplyr"]
    assert install_names('install_github("hadley/dplyr@v1.0")') == ["dplyr"]
    assert install_names('install_github("user/repo/subdir")') == ["repo"]


def test_install_target_with_no_package_is_dropped_not_blank():
    assert install_names('install_github("https://github.com/")') == []


def test_install_packages_is_unaffected():
    assert install_names('install.packages("dplyr")') == ["dplyr"]

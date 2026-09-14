"""A version written into the code is recorded, not thrown away.

Manifests are the main source of package versions and most deposits ship
none. Some state the version where they install: ``remotes::install_version``
in R, ``%pip install pandas==1.5.3`` in a notebook. The extractor already
parsed both calls and kept only the package name.
"""

from __future__ import annotations

import json

from softverse.detect.notebooks import extract_notebook
from softverse.detect.r import extract


def pins(mentions) -> dict[str, str | None]:
    return {m.raw_name: m.pinned_version for m in mentions}


def test_r_install_version_positional_and_keyword():
    source = (
        'remotes::install_version("fixest", "0.11.1")\n'
        'devtools::install_version("lfe", version = "2.8-8")\n'
        'install.packages("dplyr")\n'
    )
    assert pins(extract(source).mentions) == {
        "remotes": None,
        "fixest": "0.11.1",
        "devtools": None,
        "lfe": "2.8-8",
        "dplyr": None,
    }


def test_r_install_github_ref_is_the_pin():
    source = 'remotes::install_github("lrberge/fixest@v0.10.4")\n'
    assert pins(extract(source).mentions)["fixest"] == "v0.10.4"


def _notebook(line: str) -> str:
    return json.dumps(
        {
            "metadata": {"language_info": {"name": "python"}},
            "cells": [{"cell_type": "code", "source": [line]}],
        }
    )


def test_pip_magic_keeps_the_specifier():
    mentions = extract_notebook(
        _notebook("%pip install pandas==1.5.3 numpy>=1.24 requests\n")
    ).mentions
    assert pins(mentions) == {
        "pandas": "==1.5.3",
        "numpy": ">=1.24",
        "requests": None,
    }


def test_conda_single_equals_and_extras():
    mentions = extract_notebook(
        _notebook("!conda install -y scikit-learn=1.2.2 dask[complete]\n")
    ).mentions
    assert pins(mentions) == {"scikit-learn": "=1.2.2", "dask": None}

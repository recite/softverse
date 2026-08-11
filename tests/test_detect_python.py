"""Tests for the Python extractor.

The false-positive cases are drawn from the v1 output, which contained
``utils.plots``, ``odds_helper``, ``plot_common``, ``src.external.yolo5_face``
and ``pkgA``/``pkgB``/``pkgC`` as though they were PyPI distributions.
"""

from __future__ import annotations

import pytest

from softverse.detect.python_ import extract
from softverse.model.enums import Construct, ParseStatus


def names(source: str) -> list[str]:
    return [m.raw_name for m in extract(source).mentions]


def constructs(source: str) -> dict[str, Construct]:
    return {m.raw_name: m.construct for m in extract(source).mentions}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os", ["os"]),
        ("import numpy as np", ["numpy"]),
        ("import os, sys", ["os", "sys"]),
        ("from pandas import DataFrame", ["pandas"]),
        ("from collections.abc import Mapping", ["collections"]),
    ],
)
def test_basic_imports(source, expected):
    assert names(source) == expected


def test_submodules_collapse_to_the_top_level():
    """v1 emitted both, so sklearn and sklearn.model_selection were two
    separate 'packages' in the tally and both appeared in top-package lists."""
    assert names("import sklearn.model_selection") == ["sklearn"]
    assert names("from sklearn.linear_model import Lasso") == ["sklearn"]


def test_relative_imports_are_not_packages():
    """v1 ignored node.level, inventing a package for every relative import."""
    assert constructs("from .helpers import f")["helpers"] is Construct.LOCAL_RELATIVE
    assert constructs("from . import sibling")["."] is Construct.LOCAL_RELATIVE
    assert (
        constructs("from ..pkg.mod import thing")["pkg.mod"] is Construct.LOCAL_RELATIVE
    )


def test_absolute_import_of_a_similarly_named_module_is_still_absolute():
    assert constructs("from helpers import f")["helpers"] is Construct.IMPORT_FROM


def test_function_local_and_conditional_imports_are_found_and_flagged():
    source = "def f():\n    import scipy\n"
    assert names(source) == ["scipy"]
    result = extract("try:\n    import ujson\nexcept ImportError:\n    ujson = None\n")
    assert result.mentions[0].is_conditional


def test_future_import_is_recorded():
    """Recorded, then classified as stdlib by the resolver -- not dropped.
    Dropping rows makes the counts unreconcilable."""
    assert names("from __future__ import annotations") == ["__future__"]


def test_dynamic_import_with_a_literal():
    result = extract('import importlib\nm = importlib.import_module("torch")\n')
    dynamic = [m for m in result.mentions if m.construct is Construct.DYNAMIC_IMPORT]
    assert dynamic[0].raw_name == "torch"
    assert not dynamic[0].is_dynamic


def test_dynamic_import_with_a_variable_is_marked_unresolvable():
    result = extract("import importlib\nm = importlib.import_module(name)\n")
    dynamic = [m for m in result.mentions if m.construct is Construct.DYNAMIC_IMPORT]
    assert dynamic[0].is_dynamic


def test_imports_in_strings_and_comments_are_invisible():
    """Structural, because ast never sees inside a string literal."""
    assert names('x = "import requests"') == []
    assert names("# import requests") == []
    assert names('"""\nimport requests\n"""') == []


# -- the Python 2 problem -------------------------------------------------


def test_python2_file_recovers_its_imports_via_tokenize():
    """v1 returned zero imports for every Python 2 script, silently.

    `print "x"` is a SyntaxError to ast but tokenizes fine.
    """
    source = 'import numpy\nfrom scipy import stats\nprint "hello"\n'
    result = extract(source)
    assert result.report.status is ParseStatus.FALLBACK
    assert {m.raw_name for m in result.mentions} == {"numpy", "scipy"}


def test_fallback_still_excludes_strings():
    """The fallback reads tokens, not text, so this is not a regex."""
    source = 's = "import requests"\nimport numpy\nprint "py2"\n'
    result = extract(source)
    assert "requests" not in {m.raw_name for m in result.mentions}
    assert "numpy" in {m.raw_name for m in result.mentions}


def test_unparseable_and_untokenizable_is_reported_not_silently_empty():
    result = extract("def f(:\n  ???\n")
    assert result.mentions == []
    assert result.report.status in {ParseStatus.SYNTAX_ERROR, ParseStatus.FALLBACK}
    assert result.report.detail


def test_status_distinguishes_no_imports_from_failure():
    """The distinction v1 collapsed into one indistinguishable zero."""
    empty = extract("x = 1 + 1\n")
    assert empty.mentions == []
    assert empty.report.status is ParseStatus.OK

    broken = extract("import numpy\nprint 'py2'\n")
    assert broken.report.status is ParseStatus.FALLBACK


def test_mentions_carry_line_and_snippet():
    result = extract("x = 1\n\nimport pandas as pd\n")
    (mention,) = result.mentions
    assert mention.line == 3
    assert mention.snippet == "import pandas as pd"

"""Include edges: what a script runs, read with the same parsers as mentions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from softverse.detect.edges import includes, resolve
from softverse.model.enums import Language

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reachability import deposit_graph


def test_r_source_is_read_from_the_tree_not_the_text():
    text = (
        'source("code/clean.R")\n'
        'source(file = "setup.R")\n'
        "source(paste0(dir, 'x.R'))\n"
        '# source("commented.R")\n'
        'cat("source(\\"quoted.R\\")")\n'
    )
    found = includes(text, Language.R)
    assert sorted(found.targets) == ["code/clean.R", "setup.R"]
    assert found.n_unresolved == 1, "the computed path is counted, not guessed"


def test_stata_do_run_include_and_macros():
    text = (
        'do "01_clean.do"\n'
        "quietly do build\n"
        "run `path'/x.do\n"
        "include helpers\n"
        "* do commented.do\n"
        'display "do fake.do"\n'
    )
    found = includes(text, Language.STATA)
    assert sorted(found.targets) == ["01_clean.do", "build", "helpers"]
    assert found.n_unresolved == 1


def test_python_separates_run_magic_from_imports():
    found = includes(
        "import clean\nimport pandas as pd\n%run figs.py\n", Language.PYTHON
    )
    assert found.targets == ["figs.py"]
    assert sorted(found.imports) == ["clean", "pandas"]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("code/clean.R", "code/clean.R"),
        ("./setup.R", "setup.R"),
        ("build", "build.do"),
        ("../setup.R", "setup.R"),
        ("moved/clean.R", "code/clean.R"),
        ("pandas", None),
        ("ambiguous", None),
    ],
)
def test_resolution_prefers_the_path_as_written(target, expected):
    paths = [
        "code/clean.R",
        "setup.R",
        "build.do",
        "a/ambiguous.do",
        "b/ambiguous.do",
    ]
    assert resolve(target, "code/main.R", paths) == expected


def _deposit(tmp_path, files: dict[str, str]) -> pd.DataFrame:
    rows = []
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        rows.append(
            {
                "file_uid": relative,
                "relative_path": relative,
                "language": "r" if relative.endswith(".R") else "stata",
                "local_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def test_the_graph_finds_the_master_and_the_orphan(tmp_path):
    files = _deposit(
        tmp_path,
        {
            "master.R": 'source("clean.R")\n',
            "clean.R": 'source("helpers.R")\nlibrary(dplyr)\n',
            "helpers.R": "f <- function() 1\n",
            "old_draft.R": "library(plyr)\n",
        },
    )
    graph = deposit_graph(files, list(files["relative_path"]))
    assert graph["n_masters"] == 1
    assert sorted(graph["reachable_uids"]) == ["clean.R", "helpers.R", "master.R"]
    assert graph["n_reachable"] == 3
    assert graph["n_files"] == 4, "the orphan is counted, it just is not reached"
    assert graph["n_unfollowed_edges"] == 0


def test_a_cycle_has_no_entry_point_and_a_macro_edge_is_a_hole(tmp_path):
    files = _deposit(
        tmp_path,
        {
            "a.do": "do b.do\n",
            "b.do": "do a.do\n",
            "c.do": "do `path'/d.do\n",
            "d.do": "regress y x\n",
        },
    )
    graph = deposit_graph(files, list(files["relative_path"]))
    # a and b run each other, so neither is an entry point, and c's only edge
    # is one we cannot follow. Nothing is reachable, the walk terminates, and
    # the deposit drops out of the headline rather than reporting every file
    # as dead code.
    assert graph["n_masters"] == 0
    assert graph["n_macro_edges"] == 1
    assert graph["n_reachable"] == 0

"""A vendored package must not be mistaken for the author's own program.

Stata resolution runs `local_program` -> `builtin` -> SSC index, and the first
step is right: a command the author defined is not a package. But the
local-program pass read *every* `.do` and `.ado` in a deposit, including the
ones the hygiene rules had already marked as somebody else's library.

Replication packages routinely ship `reghdfe.ado` so their code runs without
installing anything. That file defines `program reghdfe`, so every call to
`reghdfe` in the author's own scripts resolved to their own program and the
package went uncounted.

Measured before the fix: 21,004 mentions across 110 deposits and 279 distinct
packages. `reghdfe` lost 43 deposits of 299, `gegen` 8 of 36, `binscatter` 17
of 105. The bias runs one way and lands hardest on the packages popular enough
to be worth vendoring -- which are the ones the paper ranks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softverse.build.pipeline import CorpusFile, build
from softverse.registries.resolve import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry(
        cran=frozenset(),
        cran_archive=frozenset(),
        bioconductor=frozenset(),
        pypi=frozenset(),
        julia=frozenset(),
        stata_commands={"reghdfe": ("reghdfe",), "esttab": ("estout",)},
        stata_builtins=frozenset({"regress", "use"}),
        lock_id="test",
    )


def _deposit(tmp_path: Path, files: dict[str, str]) -> list[CorpusFile]:
    out = []
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        out.append(
            CorpusFile(
                path=path,
                dataset_doi="doi:a",
                collection_id="restud",
                source="zenodo",
                relative_path=relative,
                deposit_year=2024,
            )
        )
    return out


def _resolutions(result, name: str) -> set[str]:
    return {m["resolution"] for m in result.mentions if m["normalized_name"] == name}


def test_a_vendored_package_is_not_the_authors_program(tmp_path, registry):
    """The bug, end to end."""
    corpus = _deposit(
        tmp_path,
        {
            "code/ado/reghdfe.ado": "program reghdfe, eclass\n  regress `0'\nend\n",
            "code/01_analysis.do": "use data\nreghdfe y x, absorb(id)\n",
        },
    )
    result = build(corpus, registry, ssc_shipped=frozenset({"reghdfe"}))
    assert _resolutions(result, "reghdfe") == {"known_current"}, (
        "a vendored copy of reghdfe must not turn the author's use of it "
        "into a local program"
    )


def test_the_authors_own_helper_is_still_local(tmp_path, registry):
    """The guard against over-correcting.

    This pass exists because helpers defined in one file and called from
    another were showing up as unknown packages -- `wins_top1`, `testgood`,
    `preliminaries` sat near the top of that list. That must keep working.
    """
    corpus = _deposit(
        tmp_path,
        {
            "code/helpers.do": "program define myhelper\n  regress y x\nend\n",
            "code/01_analysis.do": "use data\nmyhelper\n",
        },
    )
    result = build(corpus, registry, ssc_shipped=frozenset({"reghdfe"}))
    assert _resolutions(result, "myhelper") == {"local_program"}


def test_a_program_defined_under_an_ado_tree_is_not_the_authors(tmp_path, registry):
    """The path rule, which catches what the basename rule cannot.

    `_stubstar2names` is an official Stata internal, so it is in no SSC
    package and the basename rule never fires. One deposit vendors Stata's
    whole `ado/base` tree -- 7,094 files -- and contributed 46% of the
    corpus's unresolved Stata mentions on its own.
    """
    corpus = _deposit(
        tmp_path,
        {
            "Codes/ado/base/_/_stubstar2names.ado": (
                "program _stubstar2names, sclass\n  regress y x\nend\n"
            ),
            "Codes/01_analysis.do": "use data\nregress y x\n",
        },
    )
    result = build(corpus, registry, ssc_shipped=frozenset())
    vendored = [
        f for f in result.files if f["relative_path"].startswith("Codes/ado/base/")
    ]
    assert vendored and all(
        f["parse_status"] == "skipped_vendored" for f in vendored
    ), "a file under an ado/ tree must not be analyzed as research code"


def test_both_rules_together_on_one_deposit(tmp_path, registry):
    """A deposit that vendors an SSC package *and* part of Stata's own tree."""
    corpus = _deposit(
        tmp_path,
        {
            "packages/reghdfe.ado": "program reghdfe, eclass\nend\n",
            "packages/esttab.ado": "program esttab, rclass\nend\n",
            "ado/base/_/_parse.ado": "program _parse\nend\n",
            "code/own.do": "program define mysummary\nend\n",
            "code/run.do": "use data\nreghdfe y x\nesttab using t.tex\nmysummary\n",
        },
    )
    result = build(corpus, registry, ssc_shipped=frozenset({"reghdfe", "esttab"}))
    assert _resolutions(result, "reghdfe") == {"known_current"}
    assert _resolutions(result, "esttab") == {"known_current"}
    assert _resolutions(result, "mysummary") == {"local_program"}

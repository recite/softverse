"""Adversarial tests for the R extractor.

Each false-positive case is a defect measured in the v1 output, not a
hypothetical. v1's R tally contained ``std`` (from C++ inside a string),
``pkg``/``lib``/``x`` (loop variables from ``character.only=TRUE``), ``install``
(a ``p_load`` keyword argument), and literal parser debris such as
``lmtest, quietly = TRUE`` (46 occurrences).
"""

from __future__ import annotations

import pytest

from softverse.detect.r import extract
from softverse.model.enums import Construct, ParseStatus


def names(source: str) -> list[str]:
    return [m.raw_name for m in extract(source).mentions]


def constructs(source: str) -> dict[str, Construct]:
    return {m.raw_name: m.construct for m in extract(source).mentions}


# -- the common case ------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("library(ggplot2)", ["ggplot2"]),
        ('library("ggplot2")', ["ggplot2"]),
        ("require(dplyr)", ["dplyr"]),
        ("require('dplyr')", ["dplyr"]),
        ('requireNamespace("rstan")', ["rstan"]),
        # v1's pattern demanded `["']\\s*\\)`, so this -- the common form, in 292
        # corpus files -- was missed entirely.
        ('requireNamespace("rstan", quietly = TRUE)', ["rstan"]),
        ('loadNamespace("Matrix")', ["Matrix"]),
        ("library(ggplot2)\nlibrary(dplyr)", ["ggplot2", "dplyr"]),
    ],
)
def test_loaders(source, expected):
    assert names(source) == expected


def test_multiline_call():
    assert names("library(\n  dplyr\n)") == ["dplyr"]


def test_namespace_operators():
    assert names("MASS::ginv(m)") == ["MASS"]
    assert constructs("MASS::ginv(m)")["MASS"] is Construct.NAMESPACE_OP


def test_triple_colon_is_recorded_separately():
    """Reaching into a package's internals is a different behaviour."""
    assert constructs("pkg:::hidden()")["pkg"] is Construct.NAMESPACE_INTERNAL


def test_pacman_p_load():
    assert set(names("pacman::p_load(here, glue)")) == {"pacman", "here", "glue"}


# -- false positives v1 produced -----------------------------------------


def test_package_name_inside_a_string_is_not_a_mention():
    """The structural guarantee: string content never forms a call node."""
    assert names('cat("run library(dplyr) first")') == []
    assert names('stop("please install ggplot2 via library(ggplot2)")') == []


def test_cpp_inside_a_string_does_not_yield_std():
    """v1 reported `std` as an R package, from embedded C++."""
    assert names('cppFunction("std::vector<int> f() { return 1; }")') == []


def test_commented_code_is_not_a_mention():
    assert names("# library(commentedpkg)") == []
    assert names("library(real)  # library(fake)") == ["real"]


def test_character_only_records_dynamic_rather_than_the_loop_variable():
    """v1 emitted `pkg` as a package *and* lost the real list.

    Recording it as dynamic is both more honest and a reportable statistic:
    221 corpus R files use this idiom.
    """
    result = extract("for (pkg in pkgs) library(pkg, character.only = TRUE)")
    (mention,) = result.mentions
    assert mention.raw_name == "pkg"
    assert mention.is_dynamic
    assert mention.construct is Construct.DYNAMIC_UNRESOLVED


def test_xfun_attach_helpers_are_multi_loaders():
    """`renv`'s own test fixture lists these among the calls it expects found.

    That fixture is ground truth we did not write: it enumerates thirteen
    loader spellings and states in a comment which must be detected. Checking
    against it found two we did not handle -- `xfun::pkg_attach` and
    `pkg_attach2` -- which no amount of agreement with our own oracle would
    have surfaced, because a shared blind spot looks like consensus.
    """
    assert {m.raw_name for m in extract('pkg_attach2("i", "j")').mentions} == {"i", "j"}


def test_a_vector_of_names_inside_a_multi_loader_is_unpacked():
    """`xfun::pkg_attach(c("g", "h"))` names two packages, not zero.

    Positional arguments were read one node at a time, so a `c(...)` call was
    neither a string nor an identifier and fell through silently.
    """
    result = extract('xfun::pkg_attach(c("g", "h"))')
    assert {m.raw_name for m in result.mentions} == {"xfun", "g", "h"}


def test_a_named_argument_in_a_multi_loader_is_still_not_a_package():
    """The guard that makes the unpacking safe.

    v1 split on commas and tallied `install`, `update` and `dependencies` as
    CRAN packages; unpacking vectors must not reopen that.
    """
    result = extract("p_load(dplyr, ggplot2, install = FALSE, character.only = TRUE)")
    assert {m.raw_name for m in result.mentions} == {"dplyr", "ggplot2"}


def test_a_quoted_namespace_is_still_a_namespace():
    """`"m"::baz()` is legal R and we were dropping it.

    Found by R's own parser: renv's test fixture includes this line among the
    cases it expects to be detected, and it was the single name our extractor
    missed across 25,924 name-file pairs in the corpus.
    """
    result = extract('"m"::baz()')
    assert [m.raw_name for m in result.mentions] == ["m"]
    assert result.mentions[0].construct is Construct.NAMESPACE_OP


def test_a_variable_passed_to_package_version_is_dynamic_not_a_package():
    """The same idiom one function along, and it was being missed.

    `library(pkg, character.only = TRUE)` was already handled. But
    `packageVersion(pkg)` and `requireNamespace(pkg)` took the same fallback
    -- "if it is not a string, use the identifier's text" -- and emitted the
    *variable name* as a package. `pkg` (90 mentions) and `package` (28) were
    the two largest unresolved R names in the corpus for this reason alone,
    which reads as poor registry coverage rather than as a helper function.
    """
    for source in (
        "v <- packageVersion(pkg)",
        "requireNamespace(pkg, quietly = TRUE)",
        "if (system.file(package = pkg) == '') stop()",
    ):
        (mention,) = extract(source).mentions
        assert mention.raw_name == "pkg", source
        assert mention.construct is Construct.DYNAMIC_UNRESOLVED, source
        assert mention.is_dynamic, source


def test_a_string_passed_to_package_version_is_still_a_package():
    """The guard keys on the node type, not the name.

    1,316 of these mentions do resolve -- `plm` 144 times, `Matrix` 62,
    `lme4` 58 -- and they are written as string literals. A fix that
    suppressed the construct entirely would trade 177 false positives for
    1,316 false negatives.
    """
    for source, expected in (
        ('v <- packageVersion("plm")', "plm"),
        ('requireNamespace("dplyr", quietly = TRUE)', "dplyr"),
    ):
        (mention,) = extract(source).mentions
        assert mention.raw_name == expected
        assert not mention.is_dynamic, source


def test_character_only_false_is_a_normal_load():
    result = extract("library(dplyr, character.only = FALSE)")
    assert [m.raw_name for m in result.mentions] == ["dplyr"]
    assert not result.mentions[0].is_dynamic


def test_p_load_keyword_arguments_are_not_packages():
    """v1 stripped from the first `=`, tallying `install` as a CRAN package."""
    found = set(names("p_load(dplyr, ggplot2, install = FALSE, update = FALSE)"))
    assert found == {"dplyr", "ggplot2"}


def test_quietly_argument_never_becomes_part_of_a_name():
    """v1 shipped a package literally named `lmtest, quietly = TRUE` (46 rows)."""
    found = names("library(lmtest, quietly = TRUE)")
    assert found == ["lmtest"]


# -- installation is not use ---------------------------------------------


def test_install_calls_are_marked_as_installation():
    assert constructs('install.packages("brms")')["brms"] is Construct.INSTALL


def test_install_github_extracts_the_repo_not_the_user():
    assert names('devtools::install_github("hadley/emo")') == ["devtools", "emo"]


def test_install_github_handles_refs_and_subdirs():
    assert "emo" in names('install_github("hadley/emo@v1.0")')
    assert "emo" in names('install_github("hadley/emo/sub")')


# -- context --------------------------------------------------------------


def test_conditional_loads_are_flagged():
    result = extract('if (!requireNamespace("brms")) { install.packages("brms") }')
    brms = [m for m in result.mentions if m.raw_name == "brms"]
    assert any(m.is_conditional for m in brms)


def test_mentions_carry_line_and_snippet():
    result = extract("x <- 1\n\nlibrary(ggplot2)\n")
    (mention,) = result.mentions
    assert mention.line == 3
    assert mention.snippet == "library(ggplot2)"


def test_report_is_always_present():
    """No extractor may return an empty list without saying why."""
    result = extract("x <- 1")
    assert result.mentions == []
    assert result.report is not None
    assert result.report.status is ParseStatus.OK


def test_broken_source_is_reported_not_silently_empty():
    result = extract("library(((((")
    assert result.report is not None
    assert result.report.status in {
        ParseStatus.SYNTAX_ERROR,
        ParseStatus.OK_WITH_ERRORS,
    }


def test_r_grammar_contract():
    """Pin the node types and fields the walker depends on.

    Grammar node names change between tree-sitter-r versions (older grammars
    used `namespace_get`). Without this, an upgrade would return zero mentions
    for every R file in the corpus and look like a finding.
    """
    from tree_sitter_language_pack import get_parser

    src = b'library(x)\npkg::fun()\n"s"\n# c\n'
    root = get_parser("r").parse(src).root_node
    types = set()
    stack = [root]
    while stack:
        node = stack.pop()
        types.add(node.type)
        stack.extend(node.children)
    assert {
        "call",
        "arguments",
        "argument",
        "namespace_operator",
        "string",
        "string_content",
        "comment",
        "identifier",
    } <= types

    call = next(c for c in root.children if c.type == "call")
    assert call.child_by_field_name("function") is not None
    assert call.child_by_field_name("arguments") is not None
    ns = next(
        c
        for c in root.children
        if c.type == "call"
        and c.child_by_field_name("function").type == "namespace_operator"
    )
    op = ns.child_by_field_name("function")
    assert op.child_by_field_name("lhs") is not None
    assert op.child_by_field_name("rhs") is not None


def test_data_first_argument_is_a_dataset_not_a_package():
    """Found by the renv cross-check, not by a test I thought to write.

    `data("World")` loads the World dataset from an already-loaded package.
    Reading the first positional as a package name invented a package called
    World. Only the explicit `package=` keyword is reliable here.
    """
    assert names('data("World")') == []
    assert names("data(mtcars)") == []
    assert names('data("World", package = "tmap")') == ["tmap"]


def test_vignette_follows_the_same_rule():
    assert names('vignette("intro")') == []
    assert names('vignette("intro", package = "dplyr")') == ["dplyr"]


def test_package_version_first_argument_is_a_package():
    """The other group: here the first positional genuinely is the package."""
    assert names('packageVersion("dplyr")') == ["dplyr"]
    assert names('citation("lme4")') == ["lme4"]


#: Verbatim from `renv/tests/testthat/resources/code.R` (renv 1.2.3), which
#: states its own expected answer in comments: every lower-case name must be
#: detected, the upper-case one must not.
RENV_FIXTURE = """
# should be parsed as dependencies (use only lower-case letters for package names)
library(a)
library("b")
base::library(c)
base::library("d", character.only = TRUE)
requireNamespace("e")
base::requireNamespace("f", quietly = TRUE)
xfun::pkg_attach(c("g", "h"))
pkg_attach2("i", "j")
k::foo()
l:::bar()
"m"::baz()

# should NOT be parsed as dependencies (use only upper-case names for package names)
library(A, character.only = TRUE)
"""


def test_against_renvs_own_expectations():
    """Ground truth written by someone else, which is the point.

    Agreement with our own oracle cannot find a blind spot the two share.
    This fixture is renv's, its answer is stated in its comments, and checking
    against it found two loader spellings we did not handle and one we
    detected that we should not.
    """
    result = extract(RENV_FIXTURE)
    static = {m.raw_name for m in result.mentions if not m.is_dynamic}

    for expected in "abcdefghijklm":
        assert expected in static, f"renv expects {expected!r} to be detected"

    # `library(A, character.only = TRUE)` names a *variable* A, so the package
    # is unknowable. Recorded as dynamic rather than dropped, which is a
    # stronger claim than renv's fixture makes: it only requires that A not be
    # reported as a package.
    assert "A" not in static
    assert any(m.raw_name == "A" and m.is_dynamic for m in result.mentions), (
        "the idiom should still be visible as dynamic, not silently dropped"
    )

    # `base::` and `xfun::` are real namespace mentions in this source, and
    # resolution -- not extraction -- is what later marks `base` as base R.
    assert static - set("abcdefghijklm") == {"base", "xfun"}

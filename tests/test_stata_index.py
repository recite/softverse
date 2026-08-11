"""Tests for the Stata command -> package index.

Fixtures are verbatim excerpts of real SSC manifests, so these assert against
what the mirror actually serves rather than against an idealized format.
"""

from __future__ import annotations

from datetime import date

from softverse.stata.index import (
    Evidence,
    ambiguous_commands,
    commands_for,
    confirms_namesake,
    defined_programs,
    is_helper,
    parse_pkg,
)

ESTOUT_PKG = """\
d 'ESTOUT': module to make regression tables
d
d  estout produces a table of regression results.
d
d KW: estimates
d Requires: Stata version 8.2
d
d Distribution-Date: 20260413
d
d Author: Ben Jann, University of Bern
d Support: email jann@@soz.unibe.ch
d
f estout.ado
f estout.hlp
f esttab.ado
f eststo.ado
f estadd.ado
f ../_/_eststo.ado
"""

REGHDFE_PKG = """\
d 'REGHDFE': module for linear regression with many fixed effects
d Distribution-Date: 20260111
d Author: Sergio Correia, Board of Governors
f reghdfe.ado
f reghdfe.mata
f reghdfe_p.ado
f reghdfe_estat.ado
f reghdfe_footnote.ado
f reghdfe_header.ado
"""


def test_parses_metadata():
    pkg = parse_pkg(ESTOUT_PKG, "estout")
    assert pkg.package == "estout"
    assert pkg.distribution_date == date(2026, 4, 13)
    assert pkg.author is not None and "Ben Jann" in pkg.author


def test_ado_files_ignore_directory_prefixes():
    """Manifests use relative paths like `f ../_/_eststo.ado`."""
    pkg = parse_pkg(ESTOUT_PKG, "estout")
    assert "_eststo" in pkg.ado_files
    # Non-ado files (.hlp, .mata) are not commands.
    assert "estout.hlp" not in pkg.ado_files
    assert "reghdfe.mata" not in parse_pkg(REGHDFE_PKG, "reghdfe").ado_files


def test_one_package_can_expose_several_commands():
    """The mapping a name-only package list would miss: esttab belongs to estout."""
    commands = {
        c for c, helper in commands_for(parse_pkg(ESTOUT_PKG, "estout")) if not helper
    }
    assert commands == {"estout", "esttab", "eststo", "estadd"}


def test_leading_underscore_is_a_helper():
    """Stata's internal-subroutine convention."""
    assert is_helper("_eststo", "estout", {"estout", "_eststo"})
    helpers = {c for c, h in commands_for(parse_pkg(ESTOUT_PKG, "estout")) if h}
    assert helpers == {"_eststo"}


def test_suffix_helpers_are_excluded():
    """`f foo.ado` means the package ships a file, not that it exposes `foo`.

    reghdfe ships a predict helper, an estat hook, a footnote routine and a
    header builder. None is a command a user types; counting them would inflate
    the index with four phantom packages' worth of commands.
    """
    pairs = dict(commands_for(parse_pkg(REGHDFE_PKG, "reghdfe")))
    assert pairs["reghdfe"] is False
    for helper in ("reghdfe_p", "reghdfe_estat", "reghdfe_footnote", "reghdfe_header"):
        assert pairs[helper] is True, helper


def test_a_suffix_alone_does_not_make_a_helper():
    """`_p` only signals a helper when the stem is a sibling command.

    Otherwise a legitimately named command ending in a helper-ish suffix -- with
    no corresponding base command -- would be wrongly dropped.
    """
    assert not is_helper("bootstrap_p", "somepkg", siblings={"bootstrap_p"})
    assert is_helper("bootstrap_p", "somepkg", siblings={"bootstrap", "bootstrap_p"})


def test_defined_programs_includes_internal_subroutines():
    """Documents why namesake confirmation exists rather than raw extraction.

    esttab.ado defines sixteen programs; only one is the user command.
    """
    ado = """\
program define MakeTeXColspec
end
program define esttab
end
program CheckScalarOpt
end
"""
    assert defined_programs(ado) == {"MakeTeXColspec", "esttab", "CheckScalarOpt"}


def test_confirms_namesake_follows_ado_path_semantics():
    """Typing `foo` runs foo.ado, which must define a program named foo."""
    assert confirms_namesake("program define winsor2\nend\n", "winsor2")
    assert not confirms_namesake("program define somethingelse\nend\n", "winsor2")


def test_confirms_namesake_handles_stata_abbreviations():
    """`program`, `prog`, `pr` and an optional `define`/`def` are all legal."""
    for source in (
        "program define foo",
        "program foo",
        "prog def foo",
        "pr foo",
        "  capture program drop foo\n  program define foo",
    ):
        assert confirms_namesake(source, "foo"), source


def test_ambiguous_commands_are_reported_not_resolved():
    """Two packages claiming one command is recorded, never silently decided."""
    rows = [
        {"command": "xtabond2", "package": "xtabond2", "is_helper": False},
        {"command": "shared", "package": "pkg_a", "is_helper": False},
        {"command": "shared", "package": "pkg_b", "is_helper": False},
        {"command": "_h", "package": "pkg_a", "is_helper": True},
    ]
    assert ambiguous_commands(rows) == {"shared": ["pkg_a", "pkg_b"]}


def test_evidence_levels_are_distinct():
    assert Evidence.FILENAME != Evidence.PROGRAM_DEFINE


def test_impossible_distribution_date_does_not_abort():
    """Real manifests carry hand-typed dates like month 00 or day 32.

    Found by crawling: one bad date killed a 4,000-package run.
    """
    pkg = parse_pkg("d Distribution-Date: 20200015\nf foo.ado\n", "foo")
    assert pkg.distribution_date is None
    assert pkg.ado_files == ["foo"]

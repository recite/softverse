"""Tests for the Stata command -> package index.

Fixtures are verbatim excerpts of real SSC manifests, so these assert against
what the mirror actually serves rather than against an idealized format.
"""

from __future__ import annotations

from datetime import date

from softverse.stata.index import (
    Evidence,
    Manifest,
    _toc_entries,
    ambiguous_commands,
    commands_for,
    confirms_namesake,
    defined_programs,
    fetch_net_manifests,
    index_rows,
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
    assert pkg.author is not None
    assert "Ben Jann" in pkg.author


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


def _claim(command, package, source="ssc", *, helper=False, documented=True):
    return {
        "command": command,
        "package": package,
        "source": source,
        "is_helper": helper,
        "is_documented": documented,
    }


def test_ambiguous_commands_are_reported_not_resolved():
    """Two packages claiming one command is recorded, never silently decided."""
    rows = [
        _claim("xtabond2", "xtabond2"),
        _claim("shared", "pkg_a"),
        _claim("shared", "pkg_b"),
        _claim("_h", "pkg_a", helper=True),
    ]
    assert ambiguous_commands(rows) == {"shared": ["pkg_a", "pkg_b"]}


def test_a_journal_mirror_of_an_ssc_package_is_not_an_ambiguity():
    """`esttab` is on SSC and in SJ 14-2: one package, published twice."""
    rows = [
        _claim("esttab", "estout"),
        _claim("esttab", "st0085", "stata_journal"),
        _claim("grc1leg", "st0357", "stata_journal", documented=False),
        _claim("grc1leg", "other", "stata_journal", documented=False),
    ]
    assert ambiguous_commands(rows) == {}


def test_evidence_levels_are_distinct():
    assert Evidence.FILENAME != Evidence.PROGRAM_DEFINE


def test_impossible_distribution_date_does_not_abort():
    """Real manifests carry hand-typed dates like month 00 or day 32.

    Found by crawling: one bad date killed a 4,000-package run.
    """
    pkg = parse_pkg("d Distribution-Date: 20200015\nf foo.ado\n", "foo")
    assert pkg.distribution_date is None
    assert pkg.ado_files == ["foo"]


# -- three archives, and what a journal package may be credited with -------


RENVARS_PKG = """\
d STB-60 dm88.  Renaming variables, multiply and systematically
d STB insert by Nicholas J. Cox, University of Durham, UK
f dm88/renvars.ado
f dm88/renvars.hlp
"""

#: Verbatim shape of SJ 14-4 st0357, which bundles someone else's command.
STCOXCAL_PKG = """\
d SJ14-4 st0357. Tools for checking calibration...
f st0357/stcoxcal.ado
f st0357/stcoxcal.sthlp
f st0357/grc1leg.ado
"""

MOREMATA_PKG = """\
d 'MOREMATA': module (Mata) to provide various functions
f lmoremata.mlib
f moremata.hlp
"""

ISSUE_TOC = """\
d Stata Journal volume 26, issue 3
t .. Other Stata Journals
p dm0085_4     Update: A set of utilities for managing
p -            missing values
p gr0104       Speaking Stata
"""


def rows_for(
    text: str, package: str, source: str, issue: str | None = None
) -> list[dict]:
    return index_rows(
        [Manifest(source, package, "http://x", text, issue)], date(2026, 9, 17)
    )


def test_a_toc_lists_packages_without_its_continuation_lines():
    assert _toc_entries(ISSUE_TOC, "p") == ["dm0085_4", "gr0104"]
    assert _toc_entries(ISSUE_TOC, "t") == []


def test_a_bulletin_package_indexes_under_its_directory_prefixed_paths():
    (row,) = rows_for(RENVARS_PKG, "dm88", "stb", "stb60")
    assert (row["command"], row["package"], row["issue"]) == (
        "renvars",
        "dm88",
        "stb60",
    )
    assert row["is_documented"]


def test_a_bundled_copy_is_shipped_but_not_documented():
    """st0357 ships `grc1leg.ado` and no help for it: a dependency, not its own.

    Crediting it would have moved 486 deposits to a Cox calibration package.
    """
    documented = {
        r["command"]: r["is_documented"]
        for r in rows_for(STCOXCAL_PKG, "st0357", "stata_journal")
    }
    assert documented == {"stcoxcal": True, "grc1leg": False}


def test_a_journal_update_is_the_same_package():
    (row,) = rows_for(RENVARS_PKG, "dm88_1", "stata_journal", "sj5-4")
    assert row["package"] == "dm88"


def test_the_newest_issue_wins_and_a_command_is_listed_once():
    manifests = [
        Manifest("stata_journal", "dm88_1", "u", RENVARS_PKG, "sj5-4"),
        Manifest("stata_journal", "dm88", "u", RENVARS_PKG, "sj1-1"),
    ]
    (row,) = index_rows(manifests, date(2026, 9, 17))
    assert row["issue"] == "sj5-4"


def test_a_package_with_no_ado_is_still_a_package():
    """`ssc install moremata` named something the index had no row for."""
    (row,) = rows_for(MOREMATA_PKG, "moremata", "ssc")
    assert (row["package"], row["evidence"]) == ("moremata", Evidence.PACKAGE_ONLY)
    assert row["is_helper"], "so it is never resolved as a command"


# -- the sites `net install ..., from(URL)` names --------------------------

GRC1LEG_PKG = """\
d grc1leg.  Combine graphs into one graph with a common legend.
d Program by Vince Wiggins, StataCorp
f grc1leg.ado
f grc1leg.hlp
"""


class _Site:
    """Stands in for the network: one live site, everything else gone."""

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages, self.asked = pages, []

    def get(self, url: str) -> str | None:
        self.asked.append(url)
        return self.pages.get(url)


def test_a_net_site_is_indexed_under_its_host():
    site = _Site({"http://www.stata.com/users/vwiggins/grc1leg.pkg": GRC1LEG_PKG})
    found, dead = fetch_net_manifests(
        [("grc1leg", "http://www.stata.com/users/vwiggins/")], client=site
    )
    (row,) = index_rows(found, date(2026, 9, 17))
    assert (row["command"], row["source"], row["issue"]) == (
        "grc1leg",
        "net",
        "stata.com",
    )
    assert row["is_documented"]
    assert dead == []


def test_a_dead_site_is_reported_not_dropped():
    site = _Site({"http://gone.example/x.pkg": "<html>404</html>"})
    found, dead = fetch_net_manifests(
        [("x", "http://gone.example"), ("y", "http://gone.example")], client=site
    )
    assert found == []
    assert dead == ["http://gone.example/x.pkg", "http://gone.example/y.pkg"]


def test_a_from_pointing_at_an_archive_already_crawled_is_not_a_net_site():
    site = _Site({})
    fetch_net_manifests(
        [
            ("st0085_2", "http://www.stata-journal.com/software/sj14-2"),
            ("estout", "http://fmwww.bc.edu/repec/bocode/e"),
            ("grc1leg", "not a url"),
        ],
        client=site,
    )
    assert site.asked == []


def test_a_named_site_is_crawled_whole_from_its_own_toc():
    """The corpus names one package; the site's toc lists its neighbours."""
    base = "http://www.stata.com/users/vwiggins"
    site = _Site(
        {
            f"{base}/stata.toc": "d Materials\np grc1leg combine graphs\np finirr rate of return\n",
            f"{base}/grc1leg.pkg": GRC1LEG_PKG,
            f"{base}/finirr.pkg": "d finirr\nf finirr.ado\nf finirr.hlp\n",
        }
    )
    found, dead = fetch_net_manifests([("grc1leg", base)], client=site)
    assert sorted(m.package for m in found) == ["finirr", "grc1leg"]
    assert dead == []


def test_platform_specific_manifest_lines_name_the_file_not_the_platform():
    """`f WIN64 usespss.ado` made `usespss` a package with no command."""
    text = "f  WIN64 usespss.ado\nf  usespss.hlp\nG WIN usespss.win32 usespss.plu\ng WIN64 b_win.plugin b.plugin\n"
    pkg = parse_pkg(text, "usespss")
    assert pkg.ado_files == ["usespss"]
    assert pkg.files[-2:] == ["usespss.plu", "b.plugin"]
    assert "usespss" in pkg.documented

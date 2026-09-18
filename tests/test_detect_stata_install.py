"""Stata provisioning lines: what they name, and where they fetch it from.

Every case is a spelling measured in the corpus. Of its 12,565 `ssc install`
and `net install` lines, 3,808 were being recorded.
"""

from __future__ import annotations

import pytest

from softverse.detect.stata import extract
from softverse.model.enums import Construct


def found(source: str) -> list[tuple[str, Construct, str | None]]:
    return [
        (m.raw_name, m.construct, m.remote)
        for m in extract(source).mentions
        if m.construct in {Construct.STATA_INSTALL, Construct.STATA_WHICH}
    ]


@pytest.mark.parametrize(
    "source",
    [
        "ssc install reghdfe",
        "ssc install reghdfe, replace",
        "ssc install reghdfe,replace",
        "ssc inst reghdfe , replace all",
        "cap ssc install reghdfe, replace",
    ],
)
def test_the_option_comma_is_not_part_of_the_package_name(source):
    """`reghdfe,` is not an identifier, so 4,924 install lines named nothing."""
    assert found(source) == [("reghdfe", Construct.STATA_INSTALL, None)]


def test_install_if_missing_on_one_line_is_still_an_install():
    """The command word is `if`, so the dependency it states was lost."""
    assert found("if _rc ssc install estout, replace") == [
        ("estout", Construct.STATA_INSTALL, None)
    ]


def test_net_install_keeps_where_it_fetches_from():
    source = "net install grc1leg, from(http://www.stata.com/users/vwiggins)"
    assert found(source) == [
        ("grc1leg", Construct.STATA_INSTALL, "http://www.stata.com/users/vwiggins")
    ]


def test_a_quoted_from_is_read_too():
    source = (
        'net install st0085_2, from("http://www.stata-journal.com/software/sj14-2")'
    )
    assert found(source)[0][2] == "http://www.stata-journal.com/software/sj14-2"


def test_github_install_names_the_repository_not_a_command_called_github():
    assert found("github install haghish/rcall, stable") == [
        ("rcall", Construct.STATA_INSTALL, "github.com/haghish/rcall")
    ]
    assert "github" not in [m.raw_name for m in extract("github install a/b").mentions]


@pytest.mark.parametrize(
    "source", ["ssc hot", "ssc new", "net from http://x.org", "net set ado PLUS"]
)
def test_subcommands_that_name_no_package_record_none(source):
    """`ssc hot` lists popular packages; it does not install one called `hot`."""
    assert found(source) == []


def test_describe_is_inquiry_not_provisioning():
    assert found("ssc describe moremata") == [("moremata", Construct.STATA_WHICH, None)]


def test_net_from_sets_where_the_installs_after_it_fetch_from():
    """The two-line form is how suites on an author's site are installed."""
    source = (
        "net from https://jslsoc.sitehost.iu.edu/stata\n"
        "net install spost13_ado, replace\n"
        "ssc install estout\n"
    )
    assert found(source) == [
        (
            "spost13_ado",
            Construct.STATA_INSTALL,
            "https://jslsoc.sitehost.iu.edu/stata",
        ),
        ("estout", Construct.STATA_INSTALL, None),
    ]


def test_an_explicit_from_wins_over_an_earlier_net_from():
    source = "net from http://a.example/x\nnet install p, from(http://b.example/y)\n"
    assert found(source)[0][2] == "http://b.example/y"


def test_an_egen_function_is_a_use_of_the_file_that_implements_it():
    """`egenmore` ships no command, so reading commands alone credited it nothing."""
    source = (
        "egen n = nvals(x), by(id)\nbys id: egen double m = mean(y)\negen z = std(y)\n"
    )
    names = [(m.raw_name, m.called_function, m.line) for m in extract(source).mentions]
    assert ("_gnvals", "nvals", 1) in names
    assert ("_gmean", "mean", 2) in names
    # The `egen` call itself is still recorded, as the builtin it is.
    assert [n for n, _, _ in names].count("egen") == 3

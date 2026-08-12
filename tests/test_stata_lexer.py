"""Adversarial tests for the Stata lexer.

Every case in :func:`test_v1_false_positives_are_gone` is a real defect measured
against the corpus, not a hypothetical. v1's detector reported ``est`` (241
deposits), ``diff`` (177), ``ttest`` (135), ``did`` (114), ``is`` (88) and ``we``
(57) as Stata packages, while missing ``reghdfe`` -- the third most-used
user-written command, in 1,282 files.
"""

from __future__ import annotations

import pytest

from softverse.stata.lexer import lex, local_programs, strip_comments


def commands(source: str) -> list[str]:
    return [s.command for s in lex(source) if s.command]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # The leading token is the command, so an English word in a comment or
        # a variable name on the left of `=` can never be one.
        ("* did the main regression here", []),
        ("* we did this because matching did not work", []),
        ("gen diff = y2 - y1", ["gen"]),
        ("reg y x\nest store m1", ["reg", "est"]),
        ("ttest income, by(treat)", ["ttest"]),
        ("* is this the right specification", []),
    ],
)
def test_v1_false_positives_are_gone(source, expected):
    assert commands(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("reghdfe y x, absorb(id)", "reghdfe"),
        ("qui reghdfe y x, absorb(id)", "reghdfe"),
        ("quietly reghdfe y x", "reghdfe"),
        ("capture reghdfe y x", "reghdfe"),
        ("bysort id: reghdfe y x", "reghdfe"),
        ("by id year: reghdfe y x", "reghdfe"),
        ("qui bysort id: reghdfe y x", "reghdfe"),
        ("cap noi reghdfe y x", "reghdfe"),
    ],
)
def test_prefix_commands_are_peeled(source, expected):
    """v1 missed reghdfe entirely; a stacked prefix must not hide it."""
    assert expected in commands(source)


def test_trailing_punctuation_is_stripped():
    """`esttab, se` tokenizes as `esttab,` -- the comma is the option list."""
    assert "esttab" in commands("esttab, se")
    assert "esttab" in commands("outreg2 using t1.tex, replace\nesttab, se")


def test_line_continuation_is_joined():
    assert "reghdfe" in commands("qui reghdfe y x ///\n   , absorb(id)")


def test_block_comment_spanning_lines_is_removed():
    source = "/* this mentions reghdfe\n   and outreg2 */\nregress y x"
    assert commands(source) == ["regress"]


def test_nested_block_comments():
    """Stata's /* */ comments nest, unlike C's."""
    assert commands("/* outer /* inner */ still comment */ regress y x") == ["regress"]


def test_star_is_only_a_comment_at_statement_start():
    """Elsewhere `*` is multiplication and must not eat the line."""
    assert commands("gen z = x * y") == ["gen"]
    assert commands("* a comment") == []


def test_double_slash_inside_a_string_is_not_a_comment():
    source = 'di "see https://example.com/page"\nregress y x'
    assert commands(source) == ["di", "regress"]


def test_delimit_semicolon_mode():
    """7% of the corpus switches delimiter; getting it wrong breaks the rest."""
    source = """\
#delimit ;
regress y x;
esttab using out.tex, replace;
#delimit cr
summarize y
"""
    assert commands(source) == ["regress", "esttab", "summarize"]


def test_newline_is_not_a_terminator_in_semicolon_mode():
    source = "#delimit ;\nreghdfe y x,\n   absorb(id);\n"
    assert commands(source) == ["reghdfe"]


def test_macro_named_command_is_unresolvable_not_guessed():
    """`cmd' varlist cannot be known statically; record it, do not invent it."""
    statements = [s for s in lex("`cmd' y x") if s.is_macro_command]
    assert len(statements) == 1
    assert statements[0].command is None


def test_mata_blocks_are_marked():
    """Mata is a different language; its tokens are not Stata commands."""
    source = "mata:\nreal foo(x) { return(x) }\nend\nregress y x"
    assert "regress" in commands(source)
    assert any(s.in_mata for s in lex(source))


def test_local_programs_are_identified():
    """An author's helper program is not a package."""
    source = """\
capture program drop mytable
program define mytable
    syntax varlist
end
mytable y x
"""
    assert "mytable" in local_programs(source)


def test_statements_carry_line_numbers():
    """Line numbers must survive comment stripping, or snippets point nowhere."""
    source = "* comment\n\nregress y x\n"
    statement = next(s for s in lex(source) if s.command == "regress")
    assert statement.line == 3


def test_strip_comments_preserves_line_count():
    source = "* one\n/* two\n   three */\nregress y x\n"
    assert strip_comments(source).count("\n") == source.count("\n")


def test_real_world_snippet():
    """A composite of the idioms above, as they actually co-occur."""
    source = """\
* Table 3: main results
#delimit ;
qui bysort country: reghdfe lny lnk lnl ///
    , absorb(id year) cluster(id);
esttab using "tables/t3.tex", replace se;
#delimit cr
gen diff = lny - L.lny
* did the DID here
ttest lny, by(treated)
"""
    found = commands(source)
    assert "reghdfe" in found
    assert "esttab" in found
    for ghost in ("did", "diff", "DID", "we", "is"):
        assert ghost not in found, f"{ghost} is not a package"


def test_braces_inside_strings_do_not_split_statements():
    """Found by looking at corpus output, not by writing a test first.

    esttab options carry LaTeX: prehead("\\begin{tabular}{l*{4}{c}}"). Treating
    those braces as block delimiters shattered one statement into fragments
    whose leading tokens ("c", "tabular") were then read as commands. That one
    bug attributed 144 deposits to `fastcd`, whose only command is named `c`.
    """
    source = (
        "esttab est1 est2 using t.tex, "
        'prehead("\\begin{tabular}{l*{4}{c}} \\hline") b(a2) se(a2)'
    )
    found = commands(source)
    assert found == ["esttab"], found


def test_option_names_are_not_commands():
    """`b(a2)` is an option of esttab, not a command named b."""
    assert commands("esttab using x.tex, b(a2) se(a2) star") == ["esttab"]


def test_braces_still_delimit_blocks_at_top_level():
    source = "foreach v of varlist a b {\n    summarize `v'\n}\nregress y x"
    found = commands(source)
    assert "foreach" in found
    assert "summarize" in found
    assert "regress" in found


def test_semicolon_inside_parentheses_is_not_a_terminator():
    source = '#delimit ;\nesttab using t.tex, prehead("a;b") replace;\nregress y x;\n'
    assert commands(source) == ["esttab", "regress"]


def test_global_macro_braces_do_not_split_statements():
    """Found in the mention snippets, not by a test I thought to write.

    `${name}` is a global macro reference and its braces belong to an
    expression, not a block. Splitting on them shattered statements and
    reported the macro name as a command -- `_i`, `ind`, `output` and `Level`
    all reached the top of the unresolved list this way. Same failure as the
    LaTeX braces inside esttab options, through a different syntax.
    """
    assert commands("g d${_i}Lco = f${_i}.lnL - l.lnL") == ["g"]
    assert commands('use year ${ind}_i using "${data}", clear') == ["use"]
    assert commands("log using ${output}price_facts, text replace") == ["log"]


def test_global_macro_as_the_command_is_still_unresolvable():
    """`${cmd} varlist` is genuinely unknowable statically."""
    statements = [s for s in lex("${cmd} y x") if s.is_macro_command]
    assert len(statements) == 1
    assert statements[0].command is None

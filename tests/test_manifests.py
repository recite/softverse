"""What a deposit says about the environment it ran in.

Five file types state something no script does: the version of a package that
was installed, the version of R, Python, Julia or Stata that ran it, and the
machine it ran on. Two traps are worth pinning down, because both fail
silently and both point the same way, toward crediting a deposit with
everything it happened to bundle.
"""

from __future__ import annotations

import json

from softverse.detect.manifests import (
    is_manifest,
    read_description,
    read_manifest,
    read_renv_lock,
    read_requirements,
)
from softverse.stata.lexer import declared_version, lex

INSTALLED = """Package: sass
Version: 0.4.5
Built: R 4.2.2; x86_64-w64-mingw32; 2023-02-19 04:12:08 UTC; windows
Imports: fs, rlang (>= 0.4.10), htmltools (>= 0.5.1), R6, rappdirs
"""


def test_a_description_gives_its_package_and_version():
    read = read_description(INSTALLED)
    assert [(d.package, d.version_constraint) for d in read.declarations] == [
        ("sass", "0.4.5")
    ]
    assert read.declarations[0].dependency_role == "installed"


def test_a_description_does_not_contribute_its_own_imports():
    """The trap. `sass` depends on fs and R6; the deposit that bundled sass
    does not. Folding an installed package's `Imports:` into the deposit
    credits it with the closure under everything it shipped."""
    packages = {d.package for d in read_description(INSTALLED).declarations}
    assert packages == {"sass"}
    assert not packages & {"fs", "rlang", "htmltools", "R6", "rappdirs"}


def test_built_gives_the_r_version_and_the_os():
    assert read_description(INSTALLED).signals == {
        "r_version": "4.2.2",
        "os": "windows",
    }


def test_built_with_an_empty_platform_field_still_reads():
    """About half of them omit the platform: `R 3.6.3; ; 2020-04-20; windows`.
    The fields are positional, so an empty one must not shift the OS."""
    read = read_description(
        "Package: x\nVersion: 1.0\nBuilt: R 3.6.3; ; 2020-04-20 00:00:00 UTC; windows\n"
    )
    assert read.signals == {"r_version": "3.6.3", "os": "windows"}


def test_a_description_without_a_built_field_declares_no_version():
    """A source package says nothing about what ran it, and inventing an
    interpreter version for it would be worse than the null."""
    read = read_description("Package: scspill\nVersion: 0.1.0\n")
    assert read.signals == {}
    assert read.declarations[0].package == "scspill"


def test_a_file_named_description_that_is_not_a_package_is_not_read():
    """`sass` ships `sass-theme/DESCRIPTION` describing a CSS theme. Requiring
    `Package:` is what keeps those out of the package table."""
    assert read_description("Title: Sass CSS Theme\nAuthor: RStudio, Inc.\n") is None


def test_renv_lock_gives_the_r_version_and_a_locked_closure():
    lock = json.dumps(
        {
            "R": {"Version": "4.2.2"},
            "Packages": {
                "Deriv": {"Package": "Deriv", "Version": "4.1.6"},
                "R6": {"Package": "R6", "Version": "2.5.1"},
            },
        }
    )
    read = read_renv_lock(lock)
    assert read.signals == {"r_version": "4.2.2"}
    assert {(d.package, d.version_constraint) for d in read.declarations} == {
        ("Deriv", "4.1.6"),
        ("R6", "2.5.1"),
    }
    # renv does not record which of these the author asked for.
    assert {d.dependency_role for d in read.declarations} == {"locked"}


def test_requirements_keeps_the_constraint_not_just_the_pin():
    read = read_requirements(
        "numpy == 1.26.4\n"
        "# a comment\n"
        "-r base.txt\n"
        "\n"
        'pandas[excel]>=2.0; python_version < "3.11"\n'
        "scipy\n"
    )
    assert [(d.package, d.version_constraint) for d in read.declarations] == [
        ("numpy", "==1.26.4"),
        ("pandas", ">=2.0"),
        ("scipy", None),
    ]


def test_requirements_variants_are_requirements():
    assert is_manifest("requirements-dev.txt")
    assert is_manifest("RENV.LOCK")
    assert not is_manifest("readme.txt")


def test_read_manifest_dispatches_on_the_filename():
    assert read_manifest("DESCRIPTION", INSTALLED).kind == "description"
    assert read_manifest("analysis.R", "library(dplyr)") is None


# ---------------------------------------------------------------------------
# The Stata `version` statement
# ---------------------------------------------------------------------------


def test_declared_version_reads_the_forms_that_occur():
    assert declared_version("version 14") == "14"
    assert declared_version("version 14.2") == "14.2"
    assert declared_version("version 8, missing") == "8"
    assert declared_version("vers 15") == "15"


def test_the_prefix_use_is_not_a_declaration():
    """`version 14: regress y x` asks for one command to run under version 14
    semantics. It says nothing about the version the author ran."""
    assert declared_version("version 14: regress y x") is None


def test_the_prefix_still_lexes_as_a_command():
    """The regression that matters. `version` is a real colon prefix and
    6.7 million Stata mentions depend on `_strip_prefixes` peeling it, so a
    reader added beside that logic must leave the command untouched."""
    statements = [s for s in lex("version 14: regress y x") if s.command]
    assert [(s.command, s.prefixes) for s in statements] == [("regress", ["version"])]


def test_a_commented_version_is_not_a_declaration():
    assert declared_version("* version 12\nversion 16\n") == "16"


def test_the_first_declaration_wins():
    assert declared_version("version 14\nregress y x\nversion 16\n") == "14"


def test_a_notebook_reports_the_interpreter_that_ran_it():
    notebook = json.dumps(
        {"metadata": {"language_info": {"name": "python", "version": "3.11.7"}}}
    )
    read = read_manifest("analysis.ipynb", notebook)
    assert read.kind == "notebook"
    assert read.signals == {"python_version": "3.11.7"}
    assert read.declarations == []


def test_a_julia_notebook_is_not_a_python_version():
    """Julia writes 1.9.2 into the same field Python writes 3.11.7 into, and
    63 of this corpus's notebooks are Julia. One signal name for both would
    put Julia releases in the Python distribution."""
    notebook = json.dumps(
        {"metadata": {"language_info": {"name": "julia", "version": "1.9.2"}}}
    )
    assert read_manifest("m.ipynb", notebook).signals == {"julia_version": "1.9.2"}


def test_a_notebook_with_no_recorded_version_says_nothing():
    assert read_manifest("x.ipynb", json.dumps({"metadata": {}})) is None
    assert read_manifest("x.ipynb", "not json") is None

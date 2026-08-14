"""The construct has to reach the resolver.

`Resolution.LOCAL_RELATIVE` and `Resolution.DYNAMIC` have existed in the enum
since the model was written and nothing ever produced either. The wiring that
would have was missing: `resolve()` takes a name and a language, so by the
time it runs, `from .models import Constants` and `import models` are the same
string and get the same answer.

Measured on 1,108 deposits before this was fixed, that cost both directions at
once -- 292 mentions of a deposit's own modules credited to same-named
registry packages, and 621 more reported as unresolved third-party names.
"""

from __future__ import annotations

import pytest

from softverse.model.enums import Construct, Ecosystem, Language, Resolution
from softverse.registries.resolve import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry(
        cran=frozenset({"MASS", "ggplot2", "stats4"}),
        cran_archive=frozenset(),
        bioconductor=frozenset(),
        # `models`, `results` and `log` are real PyPI distributions and also
        # the most common names a research repo gives its own modules, which
        # is exactly why the confusion is not rare.
        pypi=frozenset({"pandas", "models", "results", "log", "stats"}),
        julia=frozenset(),
        stata_commands={"esttab": ("estout",)},
        stata_builtins=frozenset(),
        ssc_packages=frozenset({"blindschemes", "egenmore", "estout"}),
        lock_id="test",
    )


def test_a_relative_import_is_local_not_a_package(registry):
    """`from .models import Constants` is the deposit's own `models.py`.

    Resolved without the construct it matches the PyPI distribution `models`
    and is counted as third-party use -- 289 such credits in the corpus, the
    most common being `models`, `results`, `log` and `context`.
    """
    resolved = registry.resolve(
        "models", Language.PYTHON, construct=Construct.LOCAL_RELATIVE
    )
    assert resolved.resolution is Resolution.LOCAL_RELATIVE
    assert resolved.package is None, "a local module is not a package"
    assert resolved.ecosystem is None


def test_a_relative_import_that_matches_nothing_is_still_local(registry):
    """The other half: 540 of these were reported as unresolved.

    An unresolved mention says "there is a package here we could not name".
    A relative import says the opposite -- there is no package here at all --
    so counting it as unresolved inflates the very statistic the paper uses
    to describe registry coverage.
    """
    resolved = registry.resolve(
        "helpers", Language.PYTHON, construct=Construct.LOCAL_RELATIVE
    )
    assert resolved.resolution is Resolution.LOCAL_RELATIVE
    assert resolved.package is None


def test_a_bare_dot_relative_import_is_local(registry):
    """`from . import util` arrives with the name `.`."""
    resolved = registry.resolve(
        ".", Language.PYTHON, construct=Construct.LOCAL_RELATIVE
    )
    assert resolved.resolution is Resolution.LOCAL_RELATIVE


def test_an_unresolvable_dynamic_name_is_dynamic_not_unknown(registry):
    """`library(p, character.only = TRUE)` where `p` is a loop variable.

    The extractor already knows: it labels the construct `dynamic_unresolved`.
    Calling the result `unknown` claims we failed to identify a package, when
    what actually happened is that no package name was ever written down.
    """
    resolved = registry.resolve("p", Language.R, construct=Construct.DYNAMIC_UNRESOLVED)
    assert resolved.resolution is Resolution.DYNAMIC
    assert resolved.package is None


def test_a_dynamic_name_matching_a_real_package_is_still_dynamic(registry):
    """Two R mentions resolved this way by coincidence.

    A loop variable named `stats4` is not evidence that `stats4` was used.
    """
    resolved = registry.resolve(
        "MASS", Language.R, construct=Construct.DYNAMIC_UNRESOLVED
    )
    assert resolved.resolution is Resolution.DYNAMIC
    assert resolved.package is None


def test_an_ordinary_import_of_the_same_name_still_resolves(registry):
    """The guard must key on the construct, not on the name.

    `import models` at the top level really is the PyPI distribution, and a
    fix that suppressed the name everywhere would trade one error for another.
    """
    resolved = registry.resolve("models", Language.PYTHON, construct=Construct.IMPORT)
    assert resolved.resolution is Resolution.KNOWN_CURRENT
    assert resolved.package == "models"
    assert resolved.ecosystem is Ecosystem.PYPI


def test_resolution_without_a_construct_is_unchanged(registry):
    """The argument is optional, so every existing caller keeps working."""
    assert registry.resolve("pandas", Language.PYTHON).package == "pandas"
    assert registry.resolve("MASS", Language.R).package == "MASS"


def test_an_install_line_names_a_package_not_a_command(registry):
    """`ssc install blindschemes` was resolved against the *command* index.

    The index maps commands to the packages providing them, which is the
    right table for `reghdfe y x` and the wrong one for `ssc install
    reghdfe`. A package need not expose a command of its own name -- and some
    expose no commands at all: `blindschemes` ships graph schemes, `egenmore`
    ships `egen` functions. Both were reported as unidentifiable when the
    deposit had in fact named its dependency outright, which is better
    evidence than a command occurrence, not worse.
    """
    resolved = registry.resolve(
        "blindschemes", Language.STATA, construct=Construct.STATA_INSTALL
    )
    assert resolved.resolution is Resolution.KNOWN_CURRENT
    assert resolved.package == "blindschemes"
    assert resolved.ecosystem is Ecosystem.SSC


def test_an_install_line_for_something_unknown_stays_unknown(registry):
    resolved = registry.resolve(
        "notapackage", Language.STATA, construct=Construct.STATA_INSTALL
    )
    assert resolved.resolution is Resolution.UNKNOWN


def test_a_command_is_still_resolved_against_the_command_index(registry):
    """The package table must not leak into ordinary command resolution.

    `estout` the package provides `esttab`; a script saying `esttab` must
    resolve through the command index as before.
    """
    resolved = registry.resolve("esttab", Language.STATA)
    assert resolved.package == "estout"

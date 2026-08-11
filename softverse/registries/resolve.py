"""Resolve a detected name to a package, or to an honest non-answer.

Resolution never silently drops a name. Every mention leaves here with a
:class:`~softverse.model.enums.Resolution`, including ``UNKNOWN``, because a
name we cannot place is a diagnostic (usually an extractor bug) and sometimes a
finding (a GitHub-only or institutional package). v1 emitted a bare string and
left the reader to guess, which is how ``os``, ``grid``, ``std`` and
``lmtest, quietly = TRUE`` all ended up in one column labelled "package".

Two rules matter more than the rest:

- **R matching is case-sensitive.** ``grDevices`` is base R and ``MASS`` is
  recommended, but v1 lowercased the candidate and compared it against a
  mixed-case set, so ``grDevices``, ``Matrix`` and ``KernSmooth`` were all
  labelled third-party while ``MASS`` matched only because someone had typed
  ``"mass"`` in the set by hand.
- **Registry membership is coverage, not proof.** A local module can share a
  name with a real distribution. :data:`Resolution.KNOWN_CURRENT` means "this
  name exists in the registry", which is what the paper may claim from it --
  never "this is a true positive".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property

from softverse.logging_setup import get_logger
from softverse.model.enums import Ecosystem, Language, Resolution
from softverse.registries.fetch import (
    BASE_R,
    PYPI_IMPORT_ALIASES,
    RECOMMENDED_R,
    python_stdlib,
)

logger = get_logger(__name__)

_PEP503 = re.compile(r"[-_.]+")


def normalize_pypi(name: str) -> str:
    """PEP 503 normalization: lowercase, runs of ``-_.`` collapsed to ``-``."""
    return _PEP503.sub("-", name).lower()


@dataclass(frozen=True)
class Resolved:
    """The outcome of resolving one name."""

    resolution: Resolution
    package: str | None = None
    ecosystem: Ecosystem | None = None
    #: All candidates when more than one registry entry matches. Ambiguity is
    #: carried forward rather than decided here.
    candidates: tuple[str, ...] = ()
    #: How a Python name was matched: "alias", "exact", "normalized". Reported
    #: so the incompleteness of the import->distribution map stays visible.
    basis: str | None = None


@dataclass
class Registry:
    """Pinned registry snapshots, and the resolution rules over them."""

    cran: frozenset[str]
    cran_archive: frozenset[str]
    bioconductor: frozenset[str]
    pypi: frozenset[str]
    julia: frozenset[str]
    stata_commands: dict[str, tuple[str, ...]]
    stata_builtins: frozenset[str]
    lock_id: str

    @cached_property
    def _pypi_normalized(self) -> dict[str, str]:
        return {normalize_pypi(n): n for n in self.pypi}

    @cached_property
    def _stdlib(self) -> frozenset[str]:
        return python_stdlib()

    # -- R -----------------------------------------------------------------

    def resolve_r(self, name: str) -> Resolved:
        """Resolve an R package name. Case-sensitive, deliberately."""
        if name in BASE_R:
            return Resolved(Resolution.BASE_OR_STDLIB, name, Ecosystem.BASE_R)
        if name in RECOMMENDED_R:
            # Shipped with R but separately maintained and genuinely citable.
            return Resolved(Resolution.KNOWN_CURRENT, name, Ecosystem.CRAN)
        if name in self.cran:
            return Resolved(Resolution.KNOWN_CURRENT, name, Ecosystem.CRAN)
        if name in self.bioconductor:
            return Resolved(Resolution.KNOWN_CURRENT, name, Ecosystem.BIOCONDUCTOR)
        if name in self.cran_archive:
            # Real when the code was written; removed since. rgdal, maptools,
            # rgeos and Zelig are all here, and they are a finding, not noise.
            return Resolved(Resolution.KNOWN_ARCHIVED, name, Ecosystem.CRAN_ARCHIVE)
        return Resolved(Resolution.UNKNOWN, None, None)

    # -- Python ------------------------------------------------------------

    def resolve_python(self, name: str) -> Resolved:
        """Resolve a Python *import* name to a distribution.

        Import names and distribution names are different namespaces, so this
        tries, in order: stdlib, the curated alias table, an exact match, then a
        PEP 503 normalized match. The winning route is recorded in ``basis``.
        """
        top = name.split(".", 1)[0]
        if top in self._stdlib or top == "__future__":
            return Resolved(
                Resolution.BASE_OR_STDLIB, top, Ecosystem.PYTHON_STDLIB, basis="stdlib"
            )
        if alias := PYPI_IMPORT_ALIASES.get(top):
            return Resolved(
                Resolution.KNOWN_CURRENT, alias, Ecosystem.PYPI, basis="alias"
            )
        if top in self.pypi:
            return Resolved(
                Resolution.KNOWN_CURRENT, top, Ecosystem.PYPI, basis="exact"
            )
        if match := self._pypi_normalized.get(normalize_pypi(top)):
            return Resolved(
                Resolution.KNOWN_CURRENT, match, Ecosystem.PYPI, basis="normalized"
            )
        return Resolved(Resolution.UNKNOWN, None, None)

    # -- Stata -------------------------------------------------------------

    def resolve_stata(
        self, command: str, local_programs: frozenset[str] = frozenset()
    ) -> Resolved:
        """Resolve a Stata command token.

        Order matters and encodes the bias we accept. A command defined in the
        deposit is the author's own; a command official Stata provides is not a
        package. Only then do we consult the SSC index. A name claimed by both
        Stata and SSC therefore resolves to ``BUILTIN``, which *under*-counts
        user packages -- the safer direction, stated in the paper rather than
        hidden.
        """
        lowered = command.lower()
        if command in local_programs or lowered in local_programs:
            return Resolved(Resolution.LOCAL_PROGRAM, None, None)
        if lowered in self.stata_builtins:
            return Resolved(Resolution.BUILTIN, None, Ecosystem.STATA_BUILTIN)
        packages = self.stata_commands.get(lowered)
        if not packages:
            return Resolved(Resolution.UNKNOWN, None, None)
        if len(packages) > 1:
            return Resolved(
                Resolution.AMBIGUOUS, None, Ecosystem.SSC, candidates=packages
            )
        return Resolved(Resolution.KNOWN_CURRENT, packages[0], Ecosystem.SSC)

    # -- Dispatch ----------------------------------------------------------

    def resolve(
        self,
        name: str,
        language: Language,
        local_programs: frozenset[str] = frozenset(),
    ) -> Resolved:
        """Resolve ``name`` in ``language``."""
        if language is Language.R:
            return self.resolve_r(name)
        if language is Language.PYTHON:
            return self.resolve_python(name)
        if language is Language.STATA:
            return self.resolve_stata(name, local_programs)
        if language is Language.JULIA:
            if name in self.julia:
                return Resolved(Resolution.KNOWN_CURRENT, name, Ecosystem.JULIA_GENERAL)
            return Resolved(Resolution.UNKNOWN, None, None)
        return Resolved(Resolution.UNKNOWN, None, None)

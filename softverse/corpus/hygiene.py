"""Decide which files are research code and which are contamination.

The corpus contains other people's libraries. A single Zenodo record yielded
9,894 "scripts" because it shipped a vendored ``.checkpoint`` tree containing R's
own test suite; 2,349 files across the corpus sit under ``renv/``, ``packrat/``
or ``site-packages/``. Counting those measures what the R core team wrote, not
what the author used.

Every exclusion records **which rule fired**, so it is auditable and reversible.
An exclusion you cannot explain is indistinguishable from a bug.

The rules degrade gracefully. V1 needs directory paths, which the re-collected
corpus will have and the flattened legacy corpus does not; V3 and V4 work on
content and names alone, so the legacy corpus is still usable as a dev set.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from softverse.config import VENDOR_PATH_MARKERS
from softverse.model.enums import VendorRule

#: A sha256 appearing in at least this many distinct deposits is, by
#: construction, not bespoke research code. The corpus is its own reference set,
#: so this needs no external library of known files.
CROSS_DATASET_THRESHOLD = 5

#: Filenames that mark a directory as a package source tree rather than an
#: analysis directory.
_PACKAGE_MARKERS = {"DESCRIPTION", "NAMESPACE", "pyproject.toml", "setup.py"}

#: Basenames that are never research code regardless of location.
_NEVER_RESEARCH = {"__init__.py", "setup.py", "conftest.py", "activate.R"}


@dataclass
class Verdict:
    """Why a file is in or out."""

    is_vendored: bool
    rule: VendorRule | None = None
    duplicate_of: str | None = None

    @property
    def in_analysis_set(self) -> bool:
        return not self.is_vendored and self.duplicate_of is None


def vendored_by_path(relative_path: str) -> VendorRule | None:
    """V1: any path component naming a library tree.

    Requires directory structure, so it does nothing on the flattened legacy
    Dataverse corpus -- which is precisely why re-collection matters.
    """
    parts = {p.lower() for p in Path(relative_path).parts}
    if parts & VENDOR_PATH_MARKERS:
        return VendorRule.V1_PATH
    if any(p.startswith("._") for p in Path(relative_path).parts):
        return VendorRule.APPLEDOUBLE
    return None


def vendored_by_marker(relative_path: str, siblings: set[str]) -> VendorRule | None:
    """V2: the file sits in a directory that is a package source tree.

    Catches vendored packages that do not live under a conventionally named
    directory, which V1 would miss.
    """
    parent = str(Path(relative_path).parent)
    if any(f"{parent}/{marker}" in siblings for marker in _PACKAGE_MARKERS):
        return VendorRule.V2_MARKER
    return None


def vendored_by_name(filename: str, ssc_shipped: frozenset[str]) -> VendorRule | None:
    """V4: name-shaped rules that work even without directory structure.

    The precise one is membership in an SSC package's file list: a ``.ado`` whose
    basename a published package ships is a vendored copy of that package, not
    something this author wrote. That is exactly what the Stata index gives us.
    """
    base = Path(filename).name
    if base in _NEVER_RESEARCH or base.startswith("._"):
        return VendorRule.V4_NAME_SHAPE
    if base.lower().endswith(".ado") and base[:-4].lower() in ssc_shipped:
        return VendorRule.V4_NAME_SHAPE
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def cross_dataset_hashes(
    rows: Iterable[tuple[str, str]], threshold: int = CROSS_DATASET_THRESHOLD
) -> set[str]:
    """V3: hashes appearing in >= ``threshold`` distinct deposits.

    ``rows`` is ``(dataset_doi, sha256)``. Returns the hashes to exclude.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for doi, digest in rows:
        seen[digest].add(doi)
    return {d for d, dois in seen.items() if len(dois) >= threshold}


def classify(
    relative_path: str,
    sha256: str,
    *,
    siblings: set[str] = frozenset(),
    ssc_shipped: frozenset[str] = frozenset(),
    shared_hashes: frozenset[str] = frozenset(),
    seen_in_dataset: dict[str, str] | None = None,
) -> Verdict:
    """Apply the rules in order and return a single, explained verdict.

    ``seen_in_dataset`` maps sha256 -> the first file_uid with that content
    within the same deposit, so an exact duplicate is marked rather than counted
    twice. v1's ``_1``/``_2`` collision renaming meant re-runs manufactured
    duplicates that all counted.
    """
    for rule in (
        vendored_by_path(relative_path),
        vendored_by_marker(relative_path, set(siblings)),
        vendored_by_name(relative_path, ssc_shipped),
    ):
        if rule is not None:
            return Verdict(is_vendored=True, rule=rule)

    if sha256 in shared_hashes:
        return Verdict(is_vendored=True, rule=VendorRule.V3_CROSS_DATASET)

    if seen_in_dataset is not None and sha256 in seen_in_dataset:
        return Verdict(is_vendored=False, duplicate_of=seen_in_dataset[sha256])

    return Verdict(is_vendored=False)

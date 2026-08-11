"""Data and metadata models for Softverse.

The schemas in this package are the contract every other module writes against.
They are deliberately explicit: v1's headline defect was a null join key silently
dropped by a ``groupby``, which typed columns and non-nullable declarations make
loud instead of invisible.
"""

from softverse.model.enums import (
    CollectionState,
    Construct,
    Ecosystem,
    FileRole,
    Language,
    ParseStatus,
    Resolution,
    VendorRule,
    VersionState,
)
from softverse.model.schemas import (
    DATASETS,
    DECLARED_DEPENDENCIES,
    FILES,
    JOURNALS,
    MENTIONS,
    PACKAGES,
    RESOLUTION_CANDIDATES,
    SCHEMAS,
    STATA_COMMAND_INDEX,
    UNKNOWN_NAMES,
    schema_for,
)

__all__ = [
    "SCHEMAS",
    "DATASETS",
    "DECLARED_DEPENDENCIES",
    "FILES",
    "JOURNALS",
    "MENTIONS",
    "PACKAGES",
    "RESOLUTION_CANDIDATES",
    "STATA_COMMAND_INDEX",
    "UNKNOWN_NAMES",
    "schema_for",
    "CollectionState",
    "Construct",
    "Ecosystem",
    "FileRole",
    "Language",
    "ParseStatus",
    "Resolution",
    "VendorRule",
    "VersionState",
]

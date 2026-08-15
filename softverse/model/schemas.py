"""PyArrow schemas for every table Softverse emits.

Design rules, each traceable to a v1 defect:

1. **Join keys are non-nullable.** v1 derived ``repo_id`` from a path regex that
   matched nothing, produced nulls for 71,069 of 71,069 rows, and lost them all
   to ``groupby(dropna=True)``. Here a null key fails at write time.
2. **Every mention carries its provenance triple** (``extractor_version``,
   ``grammar_version``, ``registry_lock_id``) so a row can be attributed to the
   exact instrument that produced it.
3. **Every mention carries ``line``/``col``/``snippet``** so any published number
   can be traced to source text without re-running the pipeline.
4. **Denominators are columns, not derivations.** A rate whose denominator is not
   stored cannot be checked.
5. **Parquet, not CSV.** v1's CSV corrupted itself: unquoted commas inside
   captured names shifted the ``language`` column, producing values like
   ``" quietly = TRUE\"``. Typed columns make that impossible.
"""

from __future__ import annotations

import pyarrow as pa

# --------------------------------------------------------------------------
# Reusable field groups
# --------------------------------------------------------------------------

#: Stamped on every mention so a row can be traced to the instrument version
#: that produced it. Without this, a re-run silently mixes vintages.
_PROVENANCE_FIELDS = [
    pa.field("extractor_version", pa.string(), nullable=False),
    pa.field("grammar_version", pa.string(), nullable=True),
    pa.field("registry_lock_id", pa.string(), nullable=False),
]


def _dict() -> pa.DataType:
    """Dictionary-encoded string, for low-cardinality repeated values."""
    return pa.dictionary(pa.int32(), pa.string())


# --------------------------------------------------------------------------
# Frame tables
# --------------------------------------------------------------------------

JOURNALS = pa.schema(
    [
        pa.field("journal_id", pa.string(), nullable=False),
        pa.field("journal_name", pa.string(), nullable=False),
        pa.field("dataverse_url", pa.string()),
        pa.field("issn", pa.string()),
        pa.field("discipline", pa.string()),
        pa.field("impact_factor", pa.float32()),
        pa.field("impact_factor_asof", pa.date32()),
        # Policy regime: the denominator moves when a journal adopts a mandate,
        # so trends must be read within stable-policy windows.
        pa.field("mandate_adopted_date", pa.date32()),
        pa.field("collection_start_date", pa.date32()),
        pa.field("backfill_notes", pa.string()),
    ],
    metadata={"grain": "one row per journal collection", "unit": "journal"},
)

COLLECTIONS = pa.schema(
    [
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("source", _dict(), nullable=False),
        # journal | community | series | project_group. Only `journal`
        # collections enter the paper's sampling frame; the rest can be
        # collected and released without contaminating the headline estimand.
        pa.field("kind", _dict(), nullable=False),
        pa.field("collection_name", pa.string()),
        pa.field("collection_url", pa.string()),
        # Set only when the collection is a journal. Nullable by design: a
        # Zenodo community has no journal, and forcing one would either invent
        # data or push a null into a key that must never be null.
        pa.field("journal_id", pa.string()),
    ],
    metadata={
        "grain": "one row per collection across all sources",
        "note": "the join point that lets Zenodo/ICPSR/OSF coexist with the "
        "Dataverse frame without a schema migration",
    },
)

DATASETS = pa.schema(
    [
        pa.field("dataset_version_uid", pa.string(), nullable=False),
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("repo_id", pa.string(), nullable=False),
        pa.field("source", _dict(), nullable=False),
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.int64()),
        pa.field("title", pa.string()),
        # Deposit date is what Dataverse gives us; article year is what the
        # trend claim actually needs. Keep both, and record which we used.
        pa.field("deposit_date", pa.date32()),
        pa.field("deposit_year", pa.int16()),
        pa.field("article_year", pa.int16()),
        pa.field("article_year_source", _dict()),
        pa.field("article_doi", pa.string()),
        pa.field("version_number", pa.int32()),
        pa.field("version_minor", pa.int32()),
        pa.field("version_state", _dict()),
        pa.field("version_release_date", pa.date32()),
        pa.field("n_files_total", pa.int32()),
        pa.field("n_files_restricted", pa.int32()),
        pa.field("n_candidate_files", pa.int32()),
        pa.field("collection_state", _dict(), nullable=False),
        # The raw API response is archived before anything is derived from it.
        # v1 let a derived CSV be the only copy of a network response and lost
        # all 74 of them to a skip-then-overwrite bug.
        pa.field("api_json_path", pa.string()),
        pa.field("api_json_sha256", pa.string()),
        pa.field("harvest_ts", pa.timestamp("us", tz="UTC")),
    ],
    metadata={"grain": "one row per dataset version", "unit": "dataset_version"},
)

DATASET_COLLECTIONS = pa.schema(
    [
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("collection_id", pa.string(), nullable=False),
        # Dataverse supports nested and linked datasets, so membership is
        # many-to-many. v1's flat read missed 5 nested collections (12 datasets).
        pa.field("membership_kind", _dict(), nullable=False),  # direct|nested|linked
        pa.field("parent_collection_id", pa.string()),
    ],
    metadata={"grain": "one row per (dataset, collection) membership"},
)

DATASET_ARTICLES = pa.schema(
    [
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("article_doi", pa.string()),
        pa.field("article_year", pa.int16()),
        pa.field("relationship", _dict()),
        pa.field("source", _dict()),
    ],
    metadata={"grain": "one row per (dataset, article) link"},
)

# --------------------------------------------------------------------------
# Provenance spine
# --------------------------------------------------------------------------

FILES = pa.schema(
    [
        pa.field("file_uid", pa.string(), nullable=False),
        pa.field("dataset_version_uid", pa.string(), nullable=False),
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("source", _dict(), nullable=False),
        pa.field("collection_id", pa.string(), nullable=False),
        # Null only for files that came out of an archive; those carry
        # container_file_uid + path_in_container instead.
        pa.field("dataverse_file_id", pa.int64()),
        pa.field("container_file_uid", pa.string()),
        pa.field("path_in_container", pa.string()),
        pa.field("directory_label", pa.string()),
        pa.field("relative_path", pa.string(), nullable=False),
        pa.field("filename", pa.string(), nullable=False),
        pa.field("extension", pa.string()),
        pa.field("language", _dict()),
        pa.field("size_bytes", pa.int64()),
        pa.field("md5_api", pa.string()),
        pa.field("sha256_local", pa.string()),
        pa.field("md5_verified", pa.bool_()),
        pa.field("restricted", pa.bool_()),
        pa.field("encoding", pa.string()),
        pa.field("encoding_confidence", pa.float32()),
        pa.field("is_vendored", pa.bool_(), nullable=False),
        pa.field("vendor_rule", _dict()),
        pa.field("duplicate_of", pa.string()),
        pa.field("file_role", _dict()),
        pa.field("is_reachable", pa.bool_()),
        pa.field("parse_status", _dict(), nullable=False),
        pa.field("parse_error_nodes", pa.int32()),
        pa.field("n_mentions", pa.int32(), nullable=False),
        # The single boolean every derived table filters on.
        pa.field("in_analysis_set", pa.bool_(), nullable=False),
        pa.field("download_ts", pa.timestamp("us", tz="UTC")),
        pa.field("local_path", pa.string()),
    ],
    metadata={"grain": "one row per file obtained", "unit": "file"},
)

# --------------------------------------------------------------------------
# The atomic measurement
# --------------------------------------------------------------------------

MENTIONS = pa.schema(
    [
        pa.field("mention_uid", pa.string(), nullable=False),
        pa.field("file_uid", pa.string(), nullable=False),
        # Denormalized so the common queries need no join. Non-nullable because
        # a null here is exactly the v1 failure.
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("source", _dict(), nullable=False),
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("deposit_year", pa.int16()),
        pa.field("language", _dict(), nullable=False),
        pa.field("construct", _dict(), nullable=False),
        pa.field("raw_name", pa.string(), nullable=False),
        # The function called, where the source says: `select` for
        # `dplyr::select`, the command itself for Stata. Null for a bare
        # `library(dplyr)`, which names a package and no function.
        pa.field("called_function", pa.string()),
        pa.field("normalized_name", pa.string()),
        pa.field("resolved_package", pa.string()),
        pa.field("ecosystem", _dict()),
        pa.field("resolution", _dict(), nullable=False),
        # Location, so any published number is checkable against source text.
        pa.field("line", pa.int32()),
        pa.field("col", pa.int32()),
        pa.field("byte_start", pa.int64()),
        pa.field("byte_end", pa.int64()),
        pa.field("snippet", pa.string()),
        pa.field("cell_index", pa.int32()),
        pa.field("chunk_label", pa.string()),
        pa.field("is_conditional", pa.bool_()),
        pa.field("is_dynamic", pa.bool_(), nullable=False),
        pa.field("is_reachable", pa.bool_()),
        *_PROVENANCE_FIELDS,
    ],
    metadata={
        "grain": "one row per syntactic occurrence",
        "unit": "mention",
        "note": "a mention is a reference in deposited code, not evidence of execution",
    },
)

RESOLUTION_CANDIDATES = pa.schema(
    [
        pa.field("mention_uid", pa.string(), nullable=False),
        pa.field("candidate_package", pa.string(), nullable=False),
        pa.field("ecosystem", _dict(), nullable=False),
        pa.field("confidence", pa.float32()),
        pa.field("evidence", pa.string()),
        # Historical validity: a Stata command's owning package changes over
        # time, and resolving 2010 code against a 2026 index manufactures trends.
        pa.field("valid_from", pa.date32()),
        pa.field("valid_to", pa.date32()),
    ],
    metadata={"grain": "one row per candidate for an ambiguous mention"},
)

# --------------------------------------------------------------------------
# Registries and independent signals
# --------------------------------------------------------------------------

PACKAGES = pa.schema(
    [
        pa.field("package", pa.string(), nullable=False),
        pa.field("ecosystem", _dict(), nullable=False),
        pa.field("normalized_name", pa.string()),
        pa.field("first_release_date", pa.date32()),
        pa.field("latest_version", pa.string()),
        pa.field("is_archived", pa.bool_()),
        pa.field("title", pa.string()),
        pa.field("maintainer", pa.string()),
        pa.field("snapshot_date", pa.date32(), nullable=False),
    ],
    metadata={"grain": "one row per known package per registry snapshot"},
)

STATA_COMMAND_INDEX = pa.schema(
    [
        pa.field("command", pa.string(), nullable=False),
        pa.field("package", pa.string(), nullable=False),
        pa.field("source", _dict(), nullable=False),  # ssc | stata_journal | builtin
        # An `f foo.ado` line means the package ships a file, not that it
        # exposes a command named foo. Commands confirmed by parsing
        # `program define` are flagged; filename-only inferences are not.
        pa.field("evidence", _dict(), nullable=False),  # program_define | filename
        pa.field("is_helper", pa.bool_()),  # e.g. reghdfe_p, _eststo
        pa.field("author", pa.string()),
        pa.field("distribution_date", pa.date32()),
        pa.field("first_seen_date", pa.date32()),
        pa.field("snapshot_date", pa.date32(), nullable=False),
    ],
    metadata={
        "grain": "one row per (command, package) mapping",
        "note": "many-to-many by design; ambiguity is recorded, never resolved silently",
    },
)

DECLARED_DEPENDENCIES = pa.schema(
    [
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("source_file_uid", pa.string(), nullable=False),
        pa.field("manifest_kind", _dict(), nullable=False),
        pa.field("package", pa.string(), nullable=False),
        pa.field("version_constraint", pa.string()),
        pa.field("ecosystem", _dict()),
        # renv.lock ships a full closure; DESCRIPTION distinguishes roles.
        # Conflating them would misread concordance as recall.
        #
        # `direct` is what an author wrote in a requirements file, `locked` is
        # a package inside a resolved closure, and `installed` is a package
        # the deposit shipped a copy of. A deposit that vendors 200 libraries
        # is not a deposit that uses 200 libraries, so a query that treats
        # these as one thing measures how the archive was packed.
        pa.field("dependency_role", _dict()),  # direct | locked | installed
    ],
    metadata={"grain": "one row per declared dependency"},
)

ENVIRONMENT_SIGNALS = pa.schema(
    [
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("source_file_uid", pa.string(), nullable=False),
        pa.field("manifest_kind", _dict(), nullable=False),
        # r_version | python_version | stata_version | os
        pa.field("signal", _dict(), nullable=False),
        pa.field("value", pa.string(), nullable=False),
    ],
    metadata={
        "grain": "one row per environment signal a file states; "
        "sparse, and read with the coverage block in summary.json"
    },
)

UNKNOWN_NAMES = pa.schema(
    [
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("language", _dict(), nullable=False),
        pa.field("n_mentions", pa.int64(), nullable=False),
        pa.field("n_datasets", pa.int64(), nullable=False),
        pa.field("example_snippets", pa.list_(pa.string())),
    ],
    metadata={"grain": "one row per unresolved name; a diagnostic and a paper table"},
)

# --------------------------------------------------------------------------
# Derived / published tables
# --------------------------------------------------------------------------

DATASET_PACKAGES = pa.schema(
    [
        pa.field("dataset_doi", pa.string(), nullable=False),
        pa.field("collection_id", pa.string(), nullable=False),
        pa.field("year", pa.int16()),
        pa.field("language", _dict(), nullable=False),
        pa.field("package", pa.string(), nullable=False),
        pa.field("ecosystem", _dict()),
        pa.field("n_files", pa.int32(), nullable=False),
        pa.field("n_mentions", pa.int32(), nullable=False),
        pa.field("any_reachable", pa.bool_()),
        pa.field("any_use_construct", pa.bool_()),
    ],
    metadata={"grain": "one row per (dataset, package)", "unit": "the headline unit"},
)

JOURNAL_YEAR_PACKAGES = pa.schema(
    [
        pa.field("journal_id", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("language", _dict(), nullable=False),
        pa.field("package", pa.string(), nullable=False),
        pa.field("n_datasets_using", pa.int32(), nullable=False),
        # The denominator ladder, stored rather than derived. Every published
        # rate must name which rung it divides by.
        pa.field("n_datasets_in_frame", pa.int32(), nullable=False),
        pa.field("n_datasets_collected", pa.int32(), nullable=False),
        pa.field("n_datasets_with_code", pa.int32(), nullable=False),
        pa.field("n_datasets_at_risk", pa.int32(), nullable=False),
        pa.field("share", pa.float64()),
        pa.field("share_ci_lo", pa.float64()),
        pa.field("share_ci_hi", pa.float64()),
        pa.field("share_reachable_only", pa.float64()),
        pa.field("stable_mandate_window", pa.bool_()),
    ],
    metadata={
        "grain": "one row per (journal, year, language, package)",
        "estimand": (
            "share of deposits with analyzable code in L that statically "
            "reference P -- not a claim about runtime use"
        ),
    },
)

#: Registry of every table, by name. Used by the writer, the reconciliation
#: checks, and the datapackage exporter.
SCHEMAS: dict[str, pa.Schema] = {
    "journals": JOURNALS,
    "collections": COLLECTIONS,
    "datasets": DATASETS,
    "dataset_collections": DATASET_COLLECTIONS,
    "dataset_articles": DATASET_ARTICLES,
    "files": FILES,
    "mentions": MENTIONS,
    "resolution_candidates": RESOLUTION_CANDIDATES,
    "packages": PACKAGES,
    "stata_command_index": STATA_COMMAND_INDEX,
    "declared_dependencies": DECLARED_DEPENDENCIES,
    "environment_signals": ENVIRONMENT_SIGNALS,
    "unknown_names": UNKNOWN_NAMES,
    "dataset_packages": DATASET_PACKAGES,
    "journal_year_packages": JOURNAL_YEAR_PACKAGES,
}


def schema_for(table: str) -> pa.Schema:
    """Return the schema for ``table``.

    Raises:
        KeyError: if ``table`` is not a known table, listing what is.
    """
    try:
        return SCHEMAS[table]
    except KeyError:
        known = ", ".join(sorted(SCHEMAS))
        raise KeyError(f"unknown table {table!r}; known tables: {known}") from None


def required_fields(table: str) -> list[str]:
    """Names of the non-nullable fields of ``table``."""
    return [f.name for f in schema_for(table) if not f.nullable]

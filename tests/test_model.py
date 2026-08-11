"""Tests for the data and metadata models.

These guard the invariants that v1 violated silently. In particular, that join
keys are non-nullable is not a style preference: v1 produced null keys for
71,069 of 71,069 mentions and lost every one to ``groupby(dropna=True)``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from softverse.model import SCHEMAS, Construct, Language, ParseStatus, schema_for
from softverse.model.enums import (
    ANALYZABLE_STATUSES,
    FIRST_CLASS_LANGUAGES,
    NON_USE_CONSTRUCTS,
)
from softverse.model.io import (
    SchemaViolation,
    cast_to_schema,
    read_table,
    reconcile,
    validate_table,
    write_table,
)
from softverse.model.schemas import required_fields


def test_every_table_has_a_grain_declared():
    """A table whose grain is undocumented invites the wrong join."""
    for name, schema in SCHEMAS.items():
        meta = schema.metadata or {}
        assert b"grain" in meta, f"{name} does not declare its grain"


@pytest.mark.parametrize(
    ("table", "key"),
    [
        ("mentions", "file_uid"),
        ("mentions", "dataset_doi"),
        ("mentions", "collection_id"),
        ("mentions", "source"),
        ("files", "dataset_doi"),
        ("files", "collection_id"),
        ("datasets", "collection_id"),
        ("datasets", "source"),
        ("dataset_packages", "dataset_doi"),
        ("collections", "source"),
        ("collections", "kind"),
    ],
)
def test_join_keys_are_non_nullable(table, key):
    """The v1 headline bug, encoded as a test."""
    assert key in required_fields(table), (
        f"{table}.{key} is nullable; a null here is exactly the defect that "
        f"silently dropped every row of the v1 tally"
    )


def test_mentions_carry_provenance_triple():
    """Every measurement must name the instrument that produced it."""
    fields = required_fields("mentions")
    assert "extractor_version" in fields
    assert "registry_lock_id" in fields


def test_mentions_carry_source_location():
    """line/col/snippet are what make a published number checkable."""
    names = {f.name for f in schema_for("mentions")}
    assert {"line", "col", "snippet"} <= names


def test_journal_year_table_stores_the_denominator_ladder():
    """A rate whose denominator is not stored cannot be verified."""
    names = {f.name for f in schema_for("journal_year_packages")}
    assert {
        "n_datasets_in_frame",
        "n_datasets_collected",
        "n_datasets_with_code",
        "n_datasets_at_risk",
    } <= names


def test_parse_status_distinguishes_zero_from_failure():
    """v1 returned None for all of these, making the denominator unrecoverable."""
    distinct = {
        ParseStatus.OK,
        ParseStatus.SYNTAX_ERROR,
        ParseStatus.DECODE_ERROR,
        ParseStatus.UNSUPPORTED_LANGUAGE,
        ParseStatus.NOT_ANALYZED,
    }
    assert len(distinct) == 5
    assert ParseStatus.SYNTAX_ERROR not in ANALYZABLE_STATUSES


def test_provisioning_constructs_are_separable_from_use():
    """`ssc install` and `findit` are provisioning and inquiry, not use."""
    assert Construct.STATA_INSTALL in NON_USE_CONSTRUCTS
    assert Construct.STATA_WHICH in NON_USE_CONSTRUCTS
    assert Construct.LIBRARY not in NON_USE_CONSTRUCTS
    assert Construct.IMPORT not in NON_USE_CONSTRUCTS


def test_first_class_languages_include_stata():
    """v1 shipped 22,333 Stata files' worth of nothing. Never again."""
    assert Language.STATA in FIRST_CLASS_LANGUAGES
    assert FIRST_CLASS_LANGUAGES == {Language.R, Language.PYTHON, Language.STATA}
    # MATLAB has no import statement and no resolvable registry.
    assert Language.MATLAB not in FIRST_CLASS_LANGUAGES


def test_schema_for_rejects_unknown_table_helpfully():
    with pytest.raises(KeyError, match="known tables"):
        schema_for("no_such_table")


def test_empty_table_roundtrips_for_every_schema():
    """Each schema must be materializable; catches malformed type declarations."""
    for name, schema in SCHEMAS.items():
        table = pa.Table.from_pylist([], schema=schema)
        assert table.num_rows == 0, name
        assert table.schema.equals(schema), name


def _mention_row(**overrides):
    row = {
        "mention_uid": "m1",
        "file_uid": "f1",
        "dataset_doi": "doi:10.7910/DVN/ABC123",
        "collection_id": "ajps",
        "source": "dataverse",
        "language": str(Language.R),
        "construct": str(Construct.LIBRARY),
        "raw_name": "ggplot2",
        "resolution": "known_current",
        "is_dynamic": False,
        "extractor_version": "2.0.0",
        "registry_lock_id": "abc",
    }
    return {**row, **overrides}


def test_pyarrow_alone_does_not_enforce_nullability():
    """Documents *why* validate_table exists.

    `nullable=False` in a pyarrow schema is metadata, not a constraint:
    from_pylist accepts a null and Parquet stores it. Relying on the declaration
    alone would have reproduced the v1 defect while looking rigorous.
    """
    table = cast_to_schema([_mention_row(dataset_doi=None)], "mentions")
    assert table.column("dataset_doi").null_count == 1


def test_validate_table_rejects_a_null_join_key():
    """The guarantee, enforced where it is actually enforced."""
    table = cast_to_schema([_mention_row(dataset_doi=None)], "mentions")
    with pytest.raises(SchemaViolation, match="dataset_doi"):
        validate_table(table, "mentions")


def test_validate_table_accepts_a_well_formed_row():
    validate_table(cast_to_schema([_mention_row()], "mentions"), "mentions")


def test_validate_table_rejects_unexpected_columns():
    table = cast_to_schema([_mention_row()], "mentions").append_column(
        "surprise", pa.array(["x"])
    )
    with pytest.raises(SchemaViolation, match="unexpected columns"):
        validate_table(table, "mentions")


def test_write_table_roundtrips(tmp_path):
    write_table([_mention_row()], "mentions", tmp_path)
    assert read_table("mentions", tmp_path).num_rows == 1


def test_write_table_refuses_to_write_an_invalid_table(tmp_path):
    with pytest.raises(SchemaViolation):
        write_table([_mention_row(collection_id=None)], "mentions", tmp_path)
    assert not (tmp_path / "mentions.parquet").exists()


def test_reconcile_catches_a_shortfall():
    """The check whose absence let 26,681 files vanish from the v1 outputs."""
    with pytest.raises(SchemaViolation, match="unaccounted: 3"):
        reconcile(10, {"analyzed": 5, "vendored": 2}, "files")


def test_reconcile_passes_when_parts_sum():
    reconcile(10, {"analyzed": 5, "vendored": 2, "duplicate": 3}, "files")


def test_collections_table_lets_other_sources_coexist():
    """Zenodo/ICPSR/OSF must be addable without a schema migration.

    The grouping a deposit belongs to differs by source -- a journal dataverse,
    a Zenodo community, an ICPSR series -- so the fact tables key on
    `collection_id`, which is never null, and `journal_id` lives on the
    collection and is set only when the collection really is a journal.
    Hardcoding journal_id into the facts would have forced either a null in a
    key or an invented journal for every Zenodo record.
    """
    names = {f.name for f in schema_for("collections")}
    assert {"collection_id", "source", "kind", "journal_id"} <= names
    required = required_fields("collections")
    assert "collection_id" in required and "source" in required
    # Nullable by design: a Zenodo community has no journal.
    assert "journal_id" not in required


def test_facts_carry_source_so_rows_are_attributable_without_a_join():
    for table in ("datasets", "files", "mentions"):
        assert "source" in required_fields(table), table

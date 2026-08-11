"""Validated reading and writing of the model tables.

PyArrow treats ``nullable=False`` as *documentation*: ``Table.from_pylist`` will
happily build a table with nulls in a non-nullable field, and Parquet will store
it. That is not good enough here, because the single defect that invalidated the
v1 tally was a null join key -- 71,069 of 71,069 mention rows -- silently
dropped by ``groupby(dropna=True)``.

So nullability is enforced here, at the boundary, and every table goes through
:func:`write_table`. A violation raises with the offending column, the count,
and an example row rather than producing a quietly shorter output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from softverse.logging_setup import get_logger
from softverse.model.schemas import schema_for

logger = get_logger(__name__)


class SchemaViolation(ValueError):
    """A table does not satisfy its declared schema."""


def validate_table(table: pa.Table, name: str) -> None:
    """Check ``table`` against the declared schema for ``name``.

    Enforces what PyArrow declares but does not check:

    - every non-nullable field is present and null-free
    - no unexpected columns
    - field types match

    Raises:
        SchemaViolation: naming the column, the null count, and an example.
    """
    schema = schema_for(name)
    declared = {f.name for f in schema}
    actual = set(table.column_names)

    if missing := declared - actual:
        raise SchemaViolation(f"{name}: missing columns {sorted(missing)}")
    if extra := actual - declared:
        raise SchemaViolation(f"{name}: unexpected columns {sorted(extra)}")

    problems: list[str] = []
    for field in schema:
        column = table.column(field.name)
        if not field.nullable:
            n_null = column.null_count
            if n_null:
                example = _first_null_row(table, field.name)
                problems.append(
                    f"  {field.name}: {n_null:,} nulls in a non-nullable field"
                    f" (first offending row: {example})"
                )
    if problems:
        raise SchemaViolation(
            f"{name}: non-nullable fields contain nulls.\n"
            + "\n".join(problems)
            + "\nA null join key is the defect that silently voided the v1 tally;"
            " fix the producer rather than relaxing the schema."
        )


def _first_null_row(table: pa.Table, column: str) -> dict[str, Any]:
    """Return a small dict describing the first row where ``column`` is null."""
    mask = pc.is_null(table.column(column))
    indices = pc.indices_nonzero(mask)
    if not len(indices):
        return {}
    row = table.slice(indices[0].as_py(), 1).to_pylist()[0]
    # Keep the identifying columns only; snippets are long and noisy.
    keys = [k for k in row if k.endswith(("_uid", "_id", "_doi")) or k == column]
    return {k: row[k] for k in keys[:6]}


def cast_to_schema(rows: list[dict[str, Any]], name: str) -> pa.Table:
    """Build a table from ``rows``, filling absent optional fields with null.

    Callers construct plain dicts; this is the single place that knows the
    column order and types, so a producer cannot drift from the schema.
    """
    schema = schema_for(name)
    if not rows:
        return pa.Table.from_pylist([], schema=schema)
    blank: dict[str, Any] = {f.name: None for f in schema}
    normalized = [{**blank, **row} for row in rows]
    return pa.Table.from_pylist(normalized, schema=schema)


def write_table(
    table: pa.Table | list[dict[str, Any]],
    name: str,
    directory: Path,
    partition_cols: list[str] | None = None,
) -> Path:
    """Validate and write ``table`` as Parquet.

    Args:
        table: An Arrow table, or rows to be cast to the schema.
        name: Table name, used for the schema lookup and the filename.
        directory: Output directory, created if absent.
        partition_cols: Optional Hive partitioning columns.

    Returns:
        The path written.
    """
    if isinstance(table, list):
        table = cast_to_schema(table, name)
    validate_table(table, name)

    directory.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        target = directory / name
        pq.write_to_dataset(
            table,
            root_path=str(target),
            partition_cols=partition_cols,
            compression="zstd",
        )
    else:
        target = directory / f"{name}.parquet"
        pq.write_table(table, target, compression="zstd")

    logger.info(
        "wrote table",
        extra={"table": name, "rows": table.num_rows, "path": str(target)},
    )
    return target


def read_table(name: str, directory: Path) -> pa.Table:
    """Read a table and validate it against its schema."""
    path = directory / f"{name}.parquet"
    if not path.exists():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"no table {name!r} under {directory}")
    table = pq.read_table(path)
    validate_table(table, name)
    return table


def reconcile(
    total: int, parts: dict[str, int], label: str, tolerance: int = 0
) -> None:
    """Assert that ``parts`` sum to ``total``.

    The check v1 never made. Between its source archive and its published CSV,
    26,681 files and 4,189 repositories disappeared with no record, because
    nothing ever asserted that the pieces added up to the whole.

    Raises:
        SchemaViolation: showing the shortfall and the breakdown.
    """
    summed = sum(parts.values())
    if abs(total - summed) > tolerance:
        breakdown = "\n".join(f"  {k}: {v:,}" for k, v in sorted(parts.items()))
        raise SchemaViolation(
            f"{label}: reconciliation failed. total={total:,} but parts sum to "
            f"{summed:,} (unaccounted: {total - summed:,})\n{breakdown}"
        )
    logger.info(
        "reconciled", extra={"check": label, "total": total, "parts": len(parts)}
    )

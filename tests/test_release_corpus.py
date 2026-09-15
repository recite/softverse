"""The corpus release on a two-deposit world: one CC0, one under custom terms.

The checks are tested by breaking the release on purpose. A check that still
passes after its invariant is violated is not a check.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from softverse.model.io import write_table
from softverse.release.corpus import Inputs, build, check

OPEN, CLOSED = "doi:10.7910/DVN/OPEN01", "doi:10.7910/DVN/SHUT01"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def world(tmp_path):
    frame = tmp_path / "frame"
    _write_csv(
        frame / "frame.csv",
        [{"collection_id": "ajps", "journal_name": "AJPS", "discipline": "polisci"}],
    )
    _write_csv(
        frame / "dataverse_deposits.csv",
        [
            {
                "collection_id": "ajps",
                "identifier": f"DVN/{doi.rsplit('/', 1)[-1]}",
                "publication_date": "2021-03-04",
            }
            for doi in (OPEN, CLOSED)
        ],
    )

    dataverse = tmp_path / "dataverse"
    (dataverse / "raw").mkdir(parents=True)
    for doi, meta in (
        (OPEN, {"license": {"name": "CC0 1.0"}}),
        (CLOSED, {"termsOfUse": "Not to be distributed outside Harvard Dataverse."}),
    ):
        with gzip.open(
            dataverse / "raw" / f"{doi.rsplit('/', 1)[-1]}.json.gz", "wt"
        ) as h:
            json.dump({"data": meta}, h)
    (dataverse / "ledger.jsonl").write_text(
        json.dumps({"dataset_doi": OPEN, "state": "complete"})
        + "\n"
        + json.dumps(
            {
                "dataset_doi": CLOSED,
                "state": "complete",
                "skipped_archives": [{"filename": "big.tar", "size_bytes": 9}],
            }
        )
        + "\n"
    )

    files, mentions = [], []
    for doi, text in (
        (OPEN, b"library(fixest)\n"),
        (CLOSED, b"library(fixest)\n# x\n"),
    ):
        path = dataverse / "files" / doi.rsplit("/", 1)[-1] / "main.R"
        path.parent.mkdir(parents=True)
        path.write_bytes(text)
        uid = f"f-{doi[-6:]}"
        files.append(
            {
                "file_uid": uid,
                "dataset_version_uid": doi,
                "dataset_doi": doi,
                "source": "dataverse",
                "collection_id": "ajps",
                "relative_path": "main.R",
                "filename": "main.R",
                "extension": ".r",
                "language": "r",
                "size_bytes": len(text),
                "sha256_local": hashlib.sha256(text).hexdigest(),
                "encoding": "utf-8",
                "is_vendored": False,
                "parse_status": "ok",
                "n_mentions": 1,
                "in_analysis_set": True,
                "local_path": str(path),
            }
        )
        mentions.append(
            {
                "mention_uid": f"m-{doi[-6:]}",
                "file_uid": uid,
                "dataset_doi": doi,
                "source": "dataverse",
                "collection_id": "ajps",
                "deposit_year": 2021,
                "language": "r",
                "construct": "library",
                "raw_name": "fixest",
                "resolved_package": "fixest",
                "ecosystem": "cran",
                "resolution": "known_current",
                "line": 1,
                "snippet": "library(fixest)",
                "is_dynamic": False,
                "extractor_version": "test",
                "registry_lock_id": "test",
            }
        )
    tally = tmp_path / "tally"
    write_table(files, "files", tally)
    write_table(mentions, "mentions", tally)
    write_table(
        [
            {
                "dataset_doi": OPEN,
                "source_file_uid": "f-OPEN01",
                "manifest_kind": "renv_lock",
                "package": "fixest",
                "version_constraint": "0.11.1",
                "ecosystem": "cran",
                "dependency_role": "locked",
            }
        ],
        "declared_dependencies",
        tally,
    )
    write_table(
        [
            {
                "dataset_doi": OPEN,
                "source_file_uid": "f-OPEN01",
                "manifest_kind": "renv_lock",
                "signal": "r_version",
                "value": "4.2.2",
            }
        ],
        "environment_signals",
        tally,
    )
    pd.DataFrame(
        [{"package": "fixest", "language": "r", "ecosystem": "cran", "n_deposits": 2}]
    ).to_parquet(tally / "usage_by_package.parquet")
    for name in ("usage_by_package_year", "usage_by_collection", "usage_by_function"):
        pd.DataFrame([{"package": "fixest", "n_deposits": 2}]).to_parquet(
            tally / f"{name}.parquet"
        )

    inputs = Inputs(
        tally=tally,
        frame=frame,
        dataverse_corpus=dataverse,
        zenodo_corpus=tmp_path / "zenodo",
    )
    return inputs, tmp_path / "release"


def _query(sql: str):
    return duckdb.connect().execute(sql).fetchall()


def test_a_clean_release_passes_every_check(world):
    inputs, out = world
    counts = build(inputs, out)
    assert check(inputs, out) == []
    assert counts["deposits"] == 2
    assert counts["contents"] == 1, "only the CC0 deposit's text is published"

    deposits = {
        r[0]: r[1:]
        for r in _query(
            f"SELECT dataset_doi, license_id, content_redistributable, "
            f"n_archives_skipped FROM '{out / 'deposits.parquet'}'"
        )
    }
    assert deposits == {OPEN: ("CC0-1.0", True, 0), CLOSED: ("custom", False, 1)}

    snippets = dict(
        _query(f"SELECT dataset_doi, snippet FROM '{out / 'mentions.parquet'}'")
    )
    assert snippets == {OPEN: "library(fixest)", CLOSED: None}

    assert _query(
        f"SELECT dataset_doi, packages FROM '{out / 'files.parquet'}' ORDER BY 1"
    ) == [(OPEN, ["fixest"]), (CLOSED, ["fixest"])]
    assert _query(
        f"SELECT package, version, version_source "
        f"FROM '{out / 'package_versions.parquet'}'"
    ) == [("fixest", "0.11.1", "renv_lock")]
    assert pq.read_table(out / "tally_r.parquet").num_rows == 1


def test_a_leaked_closed_file_fails_the_check(world):
    inputs, out = world
    build(inputs, out)
    shard = next((out / "contents").glob("*.parquet"))
    table = pq.read_table(shard).to_pylist()
    closed = (inputs.dataverse_corpus / "files" / "SHUT01" / "main.R").read_bytes()
    table.append(
        {
            "sha256": hashlib.sha256(closed).hexdigest(),
            "content": closed.decode(),
            "src_encoding": "utf-8",
            "length_bytes": len(closed),
        }
    )
    pq.write_table(pq.read_table(shard).from_pylist(table), shard)
    assert any("no redistributable file" in p for p in check(inputs, out))


def test_a_wrong_tally_fails_the_check(world):
    inputs, out = world
    build(inputs, out)
    pd.DataFrame(
        [{"package": "fixest", "ecosystem": "cran", "n_deposits": 3}]
    ).to_parquet(out / "tally_r.parquet")
    assert any("tally_r" in p for p in check(inputs, out))

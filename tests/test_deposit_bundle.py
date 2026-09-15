"""The Zenodo bundle is CC0 tables only: no file text, no code snippets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import deposit_tally


def test_the_bundle_carries_no_code(tmp_path):
    tally, corpus = tmp_path / "tally", tmp_path / "corpus"
    tally.mkdir()
    (corpus / "contents").mkdir(parents=True)
    pd.DataFrame({"package": ["fixest"], "n_deposits": [2]}).to_csv(
        tally / "usage_by_package.csv", index=False
    )
    (tally / "README.md").write_text("tally")
    (tally / "summary.json").write_text("{}")
    pd.DataFrame({"x": [1]}).to_parquet(tally / "mentions.parquet")
    for name in deposit_tally.CORPUS_FILES:
        pd.DataFrame({"x": [1]}).to_parquet(corpus / name)
    pd.DataFrame({"mention_uid": ["m"], "snippet": ["library(fixest)"]}).to_parquet(
        corpus / "mentions.parquet"
    )
    pd.DataFrame({"sha256": ["s"], "content": ["code"]}).to_parquet(
        corpus / "contents" / "contents-00000.parquet"
    )

    files = deposit_tally.stage_bundle(tally, corpus, tmp_path / "bundle")
    names = {p.name for p in files}

    assert not any("contents" in n for n in names)
    assert "mentions.parquet" in names
    assert (
        "snippet" not in pq.read_schema(tmp_path / "bundle" / "mentions.parquet").names
    )
    assert {
        "usage_by_package.csv",
        "summary.json",
        "README.md",
        "deposits.parquet",
    } <= names

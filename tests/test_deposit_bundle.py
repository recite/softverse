"""The Zenodo bundle is CC0 tables only: no file text, no code snippets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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
    pd.DataFrame(
        {
            "mention_uid": ["m", "n"],
            "dataset_doi": ["doi:a", "doi:b"],
            "file_uid": ["f", "g"],
            "line": [1, 2],
            "snippet": ["library(fixest)", "library(dplyr)"],
        }
    ).to_parquet(corpus / "mentions.parquet")
    pd.DataFrame({"sha256": ["s"], "content": ["code"]}).to_parquet(
        corpus / "contents" / "contents-00000.parquet"
    )

    files = deposit_tally.stage_bundle(tally, corpus, tmp_path / "bundle")
    names = {p.name for p in files}

    assert not any("contents" in n for n in names)
    # Cut into parts small enough to upload, which read back as one table
    # holding every row once and no snippet.
    parts = sorted(n for n in names if n.startswith("mentions-"))
    assert len(parts) == deposit_tally.MENTION_PARTS
    assert "mentions.parquet" not in names
    whole = pd.concat(pd.read_parquet(tmp_path / "bundle" / n) for n in parts)
    assert sorted(whole["mention_uid"]) == ["m", "n"]
    assert "snippet" not in whole.columns
    assert {
        "usage_by_package.csv",
        "summary.json",
        "README.md",
        "deposits.parquet",
    } <= names

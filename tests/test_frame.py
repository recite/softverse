"""Tests for the sampling frame.

A corpus without a stated inclusion rule is a convenience sample, and no amount
of downstream care fixes that. These pin the rule.
"""

from __future__ import annotations

import csv
import json

import httpx

from softverse.acquire.http import PoliteClient, RateLimiter
from softverse.frame import (
    DCAS_VIA_DATAVERSE,
    UNLOCATED,
    ZENODO_JOURNAL_COMMUNITIES,
    FrameRow,
    unlocated_rows,
    write_frame,
    zenodo_rows,
)


def _client(monkeypatch, total: int) -> PoliteClient:
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200,
            content=json.dumps({"hits": {"total": total}}).encode(),
            request=httpx.Request("GET", url),
        ),
    )
    return client


def test_a_community_that_returns_the_archive_is_excluded(monkeypatch):
    """The guard that matters most.

    Zenodo ignores an unknown communities= filter and returns all 7.1M records.
    A mistyped slug must drop out of the frame, not silently substitute the
    entire archive for one journal.
    """
    client = _client(monkeypatch, total=7_106_477)
    assert zenodo_rows(client) == []
    client.close()


def test_verified_communities_enter_the_frame(monkeypatch):
    client = _client(monkeypatch, total=266)
    rows = zenodo_rows(client)
    assert len(rows) == len(ZENODO_JOURNAL_COMMUNITIES)
    assert all(r.inclusion_reason == "dcas" for r in rows)
    assert all(r.verified_at for r in rows), "verification must be timestamped"
    client.close()


def test_every_frame_row_states_why_it_is_included():
    """A row without a reason is a convenience sample of one."""
    rows = unlocated_rows()
    assert rows and all(r.inclusion_reason for r in rows)


def test_unlocated_journals_are_rows_not_silence():
    """A coverage gap must be visible in the artifact, not absent from it."""
    names = {j for j, _ in UNLOCATED}
    assert "American Economic Review" in names, "AEA is on openICPSR, not Zenodo"
    assert all(r.notes.startswith("unlocated") for r in unlocated_rows())


def test_dcas_journals_reaching_us_via_dataverse_are_recorded():
    """The two halves of the frame must be visibly reconciled, not assumed
    disjoint: JPE and RFS arrive through Dataverse, not Zenodo."""
    assert DCAS_VIA_DATAVERSE["Journal of Political Economy"]
    assert DCAS_VIA_DATAVERSE["Review of Financial Studies"] == ["rfs"]


def test_jeea_uses_an_underscore():
    """Recorded because guessing cost real time: the convention is mostly
    <journal>-replication, but JEEA is jeea_replication, and restud /
    jeea-replication / cje-replication all look plausible and do not exist."""
    slugs = {s for s, _, _ in ZENODO_JOURNAL_COMMUNITIES}
    assert "jeea_replication" in slugs
    assert "jeea-replication" not in slugs


def test_frame_roundtrips_to_csv(tmp_path):
    path = write_frame(
        [FrameRow("x", "zenodo", "journal", "J", "economics", "dcas")],
        tmp_path / "frame.csv",
    )
    (row,) = list(csv.DictReader(open(path)))
    assert row["collection_id"] == "x"
    assert row["inclusion_reason"] == "dcas"

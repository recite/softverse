"""Tests for the collector: rate limiting, retry policy, and state.

The state tests matter most. v1's collector wrote "success" before downloading
anything and then skipped any dataset that had a status, which is how 8,953 of
12,054 deposits ended up marked successful while holding zero bytes.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from softverse.acquire.http import RETRY_STATUSES, PoliteClient, RateLimiter
from softverse.acquire.state import DatasetRecord, Ledger, atomic_write_bytes
from softverse.model.enums import CollectionState
from softverse.sources.dataverse import (
    CandidateFile,
    candidates_from,
    is_wanted,
    repo_id_of,
)

# -- rate limiting --------------------------------------------------------


def test_limiter_caps_the_sustained_rate():
    limiter = RateLimiter(rate_per_s=20.0, burst=1)
    start = time.monotonic()
    for _ in range(10):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # 10 tokens at 20/s cannot arrive faster than ~0.45 s.
    assert elapsed >= 0.4, elapsed


def test_limiter_is_global_not_per_thread():
    """A per-worker sleep is not a rate limit: four workers each sleeping 0.5 s
    still issue 8 req/s. The bucket must bind the process."""
    limiter = RateLimiter(rate_per_s=20.0, burst=1)
    start = time.monotonic()
    threads = [
        threading.Thread(target=lambda: [limiter.acquire() for _ in range(5)])
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    # 20 acquisitions across 4 threads at 20/s is still ~1 s, not ~0.25 s.
    assert elapsed >= 0.9, elapsed


def test_penalize_only_ever_slows_down():
    limiter = RateLimiter(rate_per_s=2.0)
    limiter.penalize("429")
    assert limiter.rate == pytest.approx(1.0)
    limiter.penalize("429")
    assert limiter.rate == pytest.approx(0.5)


def test_penalize_has_a_floor():
    limiter = RateLimiter(rate_per_s=2.0)
    for _ in range(20):
        limiter.penalize("429")
    assert limiter.rate >= 0.2


# -- retry policy ---------------------------------------------------------


def test_only_transient_statuses_are_retried():
    """404 and 403 are answers, not failures.

    v1 retried on any exception including 404, and backed off up to 310 s per
    permanently forbidden file.
    """
    assert 429 in RETRY_STATUSES
    assert 503 in RETRY_STATUSES
    assert 404 not in RETRY_STATUSES
    assert 403 not in RETRY_STATUSES


def test_client_does_not_retry_a_404(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        status_code = 404
        headers: dict[str, str] = {}
        content = b""

    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(client._client, "get", fake_get)
    outcome = client.get("http://x/")
    assert calls["n"] == 1, "a 404 must not be retried"
    assert outcome.not_found and not outcome.retryable
    client.close()


def test_client_retries_a_503_then_gives_up(monkeypatch):
    calls = {"n": 0}

    class FakeResponse:
        status_code = 503
        headers: dict[str, str] = {}
        content = b""

    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000), max_retries=2)
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(client._client, "get", fake_get)
    outcome = client.get("http://x/")
    assert calls["n"] == 3
    assert not outcome.ok and outcome.retryable
    client.close()


def test_a_429_slows_the_whole_run(monkeypatch):
    class FakeResponse:
        status_code = 429
        headers = {"Retry-After": "0"}
        content = b""

    limiter = RateLimiter(rate_per_s=4.0)
    client = PoliteClient(limiter=limiter, max_retries=1)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(client._client, "get", lambda url, **kw: FakeResponse())
    client.get("http://x/")
    assert limiter.rate < 4.0
    client.close()


# -- state ----------------------------------------------------------------


def test_record_reconciles():
    record = DatasetRecord(
        dataset_doi="doi:x",
        state="complete",
        n_candidate=10,
        n_fetched=7,
        n_skipped_over_cap=1,
        n_restricted=1,
        n_failed=1,
    )
    assert record.reconciles()


def test_record_that_does_not_reconcile_is_detectable():
    """The check v1 never made anywhere."""
    record = DatasetRecord(
        dataset_doi="doi:x", state="complete", n_candidate=10, n_fetched=3
    )
    assert not record.reconciles()


def test_ledger_roundtrips(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.finish(DatasetRecord("doi:a", CollectionState.COMPLETE.value, 2, 2))
    reloaded = Ledger(tmp_path / "l.jsonl")
    assert reloaded.get("doi:a") is not None
    assert reloaded.get("doi:a").state == "complete"


def test_completed_datasets_are_skipped_on_rerun(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.finish(DatasetRecord("doi:a", CollectionState.COMPLETE.value, 1, 1))
    assert not ledger.should_process("doi:a")
    assert ledger.should_process("doi:never-seen")


@pytest.mark.parametrize("state", ["partial", "failed"])
def test_failed_and_partial_are_always_retried(tmp_path, state):
    """v1 treated any existing status as done -- including download_failed and
    forbidden -- and re-counted those failures as successes every run."""
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.finish(DatasetRecord("doi:a", state))
    assert ledger.should_process("doi:a")


@pytest.mark.parametrize(
    "state", ["complete", "no_candidate_files", "restricted", "deaccessioned"]
)
def test_terminal_states_are_not_retried(tmp_path, state):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.finish(DatasetRecord("doi:a", state))
    assert not ledger.should_process("doi:a")


def test_a_truncated_ledger_line_does_not_lose_the_rest(tmp_path):
    """A process killed mid-write must not corrupt the resume state."""
    path = tmp_path / "l.jsonl"
    path.write_text(
        json.dumps({"dataset_doi": "doi:a", "state": "complete"})
        + "\n{ this line was truncated by a kill\n"
        + json.dumps({"dataset_doi": "doi:b", "state": "complete"})
        + "\n"
    )
    ledger = Ledger(path)
    assert ledger.get("doi:a") is not None
    assert ledger.get("doi:b") is not None


def test_last_write_for_a_doi_wins(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.finish(DatasetRecord("doi:a", "failed"))
    ledger.finish(DatasetRecord("doi:a", CollectionState.COMPLETE.value, 1, 1))
    assert Ledger(tmp_path / "l.jsonl").get("doi:a").state == "complete"


def test_skipped_archives_are_reportable(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    record = DatasetRecord("doi:a", "complete", n_candidate=1, n_skipped_over_cap=1)
    record.skipped_archives = [{"filename": "big.zip", "size_bytes": 700_000_000}]
    ledger.finish(record)
    skipped = Ledger(tmp_path / "l.jsonl").skipped_archives()
    assert skipped[0]["filename"] == "big.zip"
    assert skipped[0]["dataset_doi"] == "doi:a"


def test_atomic_write_leaves_no_partial_file(tmp_path):
    target = tmp_path / "deep" / "f.bin"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    assert not list(tmp_path.glob("**/*.tmp"))


# -- candidate selection --------------------------------------------------


def test_repo_id_of():
    assert repo_id_of("doi:10.7910/DVN/ABC123") == "ABC123"


@pytest.mark.parametrize(
    ("name", "wanted"),
    [
        ("analysis.R", True),
        ("run.do", True),
        ("model.py", True),
        ("nb.ipynb", True),
        ("renv.lock", True),  # free validation data; v1 deleted these
        ("requirements.txt", True),
        ("replication.zip", True),
        ("survey.dta", False),
        ("figure.png", False),
        ("._resource.R", False),  # AppleDouble fork
    ],
)
def test_is_wanted(name, wanted):
    assert is_wanted(name) is wanted


def test_candidates_carry_provenance_and_folders():
    payload = {
        "files": [
            {
                "directoryLabel": "code/clean",
                "dataFile": {
                    "id": 42,
                    "filename": "01.R",
                    "filesize": 100,
                    "md5": "abc",
                    "restricted": False,
                },
            },
            {"dataFile": {"id": 43, "filename": "data.dta", "filesize": 9}},
        ]
    }
    (candidate,) = candidates_from(payload)
    assert candidate.file_id == 42
    assert candidate.md5 == "abc"
    # directoryLabel preserves the author's folders, which is how vendored
    # libraries stay distinguishable from research code.
    assert candidate.relative_path == "code/clean/01.R"


def test_restricted_files_are_flagged_not_dropped():
    payload = {
        "files": [
            {
                "dataFile": {
                    "id": 1,
                    "filename": "a.R",
                    "restricted": True,
                    "filesize": 1,
                }
            }
        ]
    }
    (candidate,) = candidates_from(payload)
    assert candidate.restricted


def test_archive_detection():
    assert CandidateFile(1, "x.zip", 1, None, False, None).is_archive
    assert not CandidateFile(1, "x.R", 1, None, False, None).is_archive


def test_empty_202_is_treated_as_throttling_not_success(monkeypatch):
    """Measured Harvard Dataverse behaviour, not spec.

    When it decides a client is going too fast it answers 202 Accepted with an
    empty text/html body -- no 429, no Retry-After. A client treating 2xx as
    success would record the dataset as having zero files and mark it done,
    which is exactly the shape of v1's 8,953 empty "successes".
    """

    class Throttled:
        status_code = 202
        headers: dict[str, str] = {}
        content = b""

    limiter = RateLimiter(rate_per_s=4.0)
    client = PoliteClient(limiter=limiter, max_retries=1)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(client._client, "get", lambda url, **kw: Throttled())
    outcome = client.get("http://x/")
    assert not outcome.ok, "an empty 202 must never be reported as success"
    assert outcome.retryable
    assert limiter.rate < 4.0, "throttling must slow the rest of the run"
    client.close()


def test_empty_200_is_also_treated_as_throttling(monkeypatch):
    class Empty:
        status_code = 200
        headers: dict[str, str] = {}
        content = b""

    client = PoliteClient(limiter=RateLimiter(rate_per_s=4.0), max_retries=0)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    monkeypatch.setattr(client._client, "get", lambda url, **kw: Empty())
    assert not client.get("http://x/").ok
    client.close()

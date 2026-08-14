"""Tests for the Zenodo and OSF collectors, against recorded API shapes.

No network: the fixtures are the shapes the real APIs returned during
verification, so these run in CI and do not cost either service a request.
"""

from __future__ import annotations

import json

import httpx
import pytest

from softverse.acquire.http import PoliteClient, RateLimiter, read_rate_headers
from softverse.acquire.state import Ledger
from softverse.sources import osf, zenodo


def serving(body: bytes) -> PoliteClient:
    """A client whose every request returns ``body``.

    Wired at the transport rather than by patching ``get``, because archives
    are streamed through ``stream()`` and loose files are fetched through
    ``get()``. Patching one method would leave the other talking to the real
    network, and a test that silently exercises the wrong path is worse than
    no test.
    """
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    )
    return client


def unlimited_disk():
    """A disk budget that always says yes.

    Tests that exercise the archive path must not depend on how full the
    machine happens to be. Two of them silently started failing when this
    laptop dropped below the 8 GB reserve -- not because the code broke, but
    because an unrelated project filled the volume, and the assertion was
    reading the disk rather than the collector.
    """
    from softverse.sources.zenodo import DiskBudget

    return DiskBudget(free=lambda: 1 << 60, reserve=0)


# -- rate headers ---------------------------------------------------------


def test_rate_headers_are_read_when_published():
    """Zenodo publishes its limit; use its number rather than our guess."""
    response = httpx.Response(
        200, headers={"x-ratelimit-limit": "30", "x-ratelimit-remaining": "29"}
    )
    assert read_rate_headers(response) == (29, 30)
    assert read_rate_headers(httpx.Response(200)) is None


def test_client_slows_down_before_being_refused(monkeypatch):
    """Backing off only after a 429 means the 429 already happened."""
    client = PoliteClient(limiter=RateLimiter(rate_per_s=4.0))
    before = client.limiter.rate
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200,
            content=b"{}",
            headers={"x-ratelimit-limit": "30", "x-ratelimit-remaining": "2"},
            request=httpx.Request("GET", url),
        ),
    )
    client.get("http://x/")
    assert client.limiter.rate < before
    client.close()


# -- Zenodo ---------------------------------------------------------------

ZENODO_RECORD = {
    "id": 3371190,
    "metadata": {
        "doi": "10.5281/zenodo.3371190",
        "title": "Replication package",
        "publication_date": "2019-08-20",
        "communities": [{"id": "economics"}],
    },
    "files": [
        {
            "key": "analysis.R",
            "size": 1234,
            "checksum": "md5:d41d8cd98f00b204e9800998ecf8427e",
            "links": {"self": "https://zenodo.org/api/files/x/analysis.R"},
        },
        {
            "key": "data.csv",
            "size": 99,
            "checksum": "md5:abc",
            "links": {"self": "https://zenodo.org/api/files/x/data.csv"},
        },
        {
            "key": "big.zip",
            # Expressed relative to the cap rather than as a literal, so this
            # tests the behaviour instead of whatever the cap happened to be
            # when it was written. The previous literal (900 MB) silently
            # stopped testing anything when the cap rose to 2 GB.
            "size": zenodo.ARCHIVE_CAP_BYTES + 1,
            "checksum": "md5:def",
            "links": {"self": "https://zenodo.org/api/files/x/big.zip"},
        },
    ],
}


def test_zenodo_record_parses_files_and_communities():
    record = zenodo.parse_record(ZENODO_RECORD)
    assert record.record_id == "3371190"
    assert record.year == 2019
    # Community membership is recorded, so the frame is statable. v1 built this
    # in memory and never wrote it.
    assert record.communities == ["economics"]
    assert {f.key for f in record.files} == {"analysis.R", "data.csv", "big.zip"}


def test_zenodo_checksum_prefix_is_stripped():
    record = zenodo.parse_record(ZENODO_RECORD)
    analysis = next(f for f in record.files if f.key == "analysis.R")
    assert analysis.md5 == "d41d8cd98f00b204e9800998ecf8427e"


def test_zenodo_collects_code_and_skips_data(tmp_path, monkeypatch):
    record = zenodo.parse_record(ZENODO_RECORD)
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200, content=b"", request=httpx.Request("GET", url)
        ),
    )
    state, rows = zenodo.collect_record(client, record, tmp_path)
    names = {r["filename"] for r in rows}
    assert "analysis.R" in names
    assert "data.csv" not in names, "data files are not candidates"
    # The oversized archive is recorded as a coverage gap, not silently dropped.
    assert state.n_skipped_over_cap == 1
    assert state.skipped_archives[0]["filename"] == "big.zip"
    assert state.reconciles()
    client.close()


def test_zenodo_collections_rows_record_the_frame():
    rows = zenodo.collections_rows([zenodo.parse_record(ZENODO_RECORD)])
    assert rows[0]["collection_id"] == "economics"
    assert rows[0]["kind"] == "community"
    assert rows[0]["source"] == "zenodo"


# -- OSF ------------------------------------------------------------------


def _osf_listing(entries: list[dict], next_url: str | None = None) -> bytes:
    return json.dumps({"data": entries, "links": {"next": next_url}}).encode()


def _file(name: str, path: str) -> dict:
    return {
        "attributes": {
            "kind": "file",
            "name": name,
            "materialized_path": path,
            "size": 10,
            "extra": {"hashes": {"md5": "abc"}},
        },
        "links": {"download": f"https://files.osf.io/{name}"},
    }


def _folder(name: str, href: str) -> dict:
    return {
        "attributes": {"kind": "folder", "name": name},
        "relationships": {"files": {"links": {"related": {"href": href}}}},
    }


def test_osf_recursion_finds_code_one_level_down(monkeypatch):
    """The v1 defect, stated as a test.

    v1 listed only the provider root and skipped anything whose kind was not
    'file'. A project with its code in analysis/ therefore yielded nothing and
    was recorded as no_scripts -- a false negative that looks like a finding.
    """
    pages = {
        "root": _osf_listing(
            [_folder("analysis", "sub"), _file("README.md", "/README.md")]
        ),
        "sub": _osf_listing([_file("model.R", "/analysis/model.R")]),
    }
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200, content=pages[url], request=httpx.Request("GET", url)
        ),
    )
    found = osf.walk_files(client, "root", osf.Budget())
    assert [f.name for f in found] == ["model.R"], "code one level down must be found"
    assert found[0].path == "analysis/model.R", "folder structure must survive"
    client.close()


def test_osf_recursion_is_depth_bounded(monkeypatch):
    """Deep trees are vendored libraries; paying requests to descend is waste."""
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200,
            content=_osf_listing([_folder("deeper", "same")]),
            request=httpx.Request("GET", url),
        ),
    )
    budget = osf.Budget(limit=1000)
    osf.walk_files(client, "same", budget)
    assert budget.spent <= osf.MAX_DEPTH + 2, budget.spent
    client.close()


def test_osf_budget_stops_cleanly_rather_than_hitting_the_wall():
    """10,000/day is a hard ceiling, so exhaustion is a stopping point."""
    budget = osf.Budget(limit=3)
    budget.charge()
    budget.charge()
    with pytest.raises(osf.BudgetExhausted):
        budget.charge()
    assert budget.remaining == 0


def test_osf_collect_stops_on_exhausted_budget_without_losing_state(
    tmp_path, monkeypatch
):
    nodes = [
        osf.OSFNode(node_id=f"n{i}", title="t", date_created="2020-01-01")
        for i in range(5)
    ]
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200, content=_osf_listing([]), request=httpx.Request("GET", url)
        ),
    )
    ledger = Ledger(tmp_path / "l.jsonl")
    osf.collect(nodes, tmp_path, ledger, client, budget=osf.Budget(limit=2))
    # Whatever finished is durable; the rest is simply still to do.
    assert len(ledger) < len(nodes)
    client.close()


def test_unknown_community_is_refused_not_silently_widened(monkeypatch):
    """Zenodo ignores an unknown communities= filter and returns everything.

    Measured: communities=restud and communities=aeaje (neither exists) each
    reported 7,106,477 records -- the entire repository. A typo in a frame
    definition would substitute all of Zenodo for one journal's community, and
    the error would look like abundance rather than a mistake.
    """
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200,
            content=json.dumps({"hits": {"total": 7_106_477}}).encode(),
            request=httpx.Request("GET", url),
        ),
    )
    with pytest.raises(zenodo.UnknownCommunity, match="whole"):
        zenodo.verify_community(client, "restud")
    client.close()


def test_real_community_passes_verification(monkeypatch):
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))
    monkeypatch.setattr(
        client._client,
        "get",
        lambda url, **kw: httpx.Response(
            200,
            content=json.dumps({"hits": {"total": 266}}).encode(),
            request=httpx.Request("GET", url),
        ),
    )
    assert zenodo.verify_community(client, "es-replication-repository") == 266
    client.close()


def test_extraction_failure_marks_the_deposit_retryable(tmp_path, monkeypatch):
    """An archive we fetched but could not open is a failure, not an empty
    deposit.

    Marking it complete makes it unretryable, so a later fix -- a raised limit,
    a newly supported format -- can never reach it. Six deposits were nearly
    lost this way when the anti-bomb limits turned out to be calibrated against
    an imagined attacker rather than the corpus.
    """
    record = zenodo.parse_record(
        {
            "id": 1,
            "metadata": {
                "doi": "10.5281/zenodo.1",
                "title": "t",
                "publication_date": "2020-01-01",
                "communities": [],
            },
            "files": [
                {
                    "key": "pkg.zip",
                    "size": 10,
                    "checksum": "md5:x",
                    "links": {"self": "https://zenodo.org/f/pkg.zip"},
                }
            ],
        }
    )
    client = serving(b"not a real zip")
    state, _rows = zenodo.collect_record(client, record, tmp_path)
    assert state.n_failed == 1
    assert state.state == "partial", "must be retryable, not complete"
    assert state.needs_retry
    client.close()


def test_archive_is_removed_after_successful_extraction(tmp_path, monkeypatch):
    """Disk forces this, and the ledger makes it safe.

    104 deposits held 4.4 GB of archives around 13 MB of code; the full frame
    would need ~68 GB against 10 GB free. Unlike v1 -- which deleted archives
    before knowing extraction worked and recorded nothing -- the ledger keeps
    the record id, file key, size and md5, so the archive is re-fetchable, and
    the manifests that matter for validation are extracted and kept.
    """
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("code/analysis.R", b"library(dplyr)")
    record = zenodo.parse_record(
        {
            "id": 7,
            "metadata": {
                "doi": "10.5281/zenodo.7",
                "title": "t",
                "publication_date": "2021-01-01",
                "communities": [],
            },
            "files": [
                {
                    "key": "pkg.zip",
                    "size": 10,
                    "checksum": "md5:x",
                    "links": {"self": "https://zenodo.org/f/pkg.zip"},
                }
            ],
        }
    )
    client = serving(buf.getvalue())
    state, rows = zenodo.collect_record(client, record, tmp_path, disk=unlimited_disk())
    assert state.state == "complete"
    assert any(r["filename"] == "analysis.R" for r in rows), "code must survive"
    assert not (
        tmp_path / "7" / "_archives" / "pkg.zip"
    ).exists(), "the compressed original should be gone"
    client.close()


def test_a_failed_archive_is_kept_for_diagnosis(tmp_path, monkeypatch):
    """A deposit that could not be extracted stays retryable *and* keeps the
    evidence needed to work out why."""
    record = zenodo.parse_record(
        {
            "id": 8,
            "metadata": {
                "doi": "10.5281/zenodo.8",
                "title": "t",
                "publication_date": "2021-01-01",
                "communities": [],
            },
            "files": [
                {
                    "key": "pkg.zip",
                    "size": 10,
                    "checksum": "md5:x",
                    "links": {"self": "https://zenodo.org/f/pkg.zip"},
                }
            ],
        }
    )
    client = serving(b"not a zip")
    state, _ = zenodo.collect_record(client, record, tmp_path, disk=unlimited_disk())
    assert state.needs_retry
    assert (tmp_path / "8" / "_archives" / "pkg.zip").exists()
    client.close()


def test_a_multipart_archive_is_a_coverage_gap_not_a_failure(tmp_path):
    """Zenodo record 11202896, as a test.

    A 6.4 GB package split into `.z01`, `.z02` and a 130 MB `.zip`. Only the
    `.zip` is a candidate -- `.z01` is not an archive extension -- and it is
    the segment holding the central directory, so it downloaded byte-for-byte
    against Zenodo's stated size, listed 165 members, and produced "Bad magic
    number for file header" on the first read. Recorded as `n_failed`, that
    is a deposit retried forever with no possible outcome; recorded as
    `n_spanned`, it is a number in the coverage statistic.
    """
    record = zenodo.parse_record(
        {
            "id": 11202896,
            "metadata": {
                "doi": "10.5281/zenodo.11202896",
                "title": "t",
                "publication_date": "2024-01-01",
                "communities": [],
            },
            "files": [
                {
                    "key": f"3-replication-package{ext}",
                    "size": size,
                    "checksum": "md5:x",
                    "links": {"self": f"https://zenodo.org/f/pkg{ext}"},
                }
                for ext, size in (
                    (".z01", 3145728000),
                    (".z02", 3145728000),
                    (".zip", 130173841),
                )
            ],
        }
    )
    client = serving(b"never requested")
    state, rows = zenodo.collect_record(client, record, tmp_path)
    assert state.n_spanned == 1
    assert state.n_failed == 0, "a complete download is not a transport failure"
    assert state.reconciles()
    assert not state.needs_retry, "no number of retries can assemble one segment"
    assert rows == []
    client.close()


def test_collect_honours_fresh(tmp_path, monkeypatch):
    """`fresh` must reach the filter, not just the signature.

    This exists because the parameter was accepted and silently ignored: a
    str.replace patch missed its anchor, so `fresh=True` would have quietly
    behaved as an incremental run. Three separate silent no-ops from that habit
    in one session is what retired it.
    """
    record = zenodo.parse_record(
        {
            "id": 5,
            "metadata": {
                "doi": "10.5281/zenodo.5",
                "title": "t",
                "publication_date": "2020-01-01",
                "communities": [],
            },
            "files": [],
        }
    )
    ledger = Ledger(tmp_path / "l.jsonl")
    from softverse.acquire.state import DatasetRecord

    ledger.finish(DatasetRecord("10.5281/zenodo.5", "complete", 0, 0))
    client = PoliteClient(limiter=RateLimiter(rate_per_s=1000))

    # Incremental: already complete, so nothing to do.
    assert zenodo.collect([record], tmp_path, ledger, client) == []
    # Fresh: the ledger is ignored, so it is attempted again.
    ledger2 = Ledger(tmp_path / "l.jsonl")
    zenodo.collect([record], tmp_path, ledger2, client, fresh=True)
    assert ledger2.get("10.5281/zenodo.5") is not None
    client.close()


def test_collect_runs_deposits_concurrently(tmp_path, monkeypatch):
    """Zenodo throttles per connection, so serial collection wastes the link.

    Measured against one deposit's archive: one stream sustained 284 KB/s,
    three concurrent streams 1.01 MB/s in aggregate. The limit is per
    connection, not per client, so the collector's one-deposit-at-a-time loop
    was leaving most of the available throughput unused -- and the remaining
    tail is precisely the multi-gigabyte deposits where that costs hours.
    """
    import threading
    import time

    records = [
        zenodo.parse_record(
            {
                "id": 900 + i,
                "metadata": {
                    "doi": f"10.5281/zenodo.{900 + i}",
                    "title": "t",
                    "publication_date": "2021-01-01",
                    "communities": [],
                },
                "files": [
                    {
                        "key": "analysis.R",
                        "size": 10,
                        "checksum": "md5:x",
                        "links": {"self": f"https://zenodo.org/f/{i}.R"},
                    }
                ],
            }
        )
        for i in range(6)
    ]

    inflight = 0
    peak = 0
    guard = threading.Lock()

    def slow_record(client, record, files_root, archive_cap=None, disk=None):
        nonlocal inflight, peak
        with guard:
            inflight += 1
            peak = max(peak, inflight)
        time.sleep(0.15)
        with guard:
            inflight -= 1
        from softverse.acquire.state import DatasetRecord

        return DatasetRecord(dataset_doi=record.doi, state="complete"), []

    monkeypatch.setattr(zenodo, "collect_record", slow_record)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    client = serving(b"")
    zenodo.collect(records, tmp_path, ledger, client, workers=3)
    client.close()

    assert peak > 1, "deposits were collected one at a time"
    assert len(ledger) == 6, "every deposit must reach the ledger exactly once"


def test_a_failing_deposit_does_not_take_down_its_neighbours(tmp_path, monkeypatch):
    """One bad record must not end a run, concurrently as well as serially."""
    records = [
        zenodo.parse_record(
            {
                "id": 800 + i,
                "metadata": {
                    "doi": f"10.5281/zenodo.{800 + i}",
                    "title": "t",
                    "publication_date": "2021-01-01",
                    "communities": [],
                },
                "files": [
                    {
                        "key": "a.R",
                        "size": 10,
                        "checksum": "md5:x",
                        "links": {"self": "https://zenodo.org/f/a.R"},
                    }
                ],
            }
        )
        for i in range(4)
    ]

    def sometimes_explode(client, record, files_root, archive_cap=None, disk=None):
        if record.record_id == "801":
            raise RuntimeError("boom")
        from softverse.acquire.state import DatasetRecord

        return DatasetRecord(dataset_doi=record.doi, state="complete"), []

    monkeypatch.setattr(zenodo, "collect_record", sometimes_explode)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    client = serving(b"")
    zenodo.collect(records, tmp_path, ledger, client, workers=3)
    client.close()

    assert len(ledger) == 4
    states = {r.dataset_doi: r.state for r in ledger._records.values()}
    assert states["10.5281/zenodo.801"] == "failed"
    assert sum(1 for s in states.values() if s == "complete") == 3


def test_workers_cannot_each_spend_the_last_of_the_disk(monkeypatch, tmp_path):
    """The overcommit that concurrency introduced.

    The free-space guard asks "is there room for this archive plus the
    reserve" and each worker got the same answer, because none of them could
    see what the others had already committed to. Three workers against a 5 GB
    cap can therefore pass a check made with 12 GB free and then write 15 GB.

    Observed live: 12 GB free with 6.8 GB of `.part` files already in flight.
    An ENOSPC is the one failure the whole state design exists to prevent --
    it can land between writing a file and recording it, which is the only
    window where the ledger and the disk can disagree.
    """
    import threading

    from softverse.sources.zenodo import DiskBudget

    # 12 GB free, 8 GB reserve: room for exactly one 3 GB archive. No wait:
    # this is about the arithmetic being shared between workers, and the
    # bounded wait has its own test.
    budget = DiskBudget(
        free=lambda: 12_000_000_000, reserve=8_000_000_000, wait_seconds=0
    )
    granted = []
    barrier = threading.Barrier(3)

    def ask():
        barrier.wait()
        granted.append(budget.reserve_bytes(3_000_000_000))

    threads = [threading.Thread(target=ask) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(granted) == 1, f"{sum(granted)} workers were each promised the disk"


def test_a_reservation_is_returned_when_the_download_ends(tmp_path):
    """Otherwise the first three archives permanently poison the budget."""
    from softverse.sources.zenodo import DiskBudget

    budget = DiskBudget(
        free=lambda: 12_000_000_000, reserve=8_000_000_000, wait_seconds=0
    )
    assert budget.reserve_bytes(3_000_000_000)
    assert not budget.reserve_bytes(3_000_000_000)
    budget.release(3_000_000_000)
    assert budget.reserve_bytes(3_000_000_000), "space must come back"


def test_the_disk_budget_waits_for_a_sibling_worker_before_refusing():
    """The usual shortfall is temporary and clears itself.

    A worker holding a 3 GB archive is about to extract it and delete it, so
    a refusal issued the instant the disk looks tight throws away work that
    would have succeeded a minute later. Refusing immediately is what turned
    a transient squeeze into 279 deferred deposits in 31 seconds.
    """
    import threading

    from softverse.sources.zenodo import DiskBudget

    budget = DiskBudget(free=lambda: 12_000_000_000, reserve=8_000_000_000)
    assert budget.reserve_bytes(3_000_000_000)

    # A second claim cannot fit yet. Release from another thread shortly, and
    # the waiter should get it rather than give up.
    threading.Timer(0.2, lambda: budget.release(3_000_000_000)).start()
    assert budget.reserve_bytes(3_000_000_000, wait_seconds=5.0)


def test_the_wait_is_bounded_so_a_full_disk_does_not_hang():
    """An externally full volume must still degrade to a deferral."""
    import time

    from softverse.sources.zenodo import DiskBudget

    budget = DiskBudget(free=lambda: 1_000, reserve=8_000_000_000)
    start = time.monotonic()
    assert not budget.reserve_bytes(3_000_000_000, wait_seconds=0.5)
    assert time.monotonic() - start < 5.0, "the wait must be bounded"


def test_a_low_disk_deferral_is_counted_as_such(tmp_path):
    """Not `n_failed`. The download never started; nothing broke."""
    from softverse.sources.zenodo import DiskBudget

    record = zenodo.parse_record(
        {
            "id": 4242,
            "metadata": {
                "doi": "10.5281/zenodo.4242",
                "title": "t",
                "publication_date": "2024-01-01",
                "communities": [],
            },
            "files": [
                {
                    "key": "big.zip",
                    "size": 1_000_000,
                    "checksum": "md5:x",
                    "links": {"self": "https://zenodo.org/f/big.zip"},
                }
            ],
        }
    )
    full = DiskBudget(free=lambda: 1_000, reserve=8_000_000_000, wait_seconds=0)
    client = serving(b"")
    state, _ = zenodo.collect_record(client, record, tmp_path, disk=full)
    client.close()

    assert state.n_deferred == 1
    assert state.n_failed == 0
    assert state.reconciles()
    assert state.needs_retry


def test_deposits_deferred_during_a_run_are_retried_before_it_ends(tmp_path):
    """Disk frees continuously as workers extract and delete.

    A deposit refused at minute three often fits at minute ten, so leaving it
    for a later run wastes a day. All 279 would have been caught by one more
    pass.
    """
    records = [
        zenodo.parse_record(
            {
                "id": 700 + i,
                "metadata": {
                    "doi": f"10.5281/zenodo.{700 + i}",
                    "title": "t",
                    "publication_date": "2024-01-01",
                    "communities": [],
                },
                "files": [
                    {
                        "key": "a.zip",
                        "size": 10,
                        "checksum": "md5:x",
                        "links": {"self": "https://zenodo.org/f/a.zip"},
                    }
                ],
            }
        )
        for i in range(3)
    ]

    from softverse.acquire.state import DatasetRecord

    seen: list[str] = []

    def defer_once(client, record, files_root, archive_cap=None, disk=None):
        seen.append(record.doi)
        if seen.count(record.doi) == 1:
            return DatasetRecord(
                dataset_doi=record.doi, state="partial", n_candidate=1, n_deferred=1
            ), []
        return DatasetRecord(
            dataset_doi=record.doi, state="complete", n_candidate=1, n_fetched=1
        ), []

    import pytest as _pytest

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(zenodo, "collect_record", defer_once)
    ledger = Ledger(tmp_path / "ledger.jsonl")
    client = serving(b"")
    zenodo.collect(records, tmp_path, ledger, client, workers=2)
    client.close()
    monkeypatch.undo()

    states = {r.dataset_doi: r.state for r in ledger._records.values()}
    assert all(s == "complete" for s in states.values()), states

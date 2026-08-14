"""A targeted re-run must not destroy the rest of the frame.

The frame is collected in one pass over ~75 collections, and any one of them
can come back empty. Retrying just that alias is the obvious thing to do, and
the first version of the script would have written a one-collection CSV over
the other seventy-four. It survived only because the retry happened and then
nobody looked at the file.
"""

from __future__ import annotations

import csv

import scripts_collect_dataverse_frame as frame


def _rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(collection, identifier):
    return {
        "collection_id": collection,
        "source": "dataverse",
        "dataset_id": "1",
        "persistent_id": f"https://doi.org/10.7910/{identifier}",
        "protocol": "doi",
        "authority": "10.7910",
        "identifier": identifier,
        "publication_date": "2024-01-01",
    }


def test_a_rerun_for_one_alias_keeps_the_others(tmp_path):
    out = tmp_path / "deposits.csv"
    frame.write_frame(
        [_row("ajps", "DVN/A"), _row("jop", "DVN/B")], out, collected=["ajps", "jop"]
    )
    frame.write_frame([_row("restat", "DVN/C")], out, collected=["restat"])

    rows = _rows(out)
    assert {r["collection_id"] for r in rows} == {"ajps", "jop", "restat"}
    assert len(rows) == 3


def test_a_rerun_replaces_that_collection_rather_than_appending(tmp_path):
    """Otherwise a re-run doubles the collection it was meant to refresh."""
    out = tmp_path / "deposits.csv"
    frame.write_frame([_row("ajps", "DVN/A"), _row("ajps", "DVN/B")], out, ["ajps"])
    frame.write_frame([_row("ajps", "DVN/A")], out, ["ajps"])

    rows = _rows(out)
    assert len(rows) == 1, "the stale second deposit must be gone, not duplicated"
    assert rows[0]["identifier"] == "DVN/A"


def test_a_collection_that_returned_nothing_does_not_erase_itself(tmp_path):
    """An empty result is a failure, not a statement that the journal is empty.

    Wiping the previous rows on a failed retry would turn a transient problem
    into data loss, which is the same trade v1 made when it wrote success
    before doing the work.
    """
    out = tmp_path / "deposits.csv"
    frame.write_frame([_row("restat", "DVN/C")], out, ["restat"])
    frame.write_frame([], out, collected=[])

    assert len(_rows(out)) == 1, "a failed run must leave what was already there"


# -- the search fallback --------------------------------------------------

import json  # noqa: E402

from softverse.acquire.http import FetchOutcome  # noqa: E402
from softverse.sources.dataverse import search_collection  # noqa: E402


class FakeSearch:
    """Serves paginated /api/search responses and records what was asked."""

    def __init__(self, total: int, per_page: int = 100, fail_after: int | None = None):
        self.total = total
        self.per_page = per_page
        self.fail_after = fail_after
        self.starts: list[int] = []

    def get(self, url: str, **_kw) -> FetchOutcome:
        start = int(url.split("start=")[1].split("&")[0])
        self.starts.append(start)
        if self.fail_after is not None and len(self.starts) > self.fail_after:
            return FetchOutcome(ok=False, error="no payload after 90s")
        items = [
            {
                "global_id": f"doi:10.7910/DVN/D{i}",
                "published_at": "2020-01-02T00:00:00Z",
                "name": f"dataset {i}",
                "entity_id": i,
            }
            for i in range(start, min(start + self.per_page, self.total))
        ]
        body = {"status": "OK", "data": {"total_count": self.total, "items": items}}
        return FetchOutcome(ok=True, content=json.dumps(body).encode())


def test_search_pages_through_a_collection_too_large_to_list():
    """`/contents` returns the whole collection in one response and the
    server gives up on the biggest one: `jop` came back with 466 KB in 87 s,
    `restat` with nothing in 73 s -- failing *faster*, so not a timeout.
    Paged search is the documented way, and it is lighter on their server
    than one enormous listing.
    """
    client = FakeSearch(total=1638)
    rows = search_collection(client, "restat")
    assert len(rows) == 1638
    assert client.starts == list(range(0, 1638, 100)), "each page fetched once"


def test_search_rows_carry_the_same_identifiers_as_a_contents_listing():
    """Otherwise the two paths produce rows that cannot be merged."""
    (row,) = search_collection(FakeSearch(total=1), "restat")
    assert row["protocol"] == "doi"
    assert row["authority"] == "10.7910"
    assert row["identifier"] == "DVN/D0"
    assert row["publicationDate"] == "2020-01-02"


def test_a_failed_page_stops_the_walk_rather_than_returning_a_partial():
    """A short read here would silently shrink a journal in the frame.

    Returning what we have looks like success and is indistinguishable from
    a genuinely smaller collection, which is the whole class of error this
    project exists to avoid.
    """
    client = FakeSearch(total=1638, fail_after=3)
    assert search_collection(client, "restat") == []


def test_a_collection_that_no_longer_exists_is_gone_not_failed():
    """`regionalstatistics` was in the 2024 scrape with 23 deposits and
    Harvard now answers "Can't find dataverse with identifier".

    That is a different fact from a transient failure and needs a different
    response: retrying it will never work, and counting it beside collections
    we simply could not reach would overstate the coverage gap. Six
    collections failed on one run and five recovered on retry; only this one
    is actually gone.
    """
    from softverse.sources.dataverse import collection_missing

    for body in (
        b'{"status":"ERROR","message":"Can\'t find dataverse with identifier=\'x\'"}',
        b'{"status":"ERROR","message":"Could not find dataverse with alias x"}',
    ):
        assert collection_missing(FetchOutcome(ok=True, content=body))


def test_a_transient_failure_is_not_mistaken_for_a_missing_collection():
    from softverse.sources.dataverse import collection_missing

    assert not collection_missing(FetchOutcome(ok=False, error="no payload after 90s"))
    assert not collection_missing(
        FetchOutcome(ok=True, content=b'{"status":"OK","data":[]}')
    )
    # An error about something other than a missing dataverse is still a
    # failure; only the specific one is terminal.
    assert not collection_missing(
        FetchOutcome(ok=True, content=b'{"status":"ERROR","message":"server busy"}')
    )

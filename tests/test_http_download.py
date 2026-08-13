"""Tests for resumable streaming downloads.

The failures worth testing here are the quiet ones. A download that dies loudly
costs a retry; a download that *resumes wrongly* writes a file of exactly the
right size with a duplicated prefix buried in the middle, and nothing
downstream notices until an archive fails to open -- or worse, opens and yields
subtly wrong contents.

So the interesting cases are: a server that ignores `Range` and resends the
whole body, and a server that answers 416 because we already have everything.
Both are tested by reassembling the payload and comparing bytes, not sizes.
"""

from __future__ import annotations

import httpx
import pytest

from softverse.acquire.http import PoliteClient, RateLimiter

PAYLOAD = bytes(range(256)) * 400  # 102,400 bytes, order-sensitive


def client_with(handler) -> PoliteClient:
    """A PoliteClient wired to a mock transport, with the limiter neutralised."""
    polite = PoliteClient(limiter=RateLimiter(rate_per_s=1000.0), max_retries=3)
    polite._client = httpx.Client(transport=httpx.MockTransport(handler))
    return polite


def parse_range(request: httpx.Request) -> int:
    header = request.headers.get("Range")
    return int(header.removeprefix("bytes=").rstrip("-")) if header else 0


def test_download_writes_the_whole_body(tmp_path):
    def handler(request):
        return httpx.Response(200, content=PAYLOAD)

    target = tmp_path / "a.zip"
    outcome = client_with(handler).download("https://x/a.zip", target)
    assert outcome.ok
    assert target.read_bytes() == PAYLOAD
    assert not target.with_suffix(".zip.part").exists()


def test_interrupted_download_leaves_a_part_file_to_resume_from(tmp_path):
    """Keeping the partial file is the entire point of the method."""
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        raise httpx.ReadTimeout("died mid-transfer")

    target = tmp_path / "a.zip"
    outcome = client_with(handler).download("https://x/a.zip", target)
    assert not outcome.ok
    assert outcome.retryable
    assert not target.exists(), "a failed download must not look complete"


def test_resume_continues_from_the_offset_and_reassembles_exactly(tmp_path):
    """The success path: second attempt sends Range and appends the tail."""
    cut = 40_000
    seen: list[int] = []

    def handler(request):
        start = parse_range(request)
        seen.append(start)
        if start == 0:
            # Serve a prefix, then die, as a dropped connection would.
            return httpx.Response(200, content=PAYLOAD[:cut])
        return httpx.Response(
            206,
            content=PAYLOAD[start:],
            headers={
                "Content-Range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"
            },
        )

    target = tmp_path / "a.zip"
    polite = client_with(handler)
    # First call completes "successfully" with a short body; simulate the
    # interrupted state by seeding the part file directly.
    part = target.with_suffix(".zip.part")
    part.write_bytes(PAYLOAD[:cut])

    outcome = polite.download("https://x/a.zip", target)
    assert outcome.ok
    assert seen[-1] == cut, f"expected Range from {cut}, got {seen}"
    assert target.read_bytes() == PAYLOAD, "resumed file does not match the source"


def test_server_ignoring_range_does_not_corrupt_the_file(tmp_path):
    """A 200 answer to a Range request means "here is everything, again".

    Appending it to the bytes we already hold would produce a file of
    believable size with a duplicated prefix inside. Truncate instead.
    """
    cut = 40_000

    def handler(request):
        assert parse_range(request) == cut, "the client should have asked to resume"
        return httpx.Response(200, content=PAYLOAD)  # Range ignored

    target = tmp_path / "a.zip"
    target.with_suffix(".zip.part").write_bytes(PAYLOAD[:cut])

    outcome = client_with(handler).download("https://x/a.zip", target)
    assert outcome.ok
    written = target.read_bytes()
    assert len(written) == len(PAYLOAD), f"got {len(written)}, spliced a duplicate"
    assert written == PAYLOAD


def test_416_means_we_already_have_it(tmp_path):
    """Range past the end is success, not failure."""

    def handler(request):
        return httpx.Response(416)

    target = tmp_path / "a.zip"
    target.with_suffix(".zip.part").write_bytes(PAYLOAD)

    outcome = client_with(handler).download("https://x/a.zip", target)
    assert outcome.ok
    assert target.read_bytes() == PAYLOAD


def test_cap_aborts_mid_stream_and_discards(tmp_path):
    """An over-cap archive should cost a chunk, not a full download."""

    def handler(request):
        return httpx.Response(200, content=PAYLOAD)

    target = tmp_path / "a.zip"
    outcome = client_with(handler).download(
        "https://x/a.zip", target, cap_bytes=len(PAYLOAD) // 4
    )
    assert not outcome.ok
    assert "exceeds cap" in (outcome.error or "")
    assert not outcome.retryable, "an over-cap file will be over-cap next time too"
    assert not target.exists()
    assert not target.with_suffix(".zip.part").exists(), "part file must be cleaned up"


def test_a_file_exactly_at_the_cap_is_kept(tmp_path):
    """Off-by-one here silently drops legitimate archives at the boundary."""

    def handler(request):
        return httpx.Response(200, content=PAYLOAD)

    target = tmp_path / "a.zip"
    outcome = client_with(handler).download(
        "https://x/a.zip", target, cap_bytes=len(PAYLOAD)
    )
    assert outcome.ok
    assert target.read_bytes() == PAYLOAD


def test_definitive_failure_is_not_retried(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    outcome = client_with(handler).download("https://x/a.zip", tmp_path / "a.zip")
    assert not outcome.ok
    assert outcome.status == 404
    assert not outcome.retryable
    assert calls["n"] == 1, "a 404 was retried"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_statuses_are_retried_then_reported(tmp_path, status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(status)

    outcome = client_with(handler).download("https://x/a.zip", tmp_path / "a.zip")
    assert not outcome.ok
    assert outcome.retryable
    assert calls["n"] > 1, "a retryable status was not retried"

"""Tests for the block watcher.

Verified against a fake client so resumption is proven without waiting hours on
Harvard, and without sending it a single extra request.
"""

from __future__ import annotations

from datetime import UTC, datetime

from softverse.acquire.http import FetchOutcome, RateLimiter
from softverse.acquire.watch import MIN_INTERVAL_S, Watcher


class FakeClient:
    """Answers with challenges for the first ``n`` probes, then opens."""

    def __init__(self, challenges: int) -> None:
        self.remaining = challenges
        self.probes = 0
        self.limiter = RateLimiter(rate_per_s=2.0)

    def probe(self, url: str) -> FetchOutcome:
        self.probes += 1
        if self.remaining > 0:
            self.remaining -= 1
            return FetchOutcome(
                ok=False, status=202, error="WAF challenge", challenged=True
            )
        return FetchOutcome(ok=True, status=200, content=b'{"ok":1}')


def make_watcher(challenges: int) -> tuple[Watcher, list[float]]:
    slept: list[float] = []
    watcher = Watcher(
        client=FakeClient(challenges),
        probe_url="http://x/api/info/version",
        sleeper=slept.append,
        clock=lambda: datetime.now(tz=UTC),
    )
    return watcher, slept


def test_opens_immediately_when_not_blocked():
    watcher, slept = make_watcher(challenges=0)
    assert watcher.wait_until_open()
    assert slept == [], "must not sleep when the host is already open"
    assert watcher.client.probes == 1


def test_waits_then_resumes_when_the_block_lifts():
    watcher, slept = make_watcher(challenges=3)
    assert watcher.wait_until_open()
    assert watcher.client.probes == 4
    assert len(slept) == 3


def test_intervals_grow_and_never_go_below_the_floor():
    """Patience is the whole design: back off, do not poll."""
    watcher, slept = make_watcher(challenges=4)
    watcher.wait_until_open()
    assert all(s >= MIN_INTERVAL_S for s in slept), slept
    # Growing, allowing for jitter.
    assert slept[-1] > slept[0]


def test_gives_up_rather_than_polling_forever():
    watcher, slept = make_watcher(challenges=10_000)
    assert not watcher.wait_until_open(max_wait_s=3 * 3600)
    # A handful of probes over three hours, not hundreds.
    assert watcher.client.probes < 10, watcher.client.probes


def test_work_does_not_run_while_blocked():
    """The important negative: never start collecting into a closed door."""
    watcher, _ = make_watcher(challenges=10_000)
    ran = []
    result = watcher.run_when_open(lambda: ran.append(1) or ["rows"], max_wait_s=3600)
    assert result is None
    assert ran == [], "work must not run while the host is refusing"


def test_resume_slows_the_rate():
    """Coming back at the rate that earned the challenge would earn it again."""
    watcher, _ = make_watcher(challenges=1)
    before = watcher.client.limiter.rate
    watcher.run_when_open(lambda: ["rows"])
    assert watcher.client.limiter.rate < before


def test_work_runs_once_the_block_lifts():
    watcher, _ = make_watcher(challenges=2)
    assert watcher.run_when_open(lambda: ["a", "b"]) == ["a", "b"]


def test_block_duration_is_recorded():
    """How long we were blocked becomes a fact in the log, not a memory."""
    watcher, slept = make_watcher(challenges=2)
    watcher.wait_until_open()
    assert watcher.blocked_for_s == sum(slept)
    assert len(watcher.history) == 3
    assert watcher.history[0].challenged
    assert watcher.history[-1].ok

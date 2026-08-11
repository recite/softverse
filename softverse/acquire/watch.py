"""Wait out a block, then resume collection -- patiently.

Harvard Dataverse is serving this client an AWS WAF bot challenge on every
endpoint. The challenge wants a JS-solved ``aws-waf-token`` cookie; producing
one would circumvent an access control, and their API Terms of Use ask users not
to impair the service. So the only acceptable response is to knock less often.

The loop is deliberately slow and deliberately loud:

- probe **one** cheap endpoint, unretried, on a long jittered interval that
  grows from 30 minutes to 6 hours;
- every probe is logged with its WAF header, so "how long were we blocked" is a
  recorded fact rather than someone's recollection;
- on recovery, resume at a quarter of the previous rate and re-probe before each
  batch, stopping the moment the challenge returns.

A run that gives up is not a failure state -- it is the correct outcome of
being told no, and the ledger means nothing is lost by stopping.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from softverse.acquire.http import FetchOutcome, PoliteClient, jittered
from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Never probe more often than this, whatever the backoff says.
MIN_INTERVAL_S = 15 * 60
INITIAL_INTERVAL_S = 30 * 60
MAX_INTERVAL_S = 6 * 60 * 60

#: Rate to resume at once a block lifts: a quarter of the 2/s that earned the
#: challenge. Slow is the point.
RESUME_RATE_PER_S = 0.5


@dataclass
class ProbeResult:
    """One check of whether the door is still shut."""

    at: datetime
    ok: bool
    status: int | None
    challenged: bool
    waited_s: float

    def as_log(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "challenged": self.challenged,
            "waited_s": round(self.waited_s, 1),
        }


class Watcher:
    """Probe until a host lets us back in, then hand control to a callback."""

    def __init__(
        self,
        client: PoliteClient,
        probe_url: str,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        self.client = client
        self.probe_url = probe_url
        # Injected so the loop is testable without waiting hours.
        self._sleep = sleeper
        self._now = clock
        self.history: list[ProbeResult] = []

    def probe_once(self, waited_s: float = 0.0) -> ProbeResult:
        outcome: FetchOutcome = self.client.probe(self.probe_url)
        result = ProbeResult(
            at=self._now(),
            ok=outcome.ok,
            status=outcome.status,
            challenged=outcome.challenged,
            waited_s=waited_s,
        )
        self.history.append(result)
        logger.info("probe", extra=result.as_log())
        return result

    def wait_until_open(self, max_wait_s: float = 48 * 3600) -> bool:
        """Probe on a growing interval until the host answers properly.

        Returns True if it opened, False if ``max_wait_s`` elapsed first.
        """
        interval = INITIAL_INTERVAL_S
        waited = 0.0

        first = self.probe_once(0.0)
        if first.ok:
            logger.info("host is open; no wait needed")
            return True

        while waited < max_wait_s:
            delay = max(MIN_INTERVAL_S, jittered(interval))
            logger.info(
                "blocked; sleeping before next probe",
                extra={"sleep_s": round(delay), "waited_total_s": round(waited)},
            )
            self._sleep(delay)
            waited += delay

            if self.probe_once(waited).ok:
                logger.info(
                    "block lifted",
                    extra={"after_s": round(waited), "probes": len(self.history)},
                )
                return True
            interval = min(MAX_INTERVAL_S, interval * 2)

        logger.warning(
            "still blocked after max wait",
            extra={"waited_s": round(waited), "probes": len(self.history)},
        )
        return False

    def run_when_open(
        self,
        work: Callable[[], Sequence],
        max_wait_s: float = 48 * 3600,
    ) -> Sequence | None:
        """Wait for the host, then run ``work`` once. ``None`` if it never opened."""
        if not self.wait_until_open(max_wait_s):
            return None
        self.client.limiter.penalize("resuming after a block")
        logger.info(
            "resuming collection",
            extra={"rate_per_s": round(self.client.limiter.rate, 3)},
        )
        return work()

    @property
    def blocked_for_s(self) -> float:
        """How long the block lasted, from the log rather than from memory."""
        if not self.history:
            return 0.0
        opened = next((p for p in self.history if p.ok), None)
        return opened.waited_s if opened else self.history[-1].waited_s

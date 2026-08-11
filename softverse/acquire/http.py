"""Polite HTTP: a global rate limiter and a retry policy that knows what to retry.

Harvard Dataverse publishes no rate-limit headers and its configured tier limits
are not public, so the ceiling is unknown. The response to an unknown ceiling is
to stay well under it and to back off *harder* on the first sign of pushback,
not to probe for the edge.

Two v1 behaviours this replaces:

- ``time.sleep(5.0)`` after every single download, unconditional, inside the
  worker. With ~88,000 files that is five days of pure sleeping, which is why
  the run never finished.
- Retrying on *any* ``RequestException``, including 404, and backing off
  ``retry_delay * 2**attempt`` on 403 -- 310 s burned per permanently forbidden
  file. A 404 and a 403 are answers. Only 429 and 5xx are worth repeating.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import httpx

from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Statuses worth repeating. 404 (gone) and 403 (not yours) are answers.
#:
#: 202 is here because of measured Harvard Dataverse behaviour, not the spec.
#: When it decides a client is going too fast it replies **202 Accepted with an
#: empty text/html body** rather than 429 -- no Retry-After, no error. A client
#: that treats 2xx as success therefore records a dataset with zero files and
#: calls it done, which is precisely the shape of the v1 disaster (8,953
#: deposits marked "success", holding nothing). Treat it as throttling.
RETRY_STATUSES = frozenset({202, 429, 500, 502, 503, 504})

#: A 2xx whose body is empty is never a real answer from this API.
_EMPTY_IS_THROTTLE = frozenset({200, 202})


class RateLimiter:
    """Thread-safe token bucket, shared across all workers.

    A per-worker delay is not a rate limit: four workers each sleeping 0.5 s
    still issue 8 req/s. The bucket is global so the *process* obeys the limit
    however many threads are running.

    The rate can only ever go down (:meth:`penalize`). A server that says 429
    once will say it again, and the polite response to being asked to slow down
    is to stay slow for the rest of the run rather than creep back up.
    """

    def __init__(self, rate_per_s: float = 2.0, burst: int = 2) -> None:
        self._rate = rate_per_s
        self._initial_rate = rate_per_s
        self._burst = max(1, burst)
        self._tokens = float(self._burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._burst, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = (1.0 - self._tokens) / self._rate
            # Jitter so concurrent workers do not resynchronize into a burst.
            time.sleep(deficit + random.uniform(0, 0.05))

    def penalize(self, reason: str) -> None:
        """Halve the sustained rate, permanently for this run."""
        with self._lock:
            self._rate = max(0.2, self._rate / 2)
            new = self._rate
        logger.warning(
            "rate limit reduced",
            extra={"new_rate_per_s": round(new, 3), "reason": reason},
        )


@dataclass
class FetchOutcome:
    """What happened, distinguishably.

    ``ok`` false with ``retryable`` false means a definitive negative -- gone,
    restricted, or refused. v1 collapsed those into the same zero as a timeout,
    which is why a restricted file and a network failure were indistinguishable
    in its outputs.
    """

    ok: bool
    status: int | None = None
    content: bytes | None = None
    error: str | None = None
    retryable: bool = False

    @property
    def forbidden(self) -> bool:
        return self.status == 403

    @property
    def not_found(self) -> bool:
        return self.status == 404


class PoliteClient:
    """An httpx client wrapped in the limiter and the retry policy."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        user_agent: str = "softverse/2.0 (research; github.com/recite/softverse)",
    ) -> None:
        self.limiter = limiter or RateLimiter()
        self.max_retries = max_retries
        merged = {"User-Agent": user_agent, **(headers or {})}
        self._client = httpx.Client(
            timeout=timeout, follow_redirects=True, headers=merged
        )

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, **kwargs) -> FetchOutcome:
        """GET with rate limiting and bounded retries."""
        last: FetchOutcome = FetchOutcome(ok=False, error="no attempt made")
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                response = self._client.get(url, **kwargs)
            except httpx.HTTPError as exc:
                last = FetchOutcome(
                    ok=False, error=f"{type(exc).__name__}: {exc}", retryable=True
                )
            else:
                status = response.status_code
                if status == 200 and response.content:
                    return FetchOutcome(ok=True, status=200, content=response.content)
                if status in _EMPTY_IS_THROTTLE and not response.content:
                    # Harvard's undocumented throttle: empty 200/202. Slow down
                    # for the rest of the run and wait properly before retrying.
                    self.limiter.penalize(f"empty {status} (throttle)")
                    last = FetchOutcome(
                        ok=False,
                        status=status,
                        error=f"empty {status} body (throttled)",
                        retryable=True,
                    )
                    if attempt < self.max_retries:
                        delay = min(120.0, 15.0 * (attempt + 1)) + random.uniform(0, 5)
                        logger.warning(
                            "throttled, backing off",
                            extra={"status": status, "sleep_s": round(delay, 1)},
                        )
                        time.sleep(delay)
                    continue
                if status not in RETRY_STATUSES:
                    # A definitive answer. Do not retry it 5 times.
                    return FetchOutcome(
                        ok=False,
                        status=status,
                        error=f"HTTP {status}",
                        retryable=False,
                    )
                last = FetchOutcome(
                    ok=False, status=status, error=f"HTTP {status}", retryable=True
                )
                if status == 429:
                    self.limiter.penalize("HTTP 429")
                    if retry_after := response.headers.get("Retry-After"):
                        self._wait_retry_after(retry_after)
                        continue

            if attempt < self.max_retries:
                delay = min(60.0, 2.0**attempt) + random.uniform(0, 1.0)
                logger.debug(
                    "retrying",
                    extra={
                        "url": url[:120],
                        "attempt": attempt + 1,
                        "sleep_s": round(delay, 1),
                    },
                )
                time.sleep(delay)
        return last

    @staticmethod
    def _wait_retry_after(value: str) -> None:
        """Honour ``Retry-After``. v1 never read this header at all."""
        try:
            seconds = float(value)
        except ValueError:
            return
        seconds = min(seconds, 300.0)
        logger.warning("honouring Retry-After", extra={"sleep_s": seconds})
        time.sleep(seconds)

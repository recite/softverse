"""Polite HTTP: maintained libraries for the solved parts, ours for the rest.

Retry policy and rate limiting are solved problems, so they come from libraries
rather than from forty lines of hand-rolled backoff:

- **stamina** (v26.1.0) for retries -- exponential backoff with jitter and a
  wall-clock cap, built on tenacity.
- **pyrate-limiter** (v4.4.0) for rate limiting -- a leaky bucket that enforces
  *several rates at once*, which matters because Dataverse's documented limits
  are per-hour and per-tier while politeness is naturally expressed per-second.

What is *not* delegated, because no library knows about it:

**Harvard Dataverse answers an unwelcome client with `202 Accepted`, an empty
body, `content-type: text/html`, and the header `x-amzn-waf-action: challenge`.**
That is an AWS WAF bot challenge, not a rate limit and not an error. Every HTTP
library on earth treats 202 as success, so a client built on defaults records
the dataset as having zero files and marks it complete -- which is precisely the
shape of the v1 disaster (8,953 deposits marked "success", holding nothing),
arrived at by a different route. Detecting it is ours, it is tested, and it must
not regress.

We do not attempt to satisfy the challenge. It expects a JS-solved
`aws-waf-token` cookie; producing one would be circumventing an access control,
and Harvard's API Terms of Use ask users not to impair the service. The correct
response to a closed door is to knock more slowly, which is what
:mod:`softverse.acquire.watch` does.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass

import httpx
import stamina
from pyrate_limiter import Duration, Limiter, Rate

from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Statuses worth repeating. 404 (gone) and 403 (not yours) are answers, not
#: failures -- v1 retried 404s and burned 310 s per permanently forbidden file.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: A 2xx with an empty body is never a real answer from these APIs.
_EMPTY_IS_THROTTLE = frozenset({200, 202})

#: AWS WAF names itself in the response. When present, this is definitive.
WAF_HEADER = "x-amzn-waf-action"


class Throttled(Exception):
    """The server is refusing politely: a WAF challenge or an empty 2xx."""


class RateLimiter:
    """Global limiter over pyrate-limiter, with a one-way penalty.

    Wrapping rather than using the library bare buys two things it does not
    provide: a *global* instance shared across workers (a per-worker delay is
    not a rate limit -- four workers each sleeping 0.5 s still issue 8 req/s),
    and :meth:`penalize`, which permanently lowers the rate for the rest of the
    run. A server that pushed back once will push back again, and creeping the
    rate back up is how you get blocked twice.
    """

    def __init__(self, rate_per_s: float = 2.0, per_hour: int | None = None) -> None:
        self._rate = rate_per_s
        self._per_hour = per_hour
        self._lock = threading.Lock()
        self._limiter = self._build(rate_per_s)

    def _build(self, rate_per_s: float) -> Limiter:
        """Always ``1 request per interval``, never ``N per second``.

        This distinction matters and is easy to get wrong. ``Rate(20, SECOND)``
        permits a *burst* of twenty requests in the first millisecond and then
        silence -- the average is right, the behaviour is not. Against a host
        that has already served us an AWS WAF challenge, twenty simultaneous
        requests is exactly the pattern that triggered it. Expressing the same
        average as one request per 50 ms forces even spacing.

        A test asserts the spacing rather than the average, because the average
        would have passed either way.
        """
        interval_ms = max(1, int(1000 / rate_per_s))
        rates = [Rate(1, interval_ms)]
        if self._per_hour:
            rates.append(Rate(self._per_hour, Duration.HOUR))
        return Limiter(rates)

    @property
    def rate(self) -> float:
        return self._rate

    def acquire(self, name: str = "global") -> None:
        """Block until the bucket allows another request."""
        # blocking=True makes try_acquire wait rather than return False, which
        # is what turns the bucket into a rate limiter rather than a gate.
        self._limiter.try_acquire(name, blocking=True)

    def penalize(self, reason: str) -> None:
        """Halve the sustained rate, permanently for this run."""
        with self._lock:
            self._rate = max(0.1, self._rate / 2)
            self._limiter = self._build(self._rate)
            new = self._rate
        logger.warning(
            "rate limit reduced",
            extra={"new_rate_per_s": round(new, 3), "reason": reason},
        )


@dataclass
class FetchOutcome:
    """What happened, distinguishably.

    ``ok`` false with ``retryable`` false is a definitive negative -- gone,
    restricted, refused. v1 collapsed those into the same zero as a timeout.
    """

    ok: bool
    status: int | None = None
    content: bytes | None = None
    error: str | None = None
    retryable: bool = False
    challenged: bool = False

    @property
    def forbidden(self) -> bool:
        return self.status == 403

    @property
    def not_found(self) -> bool:
        return self.status == 404


def is_waf_challenge(response: httpx.Response) -> bool:
    """Whether this response is an AWS WAF bot challenge.

    Two signals, either sufficient: the explicit header, or the shape Harvard
    actually returns (an empty 200/202 body from a JSON API).
    """
    if response.headers.get(WAF_HEADER, "").lower() == "challenge":
        return True
    return response.status_code in _EMPTY_IS_THROTTLE and not response.content


class PoliteClient:
    """httpx client with a shared limiter, stamina retries, and WAF detection."""

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
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent, **(headers or {})},
        )

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(self, url: str, **kwargs) -> FetchOutcome:
        """GET with rate limiting, retries, and challenge detection."""
        last = FetchOutcome(ok=False, error="no attempt made")

        try:
            for attempt in stamina.retry_context(
                on=(httpx.HTTPError, Throttled),
                attempts=self.max_retries + 1,
                wait_initial=2.0,
                wait_max=120.0,
                wait_jitter=5.0,
            ):
                with attempt:
                    self.limiter.acquire()
                    response = self._client.get(url, **kwargs)

                    if is_waf_challenge(response):
                        self.limiter.penalize("WAF challenge / empty 2xx")
                        last = FetchOutcome(
                            ok=False,
                            status=response.status_code,
                            error="WAF challenge (empty body)",
                            retryable=True,
                            challenged=True,
                        )
                        raise Throttled(last.error)

                    if response.status_code == 200:
                        return FetchOutcome(
                            ok=True, status=200, content=response.content
                        )

                    if response.status_code in RETRY_STATUSES:
                        if response.status_code == 429:
                            self.limiter.penalize("HTTP 429")
                        last = FetchOutcome(
                            ok=False,
                            status=response.status_code,
                            error=f"HTTP {response.status_code}",
                            retryable=True,
                        )
                        raise Throttled(last.error)

                    # A definitive answer. Returning inside the retry context
                    # exits without another attempt, which is the point.
                    return FetchOutcome(
                        ok=False,
                        status=response.status_code,
                        error=f"HTTP {response.status_code}",
                        retryable=False,
                    )
        except Throttled:
            return last
        except httpx.HTTPError as exc:
            return FetchOutcome(
                ok=False, error=f"{type(exc).__name__}: {exc}", retryable=True
            )
        return last

    def probe(self, url: str) -> FetchOutcome:
        """A single unretried request, for checking whether a block has lifted."""
        self.limiter.acquire()
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            return FetchOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
        if is_waf_challenge(response):
            return FetchOutcome(
                ok=False,
                status=response.status_code,
                error="WAF challenge",
                challenged=True,
                retryable=True,
            )
        return FetchOutcome(
            ok=response.status_code == 200 and bool(response.content),
            status=response.status_code,
            content=response.content,
        )


def jittered(seconds: float, spread: float = 0.2) -> float:
    """A delay with +/- ``spread`` jitter, so retries do not synchronize."""
    return seconds * (1 + random.uniform(-spread, spread))

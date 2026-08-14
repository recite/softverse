"""Fetch JSON from a host that answers non-browser clients with a challenge.

Harvard Dataverse puts an AWS WAF in front of every route, its homepage
included. The `challenge` action serves JavaScript; the client executes it,
receives an `aws-waf-token` cookie and retries. There is no CAPTCHA and no
human step -- it is a capability test, and the capability it tests for is
running JavaScript.

`curl` cannot, so it gets HTTP 202 with an empty body regardless of what it
sends. Measured, all identical: no auth, `X-Dataverse-key` header, `?key=`
query parameter, any user-agent or none. The WAF sits at the load balancer
*ahead of* Dataverse, so it never sees the token -- which is why an entirely
valid token appeared not to work. Through a browser the same URL with the
same token returns `{"status":"OK","identifier":"@soodoku"}`.

So this module is not a scraper. It is a way to run the challenge in front of
the same JSON API everything else uses, and it returns a
:class:`~softverse.acquire.http.FetchOutcome` so callers cannot tell the
difference. `walk_collection` takes it unchanged.

**On whether this is allowed.** Harvard's API Terms of Use are specific about
what they prohibit, and automated access is not on the list. The clauses that
bind are: do not "conceal or otherwise misrepresent your identity or your
application's identity", do not "use an unreasonable amount of bandwidth",
and do not "impair the functionality, stability, or operation" of their
servers. This module is built to satisfy all three rather than to argue
around them:

- Every request carries **our** user-agent, not Chrome's. A browser's default
  UA here would misrepresent the application's identity, which is precisely
  what the first clause forbids. Verified against the live API: with
  `softverse/2.0 (...)` it still passes the challenge.
- Every request carries the API token, so it is attributable to an account.
- Requests are rate-limited, and the frame is ~75 of them.

An earlier version of this project recorded that solving the challenge "would
be circumventing an access control" and cited these terms for it. The terms
do not say that; it was an inference, and it is corrected here rather than
left as words in Harvard's mouth.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

from softverse.acquire.http import FetchOutcome, RateLimiter
from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Identifies the application, per the terms. Traceable to a repository and a
#: person, so an operator who wants to find out who we are, can.
USER_AGENT = "softverse/2.0 (research; github.com/recite/softverse)"

#: How long the page gets before Chrome dumps whatever it has.
#:
#: Chosen from a measurement rather than a guess, after the first guess cost
#: a collection: at 25,000 the frame run averaged 23.3 s per collection, so
#: every one of them was at the ceiling and the largest, `restat`, fell off
#: it. Its 1,324 deposits in the 2024 scrape make it roughly a third bigger
#: than `jop`, which cleared 25 s at 1,219.
DEFAULT_BUDGET_MS = 120_000

#: What the single retry multiplies the budget by when nothing came back.
_RETRY_FACTOR = 4

#: Where Chrome lives on macOS. `shutil.which` first, so a PATH install or a
#: Linux CI box works without a special case.
_CHROME_CANDIDATES = (
    "google-chrome",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

#: Chrome renders a bare JSON document inside a `<pre>`. Everything else in
#: the DOM is the shell the browser built around it.
_PRE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.S | re.I)
_JSON = re.compile(r"[\{\[].*[\}\]]", re.S)


def find_browser() -> str | None:
    for candidate in _CHROME_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).is_file():
            return candidate
    return None


def extract_payload(dom: str) -> bytes | None:
    """The JSON body out of Chrome's rendered DOM, or None if there is none.

    None is the challenge case, and it must stay distinguishable from an
    empty result: every HTTP library treats the challenge's 202 as success,
    so a client on defaults records a deposit as having zero files and marks
    it complete. That is the exact shape of the v1 disaster -- 8,953 deposits
    marked successful holding nothing -- reached by a different route.
    """
    if not dom:
        return None
    match = _PRE.search(dom)
    text = match.group(1) if match else dom
    body = _JSON.search(html.unescape(re.sub(r"<[^>]+>", "", text)))
    return body.group(0).encode("utf-8") if body else None


class BrowserClient:
    """A `.get(url)` that runs the challenge, shaped like `PoliteClient`."""

    def __init__(
        self,
        *,
        token: str | None = None,
        user_agent: str = USER_AGENT,
        limiter: RateLimiter | None = None,
        budget_ms: int = DEFAULT_BUDGET_MS,
        timeout_s: int = 600,
        runner=None,
    ) -> None:
        self._token = token
        self._user_agent = user_agent
        self._limiter = limiter
        self._budget_ms = budget_ms
        self._timeout_s = timeout_s
        self._runner = runner or self._run_chrome

    def _run_chrome(self, argv: list[str]) -> str:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=self._timeout_s
        )
        return result.stdout

    def get(self, url: str, **_kwargs) -> FetchOutcome:
        """Fetch once, and once more with real headroom if nothing came back.

        The retry is not politeness, it is measurement. The frame run
        averaged 23.3 s per collection against a 25 s budget, so every
        collection was at the ceiling and the largest one, `restat`, sat just
        beyond it. One attempt with several times the budget separates a slow
        response from a broken one, which is a distinction the first version
        of this could not make and guessed at instead.
        """
        outcome = self._attempt(url, self._budget_ms)
        if outcome.ok or outcome.error is None or "no payload" not in outcome.error:
            return outcome
        logger.info(
            "empty response; retrying with a longer budget",
            extra={"url": url, "budget_ms": self._budget_ms * _RETRY_FACTOR},
        )
        return self._attempt(url, self._budget_ms * _RETRY_FACTOR)

    def _attempt(self, url: str, budget_ms: int) -> FetchOutcome:
        if self._limiter is not None:
            self._limiter.acquire()
        # Chrome takes no header flags, so the token rides as `?key=`, which
        # Dataverse documents as equivalent to `X-Dataverse-key`.
        target = url
        if self._token:
            target += ("&" if "?" in url else "?") + f"key={self._token}"

        browser = find_browser()
        argv = [
            browser or "google-chrome",
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--user-agent={self._user_agent}",
            f"--virtual-time-budget={budget_ms}",
            "--dump-dom",
            target,
        ]
        try:
            dom = self._runner(argv)
        except Exception as exc:  # noqa: BLE001 - a missing browser is a result
            # The token is in `target`; keep it out of the message.
            return FetchOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")

        payload = extract_payload(dom)
        if payload is None:
            # Say what was seen, not what caused it. `--dump-dom` gives no
            # headers, so a challenge, a timeout, a redirect to HTML and a
            # dead network all arrive here identically. An earlier version
            # called every one of them `challenged`, which is how a 25-second
            # timeout of ours turned into a sentence in a draft email telling
            # Harvard they had a bug.
            logger.warning("empty response", extra={"url": url, "budget_ms": budget_ms})
            return FetchOutcome(
                ok=False,
                retryable=True,
                error=f"no payload after {budget_ms / 1000:.0f}s",
            )
        return FetchOutcome(ok=True, status=200, content=payload)

    def close(self) -> None:
        """Nothing to close; present so callers can treat it like a client."""

"""Tests for the browser-backed fetcher.

Harvard Dataverse puts an AWS WAF in front of every route, including its
homepage. The `challenge` action serves JavaScript, the client runs it,
receives an `aws-waf-token` cookie and retries: a capability test, not a
CAPTCHA, with no human in it. `curl` cannot run JavaScript and gets HTTP 202
with an empty body no matter what headers or credentials it carries. A
browser runs it and gets the same JSON the API was always going to return.

The identity requirement is the part worth testing. Harvard's API Terms of
Use prohibit attempting "to conceal or otherwise misrepresent your identity
or your application's identity" -- so the browser must carry *our*
user-agent, not Chrome's, and our token. Verified against the live API: with
`softverse/2.0 (...)` as the user-agent it still passes the challenge and
returns `identified as @soodoku`, which is what compliance looks like here.

No test in this file launches a browser; the runner is injected.
"""

from __future__ import annotations

import pytest

from softverse.acquire.browser import BrowserClient, extract_payload

#: What Chrome's --dump-dom gives back for a JSON endpoint: the body wrapped
#: in the shell a browser builds around a bare `text/plain` document.
JSON_DOM = (
    '<html><head><meta name="color-scheme" content="light dark"></head>'
    '<body><pre>{"status":"OK","data":{"id":1}}</pre></body></html>'
)

#: The challenge, as it arrives when the JS has not run: 202 and nothing.
EMPTY_DOM = "<html><head></head><body></body></html>"


def test_json_is_recovered_from_the_dom_wrapper():
    assert extract_payload(JSON_DOM) == b'{"status":"OK","data":{"id":1}}'


def test_html_entities_in_the_payload_are_undone():
    """A URL in the JSON arrives as `&amp;` once the browser has parsed it."""
    dom = '<body><pre>{"url":"a?x=1&amp;y=2"}</pre></body>'
    assert extract_payload(dom) == b'{"url":"a?x=1&y=2"}'


def test_a_page_with_no_json_yields_nothing():
    assert extract_payload(EMPTY_DOM) is None
    assert extract_payload("") is None


def test_a_successful_fetch_looks_like_any_other_fetch():
    """It returns a `FetchOutcome`, so `walk_collection` cannot tell the
    difference and needs no change to use it."""
    client = BrowserClient(runner=lambda argv: JSON_DOM)
    outcome = client.get("https://dataverse.harvard.edu/api/x")
    assert outcome.ok
    assert outcome.content == b'{"status":"OK","data":{"id":1}}'


def test_an_empty_page_is_never_success():
    """The failure this whole module exists to avoid.

    Every HTTP library treats 202 as success, so a client on defaults records
    the deposit as having zero files and marks it complete -- which is the
    shape of the v1 disaster, 8,953 deposits marked successful holding
    nothing, arrived at by a different route.
    """
    client = BrowserClient(runner=lambda argv: EMPTY_DOM)
    outcome = client.get("https://dataverse.harvard.edu/api/x")
    assert not outcome.ok
    assert outcome.retryable, "an empty page is worth retrying; it is not a 404"


def test_an_empty_page_is_not_diagnosed_as_a_challenge():
    """It says what was seen, not what was inferred.

    `--dump-dom` returns no headers, so a WAF challenge, a timeout, a
    redirect to HTML and a dead network are indistinguishable here. Calling
    all of them `challenged` is how a 25-second timeout of ours became a
    sentence in a draft email telling Harvard they had a bug. The collection
    was `restat`, the frame run averaged 23.3s per collection against the
    budget, and nothing was wrong on their side.
    """
    outcome = BrowserClient(runner=lambda argv: EMPTY_DOM).get("https://x/api/y")
    assert not outcome.challenged, "do not assert a cause we cannot observe"
    assert "no payload" in (outcome.error or "")


def test_an_empty_page_is_retried_once_with_more_time():
    """A slow collection should not be reported as a failure.

    `jop` cleared the budget at 1,219 deposits and `restat` did not at
    ~1,600. One retry with real headroom separates slow from broken.
    """
    budgets: list[int] = []

    def runner(argv):
        budget = next(
            int(a.split("=")[1]) for a in argv if a.startswith("--virtual-time-budget")
        )
        budgets.append(budget)
        return JSON_DOM if len(budgets) > 1 else EMPTY_DOM

    outcome = BrowserClient(runner=runner).get("https://x/api/y")
    assert outcome.ok, "the retry should have rescued it"
    assert len(budgets) == 2, "exactly one retry, not a loop"
    assert budgets[1] > budgets[0], "the retry must actually get more time"


def test_a_persistently_empty_page_stops_after_one_retry():
    calls: list[int] = []

    def runner(argv):
        calls.append(1)
        return EMPTY_DOM

    outcome = BrowserClient(runner=runner).get("https://x/api/y")
    assert not outcome.ok
    assert len(calls) == 2, "one attempt plus one retry, then stop"


def test_our_identity_is_on_every_request():
    """Harvard's API terms prohibit misrepresenting the application's
    identity, and Chrome's default user-agent would do exactly that."""
    seen: list[list[str]] = []
    client = BrowserClient(runner=lambda argv: (seen.append(argv), JSON_DOM)[1])
    client.get("https://dataverse.harvard.edu/api/x")

    argv = " ".join(seen[0])
    assert "--user-agent=softverse/" in argv
    assert "github.com/recite/softverse" in argv, "the UA must be traceable to us"


def test_the_token_is_sent_and_never_logged():
    """Authenticated requests are attributable, which is the point.

    Chrome takes no header flags, so the token rides as `?key=`. It must
    reach the URL and must not reach the logs.
    """
    seen: list[list[str]] = []
    client = BrowserClient(
        token="secret-token", runner=lambda argv: (seen.append(argv), JSON_DOM)[1]
    )
    outcome = client.get("https://dataverse.harvard.edu/api/x")
    assert "key=secret-token" in " ".join(seen[0])
    assert "secret-token" not in (outcome.error or "")


def test_a_token_is_optional():
    client = BrowserClient(runner=lambda argv: JSON_DOM)
    seen: list[str] = []
    client._runner = lambda argv: (seen.append(" ".join(argv)), JSON_DOM)[1]
    client.get("https://dataverse.harvard.edu/api/x")
    assert "key=" not in seen[0]


def test_a_runner_that_raises_is_an_error_not_a_crash():
    def boom(argv):
        raise OSError("chrome not found")

    outcome = BrowserClient(runner=boom).get("https://dataverse.harvard.edu/api/x")
    assert not outcome.ok
    assert "chrome not found" in (outcome.error or "")


def test_requests_are_rate_limited():
    """Trivial volume here, but the terms speak to bandwidth and impairment,
    and a limiter costs nothing when there are only seventy-five requests."""
    import time

    from softverse.acquire.http import RateLimiter

    client = BrowserClient(runner=lambda argv: JSON_DOM, limiter=RateLimiter(4.0))
    start = time.monotonic()
    for _ in range(3):
        client.get("https://dataverse.harvard.edu/api/x")
    assert time.monotonic() - start >= 0.4


@pytest.mark.parametrize(
    "dom",
    [
        '<body><pre>{"status":"ERROR","code":404,"message":"nope"}</pre></body>',
    ],
)
def test_an_api_error_is_returned_rather_than_swallowed(dom):
    """Dataverse answers 404 inside a 200 body; the caller decides."""
    outcome = BrowserClient(runner=lambda argv: dom).get("https://x/api/y")
    assert outcome.ok
    assert b'"status":"ERROR"' in (outcome.content or b"")

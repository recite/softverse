"""Tests for the StataCorp help-server oracle.

The interesting failure here is not a crash but a silent inversion: if the
"not found" detection stops working, every string ever passed to it is reported
as an official Stata command, the builtin list swallows real packages, and the
tally loses them with no error anywhere. So the negative case is tested harder
than the positive one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from softverse.stata.official import (
    OfficialSnapshot,
    is_official,
    load,
    verify,
)

#: Trimmed from the real responses. Both arrive as HTTP 200 with the same
#: title, which is the whole reason the body has to be inspected.
REAL_PAGE = b"""<html><head><title>StataNow 19 help for ds</title></head>
<body><h2>Title</h2><p>ds -- Compactly list variables with specified properties</p>
<pre>ds [varlist] [, options]</pre></body></html>"""

STUB_PAGE = b"""<html><head><title>StataNow 19 help for notarealcmd</title></head>
<body><p>help for notarealcmd not found</p>
<p>try <b>help contents</b> or <b>search notarealcmd</b></p></body></html>"""


@dataclass
class FakeOutcome:
    ok: bool
    content: bytes | None = None


class FakeClient:
    """Serves canned pages and records what was asked."""

    def __init__(self, pages: dict[str, bytes], fail: set[str] | None = None) -> None:
        self.pages = pages
        self.fail = fail or set()
        self.asked: list[str] = []

    def get(self, url: str, **_kwargs) -> FakeOutcome:
        name = url.split("?", 1)[1]
        self.asked.append(name)
        if name in self.fail:
            return FakeOutcome(ok=False)
        body = self.pages.get(name, STUB_PAGE.replace(b"notarealcmd", name.encode()))
        return FakeOutcome(ok=True, content=body)

    def close(self) -> None:
        pass


def test_real_help_page_means_official():
    client = FakeClient({"ds": REAL_PAGE})
    assert is_official(client, "ds") is True


def test_not_found_stub_means_not_official():
    """The status code is 200 either way, so this is the only real signal."""
    client = FakeClient({})
    assert is_official(client, "notarealcmd") is False


def test_user_written_packages_are_not_official():
    """`reghdfe` and `grc1leg` are real commands but not StataCorp's.

    This is the distinction the builtin list exists to make. Getting it wrong
    for `reghdfe` would delete the third most-used Stata package in the corpus.
    """
    client = FakeClient({})
    for package_command in ("reghdfe", "grc1leg", "esttab", "outreg2"):
        assert is_official(client, package_command) is False, package_command


def test_network_failure_is_unknown_not_negative():
    """A timeout must not be recorded as "not an official command"."""
    client = FakeClient({}, fail={"ds"})
    assert is_official(client, "ds") is None


def test_unaskable_names_are_not_sent():
    """Macro fragments and sentinels are not commands; do not ask about them."""
    client = FakeClient({})
    for junk in ("`cmd'", "<dynamic>", "", "3", "a b", "\\multicolumn"):
        assert is_official(client, junk) is None, junk
    assert client.asked == []


def test_verify_caches_and_does_not_refetch(tmp_path):
    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE, "pca": REAL_PAGE})

    first = verify({"ds", "pca", "reghdfe"}, path, client=client)
    assert first.official == {"ds", "pca"}
    assert first.not_official == {"reghdfe"}
    assert sorted(client.asked) == ["ds", "pca", "reghdfe"]

    client.asked.clear()
    second = verify({"ds", "pca", "reghdfe"}, path, client=client)
    assert client.asked == [], "cached names were re-fetched"
    assert second.official == first.official


def test_verify_only_fetches_new_names(tmp_path):
    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE, "lasso": REAL_PAGE})
    verify({"ds"}, path, client=client)
    client.asked.clear()
    snapshot = verify({"ds", "lasso"}, path, client=client)
    assert client.asked == ["lasso"]
    assert snapshot.official == {"ds", "lasso"}


def test_unresolved_names_are_retried_not_cached(tmp_path):
    """A failure must leave no trace, or an outage hardens into a wrong answer."""
    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE}, fail={"ds"})
    verify({"ds"}, path, client=client)
    assert json.loads(path.read_text())["commands"] == {}

    client.fail.clear()
    assert verify({"ds"}, path, client=client).official == {"ds"}


def test_snapshot_round_trips(tmp_path):
    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE})
    verify({"ds", "reghdfe"}, path, client=client)
    loaded = load(path)
    assert loaded is not None
    assert "ds" in loaded
    assert "reghdfe" not in loaded
    assert loaded.snapshot_date


def test_load_missing_snapshot_is_none(tmp_path):
    assert load(tmp_path / "absent.json") is None


def test_names_are_matched_case_insensitively():
    snapshot = OfficialSnapshot(checked={"ds": True}, snapshot_date="2026-08-13")
    assert "DS" in snapshot
    assert "Ds" in snapshot


@pytest.mark.parametrize("name", ["ds", "pca", "regress", "summarize"])
def test_stub_detection_survives_name_substitution(name):
    """The stub echoes the queried name, so the pattern must not hardcode one."""
    client = FakeClient({})
    assert is_official(client, name) is False

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
<body><h2>Title</h2><p><b>[D] ds</b> -- Compactly list variables with properties</p>
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


def test_crawl_checkpoints_so_an_interruption_is_not_total(tmp_path, monkeypatch):
    """A twenty-minute crawl must not lose everything if it dies at minute 19.

    The cache exists so a rerun does not re-crawl someone else's server.
    Writing only at the end would defeat that in exactly the case it matters.
    """
    import softverse.stata.official as official

    monkeypatch.setattr(official, "CHECKPOINT_EVERY", 2)
    path = tmp_path / "official.json"

    class DyingClient(FakeClient):
        def get(self, url, **kwargs):
            if len(self.asked) >= 5:
                raise KeyboardInterrupt("crawl killed mid-run")
            return super().get(url, **kwargs)

    client = DyingClient({"aaa": REAL_PAGE, "bbb": REAL_PAGE, "ccc": REAL_PAGE})
    with pytest.raises(KeyboardInterrupt):
        official.verify({"aaa", "bbb", "ccc", "ddd", "eee", "fff"}, path, client=client)

    survived = json.loads(path.read_text())["commands"]
    assert len(survived) >= 4, f"only {len(survived)} answers survived"
    assert survived["aaa"] is True


def test_checkpoint_is_atomic(tmp_path):
    """A half-written cache would be parsed as authoritative on the next run."""
    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE})
    verify({"ds"}, path, client=client)
    assert json.loads(path.read_text())["commands"] == {"ds": True}
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"


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


def test_verified_snapshot_widens_the_curated_builtin_list(tmp_path):
    """The payoff: commands nobody curated stop being reported as unresolved.

    `ds` is an official Stata command absent from the curated list, so it
    resolved to `unknown` in 77 deposits and inflated the coverage statistic
    the paper reports.
    """
    from softverse.stata.builtins import builtins, curated

    assert "ds" not in curated(), "fixture is stale; pick a command still missing"

    path = tmp_path / "official.json"
    client = FakeClient({"ds": REAL_PAGE, "pca": REAL_PAGE})
    verify({"ds", "pca", "reghdfe"}, path, client=client)

    widened = builtins(verified_snapshot=path)
    assert "ds" in widened
    assert "pca" in widened
    assert "reghdfe" not in widened, "an SSC package must never become a builtin"
    assert "regress" in widened, "curated entries must survive the merge"
    assert "su" in widened, "curated abbreviations must survive the merge"


def test_verified_snapshot_is_ignored_when_absent(tmp_path):
    from softverse.stata.builtins import builtins, curated

    assert builtins(verified_snapshot=tmp_path / "nope.json").forms == curated().forms


#: The real shapes: a command reference, a User's Guide concept, a function.
COMMAND_PAGE = b"""<html><title>StataNow 19 help for ds</title><body>
<h2>Title</h2><p><b>[D] ds</b> -- Compactly list variables</p></body></html>"""

SYSTEM_VARIABLE_PAGE = b"""<html><title>StataNow 19 help for n</title><body>
<h2>Title</h2><p><b>[U] 13.4 System variables (_variables)</b></p>
<p>Expressions may also contain _variables.</p></body></html>"""

FUNCTION_PAGE = b"""<html><title>StataNow 19 help for e</title><body>
<p><b>[FN] Programming functions</b></p><p>e(name)</p></body></html>"""


def test_documented_is_not_the_same_as_being_a_command():
    """The bug this guard exists for.

    help.cgi answers for anything in the manuals. `n` and `b` return the
    User's Guide page for system variables `_n` and `_b`; `e` returns the
    programming-functions page. Treating "a page exists" as "a command
    exists" made all three official, and `n` alone is 750 corpus mentions.
    """
    client = FakeClient(
        {"ds": COMMAND_PAGE, "n": SYSTEM_VARIABLE_PAGE, "e": FUNCTION_PAGE}
    )
    assert is_official(client, "ds") is True
    assert is_official(client, "n") is False, "[U] documents concepts, not commands"
    assert is_official(client, "e") is False, "[FN] documents functions"


def test_a_dispatch_page_with_no_manual_code_is_still_a_command():
    """Requiring a `[R] name` title looked safer and was wrong.

    `npregress` and `irtgraph` are dispatch pages listing their subcommands,
    and `ivreg` is headed "Out-of-date command". All three are official Stata
    and none carries a manual code, so demanding one demoted seven real
    commands. Across the twenty names the rule changed, every page lacking a
    code was a command and every non-command carried `[U]` or `[FN]`.
    """
    pages = {
        "npregress": b"<html><body><pre>Command       Description</pre></body></html>",
        "irtgraph": b"<html><body><p>Graphs after irt commands</p></body></html>",
        "ivreg": b"<html><body><p>Out-of-date command</p></body></html>",
    }
    client = FakeClient(pages)
    for name in pages:
        assert is_official(client, name) is True, name


@pytest.mark.parametrize(
    "manual,expected",
    [
        ("R", True),
        ("D", True),
        ("MV", True),
        ("P", True),
        ("TS", True),
        ("U", False),
        ("FN", False),
    ],
)
def test_command_manuals_are_accepted_and_concept_manuals_are_not(manual, expected):
    page = f"<html><body><p><b>[{manual}] thing</b> -- desc</p></body></html>"
    client = FakeClient({"thing": page.encode()})
    assert is_official(client, "thing") is expected


def test_abbreviations_resolving_to_their_full_entry_still_count():
    """`l` is a legal abbreviation of `list`; its page is titled `[D] list`.

    Requiring the title to echo the queried name would reject every
    abbreviation, so the check is on the manual, not on the name.
    """
    page = (
        b"<html><body><p><b>[D] list</b> -- List values of variables</p></body></html>"
    )
    assert is_official(FakeClient({"l": page}), "l") is True

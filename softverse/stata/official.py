"""Ask StataCorp whether a command is official.

The curated builtin list in :mod:`softverse.stata.builtins` says of itself that
the authoritative source is a Stata installation::

    . local base : dir "`c(sysdir_base)'" files "*.ado"

which requires a licence we do not have, so the list stayed hand-written and its
incompleteness stayed a stated limitation. StataCorp's public help server closes
that gap without a licence: ``help.cgi?<name>`` serves the real help page for an
official command and a short "not found" stub for anything else.

The discriminator is the stub's text, not the status code and not the response
size. Every request returns HTTP 200 with the title ``help for <name>``,
including for names that do not exist, so a status-code check would mark every
string in the corpus official. Size separates the two cleanly today (~1.7 KB
versus 14-52 KB) but is a coincidence of styling that a site redesign would
break silently, in the direction of false positives. The stub says::

    help for notarealcmd not found

and that sentence is what we match.

**This is an oracle for "official", not for "exists".** It correctly answers no
for ``grc1leg`` and ``reghdfe``, which are real commands from user-written
packages -- exactly the distinction the builtin list needs.

Answers are cached to a snapshot so resolution does not depend on when it ran,
and so a rebuild does not re-crawl someone else's server.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient, RateLimiter

HELP_URL = "https://www.stata.com/help.cgi?{name}"

#: One request every two seconds. This is a courtesy crawl of a vendor's
#: documentation server for a few hundred names, not a bulk harvest.
DEFAULT_RATE_PER_S = 0.5

#: Persist every this many answers, so an interrupted crawl resumes rather than
#: restarting. At the rate above this is a checkpoint roughly every minute.
CHECKPOINT_EVERY = 25

#: The stub served for a name Stata does not know. Matched as a sentence
#: because the status code is always 200 and the body size is only incidentally
#: different -- see the module docstring.
_NOT_FOUND = re.compile(r"help for\s+\S+\s+not found", re.I)

#: Stata help pages open with the manual they come from: `[R] regress`,
#: `[D] list`, `[MV] pca`. The bracketed code says what *kind* of thing is
#: documented, which matters because the server answers for anything in the
#: manuals, not only for commands.
_MANUAL = re.compile(r"\[([A-Z][A-Z0-9-]*)\]\s")

#: Manuals that document concepts and functions rather than commands.
#:
#: Without this the oracle reports `n`, `b` and `e` as official commands: the
#: first two resolve to "[U] 13.4 System variables (_variables)" -- the help
#: for `_n` and `_b`, which are system variables -- and `e` to "[FN]
#: Programming functions", the help for `e()`. `n` alone accounts for 750
#: corpus mentions, so treating it as a command both deflates the unresolved
#: rate on a false basis and, where a package ships a same-named command,
#: quietly outvotes it.
#:
#: Only these two are excluded, and deliberately: every other manual (`[R]`,
#: `[D]`, `[P]`, `[MV]`, `[TS]`, ...) documents commands, and excluding one on
#: suspicion would cost real recall.
NOT_COMMAND_MANUALS = frozenset({"U", "FN"})

#: Names that cannot be asked about safely. A query string is not a command,
#: and sending these would either error or match something unrelated.
_UNASKABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


@dataclass(frozen=True)
class OfficialSnapshot:
    """Cached verdicts from StataCorp's help server."""

    checked: dict[str, bool]
    snapshot_date: str
    source: str = HELP_URL

    def __contains__(self, name: str) -> bool:
        return self.checked.get(name.lower(), False)

    @property
    def official(self) -> set[str]:
        return {name for name, ok in self.checked.items() if ok}

    @property
    def not_official(self) -> set[str]:
        return {name for name, ok in self.checked.items() if not ok}


def is_official(client: PoliteClient, name: str) -> bool | None:
    """True if StataCorp documents ``name`` as a *command*.

    Two things have to be true, and checking only the first is the mistake
    this function used to make. The name must be documented at all -- a
    missing name gets the "not found" stub -- and the manual it is documented
    in must be one that describes commands. The help server answers for system
    variables and functions too, so "the page exists" alone reports `n`, `b`
    and `e` as commands.

    Returns ``None`` when the question could not be answered -- a network
    failure or an unaskable name -- so that "we do not know" stays distinct
    from "not official". Collapsing the two would quietly turn every outage
    into a claim about Stata's command set.
    """
    if not _UNASKABLE.match(name):
        return None
    outcome = client.get(HELP_URL.format(name=name))
    if not outcome.ok or outcome.content is None:
        return None
    body = outcome.content.decode("utf-8", errors="replace")
    if _NOT_FOUND.search(body):
        return False
    manual = manual_of(body)
    # A page with no manual reference at all is not a command reference. Under
    # inclusive resolution the cost of being wrong here is recall, not
    # precision: the name falls through to the SSC index unchanged.
    return manual is not None and manual not in NOT_COMMAND_MANUALS


def manual_of(body: str) -> str | None:
    """The manual code a help page comes from -- `R`, `D`, `MV` -- or None."""
    text = re.sub(r"<[^>]*>", "\n", body)
    match = _MANUAL.search(text)
    return match.group(1) if match else None


def _write(path: Path, checked: dict[str, bool]) -> None:
    """Persist the cache. Called during the crawl, not only at the end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": HELP_URL,
        "method": (
            "GET help.cgi?<name>; a body matching 'help for <name> not found' "
            "means the name is not official Stata. Status is always 200, so it "
            "carries no information."
        ),
        "snapshot_date": datetime.now(tz=UTC).date().isoformat(),
        "n_checked": len(checked),
        "n_official": sum(checked.values()),
        "commands": dict(sorted(checked.items())),
    }
    # Write-then-rename, so an interrupted checkpoint cannot leave behind a
    # truncated file that the next run would parse as an authoritative cache.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def verify(
    names: set[str],
    path: Path,
    client: PoliteClient | None = None,
    refresh: bool = False,
) -> OfficialSnapshot:
    """Resolve ``names`` against the help server, reusing the cache on ``path``.

    Only names absent from the cache are fetched, so this is cheap to re-run as
    the corpus grows. Names that could not be resolved are left out of the
    cache rather than recorded as negative, so a transient failure is retried
    next time instead of hardening into a wrong answer.
    """
    cached: dict[str, bool] = {}
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())["commands"]

    todo = sorted({n.lower() for n in names} - set(cached))
    if todo:
        owned = client is None
        session = client or PoliteClient(
            limiter=RateLimiter(rate_per_s=DEFAULT_RATE_PER_S),
            user_agent="softverse/2.0 (research; github.com/recite/softverse)",
        )
        try:
            for done, name in enumerate(todo, 1):
                verdict = is_official(session, name)
                if verdict is not None:
                    cached[name] = verdict
                # Checkpoint, because at one request every two seconds a few
                # hundred names is a twenty-minute crawl. Writing only at the
                # end would make an interruption at minute nineteen cost the
                # whole run and re-crawl a vendor's server to recover it --
                # which is exactly what the cache exists to prevent.
                if done % CHECKPOINT_EVERY == 0:
                    _write(path, cached)
        finally:
            if owned:
                session.close()
            _write(path, cached)

    return OfficialSnapshot(
        checked=cached, snapshot_date=datetime.now(tz=UTC).date().isoformat()
    )


def load(path: Path) -> OfficialSnapshot | None:
    """Read a snapshot written by :func:`verify`, or None if absent."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return OfficialSnapshot(
        checked=payload["commands"], snapshot_date=payload["snapshot_date"]
    )

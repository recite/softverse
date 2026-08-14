"""Verify every bibliography entry against the DOI it claims.

A fabricated or subtly wrong citation is the one error a reader will catch and
never forgive, and unlike most things in a paper it is entirely machine
checkable. So it is checked, in CI, and the check fails the build.

For each entry this resolves the DOI against Crossref (falling back to DataCite,
which is where dataset and software DOIs live), compares the registered title to
the one in the .bib after normalization, and reports:

- a DOI that does not resolve -- the citation does not exist as written
- a title that does not match -- the DOI points somewhere else, which is worse
  than a dead link because it looks right
- an entry with neither DOI nor resolvable URL -- unverifiable, so it must be
  justified rather than assumed

Output goes to `paper/bib_validation.json`, recording what was checked and when,
so the claim "every reference was verified" is itself evidence rather than an
assertion.
"""

from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

CROSSREF = "https://api.crossref.org/works/{doi}"
DATACITE = "https://api.datacite.org/dois/{doi}"

#: Titles differ across registries in punctuation, case, LaTeX braces and
#: accents far more often than in substance, so comparison is on a normalized
#: form. Anything stricter produces false alarms that train you to ignore it.
_STRIP = re.compile(r"[^a-z0-9 ]+")
_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.S)
_FIELD = re.compile(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*,?\s*$", re.M | re.S)


#: LaTeX accent commands: \`e \'e \^e \"e \~n \=a \.z and \c{c} \v{s} \u{a}.
#: Removing the backslash alone leaves the accent character behind, which then
#: becomes a space and splits the word -- {\`e} turned "Mercè" into "merc e".
_LATEX_ACCENT = re.compile(r"\\[`'^\"~=.]|\\[cvuHtdbr](?=\{)")


def normalize(title: str) -> str:
    """Compare titles modulo formatting rather than substance.

    Registries differ from .bib files in punctuation, case, LaTeX braces and
    accent encoding far more often than in content. A stricter comparison
    produces false alarms, and a check that cries wolf is a check people learn
    to ignore -- which is worse than no check.
    """
    text = _LATEX_ACCENT.sub("", title)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(_STRIP.sub(" ", text.lower()).split())


@dataclass
class Entry:
    key: str
    kind: str
    title: str | None
    doi: str | None
    url: str | None


@dataclass
class Result:
    key: str
    ok: bool
    reason: str
    doi: str | None = None
    registered_title: str | None = None


def parse_bib(text: str) -> list[Entry]:
    """A deliberately small .bib reader.

    Full BibTeX parsing is a rabbit hole and the fields needed here are three.
    A malformed entry shows up as a missing DOI and gets reported, which is the
    right outcome anyway.
    """
    entries: list[Entry] = []
    for kind, key, body in _ENTRY.findall(text):
        fields = {k.lower(): " ".join(v.split()) for k, v in _FIELD.findall(body)}
        doi = fields.get("doi")
        if not doi and (url := fields.get("url", "")):
            if match := re.search(r"(10\.\d{4,9}/\S+)", url):
                doi = match.group(1)
        entries.append(
            Entry(
                key=key.strip(),
                kind=kind.lower(),
                title=fields.get("title"),
                doi=doi.strip().rstrip(".,") if doi else None,
                url=fields.get("url"),
            )
        )
    return entries


def registered_title(client: httpx.Client, doi: str) -> str | None:
    """Title as the DOI registrar has it, or None if the DOI does not resolve."""
    try:
        response = client.get(CROSSREF.format(doi=doi))
        if response.status_code == 200:
            titles = response.json()["message"].get("title") or []
            if titles:
                return titles[0]
    except (httpx.HTTPError, KeyError, json.JSONDecodeError):
        pass
    # Datasets and software register with DataCite, not Crossref.
    try:
        response = client.get(DATACITE.format(doi=doi))
        if response.status_code == 200:
            return (
                response.json()["data"]["attributes"]
                .get("titles", [{}])[0]
                .get("title")
            )
    except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError):
        pass
    return None


def page_title(client: httpx.Client, url: str) -> str | None:
    """The `<title>` of a page, for sources that carry no DOI."""
    try:
        response = client.get(url)
        if response.status_code >= 400:
            return None
        match = re.search(r"<title[^>]*>(.*?)</title>", response.text, re.S | re.I)
        return html.unescape(match.group(1)).strip() if match else None
    except httpx.HTTPError:
        return None


def validate(entries: list[Entry]) -> list[Result]:
    results: list[Result] = []
    with httpx.Client(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "softverse (research; mailto:gsood07@gmail.com)"},
    ) as client:
        for entry in entries:
            if not entry.doi:
                # Not everything worth citing has a DOI. A blog post that
                # supplies an argument is still a source, and refusing it
                # outright would push it into an untraceable aside instead.
                # So it is held to the same standard by a different route:
                # the page must exist and must carry the title we claim for
                # it. That catches the two failures a reader would care
                # about, a dead link and a citation pointing somewhere else.
                if not entry.url:
                    results.append(
                        Result(entry.key, ok=False, reason="no DOI and no URL")
                    )
                    continue
                found = page_title(client, entry.url)
                if found is None:
                    results.append(
                        Result(entry.key, ok=False, reason="URL does not resolve")
                    )
                elif entry.title and normalize(entry.title) not in normalize(found):
                    results.append(
                        Result(
                            entry.key,
                            ok=False,
                            reason="page title does not match",
                            registered_title=found,
                        )
                    )
                else:
                    results.append(Result(entry.key, ok=True, reason="verified by URL"))
                continue
            found = registered_title(client, entry.doi)
            if found is None:
                results.append(
                    Result(
                        entry.key,
                        ok=False,
                        reason="DOI does not resolve",
                        doi=entry.doi,
                    )
                )
                continue
            if entry.title and normalize(entry.title) != normalize(found):
                results.append(
                    Result(
                        entry.key,
                        ok=False,
                        # Worse than a dead link: it resolves, so it looks fine.
                        reason="title does not match the DOI",
                        doi=entry.doi,
                        registered_title=found,
                    )
                )
                continue
            results.append(Result(entry.key, ok=True, reason="verified", doi=entry.doi))
    return results


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/references.bib")
    if not path.exists():
        print(f"no bibliography at {path}")
        return 0

    entries = parse_bib(path.read_text(encoding="utf-8"))
    results = validate(entries)
    failures = [r for r in results if not r.ok]

    report = {
        "checked_at": datetime.now(tz=UTC).isoformat(),
        "bibliography": str(path),
        "n_entries": len(entries),
        "n_verified": len(results) - len(failures),
        "results": [asdict(r) for r in results],
    }
    Path("paper/bib_validation.json").write_text(json.dumps(report, indent=2))

    print(f"{len(results) - len(failures)}/{len(entries)} entries verified")
    for failure in failures:
        print(f"  FAIL {failure.key}: {failure.reason}")
        if failure.registered_title:
            print(f"       DOI resolves to: {failure.registered_title[:80]}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

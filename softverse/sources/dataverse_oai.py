"""Harvest Dataverse dataset manifests over OAI-PMH.

Harvard's JSON API is behind an AWS WAF bot challenge (``x-amzn-waf-action:
challenge``, HTTP 202, empty body) that answers every endpoint including
``/robots.txt``. **``/oai`` is the exception** -- it returns 200 normally,
because OAI-PMH is the protocol the repository deliberately exposes for machine
harvesting. Using it is the sanctioned route, not a workaround.

What it gives us, verified against a dataset whose API response we captured
before the block: ``oai_ddi`` inlines every file as an ``<otherMat>`` element
carrying the access URL -- and therefore the file id -- with the filename in its
``<labl>``::

    <otherMat URI=".../api/access/datafile/7517531"><labl>01_preprocessing.R</labl>

That is the frame and manifest stages for all 13,775 datasets without touching
the blocked API, which yields a **candidate-file census**: how many .R/.do/.py
files exist per journal and year. It replaces a 60-dataset extrapolation with a
count and reduces the eventual download to pure fetch.

What it does not give us, stated rather than discovered later: DDI carries no
md5, no file size and no restricted flag. So md5 verification moves to fetch
time, and a restricted file is indistinguishable from an open one until then.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.acquire.state import atomic_write_bytes
from softverse.logging_setup import get_logger
from softverse.sources.dataverse import is_wanted, repo_id_of

logger = get_logger(__name__)

OAI_ENDPOINT = "https://dataverse.harvard.edu/oai"

#: `<otherMat URI="...datafile/123" ...>` ... `<labl>name.R</labl>`. Non-greedy
#: and DOTALL because the label often sits on the following line.
_OTHER_MAT = re.compile(
    r'<otherMat[^>]*URI="([^"]+)"[^>]*>.*?<labl>([^<]*)</labl>', re.S
)
_ERROR = re.compile(r'<error[^>]*code="([^"]+)"')
_DATESTAMP = re.compile(r"<datestamp>([^<]+)</datestamp>")
_SETSPEC = re.compile(r"<setSpec>([^<]+)</setSpec>")


@dataclass
class OAIManifest:
    """One dataset's file listing, as OAI-PMH describes it."""

    doi: str
    datestamp: str | None = None
    sets: list[str] = field(default_factory=list)
    #: ``(file_id, filename)`` for every file the record lists.
    files: list[tuple[int, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def year(self) -> int | None:
        if self.datestamp and self.datestamp[:4].isdigit():
            return int(self.datestamp[:4])
        return None

    @property
    def candidates(self) -> list[tuple[int, str]]:
        """Files worth downloading -- scripts, notebooks, manifests, archives."""
        return [(i, n) for i, n in self.files if is_wanted(n)]


def parse_ddi(xml: str, doi: str) -> OAIManifest:
    """Parse an ``oai_ddi`` GetRecord response into a manifest."""
    manifest = OAIManifest(doi=doi)
    if error := _ERROR.search(xml):
        manifest.error = error.group(1)
        return manifest
    if stamp := _DATESTAMP.search(xml):
        manifest.datestamp = stamp.group(1)
    manifest.sets = _SETSPEC.findall(xml)
    for uri, label in _OTHER_MAT.findall(xml):
        match = re.search(r"/datafile/(\d+)", uri)
        if not match:
            continue
        name = label.strip()
        if name:
            manifest.files.append((int(match.group(1)), name))
    return manifest


def fetch_manifest(
    client: PoliteClient, doi: str, raw_root: Path | None = None
) -> OAIManifest:
    """Fetch one dataset's manifest, archiving the raw XML if asked.

    The raw response is archived before parsing for the same reason the JSON
    path archives its own: a derived file must never be the only copy of a
    network response.
    """
    outcome = client.get(
        OAI_ENDPOINT,
        params={"verb": "GetRecord", "metadataPrefix": "oai_ddi", "identifier": doi},
    )
    if not outcome.ok or outcome.content is None:
        return OAIManifest(doi=doi, error=outcome.error or f"HTTP {outcome.status}")
    if raw_root is not None:
        atomic_write_bytes(raw_root / f"{repo_id_of(doi)}.xml", outcome.content)
    return parse_ddi(outcome.content.decode("utf-8", "replace"), doi)


def harvest(
    dois: list[str],
    client: PoliteClient,
    raw_root: Path | None = None,
    journal_of: dict[str, str] | None = None,
) -> list[dict]:
    """Harvest manifests for many datasets, returning candidate-file rows.

    Each row is one *candidate* file: the census, not a download record.
    """
    rows: list[dict] = []
    errors = 0
    for index, doi in enumerate(dois, 1):
        manifest = fetch_manifest(client, doi, raw_root)
        if manifest.error:
            errors += 1
            logger.debug("oai record error", extra={"doi": doi, "err": manifest.error})
            continue
        repo = repo_id_of(doi)
        journal = (journal_of or {}).get(repo)
        for file_id, name in manifest.candidates:
            rows.append(
                {
                    "dataset_doi": doi,
                    "repo_id": repo,
                    "collection_id": journal,
                    "file_id": file_id,
                    "filename": name,
                    "extension": Path(name).suffix.lower(),
                    "oai_datestamp": manifest.datestamp,
                }
            )
        if index % 250 == 0:
            logger.info(
                "oai harvest progress",
                extra={
                    "done": index,
                    "of": len(dois),
                    "rows": len(rows),
                    "errors": errors,
                },
            )
    logger.info(
        "oai harvest complete",
        extra={"datasets": len(dois), "candidate_files": len(rows), "errors": errors},
    )
    return rows

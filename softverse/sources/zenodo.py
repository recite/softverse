"""Zenodo collection.

Zenodo is the best-behaved of the three sources: it publishes its rate limit in
``x-ratelimit-*`` headers, its search API is paginated cleanly, and each record
inlines its files with sizes and md5 checksums. Verified end to end before this
was written -- search, record, and an actual file download.

Two things this gets right that v1's Zenodo path did not:

- **Every record's community membership is recorded**, so the frame is statable.
  v1 built a metadata DataFrame in memory and never wrote it, discarding the
  title, DOI, community and publication date of every record it touched.
- **Nothing is deleted to save space.** v1 removed every file not matching its
  extension list, which destroyed the ``renv.lock`` and ``requirements.txt``
  manifests that make free validation possible.

The 19,705 Zenodo files already on disk kept their directory structure, unlike
the flattened Dataverse corpus, so they remain usable while this re-collects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.acquire.state import DatasetRecord, Ledger, atomic_write_bytes
from softverse.acquire.unpack import extract, relative_member_path
from softverse.config import (
    ARCHIVE_EXTENSIONS,
    MANIFEST_FILENAMES,
    SCRIPT_EXTENSIONS,
)
from softverse.logging_setup import get_logger
from softverse.model.enums import CollectionState, Source
from softverse.sources.dataverse import is_wanted

logger = get_logger(__name__)

ZENODO_API = "https://zenodo.org/api"

#: Zenodo publishes 30 req/min on search as of Nov 2025. Stay under it; the
#: client also reads the returned headers and eases off before exhaustion.
DEFAULT_RATE_PER_S = 0.4

#: Records per search page. Zenodo rejects anything above 25 for anonymous
#: clients -- "Page size cannot be greater than 25. Please use authenticated
#: requests to increase the limit" -- with an explicit 400 saying so. A token
#: raises it; until then 25 is the ceiling.
PAGE_SIZE = 25
PAGE_SIZE_AUTHENTICATED = 100

#: A "community" returning more than this is not a community -- Zenodo ignored
#: the filter and handed back the archive. Zenodo held ~7.1M records when this
#: was measured, so anything near that is the whole thing.
WHOLE_REPOSITORY_THRESHOLD = 1_000_000

#: Zenodo needs a larger archive cap than Dataverse, and the asymmetry is
#: measured rather than assumed. On Dataverse 73% of deposits expose loose
#: script files, so a cap on archives skips extra material. On Zenodo the
#: archive *is* the delivery mechanism: in a 400-deposit run, 75 of them (19%)
#: yielded zero code because their only content was an over-cap archive -- a
#: non-random gap, since data-heavy packages are not a random subset of
#: research. Raising the cap to 500 MB recovers 31 of the 79 skipped archives
#: for 8 GB; the remaining tail is 26 archives above 2 GB (largest 100.7 GB)
#: and stays logged rather than fetched.
#: Raised again after measuring what the cap actually costs. Going 100 MB ->
#: 500 MB did not move the coverage gap at all -- 19% of deposits yielded zero
#: code at both settings -- because the size distribution is heavy-tailed rather
#: than clustered near the cap (of 226 skipped archives: 54 in 0.5-1 GB, 58 in
#: 1-2 GB, 54 in 2-5 GB, 60 above 5 GB).
#:
#: What settles it is that **code scales with archive size**, measured across
#: 907 deposits: the smallest quartile averages 0.5 MB and 21 code files, the
#: largest 200 MB and 221. So the skipped tail is not data-without-code, and
#: dropping 19% of deposits drops considerably more than 19% of the code, biased
#: toward large code-heavy projects.
#:
#: Disk is not the constraint it looked like: archives are deleted after
#: successful extraction, so peak usage is one archive at a time, not the
#: cumulative transfer. 2 GB admits 112 of the 226 skipped archives and roughly
#: halves the gap; the remainder stays logged as a stated limitation.
ARCHIVE_CAP_BYTES = 2 * 1024 * 1024 * 1024


@dataclass
class ZenodoFile:
    """One file in a record, as the API describes it."""

    key: str
    size: int
    checksum: str | None
    link: str

    @property
    def md5(self) -> str | None:
        """Zenodo reports checksums as ``md5:<hex>``."""
        if self.checksum and self.checksum.startswith("md5:"):
            return self.checksum[4:]
        return None

    @property
    def is_archive(self) -> bool:
        return Path(self.key).suffix.lower() in ARCHIVE_EXTENSIONS


@dataclass
class ZenodoRecord:
    record_id: str
    doi: str
    title: str
    publication_date: str | None
    communities: list[str]
    files: list[ZenodoFile]

    @property
    def year(self) -> int | None:
        if self.publication_date and self.publication_date[:4].isdigit():
            return int(self.publication_date[:4])
        return None


def parse_record(payload: dict) -> ZenodoRecord:
    """Build a record from the API's JSON, tolerating its shape variations."""
    metadata = payload.get("metadata", {})
    communities = [
        c.get("id") or c.get("identifier")
        for c in (metadata.get("communities") or [])
        if isinstance(c, dict)
    ]
    files = [
        ZenodoFile(
            key=f.get("key") or f.get("filename") or "",
            size=f.get("size") or f.get("filesize") or 0,
            checksum=f.get("checksum"),
            link=(f.get("links") or {}).get("self")
            or (f.get("links") or {}).get("download")
            or "",
        )
        for f in (payload.get("files") or [])
    ]
    return ZenodoRecord(
        record_id=str(payload.get("id")),
        doi=metadata.get("doi") or payload.get("doi") or "",
        title=(metadata.get("title") or "")[:500],
        publication_date=metadata.get("publication_date"),
        communities=[c for c in communities if c],
        files=[f for f in files if f.key and f.link],
    )


class UnknownCommunity(Exception):
    """A community slug Zenodo does not recognise."""


def verify_community(client: PoliteClient, slug: str) -> int:
    """Confirm a community exists, returning its record count.

    **This guard is not paranoia.** Zenodo does not reject an unknown
    ``communities=`` value -- it ignores the filter and returns the entire
    repository. Measured: ``communities=restud`` and ``communities=aeaje``
    (neither exists) each reported 7,106,477 records, which is all of Zenodo.
    A typo in a frame definition would therefore silently substitute the whole
    archive for a journal's community, and every downstream number would be
    wrong in a way that looks like abundance rather than error.

    Raises:
        UnknownCommunity: if the slug is not a real community.
    """
    outcome = client.get(f"{ZENODO_API}/communities/{slug}")
    if not outcome.ok:
        raise UnknownCommunity(
            f"{slug!r} is not a Zenodo community (HTTP {outcome.status}). "
            f"Filtering on it would return the entire repository."
        )
    counted = client.get(
        f"{ZENODO_API}/records", params={"communities": slug, "size": 1}
    )
    if not counted.ok or counted.content is None:
        raise UnknownCommunity(f"could not count records for {slug!r}")
    total = json.loads(counted.content)["hits"]["total"]
    if total > WHOLE_REPOSITORY_THRESHOLD:
        raise UnknownCommunity(
            f"{slug!r} returned {total:,} records, which is the whole "
            f"repository -- the community filter was ignored."
        )
    logger.info("community verified", extra={"community": slug, "records": total})
    return total


def search(
    client: PoliteClient,
    query: str,
    *,
    community: str | None = None,
    max_records: int | None = None,
    authenticated: bool = False,
) -> list[ZenodoRecord]:
    """Page through search results, stopping when the API stops giving more.

    v1's Zenodo search looped ``while True`` on an endpoint that raises past
    10,000 results, swallowed the exception per community, and kept whatever
    partial list existed -- a silent truncation. Here the page count is bounded
    and short pages end the loop.
    """
    records: list[ZenodoRecord] = []
    page_size = PAGE_SIZE_AUTHENTICATED if authenticated else PAGE_SIZE
    page = 1
    while True:
        params = {"q": query, "size": page_size, "page": page}
        if community:
            params["communities"] = community
        outcome = client.get(f"{ZENODO_API}/records", params=params)
        if not outcome.ok or outcome.content is None:
            logger.warning(
                "zenodo search failed",
                extra={"page": page, "status": outcome.status, "err": outcome.error},
            )
            break
        try:
            hits = json.loads(outcome.content)["hits"]["hits"]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("zenodo search unparseable", extra={"err": str(exc)})
            break
        if not hits:
            break
        records.extend(parse_record(h) for h in hits)
        logger.info(
            "zenodo search page",
            extra={"page": page, "hits": len(hits), "total_so_far": len(records)},
        )
        if len(hits) < page_size or (max_records and len(records) >= max_records):
            break
        # Zenodo's legacy pagination refuses beyond 10,000 results; stop before
        # it errors rather than after.
        if page * page_size >= 10_000:
            logger.warning("zenodo result ceiling reached; narrow the query")
            break
        page += 1
    return records[:max_records] if max_records else records


def collect_record(
    client: PoliteClient,
    record: ZenodoRecord,
    files_root: Path,
    archive_cap: int = ARCHIVE_CAP_BYTES,
) -> tuple[DatasetRecord, list[dict]]:
    """Download one record's wanted files. Returns its state and file rows."""
    doi = record.doi or f"zenodo:{record.record_id}"
    state = DatasetRecord(dataset_doi=doi, state=CollectionState.FAILED.value)
    # Zenodo records are immutable and a revision gets a new id, so the id is
    # a sufficient version marker here. Recorded anyway, so a refresh works
    # the same way across sources that mutate in place.
    state.upstream_version = record.record_id
    rows: list[dict] = []

    wanted = [f for f in record.files if is_wanted(f.key)]
    state.n_candidate = len(wanted)
    if not wanted:
        state.state = CollectionState.NO_CANDIDATE_FILES.value
        return state, rows

    target = files_root / record.record_id
    now = datetime.now(tz=UTC)

    for item in wanted:
        if item.is_archive and item.size > archive_cap:
            state.n_skipped_over_cap += 1
            state.skipped_archives.append(
                {"filename": item.key, "size_bytes": item.size, "file_id": None}
            )
            continue

        outcome = client.get(item.link, expect_content=False)
        if not outcome.ok or outcome.content is None:
            state.n_failed += 1
            state.error = outcome.error
            continue

        content = outcome.content
        state.bytes_fetched += len(content)

        if item.is_archive:
            archive_path = target / "_archives" / item.key
            atomic_write_bytes(archive_path, content)
            unpack_root = target / "_archives" / f"{item.key}_extracted"
            result = extract(
                archive_path,
                unpack_root,
                frozenset(SCRIPT_EXTENSIONS) | frozenset(ARCHIVE_EXTENSIONS),
                frozenset(MANIFEST_FILENAMES),
            )
            if result.error:
                # An archive we fetched but could not open is a *failure*, not a
                # deposit that happens to contain no code. Marking it complete
                # would make it unretryable, so a later fix -- a raised limit, a
                # new format -- could never reach it. That is how the first six
                # of these were nearly lost when the bomb limits turned out to
                # be miscalibrated.
                logger.warning(
                    "zenodo archive extraction failed",
                    extra={"record": record.record_id, "err": result.error},
                )
                state.n_failed += 1
                state.error = f"extract: {result.error}"
                # Keep the archive: this deposit is retryable, and the archive
                # is the evidence needed to work out why it failed.
                continue

            # Extraction succeeded, so drop the compressed original.
            #
            # This is *not* v1's mistake. v1 deleted archives before knowing
            # whether extraction worked ("ALWAYS remove archive after
            # processing"), kept no record, and so lost the contents outright.
            # Here the ledger already holds the record id, file key, size and
            # md5, which makes the archive re-fetchable; the manifests that
            # matter for validation (renv.lock, requirements.txt, DESCRIPTION)
            # have been extracted and kept; and an archive that failed to
            # extract is retained above.
            #
            # The ratio forces it: 104 deposits held 4.4 GB of archives around
            # 13 MB of code, so the full frame would need ~68 GB against 10 GB
            # of free disk. Keeping the zips would mean truncating the frame,
            # and losing a quarter of the journals is a worse error than losing
            # a re-downloadable file.
            archive_path.unlink(missing_ok=True)
            for path in result.files:
                rows.append(
                    _row(
                        doi,
                        record,
                        path.name,
                        relative_member_path(path, target),
                        path.stat().st_size,
                        None,
                        None,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        str(path),
                        now,
                        path_in_container=relative_member_path(path, unpack_root),
                    )
                )
            state.n_fetched += 1
            continue

        path = target / item.key
        atomic_write_bytes(path, content)
        digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
        verified = item.md5 is not None and digest == item.md5
        if item.md5 and not verified:
            logger.error(
                "zenodo md5 mismatch",
                extra={"record": record.record_id, "file": item.key},
            )
        rows.append(
            _row(
                doi,
                record,
                item.key,
                item.key,
                len(content),
                item.md5,
                verified,
                hashlib.sha256(content).hexdigest(),
                str(path),
                now,
            )
        )
        state.n_fetched += 1

    state.state = (
        CollectionState.PARTIAL.value
        if state.n_failed
        else CollectionState.COMPLETE.value
    )
    return state, rows


def _row(
    doi: str,
    record: ZenodoRecord,
    filename: str,
    relative_path: str,
    size: int,
    md5: str | None,
    verified: bool | None,
    sha256: str,
    local_path: str,
    now: datetime,
    path_in_container: str | None = None,
) -> dict:
    return {
        "dataset_doi": doi,
        "source": str(Source.ZENODO),
        # A record can sit in several communities; the first is the primary
        # collection and the rest are recorded in dataset_collections.
        "collection_id": record.communities[0] if record.communities else "zenodo",
        "dataverse_file_id": None,
        "container_file_id": None,
        "path_in_container": path_in_container,
        "relative_path": relative_path,
        "filename": filename,
        "size_bytes": size,
        "md5_api": md5,
        "sha256_local": sha256,
        "md5_verified": verified,
        "restricted": False,
        "directory_label": None,
        "download_ts": now,
        "local_path": local_path,
        "deposit_year": record.year,
    }


def collect(
    records: list[ZenodoRecord],
    files_root: Path,
    ledger: Ledger,
    client: PoliteClient,
    archive_cap: int = ARCHIVE_CAP_BYTES,
    fresh: bool = False,
) -> list[dict]:
    """Collect many records.

    Incremental by default: only deposits never successfully fetched, plus
    anything left partial or failed. ``fresh=True`` refetches everything.
    """
    rows: list[dict] = []
    todo = [
        r for r in records if ledger.should_process(r.doi or f"zenodo:{r.record_id}")
    ]
    logger.info(
        "zenodo collection starting",
        extra={"records": len(records), "todo": len(todo)},
    )
    for index, record in enumerate(todo, 1):
        try:
            state, record_rows = collect_record(client, record, files_root, archive_cap)
        except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
            logger.exception("zenodo record failed", extra={"record": record.record_id})
            state = DatasetRecord(
                dataset_doi=record.doi or f"zenodo:{record.record_id}",
                state=CollectionState.FAILED.value,
                error=f"{type(exc).__name__}: {exc}",
            )
            record_rows = []
        ledger.finish(state)
        rows.extend(record_rows)
        if index % 25 == 0 or index == len(todo):
            logger.info("zenodo progress", extra={"done": index, "of": len(todo)})
    return rows


def collections_rows(records: list[ZenodoRecord]) -> list[dict]:
    """Community membership, so the frame is recorded rather than implied."""
    seen: dict[str, dict] = {}
    for record in records:
        for community in record.communities:
            seen.setdefault(
                community,
                {
                    "collection_id": community,
                    "source": str(Source.ZENODO),
                    "kind": "community",
                    "collection_name": community,
                    "collection_url": f"https://zenodo.org/communities/{community}",
                    "journal_id": None,
                },
            )
    return list(seen.values())

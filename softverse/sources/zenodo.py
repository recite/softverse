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

import csv
import hashlib
import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.acquire.state import DatasetRecord, Ledger, atomic_write_bytes
from softverse.acquire.unpack import (
    extract,
    relative_member_path,
    spanned_segments,
)
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
#: 1,390 deposits: median 13 code files in the smallest size quartile (0.4 MB
#: median) against 41 in the largest (163 MB median). So the skipped tail is not
#: data-without-code, and dropping 19% of deposits drops more than 19% of the
#: code, biased toward large code-heavy projects. (An earlier note here said 21
#: against 221; that was measured on a corpus a third the size and did not
#: survive recomputation. The direction held, the factor went from 10.5 to 3.2.)
#:
#: Disk is not the constraint it looked like: archives are deleted after
#: successful extraction, so peak usage is one archive at a time, not the
#: cumulative transfer. 2 GB admits 112 of the 226 skipped archives and roughly
#: halves the gap; the remainder stays logged as a stated limitation.
#: Raised to 5 GB once the nested-archive leak was fixed. At 2 GB, 53 deposits
#: (3.8%) still yielded no code at all; the 47 in the 2-5 GB band are the
#: affordable part of that tail and take the gap to roughly 1%. The remainder --
#: 6 archives above 20 GB, largest 102.7 GB -- stays logged as a stated
#: limitation, because past some point the honest move is to report the gap
#: rather than chase it.
ARCHIVE_CAP_BYTES = 5 * 1024 * 1024 * 1024

#: Refuse to start a download that would leave less than this free.
#:
#: A collection run already died of ENOSPC once today, and an out-of-space
#: failure is uniquely bad here: it can strike between writing a file and
#: recording it, which is the one window where the ledger's claims and the disk
#: can disagree. Declining to start is cheap and always recoverable; the deposit
#: stays retryable.
MIN_FREE_BYTES = 8 * 1024 * 1024 * 1024

#: Deposits fetched at once.
#:
#: An earlier comment here claimed throughput "flattens" around three. That
#: was asserted, not measured, and it is wrong: three streams sustain 6 MB/s,
#: and six *additional* streams added 5 MB/s on top -- about 11 MB/s across
#: nine, sub-linear but still climbing. Raising this would genuinely go
#: faster.
#:
#: It stays at three anyway, deliberately. The remaining work finishes
#: unattended either way, Zenodo's load does not need to grow for our
#: convenience, and each worker can hold an archive up to the cap, so N
#: workers can commit N times 5 GB before any of them lands.
DEFAULT_WORKERS = 3

#: How long a worker waits for room before deferring, and how often it looks.
#: Long enough for a sibling to finish extracting a large archive and delete
#: it; short enough that a genuinely full volume does not stall the run.
WAIT_FOR_DISK_S = 120.0
_DISK_POLL_S = 5.0


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


class DiskBudget:
    """Free space, minus what other workers have already committed to.

    Asking ``shutil.disk_usage`` directly is correct for one downloader and
    wrong for several: each worker gets the same answer, because none of them
    can see the bytes the others are about to write. Three workers against a
    5 GB cap can all pass a check made with 12 GB free and then write 15 GB.
    That was live -- 12 GB free with 6.8 GB of `.part` files already in
    flight -- and an ENOSPC is the one failure the state design exists to
    prevent: it can land between writing a file and recording it, the only
    window where the ledger and the disk can disagree.

    Checked per download rather than once per run, because free space moves
    while a collection runs, from what we write and from everything else on
    the machine.
    """

    def __init__(
        self,
        free,
        reserve: int = MIN_FREE_BYTES,
        wait_seconds: float = WAIT_FOR_DISK_S,
    ) -> None:
        self._free = free
        self._reserve = reserve
        self._wait = wait_seconds
        self._committed = 0
        self._lock = threading.Lock()

    def reserve_bytes(self, needed: int, wait_seconds: float | None = None) -> bool:
        """Claim ``needed`` bytes, waiting a bounded time for room.

        The wait is the point. A shortfall is usually a sibling worker holding
        an archive it is seconds away from extracting and deleting, so
        refusing the instant the disk looks tight throws away work that would
        have succeeded shortly after. Refusing immediately is what turned one
        transient squeeze into 279 deferred deposits inside 31 seconds.

        Bounded, because the other cause is a genuinely full volume -- an
        unrelated project, in the case that prompted this -- and then the
        right answer is to defer and let a later run pick it up, not to hang.
        """
        wait = self._wait if wait_seconds is None else wait_seconds
        deadline = time.monotonic() + max(0.0, wait)
        while True:
            with self._lock:
                if self._free() - self._committed - needed > self._reserve:
                    self._committed += needed
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(_DISK_POLL_S, max(0.0, deadline - time.monotonic())))

    def release(self, needed: int) -> None:
        with self._lock:
            self._committed = max(0, self._committed - needed)


def _disk_budget_for(root: Path) -> DiskBudget:
    import shutil

    root.mkdir(parents=True, exist_ok=True)
    return DiskBudget(lambda: shutil.disk_usage(root).free)


def collect_record(
    client: PoliteClient,
    record: ZenodoRecord,
    files_root: Path,
    archive_cap: int = ARCHIVE_CAP_BYTES,
    disk: DiskBudget | None = None,
) -> tuple[DatasetRecord, list[dict]]:
    """Download one record's wanted files. Returns its state and file rows."""
    disk = disk if disk is not None else _disk_budget_for(files_root)
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
    # Recognised from the deposit's whole file list, not just the candidates:
    # a `.z01` segment is not itself a candidate, so the tail `.zip` is the
    # only piece we would otherwise see -- and it is the piece that looks fine.
    spanned = spanned_segments(f.key for f in record.files)

    for item in wanted:
        if item.key in spanned:
            # Downloading this succeeds and extracting it cannot. Counted
            # separately from a failure so it is not retried forever, and so
            # the coverage gap is a number in the ledger rather than an
            # anomaly someone has to notice.
            state.n_spanned += 1
            state.skipped_archives.append(
                {
                    "filename": item.key,
                    "size_bytes": item.size,
                    "file_id": None,
                    "reason": "multi-part archive segment",
                }
            )
            continue

        # The cap check must come first. It is a policy decision about what we
        # collect and belongs in the coverage statistic; the disk check is a
        # transient condition of this machine. Testing disk first would record
        # an over-cap archive as a disk failure and quietly corrupt the number
        # we report as the coverage gap -- caught by a test, not by inspection.
        if item.is_archive and item.size > archive_cap:
            state.n_skipped_over_cap += 1
            state.skipped_archives.append(
                {"filename": item.key, "size_bytes": item.size, "file_id": None}
            )
            continue

        if item.is_archive and not disk.reserve_bytes(item.size):
            # Stop before the disk does. An ENOSPC can land between writing a
            # file and recording it, which is the one window where the ledger
            # and the disk can disagree -- the failure mode the whole state
            # design exists to prevent.
            state.n_deferred += 1
            state.error = "insufficient free disk; deferred"
            logger.warning(
                "deferring download, low disk",
                extra={"record": record.record_id, "size_mb": item.size // 1_000_000},
            )
            continue

        if item.is_archive:
            # Stream, and resume if a previous attempt died partway.
            #
            # Buffering these in memory and starting over on any error is what
            # made the large-archive tail uncollectable: every multi-gigabyte
            # download over this link ended in a ReadTimeout or a reset peer
            # after one to three hours, each recording bytes_fetched: 0. The
            # cap is passed down as well, so an archive whose metadata
            # understates its size is abandoned after a chunk instead of after
            # the whole transfer.
            archive_path = target / "_archives" / item.key
            try:
                outcome = client.download(
                    item.link, archive_path, cap_bytes=archive_cap
                )
            except BaseException:
                disk.release(item.size)
                raise
            if not outcome.ok:
                if "exceeds cap" in (outcome.error or ""):
                    # Metadata lied about the size. This is a coverage gap, not
                    # a transport failure, and belongs in the same statistic as
                    # the archives skipped before download.
                    state.n_skipped_over_cap += 1
                    state.skipped_archives.append(
                        {
                            "filename": item.key,
                            "size_bytes": item.size,
                            "file_id": None,
                        }
                    )
                else:
                    state.n_failed += 1
                    state.error = outcome.error
                disk.release(item.size)
                continue

            state.bytes_fetched += archive_path.stat().st_size
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
                # is the evidence needed to work out why it failed. The
                # reservation is *not* released, because the bytes are still
                # on the disk.
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
            disk.release(item.size)
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

        # Loose files are scripts and manifests -- kilobytes, not gigabytes --
        # and the md5 check wants the bytes in hand anyway, so these keep the
        # buffered path.
        outcome = client.get(item.link, expect_content=False)
        if not outcome.ok or outcome.content is None:
            state.n_failed += 1
            state.error = outcome.error
            continue

        content = outcome.content
        state.bytes_fetched += len(content)

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

    # A deferral must keep the deposit retryable. Moving deferrals out of
    # `n_failed` silently made a wholly-deferred deposit read as `complete`,
    # which is worse than the accounting bug it fixed: it would never have
    # been collected at all. Caught by a test, not by inspection.
    state.state = (
        CollectionState.PARTIAL.value
        if (state.n_failed or state.n_deferred)
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
    workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """Collect many records, several at a time.

    Incremental by default: only deposits never successfully fetched, plus
    anything left partial or failed. ``fresh=True`` refetches everything.

    Concurrency is over *deposits*, and it is about throughput rather than
    politeness. Zenodo throttles per connection: measured against one
    deposit's archive, a single stream sustained 284 KB/s while three
    concurrent streams reached 1.01 MB/s in aggregate. Collecting one deposit
    at a time therefore left most of the available bandwidth unused, and the
    deposits it left for last are the multi-gigabyte ones where that costs
    hours rather than minutes.

    The *request* rate is unchanged, which is the part Zenodo actually
    publishes a limit for: the shared `RateLimiter` is global and locked, so
    N workers still issue the same requests per second between them. What
    changes is how many byte streams are open while those requests wait.
    """
    rows: list[dict] = []
    todo = [
        r
        for r in records
        if ledger.should_process(
            r.doi or f"zenodo:{r.record_id}",
            fresh=fresh,
            upstream_version=r.record_id,
        )
    ]
    logger.info(
        "zenodo collection starting",
        extra={"records": len(records), "todo": len(todo), "workers": workers},
    )

    disk = _disk_budget_for(files_root)

    def one(record: ZenodoRecord) -> tuple[DatasetRecord, list[dict]]:
        try:
            return collect_record(client, record, files_root, archive_cap, disk)
        except Exception as exc:  # noqa: BLE001 - one bad record must not end the run
            logger.exception("zenodo record failed", extra={"record": record.record_id})
            return (
                DatasetRecord(
                    dataset_doi=record.doi or f"zenodo:{record.record_id}",
                    state=CollectionState.FAILED.value,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                [],
            )

    # The ledger is the one piece of shared mutable state, and it is the piece
    # whose corruption is unrecoverable, so writes to it are serialized rather
    # than trusted to be atomic.
    write_lock = threading.Lock()

    def run(batch: list[ZenodoRecord]) -> list[ZenodoRecord]:
        """Collect a batch; return the deposits deferred for want of disk."""
        deferred: list[ZenodoRecord] = []
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(one, record): record for record in batch}
            for future in as_completed(futures):
                state, record_rows = future.result()
                with write_lock:
                    ledger.finish(state)
                    rows.extend(record_rows)
                    if state.n_deferred and not state.n_fetched:
                        deferred.append(futures[future])
                    done += 1
                    if done % 25 == 0 or done == len(batch):
                        logger.info(
                            "zenodo progress", extra={"done": done, "of": len(batch)}
                        )
        return deferred

    deferred = run(todo)
    if deferred:
        # Disk frees continuously as workers extract and delete, so a deposit
        # refused early in a run usually fits later in the same one. Leaving
        # them to a future run is what stranded 279 deposits for a day: they
        # were all refused inside 31 seconds while the volume was briefly
        # full, and nothing looked at them again. One more pass is bounded and
        # costs nothing when there is nothing to retry.
        logger.info("retrying deferred deposits", extra={"n": len(deferred)})
        still = run(deferred)
        if still:
            logger.warning(
                "still deferred after a retry; the disk is genuinely full",
                extra={"n": len(still)},
            )
    return rows


#: Columns of `corpus/zenodo/deposits.csv`.
DEPOSIT_FIELDS = (
    "dataset_doi",
    "record_id",
    "collection_id",
    "communities",
    "publication_date",
    "deposit_year",
)


def deposit_rows(records: Iterable[ZenodoRecord]) -> list[dict]:
    """One row per deposit: which community it belongs to, and when.

    Both values are read off every record at harvest and were, until this
    existed, used for a console summary and then dropped. Nothing on disk
    carried them: `ledger.jsonl` is retry bookkeeping with no year and no
    community field, so the tally rebuilt its corpus from bare files and had
    to hardcode `collection_id="zenodo"`. That left one collection across
    178,130 files and a null year on all 3,297,235 mentions, which is why the
    per-year and per-collection tables were empty.
    """
    return [
        {
            "dataset_doi": f"zenodo:{record.record_id}",
            "record_id": record.record_id,
            # The first community is the primary collection; the rest are kept
            # so a deposit cross-listed in two is not silently reduced to one.
            "collection_id": (
                record.communities[0] if record.communities else "zenodo"
            ),
            "communities": ";".join(record.communities),
            "publication_date": record.publication_date or "",
            "deposit_year": record.year or "",
        }
        for record in records
    ]


def write_deposits(records: Iterable[ZenodoRecord], path: Path) -> int:
    """Write `deposits.csv`, merging with whatever is already there.

    Merged rather than overwritten because a partial harvest must not delete
    metadata for deposits it did not look at. Overwriting a frame file cost
    12,336 Dataverse deposits once already.
    """
    merged: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                merged[row["dataset_doi"]] = row
    for row in deposit_rows(records):
        merged[row["dataset_doi"]] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEPOSIT_FIELDS))
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda r: r["dataset_doi"]))
    return len(merged)


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

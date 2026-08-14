"""Harvard Dataverse collection: frame, manifest, fetch.

Three stages, each restartable. Measured against the live API before it was
written, which is what shaped it:

- ``GET /api/access/datafiles/{id1,id2,...}`` returns **exactly the files named**
  as one zip. 73% of deposits expose their scripts as loose datafiles, so for
  most datasets a single request gets everything we want and no survey data.
  Loose scripts across the whole frame are ~88,000 files but only ~1.1 GB.
- Archives are the opposite: 5% of them are 150 GB. Hence the size cap, with
  every skipped archive recorded so the coverage gap is reportable.
- Per-file ``id``, ``md5``, ``filesize`` and ``restricted`` all come from the
  version metadata, so provenance needs no extra requests.

What v1 did that this deliberately does not: write state before downloading,
use ``latestVersion`` (which can be a DRAFT), send ``Authorization: Bearer``
(which Dataverse ignores, making every metadata read anonymous), flatten
directories, delete non-script files, and treat a restricted file as a failure.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.acquire.state import DatasetRecord, Ledger, atomic_write_bytes
from softverse.acquire.unpack import extract, relative_member_path
from softverse.config import (
    ARCHIVE_EXTENSIONS,
    DATAVERSE_BASE_URL,
    MANIFEST_FILENAMES,
    SCRIPT_EXTENSIONS,
)
from softverse.logging_setup import get_logger
from softverse.model.enums import CollectionState

logger = get_logger(__name__)

#: Archives larger than this are recorded and skipped. 95% of archives are under
#: 50 MB and total 32 GB; the tail above is ~150 GB of mostly data.
DEFAULT_ARCHIVE_CAP_BYTES = 100 * 1024 * 1024

#: Bulk access has a URL-length ceiling; chunk the id list well under it.
MAX_IDS_PER_REQUEST = 60


def repo_id_of(doi: str) -> str:
    """`doi:10.7910/DVN/ABC123` -> `ABC123`."""
    return doi.rstrip("/").rsplit("/", 1)[-1]


def dataset_dir(root: Path, doi: str) -> Path:
    return root / repo_id_of(doi)


@dataclass
class CandidateFile:
    """A file we may want, as described by the dataset's version metadata."""

    file_id: int
    filename: str
    size: int
    md5: str | None
    restricted: bool
    directory_label: str | None

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower()

    @property
    def is_archive(self) -> bool:
        return self.suffix in ARCHIVE_EXTENSIONS

    @property
    def relative_path(self) -> str:
        """Path within the dataset, preserving the author's folders."""
        if self.directory_label:
            return f"{self.directory_label.strip('/')}/{self.filename}"
        return self.filename


def is_wanted(filename: str) -> bool:
    """Whether a datafile is worth fetching.

    Wider than "scripts": manifests are kept because ``renv.lock`` and
    ``requirements.txt`` are author-declared dependency lists and therefore free
    validation data. v1 deleted every non-script file and destroyed all of them.
    """
    base = Path(filename).name
    if base.startswith("._"):
        return False
    return (
        base.lower() in MANIFEST_FILENAMES
        or Path(base).suffix.lower() in SCRIPT_EXTENSIONS
        or Path(base).suffix.lower() in ARCHIVE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Stage 1: frame
# ---------------------------------------------------------------------------


def walk_collection(
    client: PoliteClient, alias: str, base_url: str = DATAVERSE_BASE_URL, depth: int = 0
) -> tuple[list[dict], list[str]]:
    """List a collection's datasets, recursing into sub-collections.

    ``/api/dataverses/{id}/contents`` returns **direct children only**. v1 read
    one level and silently dropped the sub-collection rows, missing 5 nested
    collections (12 datasets, enumerated during the audit).

    Returns ``(datasets, nested_aliases_visited)``.
    """
    outcome = client.get(f"{base_url}/api/dataverses/{alias}/contents")
    if not outcome.ok or outcome.content is None:
        logger.warning(
            "collection listing failed",
            extra={"alias": alias, "status": outcome.status, "err": outcome.error},
        )
        return [], []

    try:
        children = json.loads(outcome.content)["data"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            "collection listing unparseable", extra={"alias": alias, "err": str(exc)}
        )
        return [], []

    datasets: list[dict] = []
    visited: list[str] = []
    for child in children:
        if child.get("type") == "dataset":
            datasets.append(child)
        elif child.get("type") == "dataverse" and depth < 3:
            child_id = str(child.get("id") or child.get("alias") or "")
            if not child_id:
                continue
            visited.append(child_id)
            sub_datasets, sub_visited = walk_collection(
                client, child_id, base_url, depth + 1
            )
            datasets.extend(sub_datasets)
            visited.extend(sub_visited)
    return datasets, visited


#: Datasets per search page. Small enough that each request is quick and
#: cheap for the server, which is the point of using search at all.
SEARCH_PAGE_SIZE = 100

#: How Dataverse says a collection is not there. Two spellings, because
#: `/api/dataverses/{alias}` and `/api/search?subtree=` word it differently.
_MISSING = (
    "find dataverse with identifier",
    "find dataverse with alias",
)


def collection_missing(outcome) -> bool:
    """Whether the response says this collection does not exist.

    Worth separating from a failure, because the two want opposite
    responses: a transient error should be retried and a deleted collection
    never will succeed. `regionalstatistics` held 23 deposits in the 2024
    scrape and Harvard now answers "Can't find dataverse with identifier",
    while five other collections that failed in the same run recovered on a
    retry minutes later. Counting them together would overstate the coverage
    gap and hide the one fact that is actually about the frame: a journal
    collection has gone.
    """
    if not outcome.ok or not outcome.content:
        return False
    try:
        payload = json.loads(outcome.content)
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "ERROR":
        return False
    message = str(payload.get("message", "")).lower()
    return any(phrase in message for phrase in _MISSING)


def search_collection(
    client, alias: str, base_url: str = DATAVERSE_BASE_URL
) -> list[dict]:
    """List a collection's datasets by paging the search API.

    The fallback for collections `/api/dataverses/{alias}/contents` cannot
    serve. That endpoint returns the whole collection in one response, and
    the server gives up on the largest one: measured, `jop` came back with
    466 KB after 87 s while `restat` returned nothing after 73 s -- failing
    *faster* than the one that worked, so not a timeout on either side.

    Paging is also the lighter request. Seventeen 150 KB pages at ten seconds
    each is easier on their infrastructure than one response big enough to
    fall over, which matters given the terms speak to impairment.

    Returns rows shaped like `/contents` entries so the two paths merge, and
    returns nothing at all on a failed page: a short read would silently
    shrink a journal in the frame, and a truncated collection is
    indistinguishable from a small one.
    """
    rows: list[dict] = []
    start = 0
    total: int | None = None
    while total is None or start < total:
        url = (
            f"{base_url}/api/search?q=*&subtree={alias}&type=dataset"
            f"&per_page={SEARCH_PAGE_SIZE}&start={start}"
        )
        outcome = client.get(url)
        if not outcome.ok or outcome.content is None:
            logger.warning(
                "search page failed; abandoning the collection",
                extra={"alias": alias, "start": start, "err": outcome.error},
            )
            return []
        try:
            data = json.loads(outcome.content)["data"]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "search page unparseable", extra={"alias": alias, "err": str(exc)}
            )
            return []
        total = data.get("total_count", 0)
        items = data.get("items") or []
        if not items:
            break
        for item in items:
            # `doi:10.7910/DVN/24195` -> protocol, authority, identifier, so
            # the row matches what `/contents` would have produced.
            global_id = item.get("global_id") or ""
            protocol, _, rest = global_id.partition(":")
            authority, _, identifier = rest.partition("/")
            rows.append(
                {
                    "id": item.get("entity_id"),
                    "identifier": identifier,
                    "persistentUrl": f"https://doi.org/{rest}" if rest else "",
                    "protocol": protocol,
                    "authority": authority,
                    "publicationDate": (item.get("published_at") or "")[:10],
                    "type": "dataset",
                }
            )
        start += SEARCH_PAGE_SIZE
    return rows


# ---------------------------------------------------------------------------
# Stage 2: manifest
# ---------------------------------------------------------------------------


def fetch_version_metadata(
    client: PoliteClient, doi: str, raw_root: Path, base_url: str = DATAVERSE_BASE_URL
) -> tuple[dict | None, str | None, int | None]:
    """Fetch and archive a dataset's published version metadata.

    Uses ``:latest-published``, never ``latestVersion``, which can return a
    DRAFT once authenticated -- a version the article was never based on.

    The raw JSON is archived gzipped **before** anything is derived from it.
    This is the structural fix for the bug that truncated all 74 ``files_dfs``
    CSVs to one byte: a derived file must never be the only copy of a network
    response.

    Returns ``(payload, error, http_status)``.
    """
    outcome = client.get(
        f"{base_url}/api/datasets/:persistentId/versions/:latest-published",
        params={"persistentId": doi},
    )
    if not outcome.ok or outcome.content is None:
        return None, outcome.error, outcome.status

    raw_path = raw_root / f"{repo_id_of(doi)}.json.gz"
    atomic_write_bytes(raw_path, gzip.compress(outcome.content))

    try:
        return json.loads(outcome.content)["data"], None, 200
    except (json.JSONDecodeError, KeyError) as exc:
        return None, f"unparseable metadata: {exc}", 200


def candidates_from(payload: dict) -> list[CandidateFile]:
    """Extract the files worth fetching from version metadata."""
    out: list[CandidateFile] = []
    for entry in payload.get("files", []):
        data = entry.get("dataFile", {})
        filename = data.get("filename") or ""
        if not filename or not is_wanted(filename):
            continue
        out.append(
            CandidateFile(
                file_id=data.get("id"),
                filename=filename,
                size=data.get("filesize") or 0,
                md5=data.get("md5"),
                restricted=bool(data.get("restricted")),
                directory_label=entry.get("directoryLabel"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Stage 3: fetch
# ---------------------------------------------------------------------------


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def fetch_bulk(
    client: PoliteClient, ids: list[int], base_url: str = DATAVERSE_BASE_URL
) -> tuple[dict[str, bytes], str | None]:
    """Fetch several datafiles in one request, returning ``{name: bytes}``.

    The endpoint returns a zip of exactly the requested files. A single file id
    returns the bare file instead, so that case is handled separately.
    """
    if not ids:
        return {}, None
    outcome = client.get(f"{base_url}/api/access/datafiles/{','.join(map(str, ids))}")
    if not outcome.ok or outcome.content is None:
        return {}, outcome.error or f"HTTP {outcome.status}"
    body = outcome.content
    if body[:2] != b"PK":
        return {}, "response was not a zip"
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            return {
                info.filename: zf.read(info)
                for info in zf.infolist()
                if not info.is_dir() and not Path(info.filename).name.startswith("._")
            }, None
    except zipfile.BadZipFile as exc:
        return {}, f"bad zip: {exc}"


def fetch_single(
    client: PoliteClient, file_id: int, base_url: str = DATAVERSE_BASE_URL
) -> tuple[bytes | None, str | None]:
    outcome = client.get(
        f"{base_url}/api/access/datafile/{file_id}", expect_content=False
    )
    if not outcome.ok or outcome.content is None:
        return None, outcome.error or f"HTTP {outcome.status}"
    return outcome.content, None


def collect_dataset(
    client: PoliteClient,
    doi: str,
    files_root: Path,
    raw_root: Path,
    archive_cap: int = DEFAULT_ARCHIVE_CAP_BYTES,
    base_url: str = DATAVERSE_BASE_URL,
) -> tuple[DatasetRecord, list[dict]]:
    """Collect one dataset. Returns its record and the file rows it produced.

    The record is *returned*, not written -- the caller commits it to the ledger
    only after this returns, so a crash mid-dataset leaves no claim of success.
    """
    record = DatasetRecord(dataset_doi=doi, state=CollectionState.FAILED.value)
    file_rows: list[dict] = []

    payload, error, status = fetch_version_metadata(client, doi, raw_root, base_url)
    if payload is None:
        record.error = error
        record.state = (
            CollectionState.NOT_FOUND.value
            if status == 404
            else CollectionState.RESTRICTED.value
            if status == 403
            else CollectionState.FAILED.value
        )
        return record, file_rows

    state = (payload.get("versionState") or "").upper()
    record.version_number = payload.get("versionNumber")
    record.version_minor = payload.get("versionMinorNumber")
    if state == "DEACCESSIONED":
        record.state = CollectionState.DEACCESSIONED.value
        return record, file_rows

    candidates = candidates_from(payload)
    record.n_candidate = len(candidates)
    if not candidates:
        record.state = CollectionState.NO_CANDIDATE_FILES.value
        return record, file_rows

    target = dataset_dir(files_root, doi)
    now = datetime.now(tz=UTC)

    loose: list[CandidateFile] = []
    archives: list[CandidateFile] = []
    for candidate in candidates:
        if candidate.restricted:
            # Distinguishable from a failure, unlike v1 where a 403 and a
            # timeout produced the same nothing.
            record.n_restricted += 1
            continue
        if candidate.is_archive:
            if candidate.size > archive_cap:
                record.n_skipped_over_cap += 1
                record.skipped_archives.append(
                    {
                        "filename": candidate.filename,
                        "size_bytes": candidate.size,
                        "file_id": candidate.file_id,
                    }
                )
            else:
                archives.append(candidate)
        else:
            loose.append(candidate)

    by_name = {c.filename: c for c in loose}
    for chunk_start in range(0, len(loose), MAX_IDS_PER_REQUEST):
        chunk = loose[chunk_start : chunk_start + MAX_IDS_PER_REQUEST]
        if len(chunk) == 1:
            content, error = fetch_single(client, chunk[0].file_id, base_url)
            payload_files = {chunk[0].filename: content} if content is not None else {}
        else:
            payload_files, error = fetch_bulk(
                client, [c.file_id for c in chunk], base_url
            )
        if error:
            record.n_failed += len(chunk)
            record.error = error
            continue
        got = set()
        for name, content in payload_files.items():
            candidate = by_name.get(name) or by_name.get(Path(name).name)
            if candidate is None:
                continue
            got.add(candidate.filename)
            row = _write_file(target, candidate, content, doi, now)
            file_rows.append(row)
            record.n_fetched += 1
            record.bytes_fetched += len(content)
        # Anything the server did not return is a failure, counted, not ignored.
        missing = [c for c in chunk if c.filename not in got]
        record.n_failed += len(missing)

    for candidate in archives:
        content, error = fetch_single(client, candidate.file_id, base_url)
        if content is None:
            record.n_failed += 1
            record.error = error
            continue
        archive_path = target / "_archives" / candidate.filename
        atomic_write_bytes(archive_path, content)
        record.bytes_fetched += len(content)
        unpack_root = target / "_archives" / f"{candidate.filename}_extracted"
        result = extract(
            archive_path,
            unpack_root,
            frozenset(SCRIPT_EXTENSIONS) | frozenset(ARCHIVE_EXTENSIONS),
            frozenset(MANIFEST_FILENAMES),
        )
        if result.error:
            logger.warning(
                "archive extraction failed",
                extra={"doi": doi, "archive": candidate.filename, "err": result.error},
            )
        for path in result.files:
            file_rows.append(
                {
                    "dataset_doi": doi,
                    "dataverse_file_id": None,
                    "container_file_id": candidate.file_id,
                    "path_in_container": relative_member_path(path, unpack_root),
                    "relative_path": relative_member_path(path, target),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "md5_api": None,
                    "sha256_local": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "md5_verified": None,
                    "restricted": False,
                    "directory_label": None,
                    "download_ts": now,
                    "local_path": str(path),
                }
            )
        record.n_fetched += 1

    if record.n_failed:
        record.state = CollectionState.PARTIAL.value
    else:
        record.state = CollectionState.COMPLETE.value
    return record, file_rows


def _write_file(
    target: Path, candidate: CandidateFile, content: bytes, doi: str, now: datetime
) -> dict:
    """Write one datafile and build its provenance row, verifying md5."""
    path = target / candidate.relative_path
    atomic_write_bytes(path, content)
    digest = _md5(content)
    verified = (candidate.md5 is not None) and digest == candidate.md5
    if candidate.md5 and not verified:
        logger.error(
            "md5 mismatch",
            extra={"doi": doi, "file": candidate.filename, "expected": candidate.md5},
        )
    return {
        "dataset_doi": doi,
        "dataverse_file_id": candidate.file_id,
        "container_file_id": None,
        "path_in_container": None,
        "relative_path": candidate.relative_path,
        "filename": candidate.filename,
        "size_bytes": len(content),
        "md5_api": candidate.md5,
        "sha256_local": hashlib.sha256(content).hexdigest(),
        "md5_verified": verified,
        "restricted": False,
        "directory_label": candidate.directory_label,
        "download_ts": now,
        "local_path": str(path),
    }


def collect(
    dois: list[str],
    files_root: Path,
    raw_root: Path,
    ledger: Ledger,
    client: PoliteClient,
    archive_cap: int = DEFAULT_ARCHIVE_CAP_BYTES,
    base_url: str = DATAVERSE_BASE_URL,
) -> list[dict]:
    """Collect many datasets, skipping those already in a terminal state."""
    all_rows: list[dict] = []
    todo = [d for d in dois if ledger.should_process(d)]
    logger.info(
        "collection starting",
        extra={
            "requested": len(dois),
            "todo": len(todo),
            "already_done": len(dois) - len(todo),
        },
    )
    for index, doi in enumerate(todo, 1):
        try:
            record, rows = collect_dataset(
                client, doi, files_root, raw_root, archive_cap, base_url
            )
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not end the run
            logger.exception("dataset failed", extra={"doi": doi})
            record = DatasetRecord(
                dataset_doi=doi,
                state=CollectionState.FAILED.value,
                error=f"{type(exc).__name__}: {exc}",
            )
            rows = []
        ledger.finish(record)
        all_rows.extend(rows)
        if index % 25 == 0 or index == len(todo):
            logger.info(
                "progress",
                extra={"done": index, "of": len(todo), **ledger.totals()},
            )
    return all_rows


__all__ = [
    "CandidateFile",
    "DEFAULT_ARCHIVE_CAP_BYTES",
    "candidates_from",
    "collect",
    "collect_dataset",
    "dataset_dir",
    "fetch_bulk",
    "fetch_version_metadata",
    "is_wanted",
    "repo_id_of",
    "walk_collection",
]

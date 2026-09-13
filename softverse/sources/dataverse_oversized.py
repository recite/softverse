"""Recover code from Dataverse archives too large to download whole.

The collector skips any archive over its cap, because keeping the scripts from
a 20 GB replication package by downloading it means transferring 20 GB. At
4,400 deposits that was 389 archives and 578 GB, nearly all of it data, with
the code for those deposits missing from the corpus.

A zip does not have to be downloaded to be read. Its central directory -- the
member list, with each member's offset -- sits at the end of the file, and
Harvard's S3 storage answers byte-range requests, so ``remotezip`` reads the
directory and then only the members we keep. Measured on a 91 GB package:
52,883 members, 9 scripts totalling 129 KB, 9.4 MB transferred, most of that
the directory itself.

tar, 7z and rar have no such index, so those are downloaded in full, the code
extracted, and the archive deleted. ``scripts/recover_oversized.py`` caps how many
bytes one run may spend on them, so the volume can be spread across runs.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import requests
from remotezip import RemoteZip

from softverse.acquire.unpack import _wanted, extract, relative_member_path
from softverse.config import DATAVERSE_BASE_URL
from softverse.logging_setup import get_logger
from softverse.sources.dataverse import (
    ARCHIVE_EXTENSIONS,
    MANIFEST_FILENAMES,
    SCRIPT_EXTENSIONS,
    dataset_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from softverse.acquire.state import DatasetRecord

logger = get_logger(__name__)

KEEP_SUFFIXES = frozenset(SCRIPT_EXTENSIONS) | frozenset(ARCHIVE_EXTENSIONS)
KEEP_NAMES = frozenset(MANIFEST_FILENAMES)

#: A nested archive inside a big zip is opened only below this size. The 91 GB
#: package held 11,199 of them, which are compressed data far more often than a
#: `code.zip`; fetching every one would be the full download by instalments.
NESTED_CAP_BYTES = 5 * 1024 * 1024

USER_AGENT = "softverse/2.0 (research; github.com/recite/softverse)"


@dataclass
class Outcome:
    """What recovering one archive cost and produced."""

    dataset_doi: str
    filename: str
    file_id: int
    archive_bytes: int
    method: str
    transferred_bytes: int = 0
    rows: list[dict] = field(default_factory=list)
    nested_skipped: int = 0
    member_errors: int = 0
    error: str | None = None

    def summary(self) -> dict:
        """A JSON-ready line for the transfer log, without the file rows.

        Returns:
            The outcome's counts and sizes.
        """
        return {
            "dataset_doi": self.dataset_doi,
            "filename": self.filename,
            "file_id": self.file_id,
            "archive_bytes": self.archive_bytes,
            "method": self.method,
            "transferred_bytes": self.transferred_bytes,
            "n_code": len(self.rows),
            "code_bytes": sum(r["size_bytes"] for r in self.rows),
            "nested_skipped": self.nested_skipped,
            "member_errors": self.member_errors,
            "error": self.error,
        }


class _CountingSession(requests.Session):
    """A session that adds up the bytes the server says it sent."""

    def __init__(self) -> None:
        super().__init__()
        self.transferred = 0

    def request(self, method, url, *args, **kwargs):
        response = super().request(method, url, *args, **kwargs)
        self.transferred += int(response.headers.get("Content-Length") or 0)
        return response


def _row(doi: str, file_id: int, path: Path, unpack_root: Path, target: Path) -> dict:
    data = path.read_bytes()
    return {
        "dataset_doi": doi,
        "dataverse_file_id": None,
        "container_file_id": file_id,
        "path_in_container": relative_member_path(path, unpack_root),
        "relative_path": relative_member_path(path, target),
        "filename": path.name,
        "size_bytes": len(data),
        "md5_api": None,
        "sha256_local": hashlib.sha256(data).hexdigest(),
        "md5_verified": None,
        "restricted": False,
        "directory_label": None,
        "download_ts": None,
        "local_path": str(path),
    }


def storage_url(session: requests.Session, file_id: int, headers: dict) -> str:
    """The signed storage URL Dataverse redirects a file download to.

    Args:
        session: The session to ask through.
        file_id: The Dataverse datafile id.
        headers: Auth headers for the API; not sent on to storage.

    Returns:
        The ``Location`` of the redirect, or the API URL if none was given.
    """
    api = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"
    response = session.get(api, headers=headers, allow_redirects=False, timeout=60)
    response.raise_for_status()
    return response.headers.get("Location") or api


def recover_zip(
    session: _CountingSession, url: str, outcome: Outcome, files_root: Path
) -> None:
    """Read a remote zip's directory and fetch only the members worth keeping."""
    target = dataset_dir(files_root, outcome.dataset_doi)
    unpack_root = target / "_archives" / f"{outcome.filename}_extracted"
    with RemoteZip(url, session=session, timeout=300) as archive:
        for info in archive.infolist():
            if info.is_dir() or not _wanted(info.filename, KEEP_SUFFIXES, KEEP_NAMES):
                continue
            nested = PurePosixPath(info.filename).suffix.lower() in ARCHIVE_EXTENSIONS
            if nested and info.file_size > NESTED_CAP_BYTES:
                outcome.nested_skipped += 1
                continue
            try:
                path = Path(archive.extract(info, unpack_root))
            except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as exc:
                # Deflate64 or an encrypted member: that file is lost, the
                # rest of the archive is not.
                logger.warning(
                    "zip member unreadable",
                    extra={
                        "doi": outcome.dataset_doi,
                        "member": info.filename,
                        "err": str(exc),
                    },
                )
                outcome.member_errors += 1
                continue
            if not nested:
                outcome.rows.append(
                    _row(
                        outcome.dataset_doi, outcome.file_id, path, unpack_root, target
                    )
                )
                continue
            inner_root = path.parent / f"{path.name}_extracted"
            inner = extract(path, inner_root, KEEP_SUFFIXES, KEEP_NAMES)
            if inner.error is None:
                path.unlink()
            outcome.rows.extend(
                _row(outcome.dataset_doi, outcome.file_id, p, unpack_root, target)
                for p in inner.files
                if p.is_file()
            )


def recover_by_download(
    session: _CountingSession, url: str, outcome: Outcome, files_root: Path
) -> None:
    """Download an archive that has no index, keep its code, delete it."""
    target = dataset_dir(files_root, outcome.dataset_doi)
    archive_path = target / "_archives" / outcome.filename
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_name(archive_path.name + ".part")
    with session.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
    partial.replace(archive_path)
    unpack_root = target / "_archives" / f"{outcome.filename}_extracted"
    result = extract(archive_path, unpack_root, KEEP_SUFFIXES, KEEP_NAMES)
    if result.error:
        outcome.error = f"extract: {result.error}"
        return
    archive_path.unlink()
    outcome.rows.extend(
        _row(outcome.dataset_doi, outcome.file_id, p, unpack_root, target)
        for p in result.files
        if p.is_file()
    )


def recover(
    doi: str,
    skipped: dict,
    files_root: Path,
    headers: dict,
    before_request: Callable[[], None],
) -> Outcome:
    """Recover the code from one over-cap archive.

    Args:
        doi: The deposit's DOI.
        skipped: The ledger's ``skipped_archives`` entry for the archive.
        files_root: Where deposit directories live.
        headers: Auth headers for the Dataverse API.
        before_request: Called before the one request to Dataverse itself; the
            rate limiter. The range reads that follow go to S3 storage and
            carry kilobytes; throttling them as well made a zip with thirty
            scripts take a minute.

    Returns:
        What was transferred and kept, or the error that stopped it.
    """
    is_zip = skipped["filename"].lower().endswith(".zip")
    outcome = Outcome(
        dataset_doi=doi,
        filename=skipped["filename"],
        file_id=skipped["file_id"],
        archive_bytes=skipped["size_bytes"],
        method="range" if is_zip else "download",
    )
    session = _CountingSession()
    session.headers["User-Agent"] = USER_AGENT
    try:
        before_request()
        url = storage_url(session, outcome.file_id, headers)
        if is_zip:
            recover_zip(session, url, outcome, files_root)
        else:
            recover_by_download(session, url, outcome, files_root)
    except Exception as exc:  # one bad archive must not end a long run
        logger.exception("oversized recovery failed", extra={"doi": doi})
        outcome.error = f"{type(exc).__name__}: {exc}"
    finally:
        outcome.transferred_bytes = session.transferred
        session.close()
    return outcome


def apply(record: DatasetRecord, outcomes: list[Outcome]) -> None:
    """Move recovered archives from skipped to fetched on the deposit's record.

    A failure stays listed as skipped, marked with its error, so a later run
    can retry it; the counts still reconcile either way.
    """
    by_name = {o.filename: o for o in outcomes}
    remaining = []
    for entry in record.skipped_archives:
        outcome = by_name.get(entry["filename"])
        if outcome is None:
            remaining.append(entry)
        elif outcome.error:
            remaining.append({**entry, "recovery_error": outcome.error})
        else:
            record.n_skipped_over_cap -= 1
            record.n_fetched += 1
    record.skipped_archives = remaining

"""Per-dataset collection state, written after the work, never before.

This is the single most consequential file in the collector, because getting it
wrong is what destroyed the v1 corpus. v1 wrote ``{"status": "success"}`` at
*metadata* time -- before any file was downloaded -- and its resume logic then
skipped any dataset that already had a status. The result, measured: 8,953 of
12,054 dataset directories marked ``success`` with a non-zero ``script_count``
and holding **zero bytes**, unrecoverable without a full re-run, and invisible
because nothing ever compared the claim to the disk.

Three rules follow, and they are enforced here rather than left to callers:

1. **State is written only after the work it describes has completed**, via
   :meth:`Ledger.finish`.
2. **Writes are atomic** -- temp file plus ``os.replace`` -- so a process killed
   mid-write leaves the previous state, not a truncated one.
3. **A claim carries its evidence.** Every record stores the counts it asserts,
   so :func:`verify_against_disk` can check the ledger against reality instead
   of trusting it.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from softverse.logging_setup import get_logger
from softverse.model.enums import RETRYABLE_STATES, CollectionState

logger = get_logger(__name__)


@dataclass
class DatasetRecord:
    """What happened to one dataset, and the evidence for it."""

    dataset_doi: str
    state: str
    # Evidence: these must reconcile, and be checkable against the filesystem.
    n_candidate: int = 0
    n_fetched: int = 0
    n_skipped_over_cap: int = 0
    n_restricted: int = 0
    n_failed: int = 0
    bytes_fetched: int = 0
    version_number: int | None = None
    version_minor: int | None = None
    error: str | None = None
    finished_at: str | None = None
    #: Archives passed over by the size cap, kept so the coverage gap is
    #: reportable rather than invisible.
    skipped_archives: list[dict] = field(default_factory=list)

    def reconciles(self) -> bool:
        """Whether the parts account for every candidate file."""
        return self.n_candidate == (
            self.n_fetched + self.n_skipped_over_cap + self.n_restricted + self.n_failed
        )

    @property
    def needs_retry(self) -> bool:
        return self.state in {s.value for s in RETRYABLE_STATES}


class Ledger:
    """Append-only JSONL record of dataset outcomes, with an in-memory index.

    JSONL rather than one file per dataset: 13,775 tiny files is slow to scan
    and easy to leave half-written. Rather than rewriting the whole list after
    every record (v1's Zenodo checkpoint did that, O(n^2)), records are appended
    and the last write for a DOI wins on load.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, DatasetRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        bad = 0
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    self._records[payload["dataset_doi"]] = DatasetRecord(**payload)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A line truncated by a kill. Skip it; the dataset simply
                    # looks unattempted and gets retried, which is correct.
                    bad += 1
        logger.info(
            "ledger loaded",
            extra={"records": len(self._records), "unreadable_lines": bad},
        )

    def __len__(self) -> int:
        return len(self._records)

    def get(self, doi: str) -> DatasetRecord | None:
        return self._records.get(doi)

    def should_process(self, doi: str) -> bool:
        """Whether this dataset still needs work.

        A dataset in a retryable state (``partial``/``failed``) is always
        retried. v1 treated *any* existing status -- including
        ``download_failed`` and ``forbidden`` -- as done, and counted those
        failures as successes on every subsequent run.
        """
        record = self._records.get(doi)
        return record is None or record.needs_retry

    def finish(self, record: DatasetRecord) -> None:
        """Record a terminal outcome. Call this only after the work is done."""
        if not record.reconciles():
            logger.error(
                "state does not reconcile",
                extra={
                    "doi": record.dataset_doi,
                    "candidate": record.n_candidate,
                    "fetched": record.n_fetched,
                    "over_cap": record.n_skipped_over_cap,
                    "restricted": record.n_restricted,
                    "failed": record.n_failed,
                },
            )
        record.finished_at = datetime.now(tz=UTC).isoformat()
        self._records[record.dataset_doi] = record
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.state] = counts.get(record.state, 0) + 1
        return dict(sorted(counts.items()))

    def totals(self) -> dict[str, int]:
        return {
            "datasets": len(self._records),
            "candidate": sum(r.n_candidate for r in self._records.values()),
            "fetched": sum(r.n_fetched for r in self._records.values()),
            "over_cap": sum(r.n_skipped_over_cap for r in self._records.values()),
            "restricted": sum(r.n_restricted for r in self._records.values()),
            "failed": sum(r.n_failed for r in self._records.values()),
            "bytes": sum(r.bytes_fetched for r in self._records.values()),
        }

    def skipped_archives(self) -> list[dict]:
        """Every archive passed over by the size cap, for the coverage report."""
        out: list[dict] = []
        for record in self._records.values():
            for entry in record.skipped_archives:
                out.append({"dataset_doi": record.dataset_doi, **entry})
        return out

    def non_reconciling(self) -> list[DatasetRecord]:
        return [r for r in self._records.values() if not r.reconciles()]


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    A killed process must leave either the old file or the new one, never a
    half-written file that a later run mistakes for complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def verify_against_disk(
    ledger: Ledger, root: Path, sample: int | None = None
) -> dict[str, list[str]]:
    """Check the ledger's claims against what is actually on disk.

    The check whose absence let v1 report success for 8,953 empty directories.
    Returns ``{"empty_but_complete": [...], "missing_dir": [...]}``.
    """
    problems: dict[str, list[str]] = {"empty_but_complete": [], "missing_dir": []}
    records = list(ledger._records.values())
    if sample:
        records = records[:sample]
    for record in records:
        if record.state != CollectionState.COMPLETE.value or record.n_fetched == 0:
            continue
        directory = root / record.dataset_doi.replace("doi:", "").replace("/", "_")
        if not directory.exists():
            problems["missing_dir"].append(record.dataset_doi)
        elif not any(directory.rglob("*")):
            problems["empty_but_complete"].append(record.dataset_doi)
    return problems

"""Structured logging.

Two audiences, so two sinks:

- **Console**: human-readable, for watching a run.
- **JSONL file**: one object per event, for querying afterwards. A collection
  run touches ~13,775 datasets over days; "why is this deposit empty?" has to be
  answerable with ``jq``, not by scrolling.

The design point is that **counters are logged, not just messages**. v1's run
died partway through and left no record of how far it got, so nobody noticed
that 8,953 deposits were marked successful while holding zero files. Every stage
here emits a machine-readable tally at the end, and :class:`RunStats` makes the
denominators explicit rather than reconstructable.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Attributes present on every LogRecord; anything else was passed via `extra`
#: and is therefore structured context worth serializing.
_STANDARD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

#: Identifies one invocation of the pipeline across all its log lines.
RUN_ID = os.environ.get("SOFTVERSE_RUN_ID") or uuid.uuid4().hex[:12]


class JsonlFormatter(logging.Formatter):
    """One JSON object per line, with `extra=` fields promoted to top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "run_id": RUN_ID,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact human format; appends `extra` context as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')} "
            f"{record.levelname:<7} {record.name.removeprefix('softverse.'):<22} "
            f"{record.getMessage()}"
        )
        context = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        if context:
            base += "  " + " ".join(f"{k}={v}" for k, v in context.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(
    level: str | int = "INFO",
    log_dir: Path | str | None = None,
    stage: str = "run",
) -> Path | None:
    """Configure root logging. Idempotent -- safe to call more than once.

    Args:
        level: Console level. The JSONL sink always records DEBUG, because the
            detail you want after a failed run is never the detail you thought
            to enable before it.
        log_dir: Directory for the JSONL log. ``None`` disables the file sink.
        stage: Short name used in the log filename.

    Returns:
        Path to the JSONL log, or ``None`` if no file sink was configured.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if log_dir is None:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{stage}-{datetime.now(tz=UTC):%Y%m%dT%H%M%S}-{RUN_ID}.jsonl"
    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=256 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonlFormatter())
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "logging initialized", extra={"log_path": str(path), "stage": stage}
    )
    return path


def get_logger(name: str) -> logging.Logger:
    """Logger for a module. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


@dataclass
class RunStats:
    """A named tally that reconciles.

    Stages report outcomes through this rather than through log prose, so the
    denominator of every published rate is recoverable from the logs alone.
    Use :meth:`check_total` to assert that the parts sum to the whole -- that
    assertion is precisely what v1 lacked when 26,681 files vanished between
    the archive and the published CSV.
    """

    stage: str
    counts: Counter[str] = field(default_factory=Counter)
    started_at: float = field(default_factory=time.monotonic)

    def incr(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def check_total(self, total_key: str, part_keys: list[str]) -> None:
        """Assert ``total_key`` equals the sum of ``part_keys``.

        Raises:
            ValueError: if they disagree, naming the shortfall.
        """
        total = self.counts[total_key]
        parts = sum(self.counts[k] for k in part_keys)
        if total != parts:
            raise ValueError(
                f"{self.stage}: reconciliation failed -- {total_key}={total} "
                f"but {'+'.join(part_keys)}={parts} (difference {total - parts})"
            )

    def log(self, logger: logging.Logger | None = None) -> None:
        """Emit the tally as one structured record."""
        logger = logger or logging.getLogger("softverse.stats")
        logger.info(
            f"{self.stage} complete",
            extra={
                "stage": self.stage,
                "elapsed_s": round(self.elapsed_s, 1),
                **{f"n_{k}": v for k, v in sorted(self.counts.items())},
            },
        )


@contextmanager
def stage(name: str, logger: logging.Logger | None = None):
    """Context manager wrapping a pipeline stage with timing and a tally.

    Yields a :class:`RunStats` to record outcomes on. On exception the tally is
    still logged, so a crashed run leaves a record of how far it got -- the
    thing missing when the v1 collection died at 9.5% completion.
    """
    logger = logger or logging.getLogger("softverse.stage")
    stats = RunStats(stage=name)
    logger.info(f"{name} starting", extra={"stage": name})
    try:
        yield stats
    except Exception:
        logger.exception(f"{name} failed", extra={"stage": name})
        stats.log(logger)
        raise
    else:
        stats.log(logger)

"""Paths, credentials and tunables.

Deliberately small. v1's config layer created directories as a side effect of
import, spread three different output-path conventions across the entry points,
and kept a per-source option zoo for sources that collected nothing. This module
does none of that: it resolves paths, reads credentials from the environment,
and has no side effects until you ask for one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from dotenv import load_dotenv

#: Repository root, resolved from this file's location.
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Every path the pipeline writes to, in one place."""

    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def frame(self) -> Path:
        """The sampling frame: journal collection listings."""
        return self.data / "frame"

    @property
    def raw(self) -> Path:
        """Archived raw API responses. Never derived from, only read."""
        return self.root / "corpus" / "raw"

    @property
    def files(self) -> Path:
        """Downloaded files, path structure preserved."""
        return self.root / "corpus" / "files"

    @property
    def registries(self) -> Path:
        return self.root / "registries"

    @property
    def tables(self) -> Path:
        """Parquet output."""
        return self.root / "build" / "tables"

    @property
    def logs(self) -> Path:
        return self.root / "build" / "logs"

    @property
    def release(self) -> Path:
        return self.root / "build" / "release"


PATHS = Paths()


@dataclass(frozen=True)
class RateLimit:
    """Politeness settings for a host.

    Defaults are deliberate. v1 retried on 404, backed off up to 310 s per
    permanently-forbidden file, and slept 5 s unconditionally after every
    download -- which is why its run never finished.
    """

    requests_per_second: float = 3.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60.0
    timeout_s: float = 60.0
    #: Retried with backoff. 404 and 403 are deliberately absent: they are
    #: answers, not failures.
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


@cache
def _load_env() -> None:
    load_dotenv(ROOT / ".env")


def credential(name: str) -> str | None:
    """Read a credential from the environment (``.env`` is loaded once).

    Returns ``None`` rather than raising, so callers can decide whether a
    missing credential is fatal. :func:`require_credential` is the strict form.
    """
    _load_env()
    value = os.environ.get(name)
    return value.strip() or None if value else None


def require_credential(name: str, why: str) -> str:
    """Read a credential, failing loudly with an actionable message.

    Args:
        name: Environment variable name.
        why: What breaks without it, included in the error.
    """
    value = credential(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. {why}\nAdd it to {ROOT / '.env'} as {name}=<token>."
        )
    return value


#: Harvard Dataverse wants X-Dataverse-key. v1 sent `Authorization: Bearer`,
#: which Dataverse ignores, so every metadata read was silently anonymous.
DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DATAVERSE_KEY_HEADER = "X-Dataverse-key"


def dataverse_headers() -> dict[str, str]:
    """Auth headers for Dataverse, empty if no token is configured."""
    token = credential("DATAVERSE_API_KEY")
    return {DATAVERSE_KEY_HEADER: token} if token else {}


def zenodo_headers() -> dict[str, str]:
    """Auth headers for Zenodo, empty if no token is configured.

    A token is not optional in practice: anonymous clients are capped at a
    page size of 25, and Zenodo says so explicitly in a 400.
    """
    token = credential("ZENODO_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


#: Extensions collected. Wider than what we parse: manifests and READMEs are
#: kept because v1 deleted every non-script file and destroyed the
#: author-declared dependency lists that make free validation possible.
SCRIPT_EXTENSIONS = frozenset(
    {
        ".r",
        ".rscript",
        ".rmd",
        ".qmd",
        ".rnw",
        ".py",
        ".ipynb",
        ".do",
        ".ado",
        ".mata",
        ".jl",
        ".m",
        ".sas",
        ".sps",
        ".sh",
        ".bash",
        ".sql",
        ".stan",
    }
)

MANIFEST_FILENAMES = frozenset(
    {
        "renv.lock",
        "packrat.lock",
        "description",
        "namespace",
        ".rprofile",
        "requirements.txt",
        "pyproject.toml",
        "environment.yml",
        "environment.yaml",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "setup.py",
        "setup.cfg",
        "project.toml",
        "manifest.toml",
        "requires.txt",
    }
)

ARCHIVE_EXTENSIONS = frozenset({".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar"})

#: Path components that mark a vendored library tree rather than research code.
VENDOR_PATH_MARKERS = frozenset(
    {
        "renv",
        "packrat",
        ".checkpoint",
        "checkpoint",
        "site-packages",
        "dist-packages",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".julia",
        "vendor",
        "site-library",
        "__macosx",
    }
)

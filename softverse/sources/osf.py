"""OSF collection, with recursive folder traversal and a daily budget.

Hand-rolled rather than using ``osfclient``, and the reason is recorded here as
the CLAUDE.md rule asks: ``osfclient`` is v0.0.5, last released in 2021, and
does not cover the two things that actually matter for this job -- recursion and
request budgeting.

**Recursion is the defect that makes v1's OSF numbers meaningless.** Its
collector listed each storage provider's *root* and stopped, skipping anything
whose ``kind`` was not ``file``. Any project keeping code in ``analysis/`` or
``code/`` -- which is most of them -- yielded nothing and was recorded as
``no_scripts``: a false negative indistinguishable from a project with no code.
It collected 101 files in total.

**The budget is a hard constraint, not a nuisance.** OSF allows 10,000 requests
per day authenticated (100/hour anonymous), and every folder level costs a
request. A real crawl therefore spans days by construction, so the collector
tracks its own spend, stops cleanly when exhausted, and resumes -- rather than
running into a wall of 429s and recording them as failures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.acquire.state import DatasetRecord, Ledger, atomic_write_bytes
from softverse.logging_setup import get_logger
from softverse.model.enums import CollectionState, Source
from softverse.sources.dataverse import is_wanted

logger = get_logger(__name__)

OSF_API = "https://api.osf.io/v2"

#: Authenticated daily ceiling. Anonymous is 100/hour, which is unusable here.
DEFAULT_DAILY_BUDGET = 10_000

#: Depth cap on folder recursion. Deep trees are almost always vendored
#: libraries rather than analysis code, and the hygiene rules exclude those
#: anyway -- so paying requests to descend into them is waste.
MAX_DEPTH = 6


class BudgetExhausted(Exception):
    """The daily request allowance is spent. Not an error -- a stopping point."""


@dataclass
class Budget:
    """Counts requests so the collector stops before the API refuses it."""

    limit: int = DEFAULT_DAILY_BUDGET
    spent: int = 0

    def charge(self, n: int = 1) -> None:
        self.spent += n
        if self.spent >= self.limit:
            raise BudgetExhausted(f"spent {self.spent} of {self.limit} daily requests")

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)


@dataclass
class OSFFile:
    name: str
    path: str
    size: int
    download_url: str
    md5: str | None = None


@dataclass
class OSFNode:
    node_id: str
    title: str
    date_created: str | None
    doi: str | None = None
    files: list[OSFFile] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        if self.date_created and self.date_created[:4].isdigit():
            return int(self.date_created[:4])
        return None


def _get_json(client: PoliteClient, url: str, budget: Budget, **kwargs) -> dict | None:
    budget.charge()
    outcome = client.get(url, **kwargs)
    if not outcome.ok or outcome.content is None:
        logger.warning(
            "osf request failed",
            extra={"url": url[:120], "status": outcome.status, "err": outcome.error},
        )
        return None
    try:
        return json.loads(outcome.content)
    except json.JSONDecodeError:
        return None


def walk_files(
    client: PoliteClient,
    files_url: str,
    budget: Budget,
    depth: int = 0,
) -> list[OSFFile]:
    """List every file under ``files_url``, descending into folders.

    This is the part v1 was missing entirely. A folder's ``kind`` is
    ``"folder"`` and its contents live behind its own ``relationships.files``
    link, so reaching them costs one request per folder -- which is exactly why
    the budget exists.
    """
    if depth > MAX_DEPTH:
        return []

    payload = _get_json(client, files_url, budget)
    if payload is None:
        return []

    found: list[OSFFile] = []
    for entry in payload.get("data", []):
        attrs = entry.get("attributes", {})
        kind = attrs.get("kind")
        if kind == "file":
            name = attrs.get("name") or ""
            if not is_wanted(name):
                continue
            found.append(
                OSFFile(
                    name=name,
                    path=(attrs.get("materialized_path") or f"/{name}").lstrip("/"),
                    size=attrs.get("size") or 0,
                    download_url=(entry.get("links") or {}).get("download", ""),
                    md5=((attrs.get("extra") or {}).get("hashes") or {}).get("md5"),
                )
            )
        elif kind == "folder":
            nested = (
                ((entry.get("relationships") or {}).get("files") or {})
                .get("links", {})
                .get("related", {})
            )
            href = nested.get("href") if isinstance(nested, dict) else nested
            if href:
                found.extend(walk_files(client, href, budget, depth + 1))

    # Pagination within a single directory listing.
    next_url = (payload.get("links") or {}).get("next")
    if next_url:
        found.extend(walk_files(client, next_url, budget, depth))
    return found


def node_files(client: PoliteClient, node_id: str, budget: Budget) -> list[OSFFile]:
    """Every wanted file in a node, across all its storage providers."""
    providers = _get_json(client, f"{OSF_API}/nodes/{node_id}/files/", budget)
    if providers is None:
        return []
    out: list[OSFFile] = []
    for provider in providers.get("data", []):
        related = (
            ((provider.get("relationships") or {}).get("files") or {})
            .get("links", {})
            .get("related", {})
        )
        href = related.get("href") if isinstance(related, dict) else related
        if href:
            out.extend(walk_files(client, href, budget))
    return out


def collect_node(
    client: PoliteClient,
    node: OSFNode,
    files_root: Path,
    budget: Budget,
) -> tuple[DatasetRecord, list[dict]]:
    """Collect one node's code files."""
    doi = node.doi or f"osf:{node.node_id}"
    state = DatasetRecord(dataset_doi=doi, state=CollectionState.FAILED.value)
    rows: list[dict] = []

    files = node_files(client, node.node_id, budget)
    state.n_candidate = len(files)
    if not files:
        # With recursion in place this genuinely means no code, rather than
        # "the code was one directory down", which is what it meant in v1.
        state.state = CollectionState.NO_CANDIDATE_FILES.value
        return state, rows

    target = files_root / node.node_id
    now = datetime.now(tz=UTC)

    for item in files:
        if not item.download_url:
            state.n_failed += 1
            continue
        budget.charge()
        outcome = client.get(item.download_url, expect_content=False)
        if not outcome.ok or outcome.content is None:
            state.n_failed += 1
            state.error = outcome.error
            continue
        content = outcome.content
        path = target / item.path
        atomic_write_bytes(path, content)
        digest = hashlib.md5(content, usedforsecurity=False).hexdigest()
        rows.append(
            {
                "dataset_doi": doi,
                "source": str(Source.OSF),
                "collection_id": "osf",
                "dataverse_file_id": None,
                "container_file_id": None,
                "path_in_container": None,
                # The materialized path preserves the folder structure that
                # makes vendored libraries detectable later.
                "relative_path": item.path,
                "filename": item.name,
                "size_bytes": len(content),
                "md5_api": item.md5,
                "sha256_local": hashlib.sha256(content).hexdigest(),
                "md5_verified": item.md5 is not None and digest == item.md5,
                "restricted": False,
                "directory_label": str(Path(item.path).parent),
                "download_ts": now,
                "local_path": str(path),
                "deposit_year": node.year,
            }
        )
        state.n_fetched += 1
        state.bytes_fetched += len(content)

    state.state = (
        CollectionState.PARTIAL.value
        if state.n_failed
        else CollectionState.COMPLETE.value
    )
    return state, rows


def collect(
    nodes: list[OSFNode],
    files_root: Path,
    ledger: Ledger,
    client: PoliteClient,
    budget: Budget | None = None,
) -> list[dict]:
    """Collect nodes until the list is done or the daily budget runs out.

    Stopping on an exhausted budget is a normal outcome. The ledger means the
    next run picks up exactly where this one stopped.
    """
    budget = budget or Budget()
    rows: list[dict] = []
    todo = [n for n in nodes if ledger.should_process(n.doi or f"osf:{n.node_id}")]
    logger.info(
        "osf collection starting",
        extra={"nodes": len(nodes), "todo": len(todo), "budget": budget.remaining},
    )
    for index, node in enumerate(todo, 1):
        try:
            state, node_rows = collect_node(client, node, files_root, budget)
        except BudgetExhausted as exc:
            logger.warning(
                "daily budget exhausted; stopping cleanly",
                extra={"done": index - 1, "of": len(todo), "detail": str(exc)},
            )
            break
        except Exception as exc:  # noqa: BLE001 - one bad node must not end the run
            logger.exception("osf node failed", extra={"node": node.node_id})
            state = DatasetRecord(
                dataset_doi=node.doi or f"osf:{node.node_id}",
                state=CollectionState.FAILED.value,
                error=f"{type(exc).__name__}: {exc}",
            )
            node_rows = []
        ledger.finish(state)
        rows.extend(node_rows)
        if index % 25 == 0:
            logger.info(
                "osf progress",
                extra={"done": index, "of": len(todo), "budget_left": budget.remaining},
            )
    return rows

"""The sampling frame: which collections are in scope, and why.

A corpus without a stated inclusion rule is a convenience sample, and a
convenience sample cannot carry a journal-year trend claim however carefully
the rest of the pipeline is built. So the frame is a versioned artifact with a
reason attached to every row, not a list embedded in a script.

Two inclusion rules, both citable:

1. **DCAS** -- the `Data and Code Availability Standard
   <https://github.com/social-science-data-editors/DCAS>`_ journal list,
   maintained by the Social Science Data Editors: journals whose data-and-code
   policy the editors *actively verify*. This is the difference between
   "journals we happened to find" and a criterion a referee can check. Mostly
   economics and finance.
2. **Harvard Dataverse journal collections** -- 75 collections, mostly political
   science and adjacent fields, and the frame the project already used.

They dovetail rather than overlap: JPE, RFS, QJE and RESTAT reach us through
Dataverse, while Econometrica, RESTUD, the Economic Journal and JEEA reach us
through Zenodo.

**Every collection is verified before it enters the frame.** Zenodo does not
reject an unknown ``communities=`` filter -- it ignores it and returns all
7.1 million records -- so a mistyped slug would silently substitute the entire
archive for one journal. Guessing slugs is also unreliable in a way worth
recording: the convention is mostly ``<journal>-replication`` but JEEA uses an
underscore (``jeea_replication``), and ``restud``, ``aeaje``, ``jeea-replication``
and ``cje-replication`` all look plausible and do not exist.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from softverse.acquire.http import PoliteClient
from softverse.logging_setup import get_logger
from softverse.model.enums import CollectionKind, Source
from softverse.sources.zenodo import UnknownCommunity, verify_community

logger = get_logger(__name__)

DCAS_JOURNALS_URL = (
    "https://raw.githubusercontent.com/"
    "social-science-data-editors/DCAS/master/_data/journals.csv"
)


@dataclass
class FrameRow:
    """One collection in the frame, with the reason it is there."""

    collection_id: str
    source: str
    kind: str
    journal_name: str
    discipline: str
    inclusion_reason: str
    verified_at: str | None = None
    n_records_declared: int | None = None
    notes: str = ""


#: Curated journal -> Zenodo community mapping. Curated because no
#: machine-readable version exists; every slug here was verified against the
#: live API, and the counts are what it reported.
ZENODO_JOURNAL_COMMUNITIES: list[tuple[str, str, str]] = [
    # (community slug, journal(s), discipline)
    (
        "es-replication-repository",
        "Econometrica; Quantitative Economics; Theoretical Economics",
        "economics",
    ),
    ("restud-replication", "Review of Economic Studies", "economics"),
    ("ej-replication-repository", "Economic Journal", "economics"),
    ("ectj-replication-repository", "Econometrics Journal", "economics"),
    ("jeea_replication", "Journal of the European Economic Association", "economics"),
    (
        "labordynamicsinstitute",
        "AEA Data Editor's office (multiple AEA titles)",
        "economics",
    ),
]

#: DCAS journals whose replication material we have **not** located, recorded
#: so the coverage gap is a row rather than a silence. The AEA titles are on
#: openICPSR, which is deferred pending its terms of use.
UNLOCATED: list[tuple[str, str]] = [
    ("American Economic Review", "openICPSR AEA Data and Code Repository"),
    ("American Economic Review: Insights", "openICPSR"),
    ("American Economic Journal: Applied Economics", "openICPSR"),
    ("American Economic Journal: Economic Policy", "openICPSR"),
    ("American Economic Journal: Macroeconomics", "openICPSR"),
    ("American Economic Journal: Microeconomics", "openICPSR"),
    ("Journal of Economic Literature", "openICPSR"),
    ("Journal of Economic Perspectives", "openICPSR"),
    ("Canadian Journal of Economics", "not located"),
    ("Economic Inquiry", "not located"),
]

#: DCAS journals already covered by the Dataverse side of the frame, recorded
#: so the two halves are visibly reconciled rather than assumed disjoint.
DCAS_VIA_DATAVERSE = {
    "Journal of Political Economy": ["JPE", "JPE_Macroeconomics", "JPE_Microeconomics"],
    "Review of Financial Studies": ["rfs"],
}


def fetch_dcas_journals(client: PoliteClient) -> list[str]:
    """The DCAS journal list, pinned by the caller as a dated snapshot."""
    outcome = client.get(DCAS_JOURNALS_URL)
    if not outcome.ok or outcome.content is None:
        raise RuntimeError(f"could not fetch DCAS journals: {outcome.error}")
    rows = csv.DictReader(io.StringIO(outcome.content.decode("utf-8")))
    return [r["name"].strip() for r in rows if r.get("name")]


def dataverse_rows(frame_dir: Path) -> list[FrameRow]:
    """Frame rows for the Harvard Dataverse journal collections."""
    rows: list[FrameRow] = []
    for path in sorted(frame_dir.glob("*_datasets.csv")):
        alias = path.name.replace("_datasets.csv", "")
        n = max(0, sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1)
        rows.append(
            FrameRow(
                collection_id=alias,
                source=str(Source.DATAVERSE),
                kind=str(CollectionKind.JOURNAL),
                journal_name=alias,
                discipline="social science",
                inclusion_reason="dataverse_journal_collection",
                n_records_declared=n,
            )
        )
    return rows


def zenodo_rows(client: PoliteClient) -> list[FrameRow]:
    """Frame rows for verified Zenodo journal communities.

    A community that fails verification is **excluded and logged**, never
    included on the assumption that it is probably fine.
    """
    now = datetime.now(tz=UTC).isoformat()
    rows: list[FrameRow] = []
    for slug, journal, discipline in ZENODO_JOURNAL_COMMUNITIES:
        try:
            n = verify_community(client, slug)
        except UnknownCommunity as exc:
            logger.error(
                "excluded from frame: community failed verification",
                extra={"slug": slug, "err": str(exc)},
            )
            continue
        rows.append(
            FrameRow(
                collection_id=slug,
                source=str(Source.ZENODO),
                kind=str(CollectionKind.JOURNAL),
                journal_name=journal,
                discipline=discipline,
                inclusion_reason="dcas",
                verified_at=now,
                n_records_declared=n,
            )
        )
    return rows


def unlocated_rows() -> list[FrameRow]:
    """DCAS journals we have not located, as explicit rows."""
    return [
        FrameRow(
            collection_id="",
            source="",
            kind="",
            journal_name=journal,
            discipline="economics",
            inclusion_reason="dcas",
            notes=f"unlocated: {where}",
        )
        for journal, where in UNLOCATED
    ]


def build_frame(client: PoliteClient, frame_dir: Path) -> list[FrameRow]:
    """Assemble the whole frame, verified."""
    rows = zenodo_rows(client) + dataverse_rows(frame_dir) + unlocated_rows()
    located = [r for r in rows if r.collection_id]
    logger.info(
        "frame built",
        extra={
            "collections": len(located),
            "unlocated": len(rows) - len(located),
            "declared_records": sum(r.n_records_declared or 0 for r in located),
        },
    )
    return rows


def write_frame(rows: list[FrameRow], path: Path) -> Path:
    """Write the frame as a small, human-readable CSV.

    Deliberately small and readable: someone who knows these journals should be
    able to read the whole thing and say "that one is in the wrong place."
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    logger.info("frame written", extra={"path": str(path), "rows": len(rows)})
    return path


def collections_rows(rows: list[FrameRow]) -> list[dict]:
    """Frame rows as `collections` table rows, for the data model."""
    return [
        {
            "collection_id": r.collection_id,
            "source": r.source,
            "kind": r.kind,
            "collection_name": r.journal_name,
            "collection_url": None,
            "journal_id": r.collection_id or None,
        }
        for r in rows
        if r.collection_id
    ]

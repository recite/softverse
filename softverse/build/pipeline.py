"""Turn a corpus of files into the mention, file and dataset tables.

Everything here is a reconciliation. At each stage the parts must account for
the whole, and the build fails loudly if they do not. That single discipline is
what v1 lacked: between its source archive and its published CSV, 26,681 files
and 4,189 repositories disappeared with nothing to notice it, and its final
``groupby`` dropped 71,069 of 71,069 rows on a null key.

The tables carry a **corpus completeness** stamp. A number computed from a
partial corpus is a diagnostic, not a finding, and the only reliable way to stop
one being mistaken for the other later is to make the distinction a column.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from softverse import EXTRACTOR_VERSION
from softverse.corpus.hygiene import classify, cross_dataset_hashes, sha256_of
from softverse.detect.dispatch import extract_file, language_of
from softverse.stata.lexer import local_programs as stata_local_programs
from softverse.logging_setup import get_logger, stage
from softverse.model.enums import (
    ANALYZABLE_STATUSES,
    NON_USE_CONSTRUCTS,
    Language,
    ParseStatus,
    Resolution,
)
from softverse.model.io import reconcile
from softverse.registries.resolve import normalize

logger = get_logger(__name__)


@dataclass
class CorpusFile:
    """One file on disk, with the deposit it belongs to."""

    path: Path
    dataset_doi: str
    collection_id: str
    source: str
    relative_path: str
    deposit_year: int | None = None
    #: Identifies the exact deposit *version* the file came from. Defaults to
    #: the DOI, which is correct for Zenodo (records are immutable versions)
    #: and for any source without a separate version concept. Never null: it is
    #: a join key, and the schema refuses nulls in those.
    dataset_version_uid: str | None = None

    @property
    def version_uid(self) -> str:
        return self.dataset_version_uid or self.dataset_doi


@dataclass
class BuildResult:
    files: list[dict] = field(default_factory=list)
    mentions: list[dict] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    unknown: Counter = field(default_factory=Counter)

    def reconcile_files(self) -> None:
        """Assert the file dispositions account for every file seen."""
        parts = {
            k: v
            for k, v in self.counts.items()
            if k.startswith("files_") and k != "files_total"
        }
        reconcile(self.counts["files_total"], parts, "files")


def _year_of(doi_to_year: dict[str, int], doi: str) -> int | None:
    return doi_to_year.get(doi)


def build(
    corpus: Iterable[CorpusFile],
    registry,
    *,
    ssc_shipped: frozenset[str] = frozenset(),
    registry_lock_id: str = "unpinned",
    corpus_completeness: float = 1.0,
) -> BuildResult:
    """Extract and resolve every file, producing rows for `files` and `mentions`.

    Args:
        corpus: The files to process.
        registry: A :class:`softverse.registries.resolve.Registry`.
        ssc_shipped: Basenames shipped by SSC packages, for the vendored-ado rule.
        registry_lock_id: Stamped on every mention so a row names its instrument.
        corpus_completeness: Share of the intended corpus present. Below 1.0 the
            output is a diagnostic, not a result, and the stamp says so.
    """
    result = BuildResult()
    corpus = list(corpus)

    with stage("build", logger) as stats:
        # Pass 1: hash everything, so cross-deposit duplicates are detectable.
        # This needs the whole corpus before any verdict can be given, which is
        # the one genuine barrier in the pipeline.
        hashes: dict[Path, str] = {}
        for item in corpus:
            try:
                hashes[item.path] = sha256_of(item.path)
            except OSError:
                continue
        shared = frozenset(
            cross_dataset_hashes(
                (item.dataset_doi, hashes[item.path])
                for item in corpus
                if item.path in hashes
            )
        )
        stats.incr("shared_hashes", len(shared))
        logger.info(
            "cross-deposit duplicates identified",
            extra={"n_hashes": len(shared)},
        )

        siblings_by_dataset: dict[str, set[str]] = defaultdict(set)
        for item in corpus:
            siblings_by_dataset[item.dataset_doi].add(item.relative_path)

        # Stata programs are defined in one file and called from another, so
        # `program define` has to be collected across the whole deposit before
        # anything is resolved. Scanning only the current file left an author's
        # own helpers unresolved and indistinguishable from unknown packages:
        # wins_top1 (102 mentions), testgood (76) and preliminaries (97) all sat
        # near the top of the unknown list purely because of this.
        #
        # Only from files that are the *author's*. A `program define` inside
        # somebody else's library is not a local helper, and reading them all
        # inverted the rule this pass exists to enforce: replication packages
        # routinely ship `reghdfe.ado` so their code runs without installing
        # anything, that file defines `program reghdfe`, and so every call to
        # `reghdfe` in the author's own scripts resolved to their own program
        # and the package went uncounted. 21,004 mentions across 110 deposits
        # and 279 distinct packages, biased hard toward the packages popular
        # enough to be worth vendoring -- `reghdfe` lost 43 deposits of 299,
        # `gegen` 8 of 36. The hygiene rules already knew these files were
        # vendored; this pass simply never asked.
        local_programs_by_dataset: dict[str, set[str]] = defaultdict(set)
        for item in corpus:
            if item.path.suffix.lower() not in {".do", ".ado"}:
                continue
            digest = hashes.get(item.path)
            if digest is None:
                continue
            if classify(
                item.relative_path,
                digest,
                siblings=siblings_by_dataset[item.dataset_doi],
                ssc_shipped=ssc_shipped,
                shared_hashes=shared,
            ).is_vendored:
                continue
            try:
                text = item.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            local_programs_by_dataset[item.dataset_doi] |= {
                name.lower() for name in stata_local_programs(text)
            }
        stats.incr(
            "local_programs",
            sum(len(v) for v in local_programs_by_dataset.values()),
        )

        seen_by_dataset: dict[str, dict[str, str]] = defaultdict(dict)
        now = datetime.now(tz=UTC)

        for item in corpus:
            result.counts["files_total"] += 1
            digest = hashes.get(item.path)
            if digest is None:
                result.counts["files_unreadable"] += 1
                continue

            verdict = classify(
                item.relative_path,
                digest,
                siblings=siblings_by_dataset[item.dataset_doi],
                ssc_shipped=ssc_shipped,
                shared_hashes=shared,
                seen_in_dataset=seen_by_dataset[item.dataset_doi],
            )
            file_uid = uuid.uuid5(
                uuid.NAMESPACE_URL, f"{item.dataset_doi}/{item.relative_path}"
            ).hex

            if verdict.is_vendored:
                result.counts["files_vendored"] += 1
                status, mentions, language = ParseStatus.SKIPPED_VENDORED, [], language_of(item.path)
            elif verdict.duplicate_of:
                result.counts["files_duplicate"] += 1
                status, mentions, language = ParseStatus.SKIPPED_DUPLICATE, [], language_of(item.path)
            else:
                seen_by_dataset[item.dataset_doi][digest] = file_uid
                extraction, _decoded, language = extract_file(item.path)
                status = extraction.report.status if extraction.report else ParseStatus.NOT_ANALYZED
                mentions = extraction.mentions
                if status in ANALYZABLE_STATUSES:
                    result.counts["files_analyzed"] += 1
                elif status is ParseStatus.UNSUPPORTED_LANGUAGE:
                    result.counts["files_unsupported"] += 1
                elif status in {ParseStatus.SKIPPED_EMPTY, ParseStatus.SKIPPED_BINARY}:
                    result.counts["files_skipped"] += 1
                else:
                    result.counts["files_unparseable"] += 1

            deposit_locals = frozenset(
                local_programs_by_dataset.get(item.dataset_doi, ())
            )
            for mention in mentions:
                # A literate document holds chunks in several languages, so the
                # mention's own language wins over the file's when it is set.
                # Resolving an .Rmd's Python chunk against `rmarkdown` would
                # send every one of its mentions to no registry at all.
                mention_language = mention.language or language
                resolved = registry.resolve(
                    mention.raw_name,
                    mention_language,
                    deposit_locals,
                    construct=mention.construct,
                )
                if resolved.resolution is Resolution.UNKNOWN:
                    result.unknown[(mention.raw_name, str(language))] += 1
                result.mentions.append(
                    {
                        "mention_uid": uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{file_uid}/{mention.byte_start}/{mention.raw_name}",
                        ).hex,
                        "file_uid": file_uid,
                        "dataset_doi": item.dataset_doi,
                        "source": item.source,
                        "collection_id": item.collection_id,
                        "deposit_year": item.deposit_year,
                        "language": str(mention_language),
                        "construct": str(mention.construct),
                        "raw_name": mention.raw_name,
                        "normalized_name": normalize(mention.raw_name, mention_language),
                        "resolved_package": resolved.package,
                        "ecosystem": str(resolved.ecosystem) if resolved.ecosystem else None,
                        "resolution": str(resolved.resolution),
                        "line": mention.line,
                        "col": mention.col,
                        "byte_start": mention.byte_start,
                        "byte_end": mention.byte_end,
                        "snippet": mention.snippet,
                        "cell_index": mention.cell_index,
                        "chunk_label": mention.chunk_label,
                        "is_conditional": mention.is_conditional,
                        "is_dynamic": mention.is_dynamic,
                        "is_reachable": None,
                        "extractor_version": EXTRACTOR_VERSION,
                        "grammar_version": None,
                        "registry_lock_id": registry_lock_id,
                    }
                )

            result.files.append(
                {
                    "file_uid": file_uid,
                    "dataset_version_uid": item.version_uid,
                    "dataset_doi": item.dataset_doi,
                    "source": item.source,
                    "collection_id": item.collection_id,
                    "relative_path": item.relative_path,
                    "filename": item.path.name,
                    "extension": item.path.suffix.lower(),
                    "language": str(language),
                    "sha256_local": digest,
                    "is_vendored": verdict.is_vendored,
                    "vendor_rule": str(verdict.rule) if verdict.rule else None,
                    "duplicate_of": verdict.duplicate_of,
                    "parse_status": str(status),
                    "n_mentions": len(mentions),
                    "in_analysis_set": verdict.in_analysis_set
                    and status in ANALYZABLE_STATUSES,
                    "download_ts": now,
                    "local_path": str(item.path),
                }
            )

        stats.counts.update(result.counts)

    result.reconcile_files()
    logger.info(
        "build complete",
        extra={
            "mentions": len(result.mentions),
            "unknown_names": len(result.unknown),
            "corpus_completeness": corpus_completeness,
            "is_provisional": corpus_completeness < 1.0,
        },
    )
    return result


def dataset_packages(mentions: list[dict], *, use_only: bool = True) -> list[dict]:
    """Collapse mentions to the headline unit: one row per (dataset, package).

    Dataset-level presence/absence, not mention counts. Empirically necessary --
    290 deposits hold all 16,253 MATLAB files and one holds 3,607 vendored
    Python files, so any mention-weighted statistic is a statistic about a
    handful of deposits. It also caps any vendored file that survives hygiene at
    a contribution of +1.

    ``use_only`` drops provisioning and inquiry constructs (`install.packages`,
    `ssc install`, `findit`), which indicate intent rather than use.
    """
    # Filtering on "has a resolved package" is not enough, and getting this
    # wrong reproduces the v1 tally exactly. The resolver assigns a *package
    # name* to stdlib and builtin mentions too -- `os` resolves to `os` -- so a
    # presence check lets every one of them through. v1's published table had
    # os 101, sys 66, grid 268 and parallel 115 for precisely this reason, and
    # the first run here reproduced it (os 52, sys 42, time 28) until this
    # filtered on the *resolution* rather than the name.
    countable = {
        str(Resolution.KNOWN_CURRENT),
        str(Resolution.KNOWN_ARCHIVED),
    }
    non_use = {str(c) for c in NON_USE_CONSTRUCTS}

    grouped: dict[tuple, dict] = {}
    for mention in mentions:
        if not mention["resolved_package"]:
            continue
        if mention["resolution"] not in countable:
            continue
        if use_only and mention["construct"] in non_use:
            continue
        key = (
            mention["dataset_doi"],
            mention["language"],
            mention["resolved_package"],
        )
        row = grouped.setdefault(
            key,
            {
                "dataset_doi": mention["dataset_doi"],
                "collection_id": mention["collection_id"],
                "year": mention["deposit_year"],
                "language": mention["language"],
                "package": mention["resolved_package"],
                "ecosystem": mention["ecosystem"],
                "n_files": 0,
                "n_mentions": 0,
                "any_reachable": None,
                "any_use_construct": True,
                "_files": set(),
            },
        )
        row["n_mentions"] += 1
        row["_files"].add(mention["file_uid"])
    for row in grouped.values():
        row["n_files"] = len(row.pop("_files"))
    return list(grouped.values())


def language_presence(files: list[dict]) -> dict[str, int]:
    """Deposits containing at least one file of each language.

    A defensible claim even for MATLAB and SAS, where we deliberately make no
    package-level attribution: "N% of deposits contain MATLAB" is verifiable,
    "they use toolbox X" would not be.
    """
    by_language: dict[str, set[str]] = defaultdict(set)
    for row in files:
        if row["language"] and row["language"] != str(Language.UNKNOWN):
            by_language[row["language"]].add(row["dataset_doi"])
    return {lang: len(dois) for lang, dois in sorted(by_language.items())}

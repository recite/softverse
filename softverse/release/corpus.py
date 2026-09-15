"""Assemble the public corpus release from a finished tally.

Every table is rectangular Parquet, one config per table on Hugging Face. The
layout follows BigCode's The Stack v2 where the two meet -- `path`,
`language`, `extension`, `is_vendor`, `length_bytes`, a per-row license, and
file content kept apart from file metadata and keyed by its hash -- so the
usual tooling reads it without adapting.

Two rules decide what leaves the building:

- **Content follows the license.** A file's text, and a mention's snippet, is
  published only when the deposit's license allows redistribution
  (:mod:`softverse.release.licenses`). Everything derived -- which packages a
  file loads, at which version, on which interpreter -- is published for every
  deposit, because those are facts about the code rather than the code.
- **Nothing is written that the checks did not pass.** :func:`check` reads the
  written tables back and recomputes what they claim by routes that share no
  code with :func:`build`, and the script refuses to keep a release that
  fails.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from softverse.detect.dispatch import decode
from softverse.logging_setup import get_logger
from softverse.model.enums import NON_USE_CONSTRUCTS, Resolution
from softverse.release.licenses import classify

logger = get_logger(__name__)

#: Files larger than this keep their metadata row but not their text, and are
#: flagged `too_large`. The Stack uses the same cut; past it a "code" file in
#: this corpus is almost always generated -- the largest are 87 MB MATLAB
#: model dumps.
CONTENT_CAP_BYTES = 1024 * 1024

#: Target size of one `contents` shard, before compression.
SHARD_BYTES = 500 * 1024 * 1024

COUNTABLE = (str(Resolution.KNOWN_CURRENT), str(Resolution.KNOWN_ARCHIVED))
NON_USE = tuple(sorted(str(c) for c in NON_USE_CONSTRUCTS))

#: Where a pinned version came from, by the construct that stated it.
_PIN_SOURCE = {
    "install": "install_call",
    "shell_install": "pip_or_conda_install",
}


@dataclass(frozen=True)
class Inputs:
    """Everything the release reads."""

    tally: Path
    frame: Path
    dataverse_corpus: Path
    zenodo_corpus: Path


def _size(path: str | None) -> int | None:
    try:
        return Path(path).stat().st_size if path else None
    except OSError:
        return None


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def licenses(inputs: Inputs) -> dict[str, tuple[str, str | None, bool]]:
    """Deposit DOI -> (license id, terms-of-use text, redistributable).

    Args:
        inputs: Where the raw metadata lives.

    Returns:
        One entry per deposit either repository reported a license for.
    """
    out: dict[str, tuple[str, str | None, bool]] = {}
    for path in sorted((inputs.dataverse_corpus / "raw").glob("*.json.gz")):
        with gzip.open(path) as handle:
            payload = json.load(handle)
        payload = payload.get("data", payload)
        reported = payload.get("license")
        name = reported.get("name") if isinstance(reported, dict) else reported
        terms = (payload.get("termsOfUse") or "").strip() or None
        result = classify(name, terms)
        doi = f"doi:10.7910/DVN/{path.name.removesuffix('.json.gz')}"
        out[doi] = (result.license_id, terms, result.redistributable)
    deposits_csv = inputs.zenodo_corpus / "deposits.csv"
    if deposits_csv.exists():
        with deposits_csv.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                result = classify(row.get("license"))
                out[row["dataset_doi"]] = (
                    result.license_id,
                    None,
                    result.redistributable,
                )
    return out


def _ledger(path: Path, doi_of) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[doi_of(record["dataset_doi"])] = record
    return records


def _deposit_rows(inputs: Inputs, license_of: dict) -> list[dict]:
    journals: dict[str, dict] = {}
    with (inputs.frame / "frame.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["collection_id"]:
                journals[row["collection_id"]] = row

    dataverse_ledger = _ledger(inputs.dataverse_corpus / "ledger.jsonl", lambda d: d)
    zenodo_ledger = _ledger(
        inputs.zenodo_corpus / "ledger.jsonl",
        lambda d: f"zenodo:{d.rsplit('.', 1)[-1]}",
    )

    listed: list[tuple[str, str, str, str, str]] = []
    with (inputs.frame / "dataverse_deposits.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ident = row["identifier"]
            listed.append(
                (
                    f"doi:10.7910/DVN/{ident.rsplit('/', 1)[-1]}",
                    "dataverse",
                    row["collection_id"],
                    row["publication_date"],
                    f"https://doi.org/10.7910/{ident}",
                )
            )
    deposits_csv = inputs.zenodo_corpus / "deposits.csv"
    if deposits_csv.exists():
        with deposits_csv.open(encoding="utf-8") as handle:
            listed.extend(
                (
                    row["dataset_doi"],
                    "zenodo",
                    row["collection_id"],
                    row["publication_date"],
                    f"https://zenodo.org/records/{row['record_id']}",
                )
                for row in csv.DictReader(handle)
            )

    rows = []
    for doi, source, collection, published, url in listed:
        ledger = (dataverse_ledger if source == "dataverse" else zenodo_ledger).get(
            doi, {}
        )
        skipped = ledger.get("skipped_archives") or []
        license_id, terms, redistributable = license_of.get(doi, ("none", None, False))
        journal = journals.get(collection, {})
        rows.append(
            {
                "dataset_doi": doi,
                "source": source,
                "record_url": url,
                "journal_id": collection,
                "journal_name": journal.get("journal_name") or None,
                "discipline": journal.get("discipline") or None,
                "publication_date": published or None,
                "deposit_year": int(published[:4]) if published[:4].isdigit() else None,
                "license_id": license_id,
                "license_terms": terms,
                "content_redistributable": redistributable,
                "collection_state": ledger.get("state"),
                "n_archives_skipped": len(skipped),
                "bytes_skipped": sum(int(s.get("size_bytes") or 0) for s in skipped),
            }
        )
    return rows


def build(inputs: Inputs, out: Path) -> dict[str, int]:
    """Write every release table under ``out``.

    Args:
        inputs: The tally, frame and corpus directories to read.
        out: The release directory, replaced if it exists.

    Returns:
        Row counts per table.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    tally = inputs.tally
    con = duckdb.connect()
    license_of = licenses(inputs)

    deposits = pa.Table.from_pylist(_deposit_rows(inputs, license_of))
    con.register("deposits_in", deposits)
    con.execute(f"CREATE VIEW files_raw AS SELECT * FROM '{tally / 'files.parquet'}'")
    # The tally does not record file sizes, so they are read off the disk,
    # which is also the copy `contents` is built from.
    paths = con.execute("SELECT file_uid, local_path FROM files_raw").fetchall()
    sizes = pa.table(
        {
            "file_uid": [uid for uid, _ in paths],
            "disk_bytes": [_size(path) for _, path in paths],
        }
    )
    con.register("sizes", sizes)
    con.execute(
        "CREATE VIEW files_in AS SELECT f.* REPLACE "
        "(coalesce(f.size_bytes, s.disk_bytes) AS size_bytes) "
        "FROM files_raw f LEFT JOIN sizes s USING (file_uid)"
    )
    con.execute(
        f"CREATE VIEW mentions_in AS SELECT * FROM '{tally / 'mentions.parquet'}'"
    )
    con.execute(
        "CREATE TABLE redistributable AS SELECT dataset_doi, "
        "content_redistributable FROM deposits_in"
    )

    def copy(query: str, name: str) -> None:
        con.execute(
            f"COPY ({query}) TO '{out / name}.parquet' (FORMAT PARQUET, "
            "COMPRESSION ZSTD)"
        )

    copy(
        """
        SELECT d.*,
               coalesce(f.n_files, 0) AS n_files,
               coalesce(f.n_files_analyzable, 0) AS n_files_analyzable,
               f.languages
        FROM deposits_in d
        LEFT JOIN (
            SELECT dataset_doi, count(*) AS n_files,
                   count(*) FILTER (WHERE in_analysis_set) AS n_files_analyzable,
                   list(DISTINCT language ORDER BY language)
                       FILTER (WHERE in_analysis_set) AS languages
            FROM files_in GROUP BY dataset_doi
        ) f USING (dataset_doi)
        ORDER BY dataset_doi
        """,
        "deposits",
    )

    copy(
        f"""
        SELECT f.file_uid, f.dataset_doi, f.source, f.relative_path AS path,
               f.filename, f.extension, f.language,
               f.size_bytes AS length_bytes, f.sha256_local AS sha256,
               regexp_extract(f.relative_path, '_archives/(.+?)_extracted/', 1)
                   AS container,
               f.is_vendored AS is_vendor, f.vendor_rule, f.duplicate_of,
               f.parse_status, f.in_analysis_set, f.n_mentions,
               f.size_bytes > {CONTENT_CAP_BYTES} AS too_large,
               p.packages,
               coalesce(r.content_redistributable, false)
                   AS content_redistributable
        FROM files_in f
        LEFT JOIN redistributable r USING (dataset_doi)
        LEFT JOIN (
            SELECT file_uid,
                   list(DISTINCT resolved_package ORDER BY resolved_package)
                       AS packages
            FROM mentions_in
            WHERE resolved_package IS NOT NULL
              AND resolution IN ({_sql_list(COUNTABLE)})
            GROUP BY file_uid
        ) p USING (file_uid)
        ORDER BY f.dataset_doi, path
        """,
        "files",
    )

    copy(
        """
        SELECT m.* REPLACE (
            CASE WHEN coalesce(r.content_redistributable, false)
                 THEN m.snippet END AS snippet
        )
        FROM mentions_in m LEFT JOIN redistributable r USING (dataset_doi)
        """,
        "mentions",
    )

    copy(
        f"""
        SELECT file_uid, any_value(dataset_doi) AS dataset_doi, language,
               resolved_package AS package, any_value(ecosystem) AS ecosystem,
               mode(resolution) AS resolution, count(*) AS n_mentions,
               list(DISTINCT construct ORDER BY construct) AS constructs,
               list(DISTINCT called_function ORDER BY called_function)
                   FILTER (WHERE called_function IS NOT NULL) AS functions,
               list(DISTINCT pinned_version ORDER BY pinned_version)
                   FILTER (WHERE pinned_version IS NOT NULL) AS pinned_versions,
               min(line) AS first_line,
               bool_or(construct NOT IN ({_sql_list(NON_USE)})) AS is_use
        FROM mentions_in
        WHERE resolved_package IS NOT NULL
        GROUP BY file_uid, language, resolved_package
        ORDER BY dataset_doi, file_uid, package
        """,
        "file_packages",
    )

    pin_source = " ".join(
        f"WHEN construct = '{c}' THEN '{s}'" for c, s in _PIN_SOURCE.items()
    )
    copy(
        f"""
        SELECT dataset_doi, source_file_uid, ecosystem, package,
               version_constraint AS version, manifest_kind AS version_source,
               dependency_role
        FROM '{tally / "declared_dependencies.parquet"}'
        WHERE version_constraint IS NOT NULL
        UNION ALL
        SELECT dataset_doi, file_uid, ecosystem,
               coalesce(resolved_package, raw_name), pinned_version,
               CASE {pin_source} ELSE construct END, 'install'
        FROM mentions_in WHERE pinned_version IS NOT NULL
        ORDER BY dataset_doi, package
        """,
        "package_versions",
    )

    copy(
        f"SELECT * FROM '{tally / 'environment_signals.parquet'}' "
        "ORDER BY dataset_doi, signal",
        "environment",
    )

    usage = f"'{tally / 'usage_by_package.parquet'}'"
    for (language,) in con.execute(
        f"SELECT DISTINCT language FROM {usage} ORDER BY 1"
    ).fetchall():
        copy(
            f"SELECT * EXCLUDE (language) FROM {usage} "
            f"WHERE language = '{language}' ORDER BY n_deposits DESC, package",
            f"tally_{language}",
        )
    for source, name in (
        ("usage_by_package_year", "tally_by_year"),
        ("usage_by_collection", "tally_by_journal"),
        ("usage_by_function", "tally_by_function"),
    ):
        copy(f"SELECT * FROM '{tally / (source + '.parquet')}'", name)

    n_contents = _write_contents(con, out)
    counts = {
        path.stem: pq.ParquetFile(path).metadata.num_rows
        for path in sorted(out.glob("*.parquet"))
    }
    counts["contents"] = n_contents
    (out / "README.md").write_text(dataset_card(out, counts), encoding="utf-8")
    return counts


#: Tables in the order a reader meets them, with what one row is.
CARD_TABLES = {
    "deposits": "one replication deposit: DOI, journal, year, license, gaps",
    "files": "one file in a deposit: path, language, sha256, packages it loads",
    "contents": "one distinct file text, keyed by sha256 (open licenses only)",
    "file_packages": "one package a file references, with functions and pins",
    "mentions": "one reference in code: construct, line, resolution",
    "package_versions": "one stated version: manifest or install call",
    "environment": "one R/Python/Stata/Julia version or OS a deposit states",
}


def dataset_card(out: Path, counts: dict[str, int]) -> str:
    """The Hugging Face dataset card, every number read from the tables.

    Args:
        out: The release directory, already written.
        counts: Row counts per table, from :func:`build`.

    Returns:
        The card's Markdown, with YAML front matter defining one config per
        table.
    """
    con = duckdb.connect()

    def rows(query: str) -> list[tuple]:
        return con.execute(query).fetchall()

    deposits = f"'{out / 'deposits.parquet'}'"
    files = f"'{out / 'files.parquet'}'"
    tallies = sorted(p.stem for p in out.glob("tally_*.parquet"))
    configs = [*CARD_TABLES, *tallies]
    yaml = [
        "---",
        "pretty_name: Softverse",
        "license: other",
        "license_name: mixed",
        "tags:",
        "- tabular",
        "- code",
        "configs:",
    ]
    for name in configs:
        path = "contents/*.parquet" if name == "contents" else f"{name}.parquet"
        yaml += [f"- config_name: {name}", f'  data_files: "{path}"']
        if name == "deposits":
            yaml.append("  default: true")
    yaml.append("---")

    by_source = rows(
        f"SELECT source, count(*), count(*) FILTER (WHERE n_files_analyzable > 0), "
        f"count(*) FILTER (WHERE content_redistributable) FROM {deposits} "
        "GROUP BY 1 ORDER BY 1"
    )
    licenses = rows(
        f"SELECT license_id, content_redistributable, count(*) FROM {deposits} "
        "GROUP BY 1, 2 ORDER BY 3 DESC"
    )
    (archive_only,) = rows(
        f"SELECT count(*) FROM {deposits} "
        "WHERE n_archives_skipped > 0 AND n_files_analyzable = 0"
    )[0]
    states = rows(
        f"SELECT collection_state, count(*) FROM {deposits} "
        "WHERE collection_state NOT IN ('complete', 'no_candidate_files') "
        "GROUP BY 1 ORDER BY 2 DESC"
    )
    (too_large,) = rows(f"SELECT count(*) FROM {files} WHERE too_large")[0]
    top = {
        name.removeprefix("tally_"): rows(
            f"SELECT package, n_deposits FROM '{out / (name + '.parquet')}' "
            "ORDER BY n_deposits DESC LIMIT 5"
        )
        for name in tallies
        if not name.startswith("tally_by_")
    }

    def cell(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def table(header: list[str], body: list[tuple]) -> str:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in body]
        return "\n".join(lines)

    body = f"""
# Softverse: software referenced in social science replication code

Which R, Python and Stata packages the code in published replication deposits
loads, at economics and political science journals whose data-and-code policy
an editor verifies. One row per deposit, file, file text, package reference,
stated version and environment signal, plus per-language tallies.

**A reference is not a run.** A row says deposited code names a package: in
`library()`, an import, a Stata command. It does not say the code executed, and
code an author kept out of the deposit cannot be seen.

## Tables

{table(["config", "rows", "one row is"], [(n, counts.get(n, 0), CARD_TABLES[n]) for n in CARD_TABLES])}

Per-language tallies (`{"`, `".join(tallies)}`) give, per package, the deposits
that use it and the denominator: deposits with analyzable code in that language.

## Coverage

{table(["source", "deposits", "with analyzable code", "text published"], by_source)}

Most-used packages:

{chr(10).join(f"- **{lang}**: " + ", ".join(f"{p} ({n:,})" for p, n in pkgs) for lang, pkgs in top.items())}

## Licenses

File text (`contents`) and code snippets (`mentions.snippet`) are published only
for deposits whose license allows redistribution. Everything derived -- which
packages a file loads, at which version -- is published for every deposit.
Custom terms of use and unrecognised identifiers are treated as not
redistributable. Each deposit's own license is in `deposits.license_id`; NC and
ND licenses are included and flagged there.

{table(["license_id", "text published", "deposits"], licenses)}

## Known gaps

- {archive_only:,} deposits hold code only inside tar, 7z or rar archives too
  large to download here; their files are not in the corpus.
- Deposits whose collection did not complete:
  {", ".join(f"{s} {n:,}" for s, n in states) or "none"}.
- {too_large:,} files over 1 MB keep their metadata row but not their text.
- The eight AEA journals, which deposit on openICPSR, are not included.
- A package vendored into a deposit (an `renv` library, a shipped `.ado`, a
  CRAN package's source tree) is marked `is_vendor` and not counted as used.

## Source

Built by [softverse](https://github.com/recite/softverse). Package lookup and
badges: <https://recite.github.io/softverse/>.
"""
    return "\n".join(yaml) + "\n" + body.lstrip("\n")


def _write_contents(con: duckdb.DuckDBPyConnection, out: Path) -> int:
    """One row per distinct file text from a redistributable deposit.

    Args:
        con: The connection holding the `files_in` and `redistributable` views.
        out: The release directory.

    Returns:
        How many distinct texts were written.
    """
    rows = con.execute(
        f"""
        SELECT f.sha256_local, any_value(f.local_path)
        FROM files_in f JOIN redistributable r USING (dataset_doi)
        WHERE r.content_redistributable AND f.sha256_local IS NOT NULL
          AND f.size_bytes <= {CONTENT_CAP_BYTES}
        GROUP BY f.sha256_local
        ORDER BY f.sha256_local
        """
    ).fetchall()
    schema = pa.schema(
        [
            ("sha256", pa.string()),
            ("content", pa.string()),
            ("src_encoding", pa.string()),
            ("length_bytes", pa.int64()),
            # Whether encoding `content` in `src_encoding` gives back the
            # original bytes. False where the encoding was a guess or a byte
            # order mark was dropped: the text is right to read, not to hash.
            ("exact", pa.bool_()),
        ]
    )
    directory = out / "contents"
    directory.mkdir()
    shard, buffer, buffered, written = 0, [], 0, 0

    def flush() -> None:
        nonlocal shard, buffer, buffered
        if buffer:
            pq.write_table(
                pa.Table.from_pylist(buffer, schema=schema),
                directory / f"contents-{shard:05d}.parquet",
                compression="zstd",
            )
            shard, buffer, buffered = shard + 1, [], 0

    for sha, local_path in rows:
        data = Path(local_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            logger.error("file changed since the tally", extra={"path": local_path})
            continue
        decoded = decode(data)
        try:
            exact = decoded.text.encode(decoded.encoding) == data
        except (UnicodeEncodeError, LookupError):
            exact = False
        buffer.append(
            {
                "sha256": sha,
                "content": decoded.text,
                "src_encoding": decoded.encoding,
                "length_bytes": len(data),
                "exact": exact,
            }
        )
        buffered += len(data)
        written += 1
        if buffered >= SHARD_BYTES:
            flush()
    flush()
    return written


def check(inputs: Inputs, out: Path) -> list[str]:
    """Recompute what the release claims, by routes that do not share its code.

    Args:
        inputs: The same inputs :func:`build` read.
        out: The release directory :func:`build` wrote.

    Returns:
        One message per failed check; empty when the release holds up.
    """
    con = duckdb.connect()
    problems: list[str] = []

    def one(query: str) -> tuple:
        row = con.execute(query).fetchone()
        if row is None:
            raise ValueError(f"no row from: {query}")
        return row

    files, mentions = f"'{out / 'files.parquet'}'", f"'{out / 'mentions.parquet'}'"
    fp, deposits = f"'{out / 'file_packages.parquet'}'", f"'{out / 'deposits.parquet'}'"
    contents = f"'{out / 'contents' / '*.parquet'}'"

    (declared,) = one(f"SELECT coalesce(sum(n_mentions), 0) FROM {files}")
    (counted,) = one(f"SELECT count(*) FROM {mentions}")
    if declared != counted:
        problems.append(f"files.n_mentions sums to {declared}, mentions has {counted}")

    (grouped,) = one(f"SELECT coalesce(sum(n_mentions), 0) FROM {fp}")
    (resolved,) = one(
        f"SELECT count(*) FROM {mentions} WHERE resolved_package IS NOT NULL"
    )
    if grouped != resolved:
        problems.append(
            f"file_packages sums to {grouped}, resolved mentions {resolved}"
        )

    # Content: every exact row re-derived to its hash, none from a closed
    # deposit, and none missing for an open one.
    shards = sorted((out / "contents").glob("*.parquet"))
    if not shards:
        problems.append("no contents written")
        return problems
    bad_hash = 0
    for shard in shards:
        table = pq.read_table(
            shard, columns=["sha256", "content", "src_encoding", "exact"]
        )
        for sha, text, encoding, exact in zip(
            table["sha256"].to_pylist(),
            table["content"].to_pylist(),
            table["src_encoding"].to_pylist(),
            table["exact"].to_pylist(),
            strict=True,
        ):
            if exact and hashlib.sha256(text.encode(encoding)).hexdigest() != sha:
                bad_hash += 1
    if bad_hash:
        problems.append(f"{bad_hash} exact contents rows do not hash to their sha256")

    (leaked,) = one(
        f"""SELECT count(*) FROM {contents} c WHERE NOT EXISTS (
              SELECT 1 FROM {files} f
              WHERE f.sha256 = c.sha256 AND f.content_redistributable)"""
    )
    if leaked:
        problems.append(f"{leaked} contents rows belong to no redistributable file")
    (snippets,) = one(
        f"""SELECT count(*) FROM {mentions} m JOIN {deposits} d USING (dataset_doi)
            WHERE NOT d.content_redistributable AND m.snippet IS NOT NULL"""
    )
    if snippets:
        problems.append(f"{snippets} snippets published from closed deposits")
    (missing,) = one(
        f"""SELECT count(*) FROM {files} f WHERE f.content_redistributable
              AND NOT f.too_large AND f.sha256 IS NOT NULL
              AND f.sha256 NOT IN (SELECT sha256 FROM {contents})"""
    )
    if missing:
        problems.append(f"{missing} redistributable files have no contents row")

    # Licenses, recounted from the raw metadata by string matching alone.
    raw = Counter()
    for path in (inputs.dataverse_corpus / "raw").glob("*.json.gz"):
        with gzip.open(path) as handle:
            payload = json.load(handle)
        payload = payload.get("data", payload)
        reported = payload.get("license")
        name = (reported.get("name") if isinstance(reported, dict) else reported) or ""
        raw["cc0" if re.fullmatch(r"CC0 1\.0", name) else "other"] += 1
    (published_cc0,) = one(
        f"SELECT count(*) FROM {deposits} "
        "WHERE source = 'dataverse' AND license_id = 'CC0-1.0'"
    )
    (licensed,) = one(
        f"SELECT count(*) FROM {deposits} "
        "WHERE source = 'dataverse' AND license_id <> 'none'"
    )
    if published_cc0 != raw["cc0"]:
        problems.append(
            f"deposits says {published_cc0} Dataverse CC0, raw metadata {raw['cc0']}"
        )
    if licensed != sum(raw.values()):
        problems.append(
            f"{licensed} Dataverse deposits carry a license, "
            f"{sum(raw.values())} have raw metadata"
        )

    # Tallies, re-aggregated from the published mentions.
    for tally in sorted(out.glob("tally_*.parquet")):
        language = tally.stem.removeprefix("tally_")
        if language.startswith("by_"):
            continue
        (mismatched,) = one(
            f"""
            WITH recount AS (
                SELECT resolved_package AS package,
                       count(DISTINCT dataset_doi) AS n
                FROM {mentions}
                WHERE language = '{language}'
                  AND resolution IN ({_sql_list(COUNTABLE)})
                  AND construct NOT IN ({_sql_list(NON_USE)})
                  AND resolved_package IS NOT NULL
                GROUP BY resolved_package
            )
            SELECT count(*) FROM '{tally}' t FULL JOIN recount r USING (package)
            WHERE t.n_deposits IS DISTINCT FROM r.n
            """
        )
        if mismatched:
            problems.append(f"tally_{language}: {mismatched} packages disagree")
    return problems

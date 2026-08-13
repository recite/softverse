"""Extract untrusted archives safely, preserving in-archive paths.

Every archive here was uploaded by a stranger. v1 had four separate extraction
implementations and six ``extractall`` calls with no member validation at all,
which is textbook zip-slip (CVE-2007-4559 for tar). It also flattened everything
to basenames -- 5 subdirectories survived across 12,054 deposits -- which
destroyed the one signal needed to tell a vendored ``renv/`` library from
research code, and appended ``_1``/``_2`` on collisions so re-runs silently
manufactured duplicate files.

This is the only extractor. It preserves paths, rejects hostile members, and
bounds the damage a malicious or merely enormous archive can do.
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Refuse an archive claiming to expand beyond this. A zip bomb is small on the
#: wire and unbounded on disk.
#:
#: **Calibrated against the corpus, not against an imagined attacker.** The
#: first limits (2 GB, 50,000 members) were picked for safety without checking
#: what real research archives look like, and they rejected 6 of the first 100
#: Zenodo deposits: economics replication packages declaring 3.0, 5.1, 6.2 and
#: 6.6 GB uncompressed, and one with 140,219 members. None was an attack; that
#: is simply what a replication package with a large panel dataset looks like.
#: A guard that excludes 6% of the corpus is not protecting the analysis, it is
#: biasing it -- and toward exactly the data-heavy work we least want to lose.
#:
#: These bounds still stop a real bomb (the classic 42.zip expands to petabytes)
#: while admitting genuine deposits. The guards that matter for *safety* --
#: zip-slip, symlink and device rejection -- are unchanged and have cost
#: nothing, because no legitimate archive needs them.
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3
MAX_MEMBERS = 500_000
#: Nested archives are extracted this many levels deep, then left alone.
MAX_NESTING = 3

#: Ceiling on what one archive may write to disk, across all nesting levels.
#:
#: The download cap bounds the *transfer*; this bounds the *expansion*, and they
#: are not the same constraint. A 2 GB deposit made of nested data archives can
#: write far more than 2 GB before the caller's delete fires, so without this a
#: single pathological deposit can still fill the disk no matter how the
#: download cap is set. Generous enough that no legitimate replication package
#: has hit it -- the point is to bound the worst case, not to filter.
MAX_EXTRACTED_BYTES = 10 * 1024**3


class UnsafeArchive(Exception):
    """An archive member would escape the extraction root, or the archive is a bomb."""


@dataclass
class Extracted:
    """Result of unpacking one archive."""

    files: list[Path] = field(default_factory=list)
    skipped_unsafe: list[str] = field(default_factory=list)
    skipped_other: list[str] = field(default_factory=list)
    nested_archives: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def n_files(self) -> int:
        return len(self.files)


def is_safe_member(name: str, root: Path) -> bool:
    """Whether ``name`` stays inside ``root`` once joined.

    Rejects absolute paths, drive letters, and any ``..`` traversal. Checked by
    resolving the joined path rather than by pattern-matching the string, so
    encodings like ``a/../../b`` are caught too.
    """
    if not name or name.endswith("/"):
        return False
    pure = name.replace("\\", "/")
    if pure.startswith("/") or (len(pure) > 1 and pure[1] == ":"):
        return False
    target = (root / pure).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _reject_bomb(total_uncompressed: int, n_members: int, label: str) -> None:
    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        raise UnsafeArchive(
            f"{label}: declares {total_uncompressed / 1e9:.1f} GB uncompressed "
            f"(limit {MAX_UNCOMPRESSED_BYTES / 1e9:.0f} GB)"
        )
    if n_members > MAX_MEMBERS:
        raise UnsafeArchive(f"{label}: {n_members:,} members (limit {MAX_MEMBERS:,})")


def _wanted(
    name: str, keep_suffixes: frozenset[str], keep_names: frozenset[str]
) -> bool:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    if base.startswith("._"):
        # AppleDouble resource fork from a macOS-made zip. v1 counted 275 of
        # these as Python files and 15 as notebooks; they are not source.
        return False
    if "__MACOSX/" in name.replace("\\", "/"):
        return False
    return base.lower() in keep_names or Path(base).suffix.lower() in keep_suffixes


def extract_zip(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
) -> Extracted:
    """Extract wanted members of a zip, preserving their in-archive paths."""
    result = Extracted()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            _reject_bomb(sum(i.file_size for i in infos), len(infos), archive.name)
            for info in infos:
                if info.is_dir():
                    continue
                name = info.filename
                if not is_safe_member(name, dest):
                    result.skipped_unsafe.append(name)
                    continue
                if not _wanted(name, keep_suffixes, keep_names):
                    result.skipped_other.append(name)
                    continue
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    out.write(src.read())
                result.files.append(target)
    except UnsafeArchive as exc:
        result.error = str(exc)
    except (zipfile.BadZipFile, OSError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    if result.skipped_unsafe:
        logger.warning(
            "rejected unsafe archive members",
            extra={"archive": archive.name, "n": len(result.skipped_unsafe)},
        )
    return result


def extract_7z(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
) -> Extracted:
    """Extract wanted members of a .7z.

    Added because the corpus contains them: ``replication_package.7z`` was
    downloaded, could not be opened, and was recorded as an unsupported format
    -- a real deposit lost to a missing branch, which is how v1 lost every
    ``.tar``.
    """
    result = Extracted()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr

        with py7zr.SevenZipFile(archive, "r") as sz:
            names = sz.getnames()
            if len(names) > MAX_MEMBERS:
                raise UnsafeArchive(f"{archive.name}: {len(names):,} members")
            wanted = [
                n
                for n in names
                if is_safe_member(n, dest) and _wanted(n, keep_suffixes, keep_names)
            ]
            result.skipped_unsafe = [n for n in names if not is_safe_member(n, dest)]
            if wanted:
                sz.extract(path=dest, targets=wanted)
                result.files = [dest / n for n in wanted if (dest / n).is_file()]
    except UnsafeArchive as exc:
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - py7zr raises a wide variety
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def extract_tar(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
) -> Extracted:
    """Extract wanted members of a tar (optionally compressed), safely.

    Symlinks and device nodes are refused outright: a symlink member can point
    anywhere and turn a later write into an arbitrary-file overwrite.
    """
    result = Extracted()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            _reject_bomb(sum(m.size for m in members), len(members), archive.name)
            for member in members:
                if not member.isfile():
                    if member.issym() or member.islnk() or member.isdev():
                        result.skipped_unsafe.append(member.name)
                    continue
                if not is_safe_member(member.name, dest):
                    result.skipped_unsafe.append(member.name)
                    continue
                if not _wanted(member.name, keep_suffixes, keep_names):
                    result.skipped_other.append(member.name)
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                target = dest / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with extracted, open(target, "wb") as out:
                    out.write(extracted.read())
                result.files.append(target)
    except UnsafeArchive as exc:
        result.error = str(exc)
    except (tarfile.TarError, OSError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


#: Suffixes we recurse into when found inside an archive.
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}


def extract(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
    depth: int = 0,
) -> Extracted:
    """Extract an archive by type, recursing into nested archives.

    Nested archives are real: replication packages routinely ship ``code.zip``
    inside ``replication.zip``. v1's Dataverse path had no nesting support at
    all, so that code was invisible; its Zenodo path recursed but double-counted
    the results.
    """
    suffix = archive.suffix.lower()
    name = archive.name.lower()
    if suffix == ".zip":
        result = extract_zip(archive, dest, keep_suffixes, keep_names)
    elif suffix == ".7z":
        result = extract_7z(archive, dest, keep_suffixes, keep_names)
    elif suffix in {".tar", ".tgz", ".bz2", ".xz"} or name.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    ):
        result = extract_tar(archive, dest, keep_suffixes, keep_names)
    elif suffix == ".gz":
        result = extract_tar(archive, dest, keep_suffixes, keep_names)
        if result.error:
            # A bare .gz is a single compressed file, not a tar.
            result = _extract_gz(archive, dest, keep_suffixes, keep_names)
    else:
        return Extracted(error=f"unsupported archive type: {archive.name}")

    written = sum(p.stat().st_size for p in result.files if p.exists())
    if written > MAX_EXTRACTED_BYTES:
        result.error = (
            f"{archive.name}: expanded to {written / 1e9:.1f} GB on disk "
            f"(limit {MAX_EXTRACTED_BYTES / 1e9:.0f} GB); stopped extracting"
        )
        logger.warning("extraction budget exceeded", extra={"archive": archive.name})
        return result

    if depth < MAX_NESTING:
        for path in list(result.files):
            if path.suffix.lower() in _ARCHIVE_SUFFIXES:
                inner = extract(
                    path,
                    path.parent / f"{path.name}_extracted",
                    keep_suffixes,
                    keep_names,
                    depth + 1,
                )
                result.files.extend(inner.files)
                result.skipped_unsafe.extend(inner.skipped_unsafe)
                result.nested_archives.append(path)
                if inner.error is None:
                    # Its contents are now on disk uncompressed, so keeping the
                    # compressed copy stores the same bytes twice. Only the
                    # *top-level* archive is worth retaining, and that is the
                    # caller's call because it is the unit the ledger can
                    # refetch; a nested archive has no such record.
                    #
                    # Measured before this existed: 572 nested archives holding
                    # 2.2 GB and still growing mid-run, on a disk already at
                    # 99%. The outer delete had always been there, which is
                    # precisely why the leak was invisible -- "we delete
                    # archives after extracting" was true and insufficient.
                    path.unlink(missing_ok=True)
                    result.files.remove(path)
                # A nested archive that failed to extract is kept, for the same
                # reason a failed top-level one is: it is the evidence.
    return result


def _extract_gz(
    archive: Path, dest: Path, keep_suffixes: frozenset[str], keep_names: frozenset[str]
) -> Extracted:
    import gzip

    result = Extracted()
    inner_name = (
        archive.name[:-3] if archive.name.lower().endswith(".gz") else archive.stem
    )
    if not _wanted(inner_name, keep_suffixes, keep_names):
        result.skipped_other.append(inner_name)
        return result
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / inner_name
    try:
        with gzip.open(archive, "rb") as src, open(target, "wb") as out:
            written = 0
            while chunk := src.read(1 << 20):
                written += len(chunk)
                if written > MAX_UNCOMPRESSED_BYTES:
                    raise UnsafeArchive(f"{archive.name}: gz expands past limit")
                out.write(chunk)
        result.files.append(target)
    except UnsafeArchive as exc:
        target.unlink(missing_ok=True)
        result.error = str(exc)
    except OSError as exc:
        target.unlink(missing_ok=True)
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def relative_member_path(path: Path, root: Path) -> str:
    """In-archive path of an extracted file, for the provenance record."""
    try:
        return str(path.relative_to(root)).replace(os.sep, "/")
    except ValueError:
        return path.name

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
import re
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Cheap ceiling on how many members we will iterate. Unlike a size limit this
#: costs nothing to check and bounds the work of merely listing a hostile
#: archive. Calibrated against the corpus: one real deposit has 140,219
#: members.
MAX_MEMBERS = 500_000
#: Nested archives are extracted this many levels deep, then left alone.
MAX_NESTING = 3

#: Ceiling on what one archive may write to disk, across all nesting levels.
#:
#: **This is the only size guard, and it is the only one that was ever
#: measuring the right thing.** There used to be a second, checked first: the
#: archive's *declared* uncompressed size. It rejected 6 of the first 100
#: deposits at 2 GB, was raised to 20 GB, and then rejected 12 more once the
#: corpus reached 1,394 -- economics replication packages declaring 21 to 103
#: GB. Between them those twelve held 26.5 MB of `.do` and `.R`. Extraction
#: keeps only source and drops the panel data that is the other 99.994%, so
#: the declared total is three orders of magnitude away from what we write:
#: not a safety margin, a filter, and one that selects against the largest
#: empirical packages -- a bias correlated with the very thing being measured.
#: Needing to raise the number twice was the tell that the quantity was wrong.
#:
#: What remains bounds bytes *written*, which is the resource actually at risk,
#: and it is now spent incrementally rather than totalled at the end: the check
#: fires on the member that would exceed it, so the limit constrains the disk
#: and not just the report.
MAX_EXTRACTED_BYTES = 10 * 1024**3

#: Copy granularity. Also the accuracy of the budget: a member is stopped
#: within one chunk of the ceiling rather than after being read whole, which
#: matters because `ZipFile.read()` would otherwise pull a 4 GB member into
#: memory before anything got a chance to object.
_CHUNK = 1 << 20


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


def _reject_bomb(n_members: int, label: str) -> None:
    if n_members > MAX_MEMBERS:
        raise UnsafeArchive(f"{label}: {n_members:,} members (limit {MAX_MEMBERS:,})")


@dataclass
class Budget:
    """Bytes one deposit may write, shared across every nesting level.

    A per-call limit would hand each nested archive a fresh allowance, so an
    archive of archives could write an unbounded multiple of the ceiling while
    every individual extraction stayed within it.
    """

    remaining: int

    def spend(self, n: int, label: str) -> None:
        self.remaining -= n
        if self.remaining < 0:
            raise UnsafeArchive(
                f"{label}: expanded past the "
                f"{MAX_EXTRACTED_BYTES / 1e9:.0f} GB write budget; "
                f"stopped extracting"
            )


def _copy_within_budget(src, target: Path, budget: Budget, label: str) -> None:
    """Stream one member to disk, charging the budget as it goes.

    Chunked rather than `read()`-then-`write()` so that neither memory nor disk
    can be committed before the budget has a chance to refuse.
    """
    with open(target, "wb") as out:
        while chunk := src.read(_CHUNK):
            try:
                budget.spend(len(chunk), label)
            except UnsafeArchive:
                out.close()
                target.unlink(missing_ok=True)
                raise
            out.write(chunk)


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
    budget: Budget | None = None,
) -> Extracted:
    """Extract wanted members of a zip, preserving their in-archive paths."""
    result = Extracted()
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            _reject_bomb(len(infos), archive.name)
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
                with zf.open(info) as src:
                    _copy_within_budget(src, target, budget, archive.name)
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
    budget: Budget | None = None,
) -> Extracted:
    """Extract wanted members of a .7z.

    Added because the corpus contains them: ``replication_package.7z`` was
    downloaded, could not be opened, and was recorded as an unsupported format
    -- a real deposit lost to a missing branch, which is how v1 lost every
    ``.tar``.

    Symlink members are refused here as they are in the tar path. This one is
    not theoretical the way most hardening is: unlike zip and tar, where we do
    the writing, py7zr writes the members itself, so nothing in this file
    stopped it creating a link. A member named ``code/x.R`` pointing at
    ``/etc/passwd`` passes ``is_safe_member`` -- the *name* stays inside the
    root -- and a later write goes through it. The gap surfaced when a
    macOS-made archive mislabelled three ordinary files as links and py7zr
    tried to use each one's contents as a link target, which is also why the
    filter is on the flag rather than on any attempt to tell a real link from
    a mislabelled one.
    """
    result = Extracted()
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr

        with py7zr.SevenZipFile(archive, "r") as sz:
            infos = sz.list()
            names = [i.filename for i in infos]
            if len(names) > MAX_MEMBERS:
                raise UnsafeArchive(f"{archive.name}: {len(names):,} members")
            links = {i.filename for i in infos if i.is_symlink}
            wanted = [
                n
                for n in names
                if n not in links
                and is_safe_member(n, dest)
                and _wanted(n, keep_suffixes, keep_names)
            ]
            result.skipped_unsafe = [
                n for n in names if n in links or not is_safe_member(n, dest)
            ]
            if wanted:
                sz.extract(path=dest, targets=wanted)
                result.files = [dest / n for n in wanted if (dest / n).is_file()]
                # py7zr has no streaming member API, so the budget is charged
                # after the fact here. The archive-level member cap and the
                # `wanted` filter are what bound it in the meantime.
                budget.spend(sum(p.stat().st_size for p in result.files), archive.name)
    except UnsafeArchive as exc:
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - py7zr raises a wide variety
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def extract_rar(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
    budget: Budget | None = None,
) -> Extracted:
    """Extract wanted members of a .rar.

    Five deposits in the frame ship one, and until now each was recorded as an
    unsupported type -- the same way a `.7z` and, in v1, every `.tar` was lost.

    `rarfile` (4.5, August 2026) is the only maintained reader, and RAR3+ is
    patent-encumbered enough that it has no pure-Python decompressor: it shells
    out to `unar`, `bsdtar` or `unrar`. That external dependency is why the
    absence of a backend is reported as an ordinary extraction error rather
    than raised -- a machine without one should record a coverage gap and carry
    on, not stop collecting.
    """
    result = Extracted()
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import rarfile

        with rarfile.RarFile(archive) as rf:
            infos = rf.infolist()
            _reject_bomb(len(infos), archive.name)
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
                with rf.open(info) as src:
                    _copy_within_budget(src, target, budget, archive.name)
                result.files.append(target)
    except UnsafeArchive as exc:
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - rarfile wraps varied backend errors
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def extract_tar(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
    budget: Budget | None = None,
) -> Extracted:
    """Extract wanted members of a tar (optionally compressed), safely.

    Symlinks and device nodes are refused outright: a symlink member can point
    anywhere and turn a later write into an arbitrary-file overwrite.
    """
    result = Extracted()
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            _reject_bomb(len(members), archive.name)
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
                with extracted:
                    _copy_within_budget(extracted, target, budget, archive.name)
                result.files.append(target)
    except UnsafeArchive as exc:
        result.error = str(exc)
    except (tarfile.TarError, OSError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


#: Suffixes we recurse into when found inside an archive.
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"}

#: Leading bytes to format. Read in preference to the extension, because one
#: deposit shipped a 7z named `.zip`: `zipfile` reported "File is not a zip
#: file", which the collector could only record as a transport failure --
#: indistinguishable in the ledger from a timeout, so it would have been
#: retried forever and never worked.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip"),
    (b"PK\x07\x08", "zip"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"\x1f\x8b", "gz"),
    (b"BZh", "tar"),
    (b"\xfd7zXZ\x00", "tar"),
)


#: Segment name -> the archive it belongs to. `pkg.z01` is part of `pkg.zip`;
#: `pkg.7z.002` part of `pkg.7z`; `pkg.r00` part of `pkg.rar`; and a
#: `pkg.partN.rar` set names itself.
_SEGMENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(.+?)\.z\d{2}$", re.I), "{}.zip"),
    (re.compile(r"^(.+?\.7z)\.\d{3}$", re.I), "{}"),
    (re.compile(r"^(.+?)\.r\d{2}$", re.I), "{}.rar"),
    (re.compile(r"^(.+?)\.part\d+\.rar$", re.I), "{}.part1.rar"),
)


def _segment_of(key: str) -> str | None:
    """The archive ``key`` is a segment of, or None if it stands alone."""
    for pattern, template in _SEGMENT_PATTERNS:
        match = pattern.match(key)
        if match:
            return template.format(match.group(1))
    return None


def spanned_segments(keys: Iterable[str]) -> set[str]:
    """Which of ``keys`` belong to a multi-part archive set.

    A multi-part archive cannot be opened one segment at a time, and it fails
    in the worst possible way: the *last* segment carries the central
    directory, so it lists every member and errors only once one is read. A
    deposit split 6.4 GB across `.z01`, `.z02` and a 130 MB `.zip`; the `.zip`
    downloaded byte-for-byte against Zenodo's stated size, listed 165 members,
    and yielded nothing. Nothing upstream of extraction could notice, and the
    resulting error was indistinguishable from a corrupt download -- so it
    looked retryable, and no number of retries would ever have worked.

    Reassembly is deliberately not attempted: it means fetching every segment
    regardless of the size cap, which is a policy decision for the caller.
    """
    keys = list(keys)
    present = {k.lower(): k for k in keys}
    by_archive: dict[str, set[str]] = {}
    for key in keys:
        archive = _segment_of(key)
        if archive:
            by_archive.setdefault(archive.lower(), set()).add(key)

    found: set[str] = set()
    for archive, segments in by_archive.items():
        if archive in present:
            found |= segments | {present[archive]}
        elif len(segments) > 1:
            found |= segments
    return found


def archive_format(archive: Path) -> str | None:
    """Format of ``archive`` from its header, falling back to its extension.

    The fallback matters: a self-extracting zip begins with an executable stub
    (one deposit's starts ``d2 75 6f 7f``) and still opens, because a zip's
    directory is at the end of the file. Sniffing must not turn a working case
    into a failure.
    """
    try:
        with open(archive, "rb") as handle:
            head = handle.read(262)
    except OSError:
        head = b""
    for signature, fmt in _MAGIC:
        if head.startswith(signature):
            return fmt
    if head[257:262] == b"ustar":
        return "tar"

    suffix = archive.suffix.lower()
    name = archive.name.lower()
    if suffix == ".zip":
        return "zip"
    if suffix == ".7z":
        return "7z"
    if suffix == ".rar":
        return "rar"
    if suffix == ".gz":
        return "gz"
    if suffix in {".tar", ".tgz", ".bz2", ".xz"} or name.endswith(
        (".tar.gz", ".tar.bz2", ".tar.xz")
    ):
        return "tar"
    return None


def extract(
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str] = frozenset(),
    depth: int = 0,
    budget: Budget | None = None,
) -> Extracted:
    """Extract an archive by type, recursing into nested archives.

    Nested archives are real: replication packages routinely ship ``code.zip``
    inside ``replication.zip``. v1's Dataverse path had no nesting support at
    all, so that code was invisible; its Zenodo path recursed but double-counted
    the results.
    """
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    fmt = archive_format(archive)
    if fmt == "zip":
        result = extract_zip(archive, dest, keep_suffixes, keep_names, budget)
    elif fmt == "7z":
        result = extract_7z(archive, dest, keep_suffixes, keep_names, budget)
    elif fmt == "rar":
        result = extract_rar(archive, dest, keep_suffixes, keep_names, budget)
    elif fmt == "tar":
        result = extract_tar(archive, dest, keep_suffixes, keep_names, budget)
    elif fmt == "gz":
        result = extract_tar(archive, dest, keep_suffixes, keep_names, budget)
        if result.error:
            # A bare .gz is a single compressed file, not a tar.
            result = _extract_gz(archive, dest, keep_suffixes, keep_names, budget)
    else:
        return Extracted(error=f"unsupported archive type: {archive.name}")

    if result.error:
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
                    budget,
                )
                if inner.error and "write budget" in inner.error:
                    result.error = inner.error
                    return result
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
    archive: Path,
    dest: Path,
    keep_suffixes: frozenset[str],
    keep_names: frozenset[str],
    budget: Budget | None = None,
) -> Extracted:
    import gzip

    result = Extracted()
    budget = budget or Budget(MAX_EXTRACTED_BYTES)
    inner_name = (
        archive.name[:-3] if archive.name.lower().endswith(".gz") else archive.stem
    )
    if not _wanted(inner_name, keep_suffixes, keep_names):
        result.skipped_other.append(inner_name)
        return result
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / inner_name
    try:
        with gzip.open(archive, "rb") as src:
            _copy_within_budget(src, target, budget, archive.name)
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

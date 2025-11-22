"""File utilities for Softverse."""

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

from softverse.config import get_config
from softverse.utils.logging_utils import get_logger

logger = get_logger("file_utils")


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_file_newer(source_path: str | Path, target_path: str | Path) -> bool:
    """Check if source file is newer than target file.

    Args:
        source_path: Path to source file
        target_path: Path to target file

    Returns:
        True if source is newer or target doesn't exist
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    if not target_path.exists():
        return True

    if not source_path.exists():
        return False

    return source_path.stat().st_mtime > target_path.stat().st_mtime


def get_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """Get hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def is_valid_script_file(file_path: str | Path) -> bool:
    """Check if file is a valid script file based on extension.

    Args:
        file_path: Path to file

    Returns:
        True if file has valid script extension
    """
    config = get_config()
    valid_extensions = config.get("processing.valid_extensions", [".R", ".py", ".do"])

    return Path(file_path).suffix in valid_extensions


def safe_remove(path: str | Path) -> bool:
    """Safely remove file or directory.

    Args:
        path: Path to remove

    Returns:
        True if removed successfully
    """
    path = Path(path)

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")
        return False


def create_tar_gz(source_dir: str | Path, output_file: str | Path) -> bool:
    """Create tar.gz archive from directory.

    Args:
        source_dir: Directory to archive
        output_file: Output archive path

    Returns:
        True if archive created successfully
    """
    source_dir = Path(source_dir)
    output_file = Path(output_file)

    try:
        with tarfile.open(output_file, "w:gz") as tar:
            tar.add(source_dir, arcname=source_dir.name)
        logger.info(f"Created archive: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create archive {output_file}: {e}")
        return False


def extract_tar_gz(archive_path: str | Path, output_dir: str | Path) -> bool:
    """Extract tar.gz archive.

    Args:
        archive_path: Path to archive
        output_dir: Directory to extract to

    Returns:
        True if extracted successfully
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)

    try:
        ensure_directory(output_dir)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
        logger.info(f"Extracted archive: {archive_path} to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract archive {archive_path}: {e}")
        return False


class CheckpointManager:
    """Manage processing checkpoints for incremental updates."""

    def __init__(self, checkpoint_dir: str | None = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        config = get_config()
        self.checkpoint_dir = Path(
            checkpoint_dir or config.get("incremental.checkpoints_dir", "checkpoints/")
        )
        ensure_directory(self.checkpoint_dir)

    def save_checkpoint(self, name: str, data: dict) -> None:
        """Save checkpoint data.

        Args:
            name: Checkpoint name
            data: Data to save
        """
        checkpoint_file = self.checkpoint_dir / f"{name}.json"
        data["timestamp"] = datetime.now().isoformat()

        with open(checkpoint_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved checkpoint: {checkpoint_file}")

    def load_checkpoint(self, name: str) -> dict | None:
        """Load checkpoint data.

        Args:
            name: Checkpoint name

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_file = self.checkpoint_dir / f"{name}.json"

        if not checkpoint_file.exists():
            return None

        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
            logger.debug(f"Loaded checkpoint: {checkpoint_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            return None

    def clear_checkpoint(self, name: str) -> None:
        """Clear checkpoint file.

        Args:
            name: Checkpoint name
        """
        checkpoint_file = self.checkpoint_dir / f"{name}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.debug(f"Cleared checkpoint: {checkpoint_file}")


def get_file_list(
    directory: str | Path, extensions: list[str] | None = None
) -> list[Path]:
    """Get list of files in directory with specified extensions.

    Args:
        directory: Directory to search
        extensions: List of file extensions to include

    Returns:
        List of matching file paths
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if extensions is None:
        config = get_config()
        extensions = config.get("processing.valid_extensions", [".R", ".py", ".do"])

    files = []
    for root, _dirs, filenames in os.walk(directory):
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix in extensions:
                files.append(file_path)

    return sorted(files)


def copy_with_structure(src: str | Path, dst: str | Path) -> bool:
    """Copy file preserving directory structure.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        True if copied successfully
    """
    src = Path(src)
    dst = Path(dst)

    try:
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def get_disk_usage(path: str | Path) -> tuple[int, int, int]:
    """Get disk usage statistics for a path.

    Args:
        path: Directory path to check

    Returns:
        Tuple of (total, used, free) in bytes
    """
    path = Path(path)
    if path.exists():
        usage = shutil.disk_usage(path)
        return usage.total, usage.total - usage.free, usage.free
    return 0, 0, 0


def check_free_space(path: str | Path, min_gb: float = 5.0) -> bool:
    """Check if there's enough free disk space.

    Args:
        path: Directory path to check
        min_gb: Minimum free space required in GB

    Returns:
        True if enough space available
    """
    try:
        total, used, free = get_disk_usage(path)
        free_gb = float(free) / (1024**3)

        if free_gb < min_gb:
            logger.warning(f"Low disk space: {free_gb:.1f}GB free, {min_gb}GB required")
            return False

        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check


class ArchiveProcessor:
    """Handles archive processing with storage optimization."""

    def __init__(self, config_path: str | None = None):
        """Initialize archive processor.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.storage_config = self.config.get("storage", {})

    def get_temp_dir(self) -> Path:
        """Get temporary directory for downloads."""
        temp_dir = Path(self.storage_config.get("temp_dir", "temp/"))
        ensure_directory(temp_dir)
        return temp_dir

    def get_failed_archives_dir(self) -> Path:
        """Get directory for failed archives."""
        failed_dir = Path(
            self.storage_config.get("failed_archives_dir", "failed_archives/")
        )
        ensure_directory(failed_dir)
        return failed_dir

    def should_keep_archives(self) -> bool:
        """Check if archives should be kept after extraction."""
        return self.storage_config.get("keep_archives", False)

    def should_use_safe_deletion(self) -> bool:
        """Check if safe deletion is enabled."""
        return self.storage_config.get("safe_deletion", True)

    def get_max_retry_attempts(self) -> int:
        """Get maximum retry attempts for failed operations."""
        return self.storage_config.get("max_retry_attempts", 3)

    def check_storage_requirements(self) -> bool:
        """Check if storage requirements are met."""
        min_free_space = self.storage_config.get("min_free_space_gb", 5.0)
        temp_dir = self.get_temp_dir()

        return check_free_space(temp_dir, min_free_space)

    def extract_zip_safely(
        self,
        zip_path: Path,
        extract_to: Path,
        valid_extensions: list[str] | None = None,
    ) -> tuple[bool, int]:
        """Extract ZIP file safely with filtering.

        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to
            valid_extensions: List of valid file extensions to keep

        Returns:
            Tuple of (success, script_count)
        """
        if valid_extensions is None:
            valid_extensions = self.config.get(
                "processing.valid_extensions", [".R", ".py", ".do", ".ipynb"]
            )

        try:
            script_count = 0
            ensure_directory(extract_to)

            logger.debug(f"Extracting {zip_path} to {extract_to}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract all files first
                zip_ref.extractall(extract_to)

            # Filter and count script files
            for root, _dirs, files in os.walk(extract_to):
                for file in files:
                    file_path = Path(root) / file
                    if any(file.endswith(ext) for ext in valid_extensions):
                        script_count += 1
                    else:
                        # Remove non-script files
                        safe_remove(file_path)

            # Remove empty directories
            self._remove_empty_directories(extract_to)

            if script_count == 0:
                logger.debug(
                    f"No script files found in {zip_path}, removing extraction dir"
                )
                safe_remove(extract_to)

            return True, script_count

        except Exception as e:
            logger.error(f"Failed to extract {zip_path}: {e}")
            return False, 0

    def _remove_empty_directories(self, directory: Path) -> None:
        """Recursively remove empty directories."""
        try:
            for root, dirs, _files in os.walk(directory, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        if not any(dir_path.iterdir()):
                            dir_path.rmdir()
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except OSError:
                        pass
        except Exception as e:
            logger.warning(f"Error removing empty directories in {directory}: {e}")

    def process_archive_with_cleanup(
        self,
        archive_path: Path,
        extract_to: Path,
        valid_extensions: list[str] | None = None,
        preserve_on_failure: bool | None = None,
        retry_attempts: int | None = None,
    ) -> tuple[bool, int]:
        """Process archive with automatic cleanup and error recovery.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to
            valid_extensions: List of valid file extensions
            preserve_on_failure: Override config for preserving failed archives
            retry_attempts: Number of retry attempts (overrides config)

        Returns:
            Tuple of (success, script_count)
        """
        if preserve_on_failure is None:
            preserve_on_failure = self.storage_config.get(
                "preserve_failed_archives", True
            )

        if retry_attempts is None:
            retry_attempts = self.get_max_retry_attempts()

        last_error = None

        # Attempt extraction with retries
        for attempt in range(retry_attempts + 1):
            try:
                # Clean up any partial extraction from previous attempt
                if attempt > 0 and extract_to.exists():
                    logger.debug(
                        f"Cleaning up partial extraction from attempt {attempt}"
                    )
                    safe_remove(extract_to)

                success, script_count = self.extract_zip_safely(
                    archive_path, extract_to, valid_extensions
                )

                if success:
                    logger.debug(
                        f"Archive extraction successful on attempt {attempt + 1}"
                    )
                    break

                # If extraction failed but no exception was raised
                last_error = f"Extraction failed (attempt {attempt + 1})"
                if attempt < retry_attempts:
                    logger.warning(
                        f"Extraction failed for {archive_path.name}, retrying..."
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Extraction attempt {attempt + 1} failed for {archive_path.name}: {e}"
                )
                if attempt < retry_attempts:
                    logger.info(
                        f"Retrying extraction ({attempt + 1}/{retry_attempts})..."
                    )

        else:
            # All attempts failed
            success = False
            script_count = 0
            logger.error(f"All extraction attempts failed for {archive_path.name}")

        # Handle cleanup based on configuration and success
        if success:
            if self.should_use_safe_deletion():
                # Verify extraction was successful before deleting
                if self._verify_extraction(extract_to, script_count):
                    if not self.should_keep_archives():
                        safe_remove(archive_path)
                        logger.debug(
                            f"Deleted archive after successful extraction: {archive_path}"
                        )
                else:
                    logger.warning(
                        f"Extraction verification failed for {archive_path}, keeping archive"
                    )
            else:
                # Delete immediately without verification
                if not self.should_keep_archives():
                    safe_remove(archive_path)

        else:
            # Handle failed extraction with error recovery
            self._handle_failed_archive(
                archive_path,
                preserve_on_failure,
                str(last_error) if last_error else "Unknown error",
            )

        return success, script_count

    def _verify_extraction(self, extract_to: Path, expected_script_count: int) -> bool:
        """Verify that extraction was successful.

        Args:
            extract_to: Directory where files were extracted
            expected_script_count: Expected number of script files

        Returns:
            True if extraction appears successful
        """
        try:
            if not extract_to.exists():
                return False

            # Check if directory has any content
            has_content = any(extract_to.iterdir())

            # Don't require scripts - many Zenodo repos contain data only
            # Just verify that extraction produced some files
            return has_content

        except Exception as e:
            logger.warning(f"Error verifying extraction for {extract_to}: {e}")
            return False

    def _handle_failed_archive(
        self, archive_path: Path, preserve_on_failure: bool, error_msg: str
    ) -> None:
        """Handle a failed archive extraction.

        Args:
            archive_path: Path to the failed archive
            preserve_on_failure: Whether to preserve the failed archive
            error_msg: Error message describing the failure
        """
        if preserve_on_failure:
            failed_dir = self.get_failed_archives_dir()
            failed_path = failed_dir / archive_path.name

            # Create error log file alongside failed archive
            error_log_path = failed_dir / f"{archive_path.stem}_error.txt"

            try:
                if archive_path != failed_path:  # Avoid moving to same location
                    shutil.move(str(archive_path), str(failed_path))
                    logger.info(f"Moved failed archive to {failed_path}")

                # Write error details
                with open(error_log_path, "w") as f:
                    f.write(f"Archive: {archive_path.name}\n")
                    f.write(f"Error: {error_msg}\n")
                    f.write(f"Timestamp: {self._get_timestamp()}\n")

            except Exception as e:
                logger.warning(f"Could not move failed archive {archive_path}: {e}")
        else:
            safe_remove(archive_path)
            logger.debug(f"Deleted failed archive: {archive_path}")

    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def recover_failed_archives(self, failed_archives_dir: Path | None = None) -> int:
        """Attempt to recover previously failed archives.

        Args:
            failed_archives_dir: Directory containing failed archives

        Returns:
            Number of archives successfully recovered
        """
        if failed_archives_dir is None:
            failed_archives_dir = self.get_failed_archives_dir()

        if not failed_archives_dir.exists():
            logger.info("No failed archives directory found")
            return 0

        failed_archives = list(failed_archives_dir.glob("*.zip"))
        if not failed_archives:
            logger.info("No failed archives found to recover")
            return 0

        logger.info(f"Attempting to recover {len(failed_archives)} failed archives")
        recovered_count = 0
        valid_extensions = self.config.get(
            "processing.valid_extensions", [".R", ".py", ".do", ".ipynb"]
        )

        for archive_path in failed_archives:
            logger.info(f"Attempting recovery of {archive_path.name}")

            # Try to extract to a recovery directory
            recovery_dir = failed_archives_dir / "recovered" / archive_path.stem

            try:
                success, script_count = self.extract_zip_safely(
                    archive_path, recovery_dir, valid_extensions
                )

                if success:
                    logger.info(
                        f"Successfully recovered {archive_path.name} with {script_count} scripts"
                    )

                    # Remove from failed archives
                    safe_remove(archive_path)

                    # Remove error log if it exists
                    error_log = failed_archives_dir / f"{archive_path.stem}_error.txt"
                    if error_log.exists():
                        safe_remove(error_log)

                    recovered_count += 1
                else:
                    logger.warning(f"Recovery failed for {archive_path.name}")

            except Exception as e:
                logger.error(f"Error during recovery of {archive_path.name}: {e}")

        logger.info(
            f"Recovery complete: {recovered_count}/{len(failed_archives)} archives recovered"
        )
        return recovered_count

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files if enabled."""
        if self.storage_config.get("cleanup_temp_files", True):
            temp_dir = self.get_temp_dir()
            try:
                # Only clean up if temp directory exists and has content
                if temp_dir.exists() and any(temp_dir.iterdir()):
                    for item in temp_dir.iterdir():
                        safe_remove(item)
                    logger.debug(f"Cleaned up temporary files in {temp_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up temp files: {e}")

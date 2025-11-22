"""Shared utilities for handling ZIP files and script extraction."""

import os
import zipfile
from pathlib import Path

from softverse.utils.file_utils import ensure_directory
from softverse.utils.logging_utils import get_logger

logger = get_logger("zip_utils")


def extract_scripts_from_zip(
    zip_path: Path, extract_to: Path, valid_extensions: list[str] | None = None
) -> tuple[bool, int, list[str]]:
    """Extract script files from ZIP archive.

    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        valid_extensions: List of valid script file extensions

    Returns:
        Tuple of (success, script_count, script_files_list)
    """
    if valid_extensions is None:
        valid_extensions = [".R", ".py", ".do", ".ipynb", ".jl", ".m", ".sas"]

    try:
        script_files = []
        ensure_directory(extract_to)

        logger.debug(f"Extracting {zip_path} to {extract_to}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all files first
            zip_ref.extractall(extract_to)

            # Find script files
            for root_str, _dirs, files in os.walk(extract_to):
                root = Path(root_str)
                for file in files:
                    file_path = root / file
                    if any(
                        file.lower().endswith(ext.lower()) for ext in valid_extensions
                    ):
                        # Store relative path from extract_to
                        rel_path = file_path.relative_to(extract_to)
                        script_files.append(str(rel_path))

            # Clean up non-script files to save space (optional)
            # Note: Keeping all files for now in case they contain useful metadata

        logger.debug(f"Extracted {len(script_files)} script files from {zip_path.name}")
        return True, len(script_files), script_files

    except zipfile.BadZipFile:
        logger.warning(f"Bad ZIP file: {zip_path}")
        return False, 0, []
    except Exception as e:
        logger.error(f"Failed to extract ZIP {zip_path}: {e}")
        return False, 0, []


def find_nested_zips_and_extract(
    directory: Path, valid_extensions: list[str] | None = None, max_depth: int = 3
) -> tuple[int, list[str]]:
    """Recursively find and extract nested ZIP files.

    Args:
        directory: Directory to search for ZIP files
        valid_extensions: Valid script file extensions
        max_depth: Maximum recursion depth for nested ZIPs

    Returns:
        Tuple of (total_scripts_found, script_files_list)
    """
    if max_depth <= 0:
        return 0, []

    total_scripts = 0
    all_scripts = []

    # Find ZIP files in current directory
    zip_files = list(directory.glob("*.zip"))

    for zip_file in zip_files:
        try:
            # Create extraction subdirectory
            extract_subdir = directory / f"{zip_file.stem}_extracted"

            success, script_count, script_files = extract_scripts_from_zip(
                zip_file, extract_subdir, valid_extensions
            )

            if success:
                total_scripts += script_count
                all_scripts.extend(
                    [f"{extract_subdir.name}/{script}" for script in script_files]
                )

                # Remove the ZIP file after successful extraction to save space
                zip_file.unlink()

                # Recursively check for more ZIPs in extracted content
                if max_depth > 1:
                    nested_scripts, nested_files = find_nested_zips_and_extract(
                        extract_subdir, valid_extensions, max_depth - 1
                    )
                    total_scripts += nested_scripts
                    all_scripts.extend(nested_files)

        except Exception as e:
            logger.warning(f"Failed to process ZIP {zip_file}: {e}")
            continue

    return total_scripts, all_scripts


def process_directory_for_scripts(
    directory: Path,
    valid_extensions: list[str] | None = None,
    handle_nested_zips: bool = True,
) -> tuple[int, list[str]]:
    """Process a directory to find all script files, including those in ZIP archives.

    Args:
        directory: Directory to process
        valid_extensions: Valid script file extensions
        handle_nested_zips: Whether to extract and process ZIP files

    Returns:
        Tuple of (total_script_count, script_files_list)
    """
    if valid_extensions is None:
        valid_extensions = [".R", ".py", ".do", ".ipynb", ".jl", ".m", ".sas"]

    script_files = []

    # First, find direct script files
    for ext in valid_extensions:
        script_files.extend(
            [str(f.relative_to(directory)) for f in directory.rglob(f"*{ext}")]
        )

    total_scripts = len(script_files)

    # Then handle ZIP files if requested
    if handle_nested_zips:
        zip_scripts, zip_files = find_nested_zips_and_extract(
            directory, valid_extensions
        )
        total_scripts += zip_scripts
        script_files.extend(zip_files)

    return total_scripts, script_files

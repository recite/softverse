"""Base collector with shared status tracking functionality."""

import json
from datetime import datetime
from pathlib import Path

from softverse.utils.file_utils import ensure_directory


class BaseCollectorWithStatus:
    """Base collector class with shared status tracking functionality."""

    def _write_repo_status(
        self, repo_dir: Path, status: str, details: dict | None = None
    ) -> None:
        """Write repository processing status to .status file.

        Args:
            repo_dir: Repository directory path
            status: Processing status (success, failed, no_scripts, forbidden, etc.)
            details: Additional details about the processing
        """
        status_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }

        status_file = repo_dir / ".status"
        ensure_directory(repo_dir)
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    def _read_repo_status(self, repo_dir: Path) -> dict | None:
        """Read repository processing status from .status file.

        Args:
            repo_dir: Repository directory path

        Returns:
            Status data dict or None if not found
        """
        status_file = repo_dir / ".status"
        if not status_file.exists():
            return None

        try:
            with open(status_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error message into standard status types.

        Args:
            error_msg: Error message string

        Returns:
            Categorized error status
        """
        error_lower = error_msg.lower()

        if "403" in error_msg or "forbidden" in error_lower:
            return "forbidden"
        elif "404" in error_msg or "not found" in error_lower:
            return "not_found"
        elif "timeout" in error_lower or "connection" in error_lower:
            return "connection_error"
        else:
            return "error"

    def _write_error_status(
        self, repo_dir: Path, error_msg: str, identifier: str
    ) -> None:
        """Write error status with automatic categorization.

        Args:
            repo_dir: Repository directory path
            error_msg: Error message
            identifier: Repository/record identifier
        """
        status_type = self._categorize_error(error_msg)
        self._write_repo_status(
            repo_dir, status_type, {"error": error_msg, "identifier": identifier}
        )

    def _write_success_status(
        self,
        repo_dir: Path,
        script_count: int,
        identifier: str,
        extra_details: dict | None = None,
    ) -> None:
        """Write success status with script count.

        Args:
            repo_dir: Repository directory path
            script_count: Number of script files found
            identifier: Repository/record identifier
            extra_details: Additional details to include
        """
        details = {"script_count": script_count, "identifier": identifier}

        if extra_details:
            details.update(extra_details)

        if script_count > 0:
            self._write_repo_status(repo_dir, "success", details)
        else:
            self._write_repo_status(repo_dir, "no_scripts", details)

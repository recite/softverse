"""Logging utilities for Softverse."""

import logging
import logging.handlers
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from softverse.config import get_config


def setup_logging(
    name: str | None = None,
    level: str | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        name: Logger name. If None, uses root logger.
        level: Log level. If None, uses config value.
        log_file: Log file path. If None, uses config value.

    Returns:
        Configured logger
    """
    config = get_config()

    # Get configuration values
    log_level = level or config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file_path = log_file or config.get("logging.file", "logs/softverse.log")
    max_size_mb = config.get("logging.max_size_mb", 100)
    backup_count = config.get("logging.backup_count", 5)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file_path:
        # Ensure log directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=max_size_mb * 1024 * 1024, backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"softverse.{name}")


class LogProgress:
    """Context manager for logging progress of operations."""

    def __init__(
        self, logger: logging.Logger, operation: str, total: int | None = None
    ):
        """Initialize progress logger.

        Args:
            logger: Logger instance
            operation: Description of operation
            total: Total number of items (for progress tracking)
        """
        self.logger = logger
        self.operation = operation
        self.total = total
        self.processed = 0

    def __enter__(self) -> "LogProgress":
        """Enter context manager."""
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} ({self.processed} items)")
        else:
            self.logger.error(f"Failed {self.operation}: {exc_val}")

    def update(self, increment: int = 1) -> None:
        """Update progress counter.

        Args:
            increment: Number to increment by
        """
        self.processed += increment

        if self.total:
            percentage = (self.processed / self.total) * 100
            self.logger.info(
                f"{self.operation} progress: {self.processed}/{self.total} "
                f"({percentage:.1f}%)"
            )
        else:
            self.logger.info(f"{self.operation} progress: {self.processed} items")


def log_function_call(
    logger: logging.Logger,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log function calls.

    Args:
        logger: Logger instance

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed: {e}")
                raise

        return wrapper

    return decorator


class ArchiveProgressLogger:
    """Specialized progress logger for archive operations."""

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        total: int | None = None,
        show_size_info: bool = True,
    ):
        """Initialize archive progress logger.

        Args:
            logger: Logger instance
            operation: Description of operation
            total: Total number of archives (for progress tracking)
            show_size_info: Whether to show file size information
        """
        self.logger = logger
        self.operation = operation
        self.total = total
        self.show_size_info = show_size_info
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.total_size_downloaded = 0
        self.total_scripts_extracted = 0
        self.start_time: float | None = None

    def __enter__(self) -> "ArchiveProgressLogger":
        """Enter context manager."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            summary_parts = [
                f"Completed {self.operation}",
                f"{self.successful}/{self.processed} successful",
            ]

            if self.failed > 0:
                summary_parts.append(f"{self.failed} failed")

            if self.total_scripts_extracted > 0:
                summary_parts.append(
                    f"{self.total_scripts_extracted} scripts extracted"
                )

            if self.show_size_info and self.total_size_downloaded > 0:
                size_mb = float(self.total_size_downloaded) / (1024 * 1024)
                summary_parts.append(f"{size_mb:.1f}MB processed")

            summary_parts.append(f"in {duration:.1f}s")
            self.logger.info(" - ".join(summary_parts))
        else:
            self.logger.error(
                f"Failed {self.operation} after {duration:.1f}s: {exc_val}"
            )

    def update_download(self, archive_name: str, size_bytes: int = 0) -> None:
        """Update progress for a completed download.

        Args:
            archive_name: Name of the downloaded archive
            size_bytes: Size of downloaded file in bytes
        """
        self.processed += 1
        if size_bytes > 0:
            self.total_size_downloaded += size_bytes

        size_info = ""
        if self.show_size_info and size_bytes > 0:
            size_mb = float(size_bytes) / (1024 * 1024)
            size_info = f" ({size_mb:.1f}MB)"

        if self.total:
            percentage = (self.processed / self.total) * 100
            self.logger.info(
                f"Downloaded {archive_name}{size_info} - "
                f"{self.processed}/{self.total} ({percentage:.1f}%)"
            )
        else:
            self.logger.info(f"Downloaded {archive_name}{size_info}")

    def update_extraction(
        self,
        archive_name: str,
        success: bool,
        script_count: int = 0,
        error_msg: str | None = None,
    ) -> None:
        """Update progress for a completed extraction.

        Args:
            archive_name: Name of the processed archive
            success: Whether extraction was successful
            script_count: Number of script files extracted
            error_msg: Error message if extraction failed
        """
        if success:
            self.successful += 1
            self.total_scripts_extracted += script_count

            if script_count > 0:
                self.logger.info(
                    f"Extracted {archive_name}: {script_count} script files"
                )
            else:
                self.logger.debug(f"Processed {archive_name}: no script files found")
        else:
            self.failed += 1
            error_detail = f" - {error_msg}" if error_msg else ""
            self.logger.warning(f"Failed to extract {archive_name}{error_detail}")

    def log_storage_info(
        self, free_space_gb: float, temp_dir_size_gb: float = 0
    ) -> None:
        """Log storage information.

        Args:
            free_space_gb: Available free space in GB
            temp_dir_size_gb: Current size of temp directory in GB
        """
        if temp_dir_size_gb > 0:
            self.logger.info(
                f"Storage: {free_space_gb:.1f}GB free, "
                f"{temp_dir_size_gb:.1f}GB temp files"
            )
        else:
            self.logger.debug(f"Storage: {free_space_gb:.1f}GB free")

    def log_cleanup(self, files_removed: int, space_freed_mb: float) -> None:
        """Log cleanup operation results.

        Args:
            files_removed: Number of files removed
            space_freed_mb: Space freed in MB
        """
        self.logger.info(
            f"Cleanup: removed {files_removed} temp files, "
            f"freed {space_freed_mb:.1f}MB"
        )

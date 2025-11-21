"""Zenodo script file collector."""

from pathlib import Path

import pandas as pd

from softverse.config import get_config
from softverse.utils.api_utils import get_zenodo_client
from softverse.utils.file_utils import (
    ArchiveProcessor,
    CheckpointManager,
    ensure_directory,
)
from softverse.utils.logging_utils import ArchiveProgressLogger, LogProgress, get_logger

logger = get_logger("zenodo_collector")


class ZenodoScriptCollector:
    """Collects script files from Zenodo communities."""

    def __init__(self, config_path: str | None = None):
        """Initialize collector.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.client = get_zenodo_client()
        self.checkpoint_manager = CheckpointManager()
        self.archive_processor = ArchiveProcessor(config_path)

    def collect_community_metadata(
        self, communities: list[str] | None = None, max_pages: int | None = None
    ) -> pd.DataFrame:
        """Collect metadata for all records in Zenodo communities.

        Args:
            communities: List of community identifiers
            max_pages: Maximum pages to fetch per community

        Returns:
            DataFrame with all records
        """
        config = self.config.zenodo_config
        if communities is None:
            communities = config.get(
                "communities",
                [
                    "es-replication-repository",
                    "restud-replication",
                    "ej-replication-repository",
                    "pemj",
                ],
            )

        all_records = []

        with LogProgress(
            logger, "collecting Zenodo metadata", len(communities)
        ) as progress:
            for community in communities:
                try:
                    logger.info(f"Fetching records from community: {community}")
                    records = self.client.get_all_community_records(
                        community, max_pages
                    )

                    # Normalize records and add community info
                    if records:
                        df = pd.json_normalize(records)
                        df["community"] = community
                        all_records.append(df)
                        logger.info(f"Found {len(df)} records in {community}")

                except Exception as e:
                    logger.error(f"Failed to fetch records from {community}: {e}")

                progress.update()

        if all_records:
            result_df = pd.concat(all_records, ignore_index=True)
            logger.info(f"Total Zenodo records collected: {len(result_df)}")
            return result_df
        else:
            logger.warning("No Zenodo records collected")
            return pd.DataFrame()

    def download_and_process_record(
        self,
        record_id: str,
        temp_dir: Path,
        output_dir: Path,
        valid_extensions: list[str],
        progress_logger: ArchiveProgressLogger | None = None,
    ) -> tuple[bool, int]:
        """Download and process archive for a single Zenodo record.

        Args:
            record_id: Zenodo record ID
            temp_dir: Temporary directory for downloads
            output_dir: Final output directory
            valid_extensions: List of valid file extensions
            progress_logger: Optional archive progress logger

        Returns:
            Tuple of (success, script_count)
        """
        try:
            # Check storage requirements
            if not self.archive_processor.check_storage_requirements():
                logger.error("Insufficient storage space for downloads")
                return False, 0

            archive_path = temp_dir / f"{record_id}.zip"
            extract_to = output_dir / record_id

            # Skip if already processed
            if extract_to.exists():
                logger.debug(f"Skipping {record_id} - already processed")
                script_count = len(list(extract_to.rglob("*")))
                if progress_logger:
                    progress_logger.update_extraction(record_id, True, script_count)
                return True, script_count

            # Download archive to temp directory
            success = self.client.download_record_archive(record_id, archive_path)

            if not success:
                error_msg = "Download failed"
                if progress_logger:
                    progress_logger.update_extraction(record_id, False, 0, error_msg)
                return False, 0

            # Get file size for progress tracking
            archive_size = archive_path.stat().st_size if archive_path.exists() else 0
            if progress_logger:
                progress_logger.update_download(record_id, archive_size)

            # Process archive with automatic cleanup
            success, script_count = self.archive_processor.process_archive_with_cleanup(
                archive_path, extract_to, valid_extensions
            )

            # Update progress with extraction results
            if progress_logger:
                extraction_error: str | None = None if success else "Extraction failed"
                progress_logger.update_extraction(
                    record_id, success, script_count, extraction_error
                )

            return success, script_count

        except Exception as e:
            logger.error(f"Failed to download/process record {record_id}: {e}")
            if progress_logger:
                progress_logger.update_extraction(record_id, False, 0, str(e))
            return False, 0

    def collect_all_scripts(
        self,
        output_dir: str | None = None,
        communities: list[str] | None = None,
        max_records: int | None = None,
        force_refresh: bool = False,
    ) -> bool:
        """Collect all script files from Zenodo communities.

        Args:
            output_dir: Output directory for script files
            communities: List of community identifiers
            max_records: Maximum number of records to process
            force_refresh: Force refresh all downloads

        Returns:
            True if collection successful
        """
        # Get configuration
        config = self.config.zenodo_config
        output_dir_path = Path(
            output_dir or config.get("output_dir", "scripts/zenodo/")
        )
        ensure_directory(output_dir_path)

        # Use temp directory from archive processor
        temp_dir = self.archive_processor.get_temp_dir()

        try:
            # Check storage requirements before starting
            if not self.archive_processor.check_storage_requirements():
                logger.error("Insufficient storage space to begin collection")
                return False

            # Step 1: Collect metadata
            logger.info("Collecting Zenodo community metadata")
            records_df = self.collect_community_metadata(communities)

            if records_df.empty:
                logger.warning("No Zenodo records found")
                return False

            # Limit records if specified
            if max_records and len(records_df) > max_records:
                records_df = records_df.head(max_records)
                logger.info(f"Limited to {max_records} records")

            # Check checkpoint
            checkpoint_name = "zenodo_downloads"
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            processed_ids = (
                set(checkpoint.get("processed_ids", [])) if checkpoint else set()
            )

            if force_refresh:
                processed_ids = set()

            # Step 2: Process records with optimized storage
            valid_extensions = self.config.get(
                "processing.valid_extensions", [".R", ".py", ".do", ".ipynb"]
            )
            success_count = 0
            total_script_count = 0

            # Sequential processing for optimal storage usage
            storage_config = self.config.get("storage", {})
            sequential_mode = storage_config.get("sequential_processing", True)

            if sequential_mode:
                logger.info("Using sequential processing to minimize storage usage")

            # Use detailed archive progress logging
            with ArchiveProgressLogger(
                logger,
                "Zenodo archive collection",
                len(records_df),
                show_size_info=True,
            ) as archive_progress:
                # Log initial storage status
                from softverse.utils.file_utils import get_disk_usage

                total, used, free = get_disk_usage(temp_dir)
                free_gb = float(free) / (1024**3)
                archive_progress.log_storage_info(free_gb)

                for _, record in records_df.iterrows():
                    record_id = str(record.get("id", ""))

                    if record_id in processed_ids:
                        continue

                    # Download and process record with detailed progress tracking
                    success, script_count = self.download_and_process_record(
                        record_id,
                        temp_dir,
                        output_dir_path,
                        valid_extensions,
                        archive_progress,
                    )

                    if success:
                        success_count += 1
                        total_script_count += script_count
                        processed_ids.add(record_id)

                    # Update checkpoint
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_name,
                        {
                            "processed_ids": list(processed_ids),
                            "success_count": success_count,
                            "total_count": len(records_df),
                            "total_script_count": total_script_count,
                        },
                    )

            # Final cleanup
            self.archive_processor.cleanup_temp_files()

            # Clear checkpoint on completion
            if success_count == len(records_df):
                self.checkpoint_manager.clear_checkpoint(checkpoint_name)

            logger.info(
                f"Zenodo collection completed: {success_count}/{len(records_df)} records processed, "
                f"{total_script_count} script files extracted"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"Zenodo collection failed: {e}")
            return False


def main() -> None:
    """Main entry point for Zenodo script collection."""
    import argparse

    from softverse.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Collect script files from Zenodo")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for script files"
    )
    parser.add_argument("--communities", nargs="+", help="Zenodo communities to search")
    parser.add_argument(
        "--max-records", type=int, help="Maximum number of records to process"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh all downloads"
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded ZIP files after extraction",
    )
    parser.add_argument(
        "--no-sequential",
        action="store_true",
        help="Disable sequential processing mode",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Temporary directory for downloads (overrides config)",
    )
    parser.add_argument(
        "--recover-failed",
        action="store_true",
        help="Attempt to recover previously failed archives",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        help="Number of retry attempts for failed extractions",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Create collector and apply CLI overrides
    collector = ZenodoScriptCollector(args.config)

    # Override storage settings from CLI
    if args.keep_archives:
        collector.archive_processor.storage_config["keep_archives"] = True
    if args.no_sequential:
        collector.archive_processor.storage_config["sequential_processing"] = False
    if args.temp_dir:
        collector.archive_processor.storage_config["temp_dir"] = args.temp_dir
    if args.retry_attempts is not None:
        collector.archive_processor.storage_config["max_retry_attempts"] = (
            args.retry_attempts
        )

    # Handle failed archive recovery
    if args.recover_failed:
        logger.info("Attempting to recover previously failed Zenodo archives")
        recovered_count = collector.archive_processor.recover_failed_archives()
        logger.info(f"Recovery completed: {recovered_count} archives recovered")
        exit(0)

    success = collector.collect_all_scripts(
        output_dir=args.output_dir,
        communities=args.communities,
        max_records=args.max_records,
        force_refresh=args.force_refresh,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

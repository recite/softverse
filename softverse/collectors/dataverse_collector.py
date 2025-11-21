"""Dataverse dataset collector."""

from pathlib import Path

import pandas as pd

from softverse.config import get_config
from softverse.utils.api_utils import get_dataverse_client
from softverse.utils.file_utils import CheckpointManager, ensure_directory
from softverse.utils.logging_utils import LogProgress, get_logger

logger = get_logger("dataverse_collector")


class DataverseDatasetCollector:
    """Collects dataset information from Dataverse instances."""

    def __init__(self, config_path: str | None = None):
        """Initialize collector.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.client = get_dataverse_client()
        self.checkpoint_manager = CheckpointManager()

    def collect_datasets_for_dataverse(
        self, dataverse_id: str, output_dir: Path, force_refresh: bool = False
    ) -> bool:
        """Collect datasets for a single dataverse.

        Args:
            dataverse_id: Dataverse identifier
            output_dir: Output directory for CSV files
            force_refresh: Force refresh even if file exists

        Returns:
            True if collection successful
        """
        output_file = output_dir / f"{dataverse_id}_datasets.csv"

        # Check if incremental update is needed
        if not force_refresh and output_file.exists():
            logger.debug(f"Skipping {dataverse_id} - file already exists")
            return True

        try:
            logger.info(f"Collecting datasets for dataverse: {dataverse_id}")

            # Get datasets from API
            response = self.client.get_dataverse_datasets(dataverse_id)

            if "data" not in response:
                logger.warning(f"No data found for dataverse: {dataverse_id}")
                return False

            # Convert to DataFrame and save
            df = pd.DataFrame(response["data"])

            if df.empty:
                logger.warning(f"No datasets found for dataverse: {dataverse_id}")
                return False

            ensure_directory(output_dir)
            df.to_csv(output_file, index=False)

            logger.info(f"Saved {len(df)} datasets for {dataverse_id} to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to collect datasets for {dataverse_id}: {e}")
            return False

    def collect_all_datasets(
        self,
        input_csv: str | None = None,
        output_dir: str | None = None,
        force_refresh: bool = False,
    ) -> bool:
        """Collect datasets for all dataverses in input CSV.

        Args:
            input_csv: Path to CSV with dataverse information
            output_dir: Output directory for dataset files
            force_refresh: Force refresh all datasets

        Returns:
            True if collection completed successfully
        """
        # Get configuration
        config = self.config.dataverse_config
        input_csv = input_csv or config.get(
            "input_csv", "data/dataverse_socialscience.csv"
        )
        output_dir_path = Path(output_dir or config.get("output_dir", "data/datasets/"))

        # Load dataverse list
        try:
            dataverses_df = pd.read_csv(input_csv)
        except Exception as e:
            logger.error(f"Failed to load dataverse CSV {input_csv}: {e}")
            return False

        if "dataverse_id" not in dataverses_df.columns:
            logger.error(f"dataverse_id column not found in {input_csv}")
            return False

        # Get unique dataverse IDs
        dataverse_ids = dataverses_df["dataverse_id"].dropna().unique()
        logger.info(f"Found {len(dataverse_ids)} dataverses to process")

        # Check for checkpoint
        checkpoint_name = "collect_datasets"
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        processed_ids = (
            set(checkpoint.get("processed_ids", [])) if checkpoint else set()
        )

        if force_refresh:
            processed_ids = set()

        # Process each dataverse
        success_count = 0
        total_count = len(dataverse_ids)

        with LogProgress(logger, "collecting datasets", total_count) as progress:
            for dataverse_id in dataverse_ids:
                if dataverse_id in processed_ids:
                    logger.debug(f"Skipping {dataverse_id} - already processed")
                    progress.update()
                    continue

                success = self.collect_datasets_for_dataverse(
                    dataverse_id, output_dir_path, force_refresh
                )

                if success:
                    success_count += 1
                    processed_ids.add(dataverse_id)

                progress.update()

                # Update checkpoint
                self.checkpoint_manager.save_checkpoint(
                    checkpoint_name,
                    {
                        "processed_ids": list(processed_ids),
                        "success_count": success_count,
                        "total_count": total_count,
                    },
                )

        # Clear checkpoint on completion
        if success_count == total_count:
            self.checkpoint_manager.clear_checkpoint(checkpoint_name)

        logger.info(
            f"Dataset collection completed: {success_count}/{total_count} successful"
        )
        return success_count > 0


def main() -> None:
    """Main entry point for dataset collection."""
    import argparse

    from softverse.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Collect datasets from Dataverse")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--input-csv", type=str, help="Path to CSV file with dataverse information"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for dataset files"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh all datasets"
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

    # Create collector and run
    collector = DataverseDatasetCollector(args.config)
    success = collector.collect_all_datasets(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        force_refresh=args.force_refresh,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

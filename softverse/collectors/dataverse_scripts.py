"""Dataverse script file collector."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from softverse.config import get_config
from softverse.utils.api_utils import get_dataverse_client
from softverse.utils.file_utils import (
    CheckpointManager,
    ensure_directory,
    is_file_newer,
)
from softverse.utils.logging_utils import LogProgress, get_logger

logger = get_logger("dataverse_scripts")


class DataverseScriptCollector:
    """Collects script files from Dataverse datasets."""

    def __init__(self, config_path: str | None = None):
        """Initialize collector.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.client = get_dataverse_client()
        self.checkpoint_manager = CheckpointManager()

    def extract_files_info(
        self, datasets_csv: str, output_dir: Path, force_refresh: bool = False
    ) -> bool:
        """Extract script file information from datasets.

        Args:
            datasets_csv: Path to CSV with dataset information
            output_dir: Output directory for files CSV
            force_refresh: Force refresh even if output exists

        Returns:
            True if extraction successful
        """
        try:
            # Read datasets CSV
            df = pd.read_csv(datasets_csv)
            df = df.dropna(subset=["persistentUrl"])

            # Determine output file name
            file_name = Path(datasets_csv).stem
            output_file = output_dir / f"{file_name}_files.csv"

            # Check if incremental update needed
            if not force_refresh and output_file.exists():
                if is_file_newer(datasets_csv, output_file):
                    logger.debug(f"Skipping {file_name} - output is up to date")
                    return True

            logger.info(f"Extracting file info for {file_name} ({len(df)} datasets)")

            files = []
            valid_extensions = self.config.get(
                "processing.valid_extensions", [".R", ".py", ".do"]
            )

            for _, row in df.iterrows():
                try:
                    # Convert DOI format
                    doi = row.persistentUrl.replace("https://doi.org/", "doi:")

                    # Get dataset details
                    dataset = self.client.get_dataset(doi)

                    if dataset.get("status") != "OK":
                        logger.warning(f"Failed to get dataset {doi}")
                        continue

                    data = dataset.get("data", {})
                    if "latestVersion" not in data:
                        continue

                    # Extract relevant files
                    for file_info in data["latestVersion"].get("files", []):
                        datafile = file_info.get("dataFile", {})
                        filename = datafile.get("filename", "")

                        if any(filename.endswith(ext) for ext in valid_extensions):
                            files.append(
                                {"doi": doi, "fid": datafile.get("id"), "fn": filename}
                            )

                except Exception as e:
                    logger.warning(
                        f"Error processing dataset {row.get('persistentUrl', 'unknown')}: {e}"
                    )
                    continue

            # Save results
            files_df = pd.DataFrame(files)
            ensure_directory(output_dir)
            files_df.to_csv(output_file, index=False)

            logger.info(f"Found {len(files_df)} script files for {file_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract files info from {datasets_csv}: {e}")
            return False

    def download_file_wrapper(self, args: tuple[str, str, str, str, str]) -> bool:
        """Wrapper for downloading a single file (for multiprocessing).

        Args:
            args: Tuple of (file_id, filename, doi, token, output_base_dir)

        Returns:
            True if download successful
        """
        file_id, filename, doi, token, output_base_dir = args

        try:
            # Create output directory structure
            repo_id = doi.split("/")[-1]
            output_dir = Path(output_base_dir) / repo_id
            ensure_directory(output_dir)

            output_path = output_dir / filename

            # Skip if file already exists
            if output_path.exists():
                logger.debug(f"Skipping {filename} - already exists")
                return True

            # Download file
            success = self.client.download_file(file_id, output_path)
            if success:
                logger.debug(f"Downloaded: {filename}")

            return success

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    def download_files(
        self, files_csv: str, output_dir: str, num_workers: int | None = None
    ) -> bool:
        """Download files listed in CSV.

        Args:
            files_csv: Path to CSV with file information
            output_dir: Base output directory for downloads
            num_workers: Number of parallel workers

        Returns:
            True if downloads completed
        """
        try:
            # Read files CSV
            df = pd.read_csv(files_csv)
            if df.empty:
                logger.info(f"No files to download from {files_csv}")
                return True

            # Get configuration
            if num_workers is None:
                num_workers = self.config.get("processing.parallel_workers", 4)

            token = self.config.get_api_token("dataverse")

            # Prepare download arguments
            download_args = [
                (row["fid"], row["fn"], row["doi"], token, output_dir)
                for _, row in df.iterrows()
            ]

            logger.info(
                f"Downloading {len(download_args)} files using {num_workers} workers"
            )

            # Download files in parallel
            success_count = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(self.download_file_wrapper, download_args)
                success_count = sum(1 for result in results if result)

            logger.info(
                f"Download completed: {success_count}/{len(download_args)} successful"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"Failed to download files from {files_csv}: {e}")
            return False

    def collect_all_scripts(
        self,
        datasets_dir: str | None = None,
        output_dir: str | None = None,
        force_refresh: bool = False,
        num_workers: int | None = None,
    ) -> bool:
        """Collect all script files from Dataverse.

        Args:
            datasets_dir: Directory with dataset CSV files
            output_dir: Output directory for script files
            force_refresh: Force refresh all files
            num_workers: Number of parallel workers

        Returns:
            True if collection successful
        """
        # Get configuration
        config = self.config.dataverse_config
        datasets_dir_path = Path(
            datasets_dir or config.get("output_dir", "data/datasets/")
        )
        output_dir = output_dir or "scripts/dataverse/"

        ensure_directory(output_dir)
        files_dir = Path("files_dfs")
        ensure_directory(files_dir)

        # Find dataset CSV files
        dataset_files = list(datasets_dir_path.glob("*_datasets.csv"))
        if not dataset_files:
            logger.error(f"No dataset CSV files found in {datasets_dir_path}")
            return False

        logger.info(f"Processing {len(dataset_files)} dataset files")

        # Step 1: Extract file information
        with LogProgress(
            logger, "extracting file info", len(dataset_files)
        ) as progress:
            for dataset_file in dataset_files:
                success = self.extract_files_info(
                    str(dataset_file), files_dir, force_refresh
                )
                if not success:
                    logger.warning(f"Failed to extract files info from {dataset_file}")
                progress.update()

        # Step 2: Download files
        files_csvs = list(files_dir.glob("*_files.csv"))
        total_success = 0

        with LogProgress(logger, "downloading files", len(files_csvs)) as progress:
            for files_csv in files_csvs:
                success = self.download_files(str(files_csv), output_dir, num_workers)
                if success:
                    total_success += 1
                progress.update()

        logger.info(
            f"Script collection completed: {total_success}/{len(files_csvs)} file groups processed"
        )
        return total_success > 0


def main():
    """Main entry point for Dataverse script collection."""
    import argparse

    from softverse.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Collect script files from Dataverse")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--datasets-dir", type=str, help="Directory with dataset CSV files"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for script files"
    )
    parser.add_argument(
        "--force-refresh", action="store_true", help="Force refresh all files"
    )
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
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
    collector = DataverseScriptCollector(args.config)
    success = collector.collect_all_scripts(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        force_refresh=args.force_refresh,
        num_workers=args.workers,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

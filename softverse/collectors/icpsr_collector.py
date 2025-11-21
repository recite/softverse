"""ICPSR script file collector."""

from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.util.retry import Retry

from softverse.config import get_config
from softverse.utils.file_utils import (
    ArchiveProcessor,
    CheckpointManager,
    ensure_directory,
)
from softverse.utils.logging_utils import ArchiveProgressLogger, LogProgress, get_logger

logger = get_logger("icpsr_collector")


class ICPSRScriptCollector:
    """Collects script files from OpenICPSR."""

    def __init__(self, config_path: str | None = None):
        """Initialize collector.

        Args:
            config_path: Path to configuration file
        """
        self.config = get_config(config_path)
        self.checkpoint_manager = CheckpointManager()
        self.archive_processor = ArchiveProcessor(config_path)

    def _get_selenium_driver(self) -> webdriver.Chrome:
        """Get configured Selenium Chrome driver."""
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        return webdriver.Chrome(options=options)

    def search_icpsr_aea(self, max_records: int = 100) -> pd.DataFrame:
        """Search AEA OpenICPSR studies using the working approach from notebook.

        Args:
            max_records: Maximum number of records to fetch

        Returns:
            DataFrame with search results
        """
        import json

        from bs4 import BeautifulSoup

        try:
            # Use the AEA-specific URL that was working in the notebook
            url = f"https://www.openicpsr.org/openicpsr/search/aea/studies?start=1&ARCHIVE=aea&rows={max_records}"
            logger.info(f"Searching AEA ICPSR studies: {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Extract searchResults like in the notebook
            if "searchResults :" not in response.text:
                logger.warning("No searchResults found in response")
                return pd.DataFrame()

            results_text = response.text.split("searchResults : ")[1]

            # Find the starting index of the "docs" array
            start_index = results_text.find('"docs":[')
            if start_index == -1:
                logger.warning("No docs array found")
                return pd.DataFrame()

            # Extract the content from the "docs" array
            docs_content = results_text[start_index + len('"docs":[') :]

            # Find the index of the closing bracket for "docs" array
            end_index = docs_content.rfind("]")
            if end_index == -1:
                logger.warning("Closing bracket for docs not found")
                return pd.DataFrame()

            # Extract the "docs" array content
            docs_array = docs_content[: end_index + 1]
            docs = "[" + docs_array.split(',"numFound"')[0]

            # Parse JSON
            data = json.loads(docs)
            df = pd.json_normalize(data)

            if df.empty:
                logger.warning("No studies found in parsed data")
                return df

            # Clean up SUMMARY field if it exists
            if "SUMMARY" in df.columns:
                df["SUMMARY"] = (
                    df["SUMMARY"]
                    .astype(str)
                    .apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
                )
                df["SUMMARY"] = (
                    df["SUMMARY"]
                    .str.replace("[", "")
                    .str.replace("]", "")
                    .str.strip("'")
                )

            # Rename columns to match expected format
            column_mapping = {
                "ID": "study_id",
                "TITLE": "title",
                "SUMMARY": "summary",
                "AUTHOR": "author",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})

            # Add URL for each study
            if "study_id" in df.columns:
                df["url"] = df["study_id"].apply(
                    lambda x: f"https://www.openicpsr.org/openicpsr/project/{x}"
                )

            logger.info(f"Found {len(df)} AEA ICPSR studies")
            return df

        except Exception as e:
            logger.error(f"Failed to search AEA ICPSR studies: {e}")
            return pd.DataFrame()

    def search_icpsr(
        self,
        query: str = "",
        max_pages: int | None = None,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Search OpenICPSR for datasets.

        Args:
            query: Search query (empty for all)
            max_pages: Maximum pages to scrape
            output_dir: Directory to save metadata

        Returns:
            DataFrame with search results
        """
        config = self.config.icpsr_config
        output_dir_path = Path(output_dir or config.get("output_dir", "scripts/icpsr/"))
        metadata_dir = output_dir_path / "metadata"
        ensure_directory(metadata_dir)

        # Try AEA search first as it's more reliable
        if not query or query.lower() in ["aea", "all", ""]:
            logger.info("Using AEA ICPSR search (more reliable)")
            aea_results = self.search_icpsr_aea(max_records=1000)
            if not aea_results.empty:
                # Save results
                metadata_file = metadata_dir / "icpsr_aea_search_results.csv"
                aea_results.to_csv(metadata_file, index=False)
                logger.info(
                    f"Saved {len(aea_results)} AEA ICPSR studies to {metadata_file}"
                )
                return aea_results

        # Fall back to Selenium-based search for general queries
        base_url = "https://www.openicpsr.org/openicpsr/search/studies"
        all_results = []

        driver = None
        try:
            driver = self._get_selenium_driver()
            page = 1

            with LogProgress(logger, "searching ICPSR", max_pages) as progress:
                while True:
                    # Construct search URL
                    url = f"{base_url}?start={(page - 1) * 10}"
                    if query:
                        url += f"&q={query}"

                    logger.info(f"Searching ICPSR page {page}: {url}")
                    driver.get(url)

                    # Wait for page to load completely
                    WebDriverWait(driver, 15).until(
                        lambda d: d.execute_script("return document.readyState")
                        == "complete"
                    )

                    # Wait for results to load with better selectors
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located(
                                (
                                    By.CSS_SELECTOR,
                                    "[class*='result'], a[href*='project']",
                                )
                            )
                        )
                    except Exception:
                        logger.warning(f"No results found on page {page}")
                        break

                    # Extract study information using improved selectors
                    project_links = driver.find_elements(
                        By.CSS_SELECTOR, "a[href*='project']"
                    )

                    if not project_links:
                        logger.debug(f"No project links found on page {page}")
                        break

                    page_results = []
                    for link in project_links:
                        try:
                            title = link.text.strip()
                            study_url = link.get_attribute("href")

                            # Extract study ID from URL
                            if study_url and "/project/" in study_url:
                                study_id = study_url.split("/project/")[-1].split("/")[
                                    0
                                ]
                            else:
                                continue

                            if title and study_id:
                                page_results.append(
                                    {
                                        "title": title,
                                        "study_id": study_id,
                                        "url": study_url,
                                        "page": page,
                                    }
                                )

                        except Exception as e:
                            logger.debug(f"Error extracting study info: {e}")
                            continue

                    # Remove duplicates based on study_id
                    seen_ids = set()
                    unique_results = []
                    for result in page_results:
                        if result["study_id"] not in seen_ids:
                            seen_ids.add(result["study_id"])
                            unique_results.append(result)

                    all_results.extend(unique_results)
                    logger.info(
                        f"Found {len(unique_results)} unique studies on page {page}"
                    )

                    # Check if we should continue
                    if max_pages and page >= max_pages:
                        break

                    page += 1
                    progress.update()

                    # Check if there are more results
                    if len(unique_results) == 0:
                        break

        finally:
            if driver:
                driver.quit()

        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            metadata_file = metadata_dir / "icpsr_search_results.csv"
            results_df.to_csv(metadata_file, index=False)
            logger.info(f"Saved {len(results_df)} ICPSR studies to {metadata_file}")
            return results_df
        else:
            logger.warning("No ICPSR studies found")
            return pd.DataFrame()

    def _get_authenticated_session(self):
        """Get authenticated session for ICPSR downloads.

        Returns:
            Requests session with authentication
        """
        session = requests.Session()

        # Add retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Get credentials from environment variables or .env file
        import os
        from pathlib import Path

        # Load from .env file if it exists
        try:
            from dotenv import load_dotenv

            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        except ImportError:
            # Fallback to manual parsing if python-dotenv not available
            env_file = Path(__file__).parent.parent.parent / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        if line.strip() and not line.startswith("#"):
                            key, value = line.strip().split("=", 1)
                            if key in ["ICPSR_EMAIL", "ICPSR_PASSWORD"]:
                                os.environ[key] = value

        email = os.getenv("ICPSR_EMAIL")
        password = os.getenv("ICPSR_PASSWORD")

        if email and password:
            try:
                # Login to ICPSR like in the notebook
                login_page = session.get(
                    "https://login.icpsr.umich.edu/realms/icpsr/protocol/openid-connect/auth"
                    "?client_id=openicpsr-web-prod&response_type=code&login=true"
                    "&redirect_uri=https://www.openicpsr.org/openicpsr/oauth/callback"
                )

                from bs4 import BeautifulSoup

                soup = BeautifulSoup(login_page.content, "html.parser")
                form = soup.find("form", id="kc-form-login")

                if form:
                    login_url = form.get("action")
                    if login_url and isinstance(login_url, str):
                        login_data = {"username": email, "password": password}
                        session.post(login_url, data=login_data)
                        logger.info("Successfully authenticated with ICPSR")
                    else:
                        logger.warning("Login form action URL not found")
                else:
                    logger.warning("Could not find login form, proceeding without auth")

            except Exception as e:
                logger.warning(f"Authentication failed: {e}, proceeding without auth")
        else:
            logger.info("No ICPSR credentials configured, using public access")

        return session

    def download_and_process_study(
        self,
        study_id: str,
        temp_dir: Path,
        output_dir: Path,
        valid_extensions: list[str],
        progress_logger: ArchiveProgressLogger | None = None,
    ) -> tuple[bool, int]:
        """Download and process data files for a single study.

        Args:
            study_id: ICPSR study identifier
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

            # Construct download URL (like in the notebook)
            download_url = (
                f"https://www.openicpsr.org/openicpsr/project/{study_id}/version/V1/download/project"
                f"?dirPath=/openicpsr/{study_id}/fcr:versions/V1"
            )

            zip_file_path = temp_dir / f"ICPSR_{str(study_id).zfill(5)}.zip"
            extract_to = output_dir / str(study_id)

            # Check if already processed
            if extract_to.exists():
                logger.debug(f"Skipping {study_id} - already processed")
                script_count = len(list(extract_to.rglob("*")))
                if progress_logger:
                    progress_logger.update_extraction(study_id, True, script_count)
                return True, script_count

            # Get authenticated session
            session = self._get_authenticated_session()

            # Download the zip file to temp directory
            logger.debug(f"Downloading {study_id} from {download_url}")
            response = session.get(download_url, timeout=300, stream=True)

            # Check if we got a real download or an error page
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if (
                    "application/zip" in content_type
                    or "application/octet-stream" in content_type
                ):
                    # Looks like a real zip file
                    with open(zip_file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Check file size - very small files might be error pages
                    file_size = zip_file_path.stat().st_size
                    if file_size < 1000:  # Less than 1KB is probably an error
                        logger.warning(
                            f"Downloaded file for {study_id} is very small ({file_size} bytes), might be an error"
                        )
                        zip_file_path.unlink(missing_ok=True)
                        return False, 0

                    # Get file size for progress tracking
                    if progress_logger:
                        progress_logger.update_download(study_id, file_size)

                    logger.debug(f"Downloaded {file_size} bytes for study {study_id}")

                else:
                    logger.warning(
                        f"Unexpected content type for {study_id}: {content_type}"
                    )
                    return False, 0
            else:
                logger.warning(
                    f"Download failed for {study_id}: HTTP {response.status_code}"
                )
                return False, 0

            # Process archive with automatic cleanup
            success, script_count = self.archive_processor.process_archive_with_cleanup(
                zip_file_path, extract_to, valid_extensions
            )

            # Update progress with extraction results
            if progress_logger:
                error_msg = None if success else "Extraction failed"
                progress_logger.update_extraction(
                    study_id, success, script_count, error_msg
                )

            return success, script_count

        except Exception as e:
            logger.error(f"Failed to download/process study {study_id}: {e}")
            if progress_logger:
                progress_logger.update_extraction(study_id, False, 0, str(e))
            return False, 0

    def collect_all_scripts(
        self,
        output_dir: str | None = None,
        search_query: str = "",
        max_studies: int | None = None,
        force_refresh: bool = False,
    ) -> bool:
        """Collect all script files from ICPSR.

        Args:
            output_dir: Output directory for script files
            search_query: Search query for ICPSR
            max_studies: Maximum number of studies to process
            force_refresh: Force refresh all downloads

        Returns:
            True if collection successful
        """
        # Get configuration
        config = self.config.icpsr_config
        output_dir_path = Path(output_dir or config.get("output_dir", "scripts/icpsr/"))
        ensure_directory(output_dir_path)

        # Use temp directory from archive processor
        temp_dir = self.archive_processor.get_temp_dir()

        try:
            # Check storage requirements before starting
            if not self.archive_processor.check_storage_requirements():
                logger.error("Insufficient storage space to begin collection")
                return False

            # Step 1: Search for studies
            logger.info("Searching ICPSR for studies")
            studies_df = self.search_icpsr(
                query=search_query, max_pages=10, output_dir=str(output_dir_path)
            )

            if studies_df.empty:
                logger.warning("No ICPSR studies found")
                return False

            # Limit studies if specified
            if max_studies and len(studies_df) > max_studies:
                studies_df = studies_df.head(max_studies)
                logger.info(f"Limited to {max_studies} studies")

            # Check checkpoint
            checkpoint_name = "icpsr_downloads"
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
            processed_ids = (
                set(checkpoint.get("processed_ids", [])) if checkpoint else set()
            )

            if force_refresh:
                processed_ids = set()

            # Step 2: Process studies with optimized storage
            success_count = 0
            total_script_count = 0
            valid_extensions = self.config.get(
                "processing.valid_extensions", [".R", ".py", ".do", ".ipynb"]
            )

            # Sequential processing for optimal storage usage
            storage_config = self.config.get("storage", {})
            sequential_mode = storage_config.get("sequential_processing", True)

            if sequential_mode:
                logger.info("Using sequential processing to minimize storage usage")

            # Use detailed archive progress logging
            with ArchiveProgressLogger(
                logger, "ICPSR archive collection", len(studies_df), show_size_info=True
            ) as archive_progress:
                # Log initial storage status
                from softverse.utils.file_utils import get_disk_usage

                try:
                    total, used, free = get_disk_usage(str(temp_dir))
                    free_gb = float(free) / (1024**3)
                    archive_progress.log_storage_info(free_gb)
                except Exception as e:
                    logger.debug(f"Could not get disk usage info: {e}")

                for _, study in studies_df.iterrows():
                    study_id = study.get("study_id", "")

                    if study_id in processed_ids:
                        continue

                    # Download and process study with detailed progress tracking
                    success, script_count = self.download_and_process_study(
                        study_id,
                        temp_dir,
                        output_dir_path,
                        valid_extensions,
                        archive_progress,
                    )

                    if success:
                        success_count += 1
                        total_script_count += script_count
                        processed_ids.add(study_id)

                    # Update checkpoint
                    self.checkpoint_manager.save_checkpoint(
                        checkpoint_name,
                        {
                            "processed_ids": list(processed_ids),
                            "success_count": success_count,
                            "total_count": len(studies_df),
                            "total_script_count": total_script_count,
                        },
                    )

            # Final cleanup
            self.archive_processor.cleanup_temp_files()

            # Clear checkpoint on completion
            if success_count == len(studies_df):
                self.checkpoint_manager.clear_checkpoint(checkpoint_name)

            logger.info(
                f"ICPSR collection completed: {success_count}/{len(studies_df)} studies processed, "
                f"{total_script_count} script files extracted"
            )
            return success_count > 0

        except Exception as e:
            logger.error(f"ICPSR collection failed: {e}")
            return False


def main() -> None:
    """Main entry point for ICPSR script collection."""
    import argparse

    from softverse.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description="Collect script files from ICPSR")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for script files"
    )
    parser.add_argument("--search-query", type=str, default="", help="Search query")
    parser.add_argument(
        "--max-studies", type=int, help="Maximum number of studies to process"
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
    collector = ICPSRScriptCollector(args.config)

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
        logger.info("Attempting to recover previously failed ICPSR archives")
        recovered_count = collector.archive_processor.recover_failed_archives()
        logger.info(f"Recovery completed: {recovered_count} archives recovered")
        exit(0)

    success = collector.collect_all_scripts(
        output_dir=args.output_dir,
        search_query=args.search_query,
        max_studies=args.max_studies,
        force_refresh=args.force_refresh,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

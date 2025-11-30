"""ResearchBox collector for fetching research data via S3 and web scraping."""

import re
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from softverse.collectors.base_collector import BaseCollectorWithStatus
from softverse.utils.file_utils import ensure_directory
from softverse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResearchBoxCollector(BaseCollectorWithStatus):
    """Collector for ResearchBox research repositories via S3 direct access."""

    S3_BASE_URL = "https://s3.wasabisys.com/zipballs.researchbox.org/"
    WEB_BASE_URL = "https://researchbox.org/"
    DEFAULT_RATE_LIMIT_DELAY = 1.0  # seconds between requests

    def __init__(
        self,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        max_retries: int = 3,
    ):
        """Initialize ResearchBox collector.

        Args:
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Softverse ResearchBox Collector (Academic Research)"}
        )

    def _make_request(
        self, url: str, timeout: int = 30, method: str = "GET"
    ) -> requests.Response | None:
        """Make a rate-limited request.

        Args:
            url: URL to request
            timeout: Request timeout in seconds
            method: HTTP method (GET, HEAD)

        Returns:
            Response object or None if error
        """
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)

                if method.upper() == "HEAD":
                    response = self.session.head(url, timeout=timeout)
                else:
                    response = self.session.get(url, timeout=timeout)

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Failed to request {url} after {self.max_retries} attempts: {e}"
                    )
                    return None
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(self.rate_limit_delay * (attempt + 1))

        return None

    def check_researchbox_exists(self, box_id: int) -> dict[str, Any] | None:
        """Check if a ResearchBox ID exists and get its metadata.

        Args:
            box_id: ResearchBox ID to check

        Returns:
            Metadata dict or None if not found
        """
        # Check S3 ZIP file availability
        zip_url = f"{self.S3_BASE_URL}ResearchBox_{box_id}.zip"
        response = self._make_request(zip_url, method="HEAD")

        if not response:
            return None

        metadata = {
            "box_id": box_id,
            "zip_url": zip_url,
            "zip_size": int(response.headers.get("Content-Length", 0)),
            "last_modified": response.headers.get("Last-Modified"),
        }

        # Try to get additional metadata from web page
        web_metadata = self._get_web_metadata(box_id)
        if web_metadata:
            metadata.update(web_metadata)

        return metadata

    def _get_web_metadata(self, box_id: int) -> dict[str, Any] | None:
        """Get metadata from ResearchBox web page.

        Args:
            box_id: ResearchBox ID

        Returns:
            Metadata dict or None if error
        """
        web_url = f"{self.WEB_BASE_URL}{box_id}"
        response = self._make_request(web_url)

        if not response:
            return None

        try:
            soup = BeautifulSoup(response.text, "html.parser")

            metadata = {"web_url": web_url}

            # Extract title
            title_elem = soup.find("title")
            if title_elem:
                metadata["title"] = title_elem.get_text().strip()

            # Look for author information
            author_pattern = re.search(
                r"author[s]?[:\s]+([^<>\n]+)", response.text, re.IGNORECASE
            )
            if author_pattern:
                metadata["author"] = author_pattern.group(1).strip()

            # Look for publication info
            journal_pattern = re.search(
                r"journal[:\s]+([^<>\n]+)", response.text, re.IGNORECASE
            )
            if journal_pattern:
                metadata["journal"] = journal_pattern.group(1).strip()

            # Count files in bingo table
            file_links = soup.find_all("a", href=True)
            file_count = len(
                [link for link in file_links if "download" in str(link.get("href", ""))]
            )
            if file_count > 0:
                metadata["file_count"] = str(file_count)

            # Look for code/script files
            script_extensions = [".R", ".py", ".do", ".ipynb", ".m", ".sas"]
            script_files = []
            for link in file_links:
                href = str(link.get("href", ""))
                for ext in script_extensions:
                    if ext.lower() in href.lower():
                        script_files.append(link.get_text().strip())

            if script_files:
                metadata["script_files"] = str(script_files)
                metadata["script_count"] = str(len(script_files))

            return metadata

        except Exception as e:
            logger.warning(f"Failed to parse web metadata for box {box_id}: {e}")
            return None

    def discover_available_boxes(
        self, start_id: int = 1, end_id: int = 1000, batch_size: int = 50
    ) -> list[dict[str, Any]]:
        """Discover available ResearchBox IDs in a range.

        Args:
            start_id: Starting ID to check
            end_id: Ending ID to check
            batch_size: Number of IDs to check in parallel batches

        Returns:
            List of available box metadata
        """
        logger.info(f"Discovering ResearchBox IDs from {start_id} to {end_id}")
        available_boxes = []

        for batch_start in range(start_id, end_id + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_id)
            logger.info(f"Checking batch: {batch_start}-{batch_end}")

            batch_boxes = []
            for box_id in range(batch_start, batch_end + 1):
                metadata = self.check_researchbox_exists(box_id)
                if metadata:
                    batch_boxes.append(metadata)
                    logger.debug(
                        f"Found ResearchBox {box_id} (Size: {metadata['zip_size']} bytes)"
                    )

            available_boxes.extend(batch_boxes)
            logger.info(
                f"Batch {batch_start}-{batch_end}: Found {len(batch_boxes)} boxes"
            )

        logger.info(f"Discovery complete: Found {len(available_boxes)} total boxes")
        return available_boxes

    def download_researchbox(
        self,
        box_id: int,
        output_dir: Path,
        extract: bool = True,
        keep_zip: bool = False,
    ) -> dict[str, Any]:
        """Download a ResearchBox and optionally extract it.

        Args:
            box_id: ResearchBox ID to download
            output_dir: Directory to save files
            extract: Whether to extract the ZIP file
            keep_zip: Whether to keep ZIP file after extraction

        Returns:
            Download statistics
        """
        ensure_directory(output_dir)
        box_dir = output_dir / f"ResearchBox_{box_id}"
        ensure_directory(box_dir)

        stats = {
            "box_id": box_id,
            "downloaded": False,
            "extracted": False,
            "file_count": 0,
            "script_count": 0,
        }

        try:
            # Get metadata first
            metadata = self.check_researchbox_exists(box_id)
            if not metadata:
                logger.warning(f"ResearchBox {box_id} not found")
                self._write_error_status(box_dir, "ResearchBox not found", str(box_id))
                return stats

            # Download ZIP file
            zip_url = metadata["zip_url"]
            zip_path = box_dir / f"ResearchBox_{box_id}.zip"

            logger.info(
                f"Downloading ResearchBox {box_id} ({metadata['zip_size']} bytes)"
            )

            response = self._make_request(zip_url)
            if not response:
                self._write_error_status(box_dir, "Failed to download ZIP", str(box_id))
                return stats

            with open(zip_path, "wb") as f:
                f.write(response.content)

            stats["downloaded"] = True
            logger.info(f"Downloaded ResearchBox {box_id} to {zip_path}")

            # Extract if requested
            if extract:
                extract_dir = box_dir / "extracted"
                ensure_directory(extract_dir)

                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)

                    stats["extracted"] = True

                    # Count files
                    all_files = list(extract_dir.rglob("*"))
                    file_count = len([f for f in all_files if f.is_file()])
                    stats["file_count"] = file_count

                    # Count script files
                    script_extensions = [".R", ".py", ".do", ".ipynb", ".m", ".sas"]
                    script_files = []
                    for file_path in all_files:
                        if file_path.is_file() and any(
                            file_path.suffix.lower() == ext.lower()
                            for ext in script_extensions
                        ):
                            script_files.append(file_path)

                    stats["script_count"] = len(script_files)
                    logger.info(
                        f"Extracted {file_count} files, {len(script_files)} scripts"
                    )

                    # Save metadata
                    metadata_file = box_dir / "metadata.json"
                    import json

                    with open(metadata_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

                except Exception as e:
                    logger.error(f"Failed to extract ResearchBox {box_id}: {e}")
                    self._write_error_status(
                        box_dir, f"Extraction failed: {e}", str(box_id)
                    )
                    return stats

            # Remove ZIP if not keeping it
            if extract and not keep_zip:
                zip_path.unlink()

            # Write success status
            self._write_success_status(
                box_dir,
                stats["script_count"],
                str(box_id),
                {"total_files": stats["file_count"], "zip_size": metadata["zip_size"]},
            )

        except Exception as e:
            logger.error(f"Error processing ResearchBox {box_id}: {e}")
            self._write_error_status(box_dir, str(e), str(box_id))

        return stats

    def collect_researchboxes(
        self,
        box_ids: list[int],
        output_dir: Path,
        extract: bool = True,
        keep_zip: bool = False,
    ) -> dict[str, Any]:
        """Collect multiple ResearchBox repositories.

        Args:
            box_ids: List of ResearchBox IDs to collect
            output_dir: Directory to save collected data
            extract: Whether to extract ZIP files
            keep_zip: Whether to keep ZIP files after extraction

        Returns:
            Collection statistics
        """
        ensure_directory(output_dir)

        stats = {
            "total_requested": len(box_ids),
            "successful_downloads": 0,
            "successful_extractions": 0,
            "total_files": 0,
            "total_scripts": 0,
            "errors": [],
        }

        for box_id in tqdm(box_ids, desc="Collecting ResearchBoxes"):
            try:
                result = self.download_researchbox(
                    box_id, output_dir, extract=extract, keep_zip=keep_zip
                )

                if result["downloaded"]:
                    stats["successful_downloads"] += 1

                if result["extracted"]:
                    stats["successful_extractions"] += 1

                stats["total_files"] += result["file_count"]
                stats["total_scripts"] += result["script_count"]

            except Exception as e:
                error_msg = f"Error processing ResearchBox {box_id}: {e}"
                logger.error(error_msg)
                stats["errors"].append({"box_id": box_id, "error": str(e)})

        return stats

    def collect_range(
        self,
        start_id: int,
        end_id: int,
        output_dir: Path,
        discover_first: bool = True,
        extract: bool = True,
        keep_zip: bool = False,
    ) -> dict[str, Any]:
        """Collect ResearchBoxes in a range of IDs.

        Args:
            start_id: Starting ResearchBox ID
            end_id: Ending ResearchBox ID
            output_dir: Directory to save collected data
            discover_first: Whether to discover available IDs first
            extract: Whether to extract ZIP files
            keep_zip: Whether to keep ZIP files after extraction

        Returns:
            Collection statistics
        """
        logger.info(f"Collecting ResearchBoxes from ID {start_id} to {end_id}")

        if discover_first:
            available_boxes = self.discover_available_boxes(start_id, end_id)
            box_ids = [box["box_id"] for box in available_boxes]
            logger.info(f"Found {len(box_ids)} available ResearchBoxes to collect")
        else:
            box_ids = list(range(start_id, end_id + 1))
            logger.info(
                f"Attempting to collect all IDs in range ({len(box_ids)} total)"
            )

        return self.collect_researchboxes(
            box_ids, output_dir, extract=extract, keep_zip=keep_zip
        )

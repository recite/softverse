"""API utilities for Softverse."""

import time
from pathlib import Path
from typing import Any

import requests

from softverse.config import get_config
from softverse.utils.logging_utils import get_logger

logger = get_logger("api_utils")


class APIClient:
    """Base API client with retry logic and rate limiting."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        """Initialize API client.

        Args:
            base_url: Base URL for API
            token: API token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        config = get_config()

        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout or config.get("api.request_timeout", 30)
        self.max_retries = max_retries or config.get("api.max_retries", 3)
        self.retry_delay = retry_delay or config.get("api.retry_delay", 1.0)

        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        **kwargs,
    ) -> requests.Response:
        """Make API request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request data
            **kwargs: Additional request arguments

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response

            except requests.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"API request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

                logger.warning(f"API request attempt {attempt + 1} failed: {e}")
                time.sleep(self.retry_delay * (attempt + 1))

        # This should never be reached
        raise requests.RequestException("Unexpected error in request retry logic")

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None, **kwargs
    ) -> requests.Response:
        """Make GET request."""
        return self._make_request("GET", endpoint, params=params, **kwargs)

    def post(
        self, endpoint: str, data: dict[str, Any] | None = None, **kwargs
    ) -> requests.Response:
        """Make POST request."""
        return self._make_request("POST", endpoint, data=data, **kwargs)

    def download_file(
        self,
        endpoint: str,
        output_path: str | Path,
        params: dict[str, Any] | None = None,
        chunk_size: int | None = None,
    ) -> bool:
        """Download file from API endpoint.

        Args:
            endpoint: API endpoint
            output_path: Path to save file
            params: Query parameters
            chunk_size: Download chunk size

        Returns:
            True if download successful
        """
        config = get_config()
        chunk_size = chunk_size or config.get("processing.chunk_size", 8192)
        output_path = Path(output_path)

        try:
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream download
            response = self._make_request("GET", endpoint, params=params, stream=True)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

            logger.debug(f"Downloaded file: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download file to {output_path}: {e}")
            return False


class DataverseClient(APIClient):
    """Client for Dataverse API."""

    def __init__(self, token: str | None = None):
        """Initialize Dataverse client.

        Args:
            token: Dataverse API token
        """
        config = get_config()
        base_url = config.get(
            "data_sources.dataverse.base_url", "https://dataverse.harvard.edu/api"
        )

        if token is None:
            token = config.get_api_token("dataverse")

        super().__init__(base_url, token)

    def get_dataverse_datasets(self, dataverse_id: str) -> dict[str, Any]:
        """Get datasets from a dataverse.

        Args:
            dataverse_id: Dataverse identifier

        Returns:
            API response data
        """
        endpoint = f"dataverses/{dataverse_id}/contents"
        response = self.get(endpoint)
        return response.json()

    def get_dataset(self, doi: str) -> dict[str, Any]:
        """Get dataset details by DOI.

        Args:
            doi: Dataset DOI

        Returns:
            API response data
        """
        endpoint = "datasets/:persistentId"
        params = {"persistentId": doi}
        response = self.get(endpoint, params=params)
        return response.json()

    def download_file(
        self,
        file_id: str,
        output_path: str | Path,
        params: dict[str, Any] | None = None,
        chunk_size: int | None = None,
    ) -> bool:
        """Download file from Dataverse.

        Args:
            file_id: File identifier
            output_path: Path to save file

        Returns:
            True if download successful
        """
        endpoint = f"access/datafile/{file_id}"
        file_params = {"key": self.token} if self.token else None
        # Merge with any additional params passed
        if params:
            file_params = {**(file_params or {}), **params}
        return super().download_file(endpoint, output_path, file_params, chunk_size)


class ZenodoClient(APIClient):
    """Client for Zenodo API."""

    def __init__(self, token: str | None = None):
        """Initialize Zenodo client.

        Args:
            token: Zenodo API token
        """
        config = get_config()
        base_url = config.get(
            "data_sources.zenodo.base_url", "https://zenodo.org/api/records"
        )

        if token is None:
            token = config.get_api_token("zenodo")

        super().__init__(base_url, token)

    def search_community(
        self, community: str, size: int = 100, page: int = 1
    ) -> dict[str, Any]:
        """Search records in Zenodo community.

        Args:
            community: Community identifier
            size: Number of results per page
            page: Page number

        Returns:
            API response data
        """
        params = {"communities": community, "size": size, "page": page}
        if self.token:
            params["access_token"] = self.token

        response = self.get("", params=params)
        return response.json()

    def get_all_community_records(
        self, community: str, max_pages: int | None = None
    ) -> list:
        """Get all records from a Zenodo community.

        Args:
            community: Community identifier
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all records
        """
        all_records = []
        page = 1

        while True:
            result = self.search_community(community, page=page)

            if not result["hits"]["hits"]:
                break

            all_records.extend(result["hits"]["hits"])

            if max_pages and page >= max_pages:
                break

            page += 1

        return all_records

    def download_record_archive(self, record_id: str, output_path: str | Path) -> bool:
        """Download record files archive.

        Args:
            record_id: Record identifier
            output_path: Path to save archive

        Returns:
            True if download successful
        """
        url = f"https://zenodo.org/api/records/{record_id}/files-archive"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.debug(f"Downloaded Zenodo archive: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download Zenodo archive {record_id}: {e}")
            return False


def get_dataverse_client() -> DataverseClient:
    """Get configured Dataverse client."""
    return DataverseClient()


def get_zenodo_client() -> ZenodoClient:
    """Get configured Zenodo client."""
    return ZenodoClient()

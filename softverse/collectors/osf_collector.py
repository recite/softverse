"""OSF (Open Science Framework) collector for fetching research data."""

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from tqdm import tqdm

from softverse.collectors.base_collector import BaseCollectorWithStatus
from softverse.config import get_config
from softverse.utils.file_utils import ensure_directory
from softverse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OSFCollector(BaseCollectorWithStatus):
    """Collector for OSF (Open Science Framework) repositories."""

    BASE_URL = "https://api.osf.io/v2/"
    DEFAULT_PAGE_SIZE = 100
    DEFAULT_RATE_LIMIT_DELAY = 0.5  # seconds between requests

    def __init__(
        self,
        api_token: str | None = None,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    ):
        """Initialize OSF collector.

        Args:
            api_token: OSF Personal Access Token for authentication
            rate_limit_delay: Delay between API requests in seconds
        """
        self.api_token = api_token or get_config().get_api_token("osf")
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})
            logger.info("OSF API token configured")
        else:
            logger.warning("No OSF API token provided. Rate limits will be restricted.")

    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Make a rate-limited request to the OSF API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data or None if error
        """
        url = urljoin(self.BASE_URL, endpoint)

        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from OSF API: {e}")
            return None

    def _paginate_results(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Paginate through all results from an endpoint.

        Args:
            endpoint: API endpoint path
            params: Initial query parameters

        Returns:
            List of all data items from paginated results
        """
        if params is None:
            params = {}

        params.setdefault("page[size]", self.DEFAULT_PAGE_SIZE)
        all_results = []
        next_url = None

        while True:
            if next_url:
                # Use the full next URL directly
                response = self.session.get(next_url)
                try:
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error during pagination: {e}")
                    break
            else:
                data = self._make_request(endpoint, params)
                if not data:
                    break

            # Extract items from data array
            if "data" in data:
                all_results.extend(data["data"])

            # Check for next page
            if "links" in data and "next" in data["links"]:
                next_url = data["links"]["next"]
                if next_url:
                    time.sleep(self.rate_limit_delay)
                else:
                    break
            else:
                break

        return all_results

    def get_nodes(
        self,
        filters: dict[str, str] | None = None,
        institution_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get nodes (projects) from OSF.

        Args:
            filters: Filter parameters (e.g., {"title": "keyword", "public": "true"})
            institution_id: Filter by institution ID

        Returns:
            List of node data
        """
        params = {}

        if filters:
            for key, value in filters.items():
                params[f"filter[{key}]"] = value

        if institution_id:
            endpoint = f"institutions/{institution_id}/nodes/"
        else:
            endpoint = "nodes/"

        logger.info(f"Fetching nodes from OSF with filters: {filters}")
        nodes = self._paginate_results(endpoint, params)
        logger.info(f"Retrieved {len(nodes)} nodes")

        return nodes

    def get_node_children(
        self, node_id: str, recursive: bool = False
    ) -> list[dict[str, Any]]:
        """Get children of a node.

        Args:
            node_id: OSF node ID
            recursive: Whether to recursively fetch all descendants

        Returns:
            List of child node data
        """
        endpoint = f"nodes/{node_id}/children/"
        children = self._paginate_results(endpoint)

        if recursive and children:
            all_children = list(children)
            for child in children:
                child_id = child.get("id")
                if child_id:
                    grandchildren = self.get_node_children(child_id, recursive=True)
                    all_children.extend(grandchildren)
            return all_children

        return children

    def get_node_files(self, node_id: str) -> list[dict[str, Any]]:
        """Get files associated with a node.

        Args:
            node_id: OSF node ID

        Returns:
            List of file metadata
        """
        endpoint = f"nodes/{node_id}/files/"
        providers = self._paginate_results(endpoint)

        all_files = []
        for provider in providers:
            if "relationships" in provider and "files" in provider["relationships"]:
                files_link = provider["relationships"]["files"]["links"]["related"][
                    "href"
                ]
                # Extract endpoint from full URL
                parsed = urlparse(files_link)
                files_endpoint = parsed.path.replace("/v2/", "")

                files = self._paginate_results(files_endpoint)
                all_files.extend(files)

        return all_files

    def traverse_project_tree(
        self, root_node_id: str, include_files: bool = True
    ) -> dict[str, Any]:
        """Traverse entire project tree starting from root node.

        Args:
            root_node_id: Root node ID to start traversal
            include_files: Whether to include file information

        Returns:
            Tree structure with node and file information
        """
        logger.info(f"Traversing project tree from node: {root_node_id}")

        # Get root node info
        root_data = self._make_request(f"nodes/{root_node_id}/")
        if not root_data or "data" not in root_data:
            logger.error(f"Could not fetch root node: {root_node_id}")
            return {}

        root = root_data["data"]
        tree = {
            "id": root["id"],
            "type": root["type"],
            "attributes": root.get("attributes", {}),
            "children": [],
            "files": [],
        }

        # Get children recursively
        children = self.get_node_children(root_node_id, recursive=False)
        for child in children:
            child_tree = self.traverse_project_tree(
                child["id"], include_files=include_files
            )
            tree["children"].append(child_tree)

        # Get files if requested
        if include_files:
            tree["files"] = self.get_node_files(root_node_id)

        return tree

    def collect_from_nodes(
        self,
        node_ids: list[str],
        output_dir: Path,
        download_files: bool = True,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """Collect data from specified OSF nodes.

        Args:
            node_ids: List of OSF node IDs to collect
            output_dir: Directory to save collected data
            download_files: Whether to download actual files
            recursive: Whether to include child components

        Returns:
            Collection statistics
        """
        ensure_directory(output_dir)
        stats = {"total_nodes": 0, "total_files": 0, "errors": []}

        for node_id in tqdm(node_ids, desc="Processing nodes"):
            node_dir = output_dir / node_id
            ensure_directory(node_dir)

            try:
                # Get node tree
                if recursive:
                    tree = self.traverse_project_tree(node_id)
                    self._save_tree_structure(node_dir, tree)
                    stats["total_nodes"] += self._count_nodes_in_tree(tree)
                else:
                    node_data = self._make_request(f"nodes/{node_id}/")
                    if node_data:
                        with open(node_dir / "node_metadata.json", "w") as f:
                            json.dump(node_data, f, indent=2)
                        stats["total_nodes"] += 1

                # Download files if requested
                if download_files:
                    files = self.get_node_files(node_id)
                    file_count = self._download_node_files(node_dir, files)
                    stats["total_files"] += file_count
                    self._write_success_status(node_dir, file_count, node_id)
                else:
                    self._write_success_status(node_dir, 0, node_id)

            except Exception as e:
                logger.error(f"Error processing node {node_id}: {e}")
                stats["errors"].append({"node_id": node_id, "error": str(e)})
                self._write_error_status(node_dir, str(e), node_id)

        return stats

    def collect_from_institution(
        self,
        institution_id: str,
        output_dir: Path,
        max_nodes: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Collect all nodes from an institution.

        Args:
            institution_id: OSF institution ID
            output_dir: Directory to save collected data
            max_nodes: Maximum number of nodes to collect
            **kwargs: Additional arguments for collect_from_nodes

        Returns:
            Collection statistics
        """
        logger.info(f"Collecting nodes from institution: {institution_id}")

        nodes = self.get_nodes(institution_id=institution_id)

        if max_nodes:
            nodes = nodes[:max_nodes]

        node_ids = [node["id"] for node in nodes if "id" in node]
        logger.info(f"Found {len(node_ids)} nodes to collect")

        return self.collect_from_nodes(node_ids, output_dir, **kwargs)

    def search_nodes(self, query: str) -> list[dict[str, Any]]:
        """Search for nodes using OSF search API.

        Args:
            query: Search query string

        Returns:
            List of matching nodes
        """
        # OSF search uses a different endpoint
        search_url = "https://api.osf.io/v2/search/"
        params = {"q": query, "page[size]": self.DEFAULT_PAGE_SIZE}

        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collections(
        self, filters: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Get collections from OSF.

        Args:
            filters: Optional filter parameters

        Returns:
            List of collection data
        """
        params = {}
        if filters:
            for key, value in filters.items():
                params[f"filter[{key}]"] = value

        logger.info("Fetching collections from OSF")
        collections = self._paginate_results("collections/", params)
        logger.info(f"Retrieved {len(collections)} collections")

        return collections

    def get_collection_projects(self, collection_id: str) -> list[dict[str, Any]]:
        """Get projects associated with a collection.

        Args:
            collection_id: OSF collection ID

        Returns:
            List of project data from the collection
        """
        endpoint = f"collections/{collection_id}/relationships/linked_nodes/"
        logger.info(f"Fetching projects for collection {collection_id}")

        projects = self._paginate_results(endpoint)
        logger.info(
            f"Retrieved {len(projects)} projects from collection {collection_id}"
        )

        return projects

    def analyze_journal_collections(
        self, output_path: Path | None = None
    ) -> dict[str, Any]:
        """Analyze OSF collections to identify journal partnerships.

        Args:
            output_path: Optional path to save analysis results

        Returns:
            Analysis results with journal mappings
        """
        logger.info("Starting journal collections analysis")

        # Get all collections
        collections = self.get_collections()

        journal_collections = []
        journal_keywords = [
            "journal",
            "publication",
            "manuscript",
            "article",
            "review",
            "proceedings",
            "conference",
            "society",
            "press",
            "publisher",
        ]

        for collection in collections:
            attributes = collection.get("attributes", {})
            title = attributes.get("title", "").lower()
            description = attributes.get("description", "").lower()

            # Check if collection appears to be journal-related
            is_journal = any(
                keyword in title or keyword in description
                for keyword in journal_keywords
            )

            if is_journal:
                # Get projects in this collection
                collection_id = collection["id"]
                try:
                    projects = self.get_collection_projects(collection_id)

                    journal_info = {
                        "collection_id": collection_id,
                        "title": attributes.get("title"),
                        "description": attributes.get("description"),
                        "project_count": len(projects),
                        "created": attributes.get("date_created"),
                        "modified": attributes.get("date_modified"),
                        "projects": projects[:10],  # Sample of projects
                    }
                    journal_collections.append(journal_info)

                    logger.info(
                        f"Found journal collection: {attributes.get('title')} ({len(projects)} projects)"
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to get projects for collection {collection_id}: {e}"
                    )

        analysis = {
            "total_collections": len(collections),
            "journal_collections": journal_collections,
            "journal_count": len(journal_collections),
            "total_journal_projects": sum(
                jc["project_count"] for jc in journal_collections
            ),
        }

        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved journal collections analysis to {output_path}")

        return analysis

    def find_projects_with_dois(
        self, sample_size: int | None = None
    ) -> list[dict[str, Any]]:
        """Find OSF projects that have publication DOIs.

        Args:
            sample_size: Optional limit on number of projects to analyze

        Returns:
            List of projects with DOI information
        """
        logger.info("Searching for projects with DOIs")

        # Search for projects that mention DOI-related terms
        doi_searches = [
            "doi",
            "publication",
            "journal article",
            "peer reviewed",
            "published in",
        ]

        projects_with_dois = []
        seen_ids = set()

        for search_term in doi_searches:
            try:
                results = self.search_nodes(search_term)

                for project in results:
                    project_id = project.get("id")
                    if project_id in seen_ids:
                        continue

                    seen_ids.add(project_id)

                    # Get detailed project info
                    project_data = self._make_request(f"nodes/{project_id}/")
                    if not project_data:
                        continue

                    project_attrs = project_data["data"]["attributes"]
                    description = project_attrs.get("description", "")
                    title = project_attrs.get("title", "")

                    # Look for DOI patterns
                    import re

                    doi_pattern = r"10\.\d{4,}/[^\s]+"
                    dois = re.findall(doi_pattern, description + " " + title)

                    if dois:
                        projects_with_dois.append(
                            {
                                "project_id": project_id,
                                "title": title,
                                "description": description,
                                "dois": list(set(dois)),  # Remove duplicates
                                "created": project_attrs.get("date_created"),
                                "modified": project_attrs.get("date_modified"),
                            }
                        )

                        logger.debug(
                            f"Found project with DOIs: {title} ({len(dois)} DOIs)"
                        )

                        if sample_size and len(projects_with_dois) >= sample_size:
                            break

            except Exception as e:
                logger.warning(f"Search failed for term '{search_term}': {e}")
                continue

            if sample_size and len(projects_with_dois) >= sample_size:
                break

        logger.info(f"Found {len(projects_with_dois)} projects with DOIs")
        return projects_with_dois

    def _save_tree_structure(self, output_dir: Path, tree: dict[str, Any]) -> None:
        """Save project tree structure to JSON file.

        Args:
            output_dir: Directory to save tree structure
            tree: Tree structure data
        """
        tree_file = output_dir / "project_tree.json"
        with open(tree_file, "w") as f:
            json.dump(tree, f, indent=2)
        logger.info(f"Saved tree structure to {tree_file}")

    def _count_nodes_in_tree(self, tree: dict[str, Any]) -> int:
        """Count total nodes in a tree structure.

        Args:
            tree: Tree structure

        Returns:
            Total node count
        """
        count = 1  # Count current node
        for child in tree.get("children", []):
            count += self._count_nodes_in_tree(child)
        return count

    def _download_node_files(self, node_dir: Path, files: list[dict[str, Any]]) -> int:
        """Download files for a node.

        Args:
            node_dir: Directory to save files
            files: List of file metadata

        Returns:
            Number of files downloaded
        """
        if not files:
            return 0

        files_dir = node_dir / "files"
        ensure_directory(files_dir)

        downloaded = 0
        for file_data in files:
            if file_data.get("attributes", {}).get("kind") != "file":
                continue

            file_name = file_data["attributes"].get("name", "unknown")
            download_url = file_data["links"].get("download")

            if download_url:
                file_path = files_dir / file_name
                if self._download_file(download_url, file_path):
                    downloaded += 1
                    logger.debug(f"Downloaded: {file_name}")
                else:
                    logger.warning(f"Failed to download: {file_name}")

        return downloaded

    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download a file from URL.

        Args:
            url: Download URL
            output_path: Path to save file

        Returns:
            True if download successful
        """
        try:
            time.sleep(self.rate_limit_delay)
            response = self.session.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            return True
        except Exception as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return False

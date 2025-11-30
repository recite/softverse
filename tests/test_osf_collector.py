"""Tests for OSF collector."""

import json
from unittest.mock import MagicMock, patch

import pytest

from softverse.collectors.osf_collector import OSFCollector


class TestOSFCollector:
    """Test OSF collector functionality."""

    @pytest.fixture
    def collector(self):
        """Create OSF collector instance."""
        return OSFCollector(api_token="test_token", rate_limit_delay=0)

    @pytest.fixture
    def mock_node_data(self):
        """Sample node data."""
        return {
            "data": {
                "id": "abc123",
                "type": "nodes",
                "attributes": {
                    "title": "Test Project",
                    "description": "Test Description",
                    "public": True,
                },
                "relationships": {
                    "files": {
                        "links": {
                            "related": {
                                "href": "https://api.osf.io/v2/nodes/abc123/files/"
                            }
                        }
                    }
                },
            }
        }

    @pytest.fixture
    def mock_paginated_response(self):
        """Sample paginated response."""
        return {
            "data": [
                {"id": "node1", "type": "nodes", "attributes": {"title": "Node 1"}},
                {"id": "node2", "type": "nodes", "attributes": {"title": "Node 2"}},
            ],
            "links": {
                "first": "https://api.osf.io/v2/nodes/?page=1",
                "last": "https://api.osf.io/v2/nodes/?page=5",
                "next": None,
            },
        }

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.api_token == "test_token"
        assert collector.rate_limit_delay == 0
        assert "Authorization" in collector.session.headers
        assert collector.session.headers["Authorization"] == "Bearer test_token"

    def test_initialization_without_token(self):
        """Test collector initialization without token."""
        collector = OSFCollector(rate_limit_delay=0)
        assert collector.api_token is None
        assert "Authorization" not in collector.session.headers

    @patch("softverse.collectors.osf_collector.requests.Session.get")
    def test_make_request_success(self, mock_get, collector, mock_node_data):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_node_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = collector._make_request("nodes/abc123/")

        assert result == mock_node_data
        mock_get.assert_called_once()

    @patch("softverse.collectors.osf_collector.OSFCollector._make_request")
    def test_make_request_error(self, mock_request, collector):
        """Test API request error handling."""
        mock_request.return_value = None

        result = collector._make_request("nodes/abc123/")

        assert result is None

    @patch("softverse.collectors.osf_collector.OSFCollector._make_request")
    def test_paginate_results_single_page(
        self, mock_request, collector, mock_paginated_response
    ):
        """Test pagination with single page."""
        mock_request.return_value = mock_paginated_response

        results = collector._paginate_results("nodes/")

        assert len(results) == 2
        assert results[0]["id"] == "node1"
        assert results[1]["id"] == "node2"

    @patch("softverse.collectors.osf_collector.requests.Session.get")
    @patch("softverse.collectors.osf_collector.OSFCollector._make_request")
    def test_paginate_results_multiple_pages(self, mock_request, mock_get, collector):
        """Test pagination with multiple pages."""
        # First page
        page1 = {
            "data": [{"id": "node1"}, {"id": "node2"}],
            "links": {"next": "https://api.osf.io/v2/nodes/?page=2"},
        }

        # Second page
        page2 = {
            "data": [{"id": "node3"}, {"id": "node4"}],
            "links": {"next": None},
        }

        mock_request.return_value = page1

        mock_response = MagicMock()
        mock_response.json.return_value = page2
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = collector._paginate_results("nodes/")

        assert len(results) == 4
        assert results[2]["id"] == "node3"

    @patch("softverse.collectors.osf_collector.OSFCollector._paginate_results")
    def test_get_nodes_with_filters(self, mock_paginate, collector):
        """Test getting nodes with filters."""
        mock_paginate.return_value = [{"id": "filtered_node"}]

        nodes = collector.get_nodes(filters={"title": "test", "public": "true"})

        assert len(nodes) == 1
        assert nodes[0]["id"] == "filtered_node"

        # Check that filters were passed correctly
        call_args = mock_paginate.call_args
        assert "filter[title]" in call_args[0][1]
        assert call_args[0][1]["filter[title]"] == "test"
        assert "filter[public]" in call_args[0][1]
        assert call_args[0][1]["filter[public]"] == "true"

    @patch("softverse.collectors.osf_collector.OSFCollector._paginate_results")
    def test_get_nodes_from_institution(self, mock_paginate, collector):
        """Test getting nodes from institution."""
        mock_paginate.return_value = [{"id": "inst_node"}]

        nodes = collector.get_nodes(institution_id="cos")

        assert len(nodes) == 1
        assert nodes[0]["id"] == "inst_node"

        # Check correct endpoint was used
        call_args = mock_paginate.call_args
        assert call_args[0][0] == "institutions/cos/nodes/"

    @patch("softverse.collectors.osf_collector.OSFCollector._paginate_results")
    def test_get_node_children_non_recursive(self, mock_paginate, collector):
        """Test getting node children without recursion."""
        mock_paginate.return_value = [
            {"id": "child1"},
            {"id": "child2"},
        ]

        children = collector.get_node_children("parent_id", recursive=False)

        assert len(children) == 2
        assert children[0]["id"] == "child1"
        mock_paginate.assert_called_once_with("nodes/parent_id/children/")

    @patch("softverse.collectors.osf_collector.OSFCollector._paginate_results")
    def test_get_node_children_recursive(self, mock_paginate, collector):
        """Test getting node children with recursion."""

        # Mock different responses for parent and children
        def paginate_side_effect(endpoint):
            if endpoint == "nodes/parent_id/children/":
                return [{"id": "child1"}, {"id": "child2"}]
            elif endpoint == "nodes/child1/children/":
                return [{"id": "grandchild1"}]
            elif endpoint == "nodes/child2/children/":
                return []
            elif endpoint == "nodes/grandchild1/children/":
                return []
            return []

        mock_paginate.side_effect = paginate_side_effect

        children = collector.get_node_children("parent_id", recursive=True)

        assert len(children) == 3
        assert any(c["id"] == "child1" for c in children)
        assert any(c["id"] == "child2" for c in children)
        assert any(c["id"] == "grandchild1" for c in children)

    @patch("softverse.collectors.osf_collector.OSFCollector._paginate_results")
    def test_get_node_files(self, mock_paginate, collector):
        """Test getting node files."""

        # Mock provider response
        def paginate_side_effect(endpoint):
            if endpoint == "nodes/test_node/files/":
                return [
                    {
                        "relationships": {
                            "files": {
                                "links": {
                                    "related": {
                                        "href": "https://api.osf.io/v2/nodes/test_node/files/osfstorage/"
                                    }
                                }
                            }
                        }
                    }
                ]
            elif "osfstorage" in endpoint:
                return [
                    {"id": "file1", "attributes": {"name": "test.py"}},
                    {"id": "file2", "attributes": {"name": "data.csv"}},
                ]
            return []

        mock_paginate.side_effect = paginate_side_effect

        files = collector.get_node_files("test_node")

        assert len(files) == 2
        assert files[0]["attributes"]["name"] == "test.py"
        assert files[1]["attributes"]["name"] == "data.csv"

    @patch("softverse.collectors.osf_collector.OSFCollector.get_node_files")
    @patch("softverse.collectors.osf_collector.OSFCollector.get_node_children")
    @patch("softverse.collectors.osf_collector.OSFCollector._make_request")
    def test_traverse_project_tree(
        self, mock_request, mock_children, mock_files, collector, mock_node_data
    ):
        """Test project tree traversal."""
        mock_request.return_value = mock_node_data
        mock_children.return_value = []
        mock_files.return_value = [{"id": "file1"}]

        tree = collector.traverse_project_tree("abc123")

        assert tree["id"] == "abc123"
        assert tree["type"] == "nodes"
        assert "title" in tree["attributes"]
        assert len(tree["files"]) == 1
        assert len(tree["children"]) == 0

    def test_count_nodes_in_tree(self, collector):
        """Test counting nodes in tree structure."""
        tree = {
            "id": "root",
            "children": [
                {"id": "child1", "children": []},
                {
                    "id": "child2",
                    "children": [
                        {"id": "grandchild1", "children": []},
                    ],
                },
            ],
        }

        count = collector._count_nodes_in_tree(tree)
        assert count == 4  # root + child1 + child2 + grandchild1

    @patch("softverse.collectors.osf_collector.OSFCollector._download_file")
    def test_download_node_files(self, mock_download, collector, tmp_path):
        """Test downloading node files."""
        files = [
            {
                "attributes": {"kind": "file", "name": "test.py"},
                "links": {"download": "https://osf.io/download/123"},
            },
            {
                "attributes": {"kind": "folder", "name": "folder"},
                "links": {},
            },
            {
                "attributes": {"kind": "file", "name": "data.csv"},
                "links": {"download": "https://osf.io/download/456"},
            },
        ]

        mock_download.return_value = True

        count = collector._download_node_files(tmp_path, files)

        assert count == 2  # Only files, not folders
        assert mock_download.call_count == 2
        assert (tmp_path / "files").exists()

    @patch("softverse.collectors.osf_collector.OSFCollector._download_node_files")
    @patch("softverse.collectors.osf_collector.OSFCollector.traverse_project_tree")
    @patch("softverse.collectors.osf_collector.OSFCollector._make_request")
    def test_collect_from_nodes(
        self, mock_request, mock_traverse, mock_download, collector, tmp_path
    ):
        """Test collecting from specific nodes."""
        mock_request.return_value = {"data": {"id": "node1"}}
        mock_traverse.return_value = {"id": "node1", "children": []}
        mock_download.return_value = 3

        stats = collector.collect_from_nodes(
            ["node1"], tmp_path, download_files=True, recursive=True
        )

        assert stats["total_nodes"] == 1
        assert stats["total_files"] == 3
        assert len(stats["errors"]) == 0
        assert (tmp_path / "node1").exists()

    @patch("softverse.collectors.osf_collector.requests.Session.get")
    def test_search_nodes(self, mock_get, collector):
        """Test searching nodes."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "result1", "attributes": {"title": "Test Result 1"}},
                {"id": "result2", "attributes": {"title": "Test Result 2"}},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        results = collector.search_nodes("test query")

        assert len(results) == 2
        assert results[0]["id"] == "result1"
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["params"]["q"] == "test query"

    def test_save_tree_structure(self, collector, tmp_path):
        """Test saving tree structure to file."""
        tree = {
            "id": "root",
            "attributes": {"title": "Test Project"},
            "children": [],
            "files": [],
        }

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        collector._save_tree_structure(output_dir, tree)

        tree_file = output_dir / "project_tree.json"
        assert tree_file.exists()

        with open(tree_file) as f:
            saved_tree = json.load(f)

        assert saved_tree["id"] == "root"
        assert saved_tree["attributes"]["title"] == "Test Project"

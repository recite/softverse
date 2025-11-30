"""Tests for ResearchBox collector."""

from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import requests

from softverse.collectors.researchbox_collector import ResearchBoxCollector


class TestResearchBoxCollector:
    """Test ResearchBox collector functionality."""

    @pytest.fixture
    def collector(self):
        """Create ResearchBox collector instance."""
        return ResearchBoxCollector(rate_limit_delay=0, max_retries=1)

    @pytest.fixture
    def mock_zip_response(self):
        """Mock ZIP file response."""
        response = MagicMock()
        response.status_code = 200
        response.headers = {
            "Content-Length": "1000000",
            "Last-Modified": "Wed, 21 Oct 2023 07:28:00 GMT",
        }
        response.content = b"fake zip content"
        return response

    @pytest.fixture
    def mock_web_response(self):
        """Mock web page response."""
        response = MagicMock()
        response.status_code = 200
        response.text = """
        <html>
            <head><title>ResearchBox 15: Test Study</title></head>
            <body>
                <div>Authors: John Doe, Jane Smith</div>
                <div>Journal: Test Journal</div>
                <a href="download/test.R">test.R</a>
                <a href="download/data.csv">data.csv</a>
                <a href="download/analysis.py">analysis.py</a>
            </body>
        </html>
        """
        return response

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.rate_limit_delay == 0
        assert collector.max_retries == 1
        assert "User-Agent" in collector.session.headers

    @patch("softverse.collectors.researchbox_collector.requests.Session.head")
    def test_check_researchbox_exists_success(
        self, mock_head, collector, mock_zip_response
    ):
        """Test checking if ResearchBox exists - success case."""
        mock_head.return_value = mock_zip_response

        with patch.object(collector, "_get_web_metadata") as mock_web:
            mock_web.return_value = {"title": "Test Study", "author": "John Doe"}

            metadata = collector.check_researchbox_exists(15)

            assert metadata is not None
            assert metadata["box_id"] == 15
            assert metadata["zip_size"] == 1000000
            assert metadata["title"] == "Test Study"
            assert metadata["author"] == "John Doe"
            mock_head.assert_called_once()

    @patch("softverse.collectors.researchbox_collector.requests.Session.head")
    def test_check_researchbox_exists_not_found(self, mock_head, collector):
        """Test checking if ResearchBox exists - not found case."""
        mock_head.side_effect = requests.exceptions.HTTPError("404 Not Found")

        metadata = collector.check_researchbox_exists(999)

        assert metadata is None
        mock_head.assert_called_once()

    @patch("softverse.collectors.researchbox_collector.requests.Session.get")
    def test_get_web_metadata_success(self, mock_get, collector, mock_web_response):
        """Test extracting metadata from web page."""
        mock_get.return_value = mock_web_response

        metadata = collector._get_web_metadata(15)

        assert metadata is not None
        assert "ResearchBox 15: Test Study" in metadata["title"]
        assert metadata["web_url"] == "https://researchbox.org/15"
        # Should find script files
        assert "script_files" in metadata
        assert len(metadata["script_files"]) >= 1

    @patch("softverse.collectors.researchbox_collector.requests.Session.get")
    def test_get_web_metadata_error(self, mock_get, collector):
        """Test web metadata extraction error handling."""
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        metadata = collector._get_web_metadata(15)

        assert metadata is None

    def test_discover_available_boxes(self, collector):
        """Test discovering available ResearchBox IDs."""
        with patch.object(collector, "check_researchbox_exists") as mock_check:
            # Mock some existing and non-existing boxes
            def mock_exists(box_id):
                if box_id in [1, 15, 17]:
                    return {
                        "box_id": box_id,
                        "zip_size": 1000000,
                        "zip_url": f"https://s3.wasabisys.com/zipballs.researchbox.org/ResearchBox_{box_id}.zip",
                    }
                return None

            mock_check.side_effect = mock_exists

            available = collector.discover_available_boxes(1, 20, batch_size=10)

            assert len(available) == 3
            assert all(box["box_id"] in [1, 15, 17] for box in available)

    @patch("zipfile.ZipFile")
    def test_download_researchbox_success(self, mock_zipfile, collector, tmp_path):
        """Test downloading and extracting ResearchBox."""
        # Mock successful metadata check
        with patch.object(collector, "check_researchbox_exists") as mock_check:
            mock_check.return_value = {
                "box_id": 15,
                "zip_url": "https://s3.wasabisys.com/zipballs.researchbox.org/ResearchBox_15.zip",
                "zip_size": 1000000,
            }

            # Mock successful download
            with patch.object(collector, "_make_request") as mock_request:
                mock_response = MagicMock()
                mock_response.content = b"fake zip content"
                mock_request.return_value = mock_response

                # Mock ZIP extraction
                mock_zip = MagicMock()
                mock_zipfile.return_value.__enter__.return_value = mock_zip

                # Mock extracted files
                extract_dir = tmp_path / "ResearchBox_15" / "extracted"
                extract_dir.mkdir(parents=True)
                (extract_dir / "test.R").write_text("# R script")
                (extract_dir / "data.csv").write_text("data")

                stats = collector.download_researchbox(15, tmp_path, extract=True)

                assert stats["downloaded"] is True
                assert stats["extracted"] is True
                assert stats["box_id"] == 15

    def test_download_researchbox_not_found(self, collector, tmp_path):
        """Test downloading non-existent ResearchBox."""
        with patch.object(collector, "check_researchbox_exists") as mock_check:
            mock_check.return_value = None

            stats = collector.download_researchbox(999, tmp_path)

            assert stats["downloaded"] is False
            assert stats["extracted"] is False

    def test_collect_researchboxes(self, collector, tmp_path):
        """Test collecting multiple ResearchBoxes."""
        box_ids = [1, 15, 17]

        with patch.object(collector, "download_researchbox") as mock_download:
            # Mock different outcomes for each box
            def mock_download_result(box_id, *args, **kwargs):
                if box_id == 1:
                    return {
                        "downloaded": True,
                        "extracted": True,
                        "file_count": 5,
                        "script_count": 2,
                    }
                elif box_id == 15:
                    return {
                        "downloaded": True,
                        "extracted": True,
                        "file_count": 10,
                        "script_count": 3,
                    }
                else:  # box_id == 17
                    return {
                        "downloaded": False,
                        "extracted": False,
                        "file_count": 0,
                        "script_count": 0,
                    }

            mock_download.side_effect = mock_download_result

            stats = collector.collect_researchboxes(box_ids, tmp_path)

            assert stats["total_requested"] == 3
            assert stats["successful_downloads"] == 2
            assert stats["successful_extractions"] == 2
            assert stats["total_files"] == 15
            assert stats["total_scripts"] == 5

    def test_collect_range_with_discovery(self, collector, tmp_path):
        """Test collecting a range with discovery."""
        with patch.object(collector, "discover_available_boxes") as mock_discover:
            mock_discover.return_value = [{"box_id": 1}, {"box_id": 15}, {"box_id": 17}]

            with patch.object(collector, "collect_researchboxes") as mock_collect:
                mock_collect.return_value = {"total_requested": 3}

                collector.collect_range(1, 20, tmp_path, discover_first=True)

                mock_discover.assert_called_once_with(1, 20)
                mock_collect.assert_called_once_with(
                    [1, 15, 17], tmp_path, extract=True, keep_zip=False
                )

    def test_collect_range_without_discovery(self, collector, tmp_path):
        """Test collecting a range without discovery."""
        with patch.object(collector, "collect_researchboxes") as mock_collect:
            mock_collect.return_value = {"total_requested": 5}

            collector.collect_range(1, 5, tmp_path, discover_first=False)

            mock_collect.assert_called_once_with(
                [1, 2, 3, 4, 5], tmp_path, extract=True, keep_zip=False
            )

    @patch("softverse.collectors.researchbox_collector.requests.Session.head")
    def test_make_request_with_retries(self, mock_head, collector):
        """Test request retry mechanism."""
        # First two attempts fail, third succeeds
        mock_head.side_effect = [
            requests.exceptions.RequestException("Error 1"),
            requests.exceptions.RequestException("Error 2"),
            MagicMock(status_code=200),
        ]

        # Override max_retries for this test
        collector.max_retries = 3

        response = collector._make_request("http://test.com", method="HEAD")

        assert response is not None
        assert mock_head.call_count == 3

    @patch("softverse.collectors.researchbox_collector.requests.Session.head")
    def test_make_request_all_retries_fail(self, mock_head, collector):
        """Test request when all retries fail."""
        mock_head.side_effect = requests.exceptions.RequestException("Persistent error")

        response = collector._make_request("http://test.com", method="HEAD")

        assert response is None
        assert mock_head.call_count == 1  # max_retries is 1 in fixture

    def test_s3_url_patterns(self, collector):
        """Test S3 URL pattern generation."""
        box_id = 15
        expected_url = (
            "https://s3.wasabisys.com/zipballs.researchbox.org/ResearchBox_15.zip"
        )

        actual_url = f"{collector.S3_BASE_URL}ResearchBox_{box_id}.zip"

        assert actual_url == expected_url

        # Verify URL is well-formed
        parsed = urlparse(actual_url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "s3.wasabisys.com"
        assert "ResearchBox_15.zip" in parsed.path

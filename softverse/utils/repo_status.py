"""Repository status analysis utilities."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from softverse.utils.logging_utils import get_logger

logger = get_logger("repo_status")


class RepoStatusAnalyzer:
    """Analyzes repository processing status across collections."""

    def __init__(self, base_scripts_dir: str = "scripts"):
        self.base_scripts_dir = Path(base_scripts_dir)

    def _read_status_file(self, status_file: Path) -> dict | None:
        """Read a single .status file."""
        if not status_file.exists():
            return None

        try:
            with open(status_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read status file {status_file}: {e}")
            return None

    def collect_status_data(self, source: str = "dataverse") -> list[dict]:
        """Collect all status data for a given source.

        Args:
            source: Source name (e.g., 'dataverse', 'zenodo', 'icpsr')

        Returns:
            List of status data dictionaries
        """
        source_dir = self.base_scripts_dir / source
        if not source_dir.exists():
            logger.warning(f"Source directory {source_dir} does not exist")
            return []

        status_data = []

        # Find all .status files
        for status_file in source_dir.rglob(".status"):
            repo_id = status_file.parent.name
            data = self._read_status_file(status_file)
            if data:
                data["repo_id"] = repo_id
                data["source"] = source
                status_data.append(data)

        return status_data

    def analyze_status_summary(self, source: str = "dataverse") -> dict:
        """Generate summary statistics for repository processing status.

        Args:
            source: Source name

        Returns:
            Summary statistics dictionary
        """
        status_data = self.collect_status_data(source)

        if not status_data:
            return {"total": 0, "by_status": {}, "errors": {}}

        # Count by status
        status_counts = Counter(item["status"] for item in status_data)

        # Analyze error types
        error_details = defaultdict(list)
        for item in status_data:
            if item["status"] in ["forbidden", "error", "not_found"]:
                error_type = item["details"].get("error", item["status"])
                error_details[item["status"]].append(
                    {
                        "repo_id": item["repo_id"],
                        "doi": item["details"].get("doi"),
                        "error": error_type,
                        "timestamp": item["timestamp"],
                    }
                )

        # Success metrics
        successful = [item for item in status_data if item["status"] == "success"]
        total_scripts = sum(
            item["details"].get("script_count", 0) for item in successful
        )

        return {
            "total_repos": len(status_data),
            "by_status": dict(status_counts),
            "success_rate": (
                len(successful) / len(status_data) * 100 if status_data else 0
            ),
            "total_scripts_found": total_scripts,
            "avg_scripts_per_successful_repo": (
                total_scripts / len(successful) if successful else 0
            ),
            "error_details": dict(error_details),
        }

    def get_retryable_repos(
        self, source: str = "dataverse", retry_statuses: list[str] | None = None
    ) -> list[dict]:
        """Get repositories that could be retried.

        Args:
            source: Source name
            retry_statuses: List of status types to consider for retry

        Returns:
            List of retryable repository data
        """
        if retry_statuses is None:
            retry_statuses = ["forbidden", "error", "not_found"]

        status_data = self.collect_status_data(source)
        retryable = [item for item in status_data if item["status"] in retry_statuses]

        return retryable

    def print_summary_report(self, source: str = "dataverse") -> None:
        """Print a human-readable summary report."""
        summary = self.analyze_status_summary(source)

        print(f"\n=== Repository Status Summary for {source.upper()} ===")
        print(f"Total repositories processed: {summary['total_repos']:,}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total script files found: {summary['total_scripts_found']:,}")

        if summary["total_repos"] > 0:
            print(
                f"Average scripts per successful repo: {summary['avg_scripts_per_successful_repo']:.1f}"
            )

        print("\nStatus breakdown:")
        for status, count in summary["by_status"].items():
            percentage = (count / summary["total_repos"]) * 100
            print(f"  {status}: {count:,} ({percentage:.1f}%)")

        # Error details
        if summary["error_details"]:
            print("\nError details:")
            for status, errors in summary["error_details"].items():
                print(f"  {status}: {len(errors)} repos")
                if status == "forbidden":
                    print(
                        f"    Example DOIs: {', '.join(e['doi'] for e in errors[:3] if e['doi'])}"
                    )


def main():
    """Command line entry point for status analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze repository processing status")
    parser.add_argument("--source", default="dataverse", help="Source to analyze")
    parser.add_argument(
        "--retryable", action="store_true", help="Show retryable repositories"
    )

    args = parser.parse_args()

    analyzer = RepoStatusAnalyzer()

    if args.retryable:
        retryable = analyzer.get_retryable_repos(args.source)
        print(f"\nRetryable repositories ({len(retryable)} total):")
        for repo in retryable[:10]:  # Show first 10
            print(
                f"  {repo['repo_id']} ({repo['status']}): {repo['details'].get('doi')}"
            )
        if len(retryable) > 10:
            print(f"  ... and {len(retryable) - 10} more")
    else:
        analyzer.print_summary_report(args.source)


if __name__ == "__main__":
    main()

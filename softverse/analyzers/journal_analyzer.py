"""Journal analyzer for identifying journals hosting data on OSF."""

import re
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from softverse.collectors.osf_collector import OSFCollector
from softverse.utils.file_utils import ensure_directory
from softverse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class JournalAnalyzer:
    """Analyzer for identifying journals and publications hosting data on OSF."""

    CROSSREF_API_BASE = "https://api.crossref.org/works/"
    CROSSREF_RATE_LIMIT = 1.0  # seconds between requests

    def __init__(
        self,
        osf_collector: OSFCollector | None = None,
        crossref_rate_limit: float = CROSSREF_RATE_LIMIT,
    ):
        """Initialize journal analyzer.

        Args:
            osf_collector: OSF collector instance, creates new if None
            crossref_rate_limit: Delay between Crossref API requests
        """
        self.osf_collector = osf_collector or OSFCollector()
        self.crossref_rate_limit = crossref_rate_limit
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Softverse Journal Analyzer (mailto:research@example.com)"}
        )

    def resolve_doi_to_journal(self, doi: str) -> dict[str, Any] | None:
        """Resolve DOI to journal metadata using Crossref API.

        Args:
            doi: DOI string to resolve

        Returns:
            Journal metadata or None if resolution fails
        """
        try:
            time.sleep(self.crossref_rate_limit)

            # Clean DOI
            clean_doi = doi.replace("https://doi.org/", "").replace(
                "http://dx.doi.org/", ""
            )

            url = f"{self.CROSSREF_API_BASE}{clean_doi}"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            work = data.get("message", {})

            # Extract journal information
            container_title = work.get("container-title", [])
            journal_name = container_title[0] if container_title else "Unknown"

            issn_list = work.get("ISSN", [])
            publisher = work.get("publisher", "Unknown")

            # Extract additional metadata
            published_date = work.get(
                "published-print", work.get("published-online", {})
            )
            year = published_date.get("date-parts", [[]])[0]
            publication_year = year[0] if year else None

            journal_info = {
                "journal_name": journal_name,
                "publisher": publisher,
                "issn": issn_list,
                "publication_year": publication_year,
                "title": work.get("title", [None])[0],
                "authors": [
                    f"{author.get('given', '')} {author.get('family', '')}"
                    for author in work.get("author", [])
                ],
                "type": work.get("type"),
                "doi": clean_doi,
            }

            logger.debug(f"Resolved DOI {clean_doi} to journal: {journal_name}")
            return journal_info

        except Exception as e:
            logger.warning(f"Failed to resolve DOI {doi}: {e}")
            return None

    def analyze_doi_based_journals(
        self,
        projects_with_dois: list[dict[str, Any]] | None = None,
        sample_size: int = 100,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Analyze journals based on DOIs found in OSF projects.

        Args:
            projects_with_dois: Pre-collected projects with DOIs, fetches if None
            sample_size: Number of projects to analyze if fetching
            output_path: Optional path to save results

        Returns:
            Journal analysis results
        """
        logger.info("Starting DOI-based journal analysis")

        if projects_with_dois is None:
            logger.info(f"Fetching projects with DOIs (sample size: {sample_size})")
            projects_with_dois = self.osf_collector.find_projects_with_dois(sample_size)

        journal_mappings = {}
        project_journal_map = {}
        failed_dois = []

        logger.info(f"Analyzing {len(projects_with_dois)} projects with DOIs")

        for project in tqdm(projects_with_dois, desc="Resolving DOIs to journals"):
            project_id = project["project_id"]
            project_journals = []

            for doi in project["dois"]:
                journal_info = self.resolve_doi_to_journal(doi)

                if journal_info:
                    journal_name = journal_info["journal_name"]

                    if journal_name not in journal_mappings:
                        journal_mappings[journal_name] = {
                            "name": journal_name,
                            "publisher": journal_info["publisher"],
                            "issn": journal_info["issn"],
                            "projects": [],
                            "project_count": 0,
                            "dois": [],
                        }

                    # Add project to journal
                    if project_id not in [
                        p["project_id"]
                        for p in journal_mappings[journal_name]["projects"]
                    ]:
                        journal_mappings[journal_name]["projects"].append(
                            {
                                "project_id": project_id,
                                "title": project["title"],
                                "created": project["created"],
                            }
                        )
                        journal_mappings[journal_name]["project_count"] += 1

                    journal_mappings[journal_name]["dois"].append(doi)
                    project_journals.append(journal_name)

                else:
                    failed_dois.append(doi)

            if project_journals:
                project_journal_map[project_id] = {
                    "project_title": project["title"],
                    "journals": project_journals,
                }

        # Calculate statistics
        total_journals = len(journal_mappings)
        total_projects_mapped = len(project_journal_map)
        total_dois_resolved = sum(len(j["dois"]) for j in journal_mappings.values())

        # Sort journals by project count
        sorted_journals = sorted(
            journal_mappings.items(), key=lambda x: x[1]["project_count"], reverse=True
        )

        analysis = {
            "summary": {
                "total_journals_identified": total_journals,
                "total_projects_mapped": total_projects_mapped,
                "total_dois_resolved": total_dois_resolved,
                "failed_dois": len(failed_dois),
                "success_rate": (
                    (total_dois_resolved / (total_dois_resolved + len(failed_dois)))
                    * 100
                    if (total_dois_resolved + len(failed_dois)) > 0
                    else 0
                ),
            },
            "journals": dict(sorted_journals),
            "project_journal_map": project_journal_map,
            "failed_dois": failed_dois[:50],  # Sample of failed DOIs
        }

        if output_path:
            import json

            ensure_directory(output_path.parent)
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved DOI-based journal analysis to {output_path}")

        logger.info(
            f"Analysis complete: {total_journals} journals identified from {total_projects_mapped} projects"
        )
        return analysis

    def analyze_text_based_journals(
        self,
        search_terms: list[str] | None = None,
        max_projects: int = 500,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Analyze journals based on text mentions in OSF project metadata.

        Args:
            search_terms: Journal-related search terms, uses defaults if None
            max_projects: Maximum projects to analyze
            output_path: Optional path to save results

        Returns:
            Text-based journal analysis results
        """
        logger.info("Starting text-based journal analysis")

        if search_terms is None:
            search_terms = [
                "published in",
                "journal of",
                "proceedings",
                "nature",
                "science",
                "plos",
                "cell",
                "jama",
                "nejm",
                "psychological science",
            ]

        journal_mentions = {}
        analyzed_projects = []

        for search_term in tqdm(search_terms, desc="Searching for journal mentions"):
            try:
                results = self.osf_collector.search_nodes(search_term)

                for project in results[: max_projects // len(search_terms)]:
                    project_id = project.get("id")
                    if any(p["project_id"] == project_id for p in analyzed_projects):
                        continue

                    # Get detailed project info
                    project_data = self.osf_collector._make_request(
                        f"nodes/{project_id}/"
                    )
                    if not project_data:
                        continue

                    project_attrs = project_data["data"]["attributes"]
                    title = project_attrs.get("title", "")
                    description = project_attrs.get("description", "")

                    # Extract journal mentions from text
                    journal_patterns = self._extract_journal_patterns(
                        title + " " + description
                    )

                    if journal_patterns:
                        analyzed_projects.append(
                            {
                                "project_id": project_id,
                                "title": title,
                                "journal_mentions": journal_patterns,
                                "created": project_attrs.get("date_created"),
                            }
                        )

                        for journal_name in journal_patterns:
                            if journal_name not in journal_mentions:
                                journal_mentions[journal_name] = {
                                    "name": journal_name,
                                    "projects": [],
                                    "mention_count": 0,
                                }

                            journal_mentions[journal_name]["projects"].append(
                                {
                                    "project_id": project_id,
                                    "title": title,
                                }
                            )
                            journal_mentions[journal_name]["mention_count"] += 1

            except Exception as e:
                logger.warning(f"Search failed for term '{search_term}': {e}")
                continue

        # Sort journals by mention count
        sorted_journals = sorted(
            journal_mentions.items(),
            key=lambda x: int(x[1]["mention_count"]),
            reverse=True,
        )

        analysis = {
            "summary": {
                "total_journals_mentioned": len(journal_mentions),
                "total_projects_analyzed": len(analyzed_projects),
                "search_terms_used": search_terms,
            },
            "journals": dict(sorted_journals),
            "projects": analyzed_projects,
        }

        if output_path:
            import json

            ensure_directory(output_path.parent)
            with open(output_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved text-based journal analysis to {output_path}")

        logger.info(
            f"Analysis complete: {len(journal_mentions)} journals mentioned in {len(analyzed_projects)} projects"
        )
        return analysis

    def _extract_journal_patterns(self, text: str) -> list[str]:
        """Extract journal name patterns from text.

        Args:
            text: Text to analyze

        Returns:
            List of potential journal names
        """
        if not text:
            return []

        text_lower = text.lower()
        journal_patterns = []

        # Common journal name patterns
        journal_regexes = [
            r"journal of [\w\s&]+",
            r"proceedings of [\w\s&]+",
            r"nature[\w\s]*",
            r"science[\w\s]*",
            r"cell[\w\s]*",
            r"plos [\w\s]+",
            r"jama[\w\s]*",
            r"nejm",
            r"psychological science",
            r"american economic review",
            r"quarterly journal of economics",
            r"journal of political economy",
            r"review of economic studies",
            r"econometrica",
        ]

        for pattern in journal_regexes:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                journal_name = match.group().strip()
                # Clean up and normalize
                journal_name = re.sub(r"\s+", " ", journal_name)
                journal_name = journal_name.title()
                if len(journal_name) > 5:  # Filter out very short matches
                    journal_patterns.append(journal_name)

        # Look for "published in" patterns
        pub_pattern = r"published in ([^.;,\n]+)"
        pub_matches = re.finditer(pub_pattern, text_lower)
        for match in pub_matches:
            potential_journal = match.group(1).strip()
            potential_journal = re.sub(r"\s+", " ", potential_journal)
            potential_journal = potential_journal.title()
            if 5 < len(potential_journal) < 100:  # Reasonable length
                journal_patterns.append(potential_journal)

        return list(set(journal_patterns))  # Remove duplicates

    def generate_comprehensive_report(
        self,
        output_dir: Path,
        include_collections: bool = True,
        include_dois: bool = True,
        include_text: bool = True,
        doi_sample_size: int = 100,
    ) -> dict[str, Any]:
        """Generate comprehensive journal analysis report.

        Args:
            output_dir: Directory to save analysis outputs
            include_collections: Whether to analyze collections
            include_dois: Whether to analyze DOI-based journals
            include_text: Whether to analyze text-based journals
            doi_sample_size: Sample size for DOI analysis

        Returns:
            Comprehensive analysis results
        """
        ensure_directory(output_dir)
        logger.info(f"Generating comprehensive journal analysis report in {output_dir}")

        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "include_collections": include_collections,
                "include_dois": include_dois,
                "include_text": include_text,
                "doi_sample_size": doi_sample_size,
            },
        }

        # Collection-based analysis
        if include_collections:
            logger.info("Running collection-based analysis...")
            collections_analysis = self.osf_collector.analyze_journal_collections(
                output_path=output_dir / "journal_collections.json"
            )
            report["collections_analysis"] = collections_analysis

        # DOI-based analysis
        if include_dois:
            logger.info("Running DOI-based analysis...")
            doi_analysis = self.analyze_doi_based_journals(
                sample_size=doi_sample_size,
                output_path=output_dir / "doi_based_journals.json",
            )
            report["doi_analysis"] = doi_analysis

        # Text-based analysis
        if include_text:
            logger.info("Running text-based analysis...")
            text_analysis = self.analyze_text_based_journals(
                output_path=output_dir / "text_based_journals.json"
            )
            report["text_analysis"] = text_analysis

        # Generate summary report
        summary = self._generate_summary_report(report)
        report["summary"] = summary

        # Save comprehensive report
        report_path = output_dir / "comprehensive_journal_report.json"
        with open(report_path, "w") as f:
            import json

            json.dump(report, f, indent=2)

        # Generate human-readable summary
        self._generate_readable_summary(report, output_dir / "journal_summary.txt")

        logger.info(
            f"Comprehensive journal analysis complete. Results saved to {output_dir}"
        )
        return report

    def _generate_summary_report(self, analysis_data: dict[str, Any]) -> dict[str, Any]:
        """Generate summary statistics from analysis data.

        Args:
            analysis_data: Complete analysis results

        Returns:
            Summary statistics
        """
        summary = {
            "total_unique_journals": 0,
            "total_projects_mapped": 0,
            "analysis_methods": [],
        }

        all_journals = set()
        all_projects = set()

        # Collections data
        if "collections_analysis" in analysis_data:
            collections = analysis_data["collections_analysis"]
            summary["analysis_methods"].append("collections")
            summary["collections"] = {
                "journal_collections": collections.get("journal_count", 0),
                "projects_in_collections": collections.get("total_journal_projects", 0),
            }

            # Add to totals
            for journal_collection in collections.get("journal_collections", []):
                all_journals.add(journal_collection.get("title", ""))
                for project in journal_collection.get("projects", []):
                    all_projects.add(project.get("id", ""))

        # DOI data
        if "doi_analysis" in analysis_data:
            doi_data = analysis_data["doi_analysis"]
            summary["analysis_methods"].append("doi_resolution")
            summary["doi_analysis"] = {
                "journals_from_dois": doi_data["summary"].get(
                    "total_journals_identified", 0
                ),
                "projects_with_dois": doi_data["summary"].get(
                    "total_projects_mapped", 0
                ),
                "success_rate": doi_data["summary"].get("success_rate", 0),
            }

            # Add to totals
            all_journals.update(doi_data.get("journals", {}).keys())
            all_projects.update(doi_data.get("project_journal_map", {}).keys())

        # Text data
        if "text_analysis" in analysis_data:
            text_data = analysis_data["text_analysis"]
            summary["analysis_methods"].append("text_analysis")
            summary["text_analysis"] = {
                "journals_mentioned": text_data["summary"].get(
                    "total_journals_mentioned", 0
                ),
                "projects_analyzed": text_data["summary"].get(
                    "total_projects_analyzed", 0
                ),
            }

            # Add to totals
            all_journals.update(text_data.get("journals", {}).keys())
            for project in text_data.get("projects", []):
                all_projects.add(project.get("project_id", ""))

        summary["total_unique_journals"] = len([j for j in all_journals if j])
        summary["total_projects_mapped"] = len(all_projects)

        return summary

    def _generate_readable_summary(
        self, report: dict[str, Any], output_path: Path
    ) -> None:
        """Generate human-readable summary report.

        Args:
            report: Complete analysis report
            output_path: Path to save readable summary
        """
        with open(output_path, "w") as f:
            f.write("OSF JOURNAL ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Date: {report.get('analysis_timestamp', 'Unknown')}\n\n")

            summary = report.get("summary", {})
            f.write("OVERALL STATISTICS:\n")
            f.write(
                f"  Total Unique Journals: {summary.get('total_unique_journals', 0)}\n"
            )
            f.write(
                f"  Total Projects Mapped: {summary.get('total_projects_mapped', 0)}\n"
            )
            f.write(
                f"  Analysis Methods Used: {', '.join(summary.get('analysis_methods', []))}\n\n"
            )

            # Collections summary
            if "collections" in summary:
                collections = summary["collections"]
                f.write("COLLECTIONS ANALYSIS:\n")
                f.write(
                    f"  Journal Collections Found: {collections.get('journal_collections', 0)}\n"
                )
                f.write(
                    f"  Projects in Collections: {collections.get('projects_in_collections', 0)}\n\n"
                )

            # DOI summary
            if "doi_analysis" in summary:
                doi_summary = summary["doi_analysis"]
                f.write("DOI-BASED ANALYSIS:\n")
                f.write(
                    f"  Journals from DOIs: {doi_summary.get('journals_from_dois', 0)}\n"
                )
                f.write(
                    f"  Projects with DOIs: {doi_summary.get('projects_with_dois', 0)}\n"
                )
                f.write(
                    f"  DOI Resolution Success Rate: {doi_summary.get('success_rate', 0):.1f}%\n\n"
                )

            # Text summary
            if "text_analysis" in summary:
                text_summary = summary["text_analysis"]
                f.write("TEXT-BASED ANALYSIS:\n")
                f.write(
                    f"  Journals Mentioned: {text_summary.get('journals_mentioned', 0)}\n"
                )
                f.write(
                    f"  Projects Analyzed: {text_summary.get('projects_analyzed', 0)}\n\n"
                )

            # Top journals from DOI analysis
            if "doi_analysis" in report:
                f.write("TOP JOURNALS BY PROJECT COUNT (DOI-based):\n")
                journals = report["doi_analysis"].get("journals", {})
                top_journals = sorted(
                    journals.items(),
                    key=lambda x: x[1].get("project_count", 0),
                    reverse=True,
                )
                for i, (journal_name, data) in enumerate(top_journals[:15]):
                    f.write(
                        f"  {i + 1:2d}. {journal_name} ({data.get('project_count', 0)} projects)\n"
                    )
                f.write("\n")

        logger.info(f"Generated readable summary: {output_path}")

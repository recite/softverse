"""The README's numbers are the shipped summary's, not a copy typed once."""

from __future__ import annotations

import json
from pathlib import Path

from release_tally import README_END, README_START, readme_numbers, write_readme

# The checkout, not `PATHS.root`: the wheel job installs the package elsewhere
# and runs these tests against the repository's files.
REPO = Path(__file__).resolve().parent.parent


def test_readme_block_matches_the_shipped_summary():
    summary = json.loads((REPO / "data" / "tally" / "summary.json").read_text())
    text = (REPO / "README.md").read_text(encoding="utf-8")
    block = text[text.index(README_START) : text.index(README_END) + len(README_END)]
    assert block == readme_numbers(summary)


def test_write_readme_replaces_only_the_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nstale\n{README_END}\noutro\n")
    summary = {
        "built": "2026-09-15",
        "n_deposits": 3,
        "n_deposits_analyzable": 2,
        "n_files_analyzable": 5,
        "n_packages": 4,
        "deposits_by_language": {"r": 2},
        "trawl": {
            "collections_by_source": {"dataverse": 1, "zenodo": 1},
            "deposits_in_frame_by_source": {"dataverse": 7, "zenodo": 1},
            "n_files": 9,
            "n_mentions": 1234,
        },
    }
    write_readme(summary, readme)
    text = readme.read_text()
    assert text.startswith("intro\n")
    assert text.endswith("\noutro\n")
    assert "stale" not in text
    assert "8 deposits in all" in text
    assert "| R | 2 |" in text
    assert "**1,234** package references" in text

"""The paper must execute.

Its central claim is that no number in it is typed. That is only worth
something if the expressions run, and for a while nothing ran them: rendering
needs Quarto, Quarto is not installed here, and the figures were being checked
against a block copied out of the file by hand -- hand-checking a document
whose whole point is that it is not hand-checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PAPER_DIR = Path(__file__).parent.parent / "paper"
TALLY = Path(__file__).parent.parent / "build" / "tally" / "mentions.parquet"


@pytest.mark.skipif(not TALLY.exists(), reason="no tally built")
def test_every_computed_value_in_the_paper_evaluates():
    result = subprocess.run(
        [sys.executable, "check_paper.py"],
        cwd=PAPER_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not TALLY.exists(), reason="no tally built")
def test_the_paper_types_no_numbers_it_could_compute():
    """A guard against the failure this whole arrangement exists to prevent.

    One literal did go stale -- "21 files to 221", measured on a corpus a
    third the size, which recomputed to 13 to 41. The check is deliberately
    crude: any bare digit group of three or more in the prose, minus the
    handful that are genuinely external facts rather than our measurements.
    """
    import re

    text = (PAPER_DIR / "paper.qmd").read_text()
    prose = re.sub(r"^```\{python\}\n.*?^```", "", text, flags=re.M | re.S)
    prose = re.sub(r"`\{python\}[^`]*`", "", prose)
    prose = re.sub(r"^\s*[-*]?\s*\[.*?\]\(.*?\)", "", prose, flags=re.M)

    #: Facts about the world, not measurements of our corpus.
    allowed = {"7.1", "2010", "1.0", "4559", "2007"}
    found = {
        m.group(0)
        for m in re.finditer(r"\b\d[\d,.]{2,}\b", prose)
        if m.group(0) not in allowed
    }
    assert not found, f"literal numbers in the prose: {sorted(found)}"

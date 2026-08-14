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
    # Inline code spans hold identifiers, not claims: `.7z.002` is a filename
    # extension and `renv 1.2.3` a version. Stripping them keeps the check on
    # prose, where a number is an assertion about the corpus.
    prose = re.sub(r"`[^`\n]*`", "", prose)
    prose = re.sub(r"^\s*[-*]?\s*\[.*?\]\(.*?\)", "", prose, flags=re.M)

    #: Facts about the world rather than measurements of our corpus: Zenodo's
    #: size, a CVE, and another paper's deposit count.
    allowed = {"7.1", "2,000", "200", "1.0", "4559"}
    #: A four-digit year is a date, never a measurement of this corpus, so it
    #: is admitted by rule rather than by adding each one to the list above as
    #: it appears -- which is how an allowlist quietly becomes a way of
    #: silencing the check it belongs to.
    year = re.compile(r"^(19|20)\d\d$")
    found = {
        m.group(0)
        for m in re.finditer(r"\b\d[\d,.]{2,}\b", prose)
        if m.group(0) not in allowed and not year.match(m.group(0))
    }
    assert not found, f"literal numbers in the prose: {sorted(found)}"

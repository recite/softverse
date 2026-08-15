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
    # Two digits is enough to be a claim. The first version of this pattern
    # demanded three or more characters, so every two-digit number walked
    # through it: "ahead of all 47 estimators" was invisible to the check
    # that exists to catch exactly that. Found by mutating the paper and
    # watching the build succeed.
    found = {
        m.group(0)
        for m in re.finditer(r"\b\d[\d,.]*\d\b|\b\d+%", prose)
        if m.group(0) not in allowed and not year.match(m.group(0))
    }
    assert not found, f"literal numbers in the prose: {sorted(found)}"


def _prose() -> str:
    """The paper with code chunks and inline expressions removed."""
    import re

    text = (PAPER_DIR / "paper.qmd").read_text()
    text = re.sub(r"^```\{python\}\n.*?^```", "", text, flags=re.M | re.S)
    return re.sub(r"`\{python\}[^`]*`", "", text)


def test_the_prose_carries_no_em_or_en_dashes():
    """A hard ban, and one that fails silently.

    An em dash is the single most reliable mark of machine prose, and the
    draft this rewrite replaced had forty of them. Nothing about the rendered
    PDF would tell you.
    """
    import re

    prose = _prose()
    # Search for the character itself. An earlier version of this test asked
    # for forty characters of context either side, and `.` does not match a
    # newline, so any dash near a line break passed unnoticed. A check that
    # fails silently is worse than no check, and this one exists precisely
    # because the defect it looks for is invisible in the rendered output.
    hits = [m.start() for m in re.finditer(r"[\u2013\u2014]", prose)]
    context = [prose[max(0, i - 40) : i + 40].replace("\n", " ") for i in hits[:3]]
    assert not hits, f"{len(hits)} em/en dashes, first: ...{context}..."


def test_no_bold_header_list_items():
    """`- **Thing:** description` restates its own label and reads as generated.

    Seven of these were in Limitations and Validation; they are prose now.
    """
    import re

    found = re.findall(r"^\s*[-*]\s+\*\*[^*]+\*\*", _prose(), re.M)
    assert not found, f"bold-header list items: {found}"


def test_no_sentence_opens_with_however_also_or_therefore():
    import re

    found = re.findall(r"(?:^|\. )(However|Also|Therefore),", _prose())
    assert not found, f"sentence-initial connectives: {found}"


def test_the_paper_opens_on_a_finding_not_a_topic():
    """Shafer's test: give away the murderer in the first paragraph.

    The previous draft opened on "What this measures, and what it does not",
    which is a caveat about a contribution the reader has not been given yet.
    The first heading should name the problem and the first sentence should
    state a result.
    """
    import re

    prose = _prose()
    first_heading = re.search(r"^## (.+)$", prose, re.M)
    assert first_heading, "no sections found"
    assert "what this measures" not in first_heading.group(1).lower(), (
        "the paper opens on its own caveats again"
    )

    after = prose[first_heading.end() :].strip()
    opening = " ".join(after.split("\n\n")[0].split())
    # A finding names something concrete. A topic sentence says a field is
    # important. The cheapest discriminator is whether the paper's own
    # subject matter appears before the reader is asked to care.
    assert len(opening) > 40, "the opening paragraph is missing"
    assert not opening.lower().startswith(
        ("research software is", "software is", "in recent years", "it has long")
    ), f"opens on a topic rather than a finding: {opening[:90]}"

"""Tests for the bibliography validator.

The validator itself must be trustworthy, because it is the thing standing
between a fabricated citation and a published paper.
"""

from __future__ import annotations

from paper.validate_bib import normalize, parse_bib

BIB = """\
@article{good,
  author  = {Someone},
  title   = {A real title},
  journal = {A journal},
  year    = {2022},
  doi     = {10.1038/s41597-022-01143-6}
}

@misc{nodoi,
  author = {Nobody},
  title  = {Unverifiable}
}

@article{doi_in_url,
  title = {From the URL},
  url   = {https://doi.org/10.7717/peerj-cs.86}
}
"""


def test_entries_and_dois_are_parsed():
    entries = {e.key: e for e in parse_bib(BIB)}
    assert entries["good"].doi == "10.1038/s41597-022-01143-6"
    assert entries["good"].title == "A real title"


def test_an_entry_without_a_doi_is_visible_as_such():
    """Unverifiable must be a reported state, not an assumed pass."""
    entries = {e.key: e for e in parse_bib(BIB)}
    assert entries["nodoi"].doi is None


def test_a_doi_is_recovered_from_a_url():
    entries = {e.key: e for e in parse_bib(BIB)}
    assert entries["doi_in_url"].doi == "10.7717/peerj-cs.86"


def test_titles_compare_modulo_formatting_not_substance():
    """Registries differ in braces, case, punctuation and accents constantly.
    Being stricter produces false alarms, which train you to ignore the check."""
    assert normalize("A Large-Scale Study on {Research} Code") == normalize(
        "a large scale study on research code"
    )
    assert normalize("Merc{\\`e} Crosas") == normalize("Merce Crosas")
    # ...but a genuinely different title still differs.
    assert normalize("Scientific software production") != normalize(
        "Scientific software production: incentives and collaboration"
    )


def test_the_subtitle_case_that_actually_occurred():
    """Caught on the validator's first run, in this repository's own .bib:
    the DOI resolved, but to a title without the subtitle I had added."""
    assert normalize("Scientific software production") != normalize(
        "Scientific software production: incentives and collaboration"
    )

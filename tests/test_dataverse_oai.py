"""Tests for the OAI-PMH manifest harvest.

The fixture is a trimmed copy of a real Harvard response, and the expected file
ids are the ones the JSON API returned for the same dataset *before* the WAF
block -- so this is validated against ground truth from another channel, not
against itself.
"""

from __future__ import annotations

from softverse.sources.dataverse_oai import parse_ddi

# Real shape: the access URL carries the file id, the label carries the name.
DDI = """<OAI-PMH><GetRecord><record>
<header><identifier>doi:10.7910/DVN/RE0FZS</identifier>
<datestamp>2023-11-19T01:12:48Z</datestamp>
<setSpec>social_science_and_humanities</setSpec></header>
<metadata><codeBook>
<otherMat level="datafile" URI="https://dataverse.harvard.edu/api/access/datafile/7517531">
  <labl>01_preprocessing.R</labl></otherMat>
<otherMat level="datafile" URI="https://dataverse.harvard.edu/api/access/datafile/7517528">
  <labl>04_fh.R</labl></otherMat>
<otherMat level="datafile" URI="https://dataverse.harvard.edu/api/access/datafile/7517540">
  <labl>codebook.pdf</labl></otherMat>
</codeBook></metadata></record></GetRecord></OAI-PMH>"""

ERROR = """<OAI-PMH><error code="idDoesNotExist">no such record</error></OAI-PMH>"""


def test_file_ids_and_names_are_recovered():
    """These ids are what the JSON API returned before it was blocked."""
    m = parse_ddi(DDI, "doi:10.7910/DVN/RE0FZS")
    assert (7517531, "01_preprocessing.R") in m.files
    assert (7517528, "04_fh.R") in m.files
    assert m.datestamp == "2023-11-19T01:12:48Z"
    assert m.sets == ["social_science_and_humanities"]


def test_only_code_like_files_are_candidates():
    """A PDF codebook is listed but is not something to download and parse."""
    m = parse_ddi(DDI, "doi:x")
    assert len(m.files) == 3
    assert {n for _, n in m.candidates} == {"01_preprocessing.R", "04_fh.R"}


def test_an_oai_error_is_reported_not_read_as_an_empty_dataset():
    """`idDoesNotExist` must not look like a dataset containing no code."""
    m = parse_ddi(ERROR, "doi:missing")
    assert m.error == "idDoesNotExist"
    assert m.files == []


def test_a_record_with_no_files_is_distinguishable_from_an_error():
    m = parse_ddi(
        "<OAI-PMH><GetRecord><record></record></GetRecord></OAI-PMH>", "doi:x"
    )
    assert m.error is None
    assert m.files == []

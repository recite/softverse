"""License strings as the repositories actually report them."""

from __future__ import annotations

import pytest

from softverse.release.licenses import classify


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Dataverse display names, every one present in the 2026 collection.
        ("CC0 1.0", "CC0-1.0"),
        ("CC BY 4.0", "CC-BY-4.0"),
        ("MIT", "MIT"),
        ("CC BY-NC 4.0", "CC-BY-NC-4.0"),
        ("CC BY-NC-ND 4.0", "CC-BY-NC-ND-4.0"),
        ("CC BY-NC-SA 4.0", "CC-BY-NC-SA-4.0"),
        ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
        ("CC BY-ND 4.0", "CC-BY-ND-4.0"),
        # Zenodo identifiers.
        ("cc-by-4.0", "CC-BY-4.0"),
        ("cc-zero", "CC0-1.0"),
        ("cc-by-nc-sa-4.0", "CC-BY-NC-SA-4.0"),
        ("bsd-3-clause", "BSD-3-Clause"),
        ("gpl-3.0-or-later", "GPL-3.0-or-later"),
        # Zenodo identifiers seen in the 2026 collection.
        ("isc-license", "ISC"),
        ("apache2.0", "Apache-2.0"),
    ],
)
def test_known_licenses_are_redistributable(name, expected):
    result = classify(name)
    assert result.license_id == expected
    assert result.redistributable


def test_custom_terms_are_not_redistributable_even_when_they_mention_cc0():
    terms = (
        "This dataset is made available under a Creative Commons CC0 license "
        "with the following additional/modified terms and conditions: ..."
    )
    assert classify(None, terms) == classify("", terms)
    assert classify(None, terms).license_id == "custom"
    assert not classify(None, terms).redistributable


def test_no_license_and_no_terms_is_none():
    assert classify(None, None).license_id == "none"
    assert not classify(None, "   ").redistributable


def test_an_unknown_identifier_is_kept_but_not_trusted():
    result = classify("other-closed")
    assert result.license_id == "other-closed"
    assert not result.redistributable

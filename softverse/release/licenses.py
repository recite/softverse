"""What a deposit's license lets us republish.

The corpus release publishes file content only for deposits whose license
allows redistribution, and derived rows -- which packages a file loads, at what
version -- for every deposit. This module turns what each repository reports
into one SPDX-style identifier and that one decision.

The two repositories report licenses differently. Harvard Dataverse gives a
display name (``CC0 1.0``, ``CC BY-NC 4.0``) or, instead of any license, free
text in ``termsOfUse``; in 14,618 deposits the two never co-occur. Zenodo gives
a lowercase identifier (``cc-by-4.0``). Both reduce to the same identifiers
here.

The rule is deliberately conservative. Custom terms are not redistributable,
even the 144 that begin "CC0 ... with the following additional terms", because
whether an added term forbids reposting has to be read, and 248 of those texts
say outright that the files are "not to be distributed/posted outside of the
Harvard Dataverse". An identifier this module does not know is also not
redistributable until someone adds it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: License families whose terms permit sharing the files unmodified. NC and ND
#: variants qualify: publishing a file verbatim in a non-commercial research
#: dataset is sharing, not adapting, and the row keeps its identifier so a
#: user who needs commercial terms can filter them out.
_REDISTRIBUTABLE = re.compile(
    r"^(CC0-1\.0|CC-BY(-SA|-NC|-ND|-NC-SA|-NC-ND)?-[0-9.]+|MIT|ISC|Unlicense|"
    r"Apache-2\.0|BSD-[0-9]-Clause|GPL-[23]\.0(-only|-or-later)?|"
    r"LGPL-[23]\.[01](-only|-or-later)?|MPL-2\.0|ODC-By-1\.0|ODbL-1\.0|PDDL-1\.0)$"
)

#: Names Zenodo and Dataverse use that do not map mechanically.
_ALIASES = {
    "cc-zero": "CC0-1.0",
    "cc0": "CC0-1.0",
    "cc0-1.0": "CC0-1.0",
    "cc0 1.0": "CC0-1.0",
    "mit": "MIT",
    "mit-license": "MIT",
    "apache-2.0": "Apache-2.0",
    "isc": "ISC",
    "unlicense": "Unlicense",
    "mpl-2.0": "MPL-2.0",
    "odc-by-1.0": "ODC-By-1.0",
    "odbl-1.0": "ODbL-1.0",
    "pddl-1.0": "PDDL-1.0",
}


@dataclass(frozen=True)
class License:
    """A deposit's license, reduced to what the release needs."""

    #: SPDX-style identifier, or ``custom`` for terms-of-use text, or ``none``.
    license_id: str
    redistributable: bool


def _spdx(name: str) -> str:
    """Map a Dataverse display name or Zenodo id to an SPDX-style id.

    Returns:
        The identifier, or the input tidied if it matches no known form.
    """
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # `CC BY-NC-SA 4.0` (Dataverse) and `cc-by-nc-sa-4.0` (Zenodo).
    creative = re.fullmatch(r"cc[- ]by((?:-(?:nc|nd|sa))*)[- ]([0-9.]+)", key)
    if creative:
        return f"CC-BY{creative.group(1).upper()}-{creative.group(2)}"
    bsd = re.fullmatch(r"bsd-([0-9])-clause", key)
    if bsd:
        return f"BSD-{bsd.group(1)}-Clause"
    gpl = re.fullmatch(r"(l?gpl)-([0-9.]+)(-only|-or-later)?", key)
    if gpl:
        return f"{gpl.group(1).upper()}-{gpl.group(2)}{gpl.group(3) or ''}"
    return name.strip()


def classify(name: str | None, terms_of_use: str | None = None) -> License:
    """Reduce a repository's license report to an identifier and a decision.

    Args:
        name: The license name or identifier the repository gives, if any.
        terms_of_use: Free-text terms, which Dataverse gives instead of a name.

    Returns:
        The license identifier and whether file content may be republished.
    """
    if name and name.strip():
        license_id = _spdx(name)
        return License(license_id, bool(_REDISTRIBUTABLE.match(license_id)))
    if terms_of_use and terms_of_use.strip():
        return License("custom", False)
    return License("none", False)

"""Read what a deposit says about the environment it ran in.

Four file types state something a script cannot: which version of a package
was installed, which version of R or Python or Stata ran it, and on which
operating system. None of this was read before, and all of it sits in files
the corpus already holds.

The signals are sparse and that is the normal case. A reader who wants
"papers running Stata 14" is entitled to see, in the same breath, how many
deposits said anything at all, which is why `build()` publishes a denominator
beside every count here.

Two grains, kept apart. A *declaration* names a package and a version: what
the author asked for, or what the deposit shipped. A *signal* names the
interpreter or the machine. Both are per file, because the file is what said
it, and a deposit can contradict itself.

The DESCRIPTION rule worth stating: a DESCRIPTION contributes **its own
package and version, and nothing else**. Its `Imports:` line lists what *that
package* depends on, not what the deposit uses. Folding those in would credit
a deposit that vendored one library tree with the full closure underneath it,
which is the same error, in the same direction, as counting calls inside
bundled packages as the author's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from email.parser import Parser

from softverse.model.enums import Ecosystem

#: `Built: R 4.2.2; x86_64-w64-mingw32; 2023-02-19 04:12:08 UTC; windows`
#: R version first, platform second, build date third, OS last. The platform
#: field is empty in about half of these, so the fields are addressed by
#: position from a split rather than matched as a whole.
_BUILT_R_VERSION = re.compile(r"^R\s+([0-9][0-9.-]*)")

#: A DESCRIPTION-like file that never names a package is not a package
#: manifest. `sass` ships `sass-theme/DESCRIPTION` and `sass-font/DESCRIPTION`
#: describing CSS themes, and 800 of the 1,845 files named DESCRIPTION in this
#: corpus are that sort of thing. Requiring `Package:` is what separates them.
_REQUIRED_DESCRIPTION_FIELDS = ("Package", "Version")


@dataclass(frozen=True)
class Declaration:
    """One package a manifest names, at whatever version it names."""

    package: str
    version_constraint: str | None
    ecosystem: Ecosystem
    #: `installed` is what a deposit shipped, `locked` is a resolved closure
    #: in which direct and transitive are indistinguishable, and `direct` is
    #: what an author wrote down. The three answer different questions and
    #: averaging them would answer none.
    dependency_role: str


@dataclass(frozen=True)
class ManifestRead:
    kind: str
    declarations: list[Declaration] = field(default_factory=list)
    #: `r_version`, `python_version`, `stata_version`, `os`.
    signals: dict[str, str] = field(default_factory=dict)


def read_description(text: str) -> ManifestRead | None:
    """An R `DESCRIPTION`: the package it describes, and how it was built.

    Debian Control Format, which is RFC 822 with folded continuation lines, so
    `email.parser` reads it correctly out of the standard library including the
    multi-line `Imports:` this function then declines to use.
    """
    message = Parser().parsestr(text)
    if any(not message.get(f) for f in _REQUIRED_DESCRIPTION_FIELDS):
        return None

    signals: dict[str, str] = {}
    if built := message.get("Built"):
        parts = [p.strip() for p in built.replace("\r", "").split(";")]
        if parts and (match := _BUILT_R_VERSION.match(parts[0])):
            signals["r_version"] = match.group(1)
        if len(parts) >= 4 and parts[3]:
            signals["os"] = parts[3].lower()

    return ManifestRead(
        kind="description",
        declarations=[
            Declaration(
                package=str(message["Package"]).strip(),
                version_constraint=str(message["Version"]).strip(),
                ecosystem=Ecosystem.CRAN,
                dependency_role="installed",
            )
        ],
        signals=signals,
    )


def read_renv_lock(text: str) -> ManifestRead | None:
    """An `renv.lock`: the R version and the closure renv resolved.

    Every package here is `locked` rather than `direct`. renv records the
    closure it installed and does not mark which packages the author asked for,
    so calling any of them direct would be an inference dressed as a reading.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    signals: dict[str, str] = {}
    version = (payload.get("R") or {}).get("Version")
    if isinstance(version, str) and version.strip():
        signals["r_version"] = version.strip()

    declarations = []
    for name, entry in (payload.get("Packages") or {}).items():
        if not isinstance(entry, dict):
            continue
        declarations.append(
            Declaration(
                package=str(entry.get("Package") or name),
                version_constraint=(
                    str(entry["Version"]) if entry.get("Version") else None
                ),
                ecosystem=Ecosystem.CRAN,
                dependency_role="locked",
            )
        )
    if not declarations and not signals:
        return None
    return ManifestRead(kind="renv_lock", declarations=declarations, signals=signals)


def read_requirements(text: str) -> ManifestRead | None:
    """A `requirements.txt`: what the author wrote down, pinned or not.

    Parsed with `packaging`, the PyPA library that defines the grammar, rather
    than by splitting on `==`. The corpus contains extras, environment markers
    and `>=` alongside the exact pins, and a split on `==` reads
    `pandas[excel]>=2.0; python_version<"3.11"` as a package named
    `pandas[excel]>=2.0; python_version<"3.11"`.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    declarations = []
    for line in text.splitlines():
        line = line.split(" #")[0].strip()
        # `-r base.txt`, `-e .`, `--index-url ...`: directives, not packages.
        if not line or line.startswith(("#", "-")):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            continue
        declarations.append(
            Declaration(
                package=requirement.name,
                version_constraint=str(requirement.specifier) or None,
                ecosystem=Ecosystem.PYPI,
                dependency_role="direct",
            )
        )
    if not declarations:
        return None
    return ManifestRead(kind="requirements_txt", declarations=declarations)


#: Kernel name in a notebook's metadata -> the signal its version is.
#: A Julia notebook reports version 1.9.2 in the same field a Python notebook
#: reports 3.11.7, so filing them under one name would put Julia releases in
#: the Python distribution.
_KERNEL_SIGNAL = {
    "python": "python_version",
    "r": "r_version",
    "julia": "julia_version",
    "stata": "stata_version",
    "matlab": "matlab_version",
}


def read_notebook_environment(text: str) -> ManifestRead | None:
    """A notebook's `language_info.version`: the interpreter that really ran.

    Stronger evidence than anything else here. A `requirements.txt` records
    what an author meant to install and a `DESCRIPTION` records what was
    installed at some point; Jupyter writes this field when the notebook is
    executed, so it names the interpreter that produced the output in the
    file. It is also the only source of a Python version in this corpus:
    requirements files pin packages and say nothing about Python itself.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    info = (payload.get("metadata") or {}).get("language_info") or {}
    version = info.get("version")
    signal = _KERNEL_SIGNAL.get(str(info.get("name") or "").lower())
    if not signal or not isinstance(version, str) or not version.strip():
        return None
    return ManifestRead(kind="notebook", signals={signal: version.strip()})


#: Matched on the filename, lowercased. `requirements-dev.txt` and
#: `requirements_prod.txt` both occur and both are requirements files.
_READERS = {
    "description": read_description,
    "renv.lock": read_renv_lock,
}


def read_manifest(filename: str, text: str) -> ManifestRead | None:
    """Dispatch on the filename. Returns None when the file says nothing."""
    name = filename.lower()
    if reader := _READERS.get(name):
        return reader(text)
    if name.startswith("requirements") and name.endswith(".txt"):
        return read_requirements(text)
    if name.endswith(".ipynb"):
        return read_notebook_environment(text)
    return None


def is_manifest(filename: str) -> bool:
    """Whether `read_manifest` would try. Cheap enough to call per file."""
    name = filename.lower()
    return (
        name in _READERS
        or name.endswith(".ipynb")
        or (name.startswith("requirements") and name.endswith(".txt"))
    )

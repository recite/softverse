"""Fetch and pin package-registry snapshots.

Resolution has to be reproducible: the same corpus resolved next year must give
the same answer, and a referee must be able to check it. So a registry is never
queried live during analysis. It is fetched once into a dated snapshot, hashed,
and recorded in ``registries.lock.json``; the analysis reads only through the
lock.

Two registry choices are load-bearing and worth stating:

**CRAN Archive is included.** ``rgdal``, ``maptools``, ``rgeos`` and ``Zelig``
are gone from current CRAN but were entirely real when the code that uses them
was written -- they appear 135, 116, 81 and 115 times in the v1 tally.
Validating against current CRAN alone would misclassify a decade of legitimate
use as noise. The distinction is preserved (``known_archived``) because
disappearing packages are themselves a finding.

**Python import names are not distribution names.** ``import sklearn`` comes
from ``scikit-learn``, ``cv2`` from ``opencv-python``, ``yaml`` from ``PyYAML``.
A PyPI lookup of the raw import name is not resolution, so an explicit alias
table carries the common cases and the resolution basis is recorded per mention
rather than papered over.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import stdlib_list

from softverse.logging_setup import get_logger, stage

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

CRANDB_DESC = "https://crandb.r-pkg.org/-/desc"
CRAN_ARCHIVE = "https://cran.r-project.org/src/contrib/Archive/"
BIOC_VIEWS = "https://bioconductor.org/packages/release/bioc/VIEWS"
PYPI_SIMPLE = "https://pypi.org/simple/"
STATA_BASE_ADO = "https://api.github.com/repos/opensafely-core/stata-docker/git/trees/main?recursive=1"
CRAN_PACKAGES = "https://cran.r-project.org/src/contrib/PACKAGES.gz"
TASK_VIEWS = ("Econometrics", "CausalInference")
TASK_VIEW_SOURCE = (
    "https://raw.githubusercontent.com/cran-task-views/{view}/main/{view}.md"
)
PIPREQS_MAPPING = (
    "https://raw.githubusercontent.com/bndr/pipreqs/master/pipreqs/mapping"
)
JULIA_REGISTRY = (
    "https://raw.githubusercontent.com/JuliaRegistries/General/master/Registry.toml"
)

#: R packages shipped with R itself. Never a citation to a contributed package.
#: v1 counted `grid` 268 times and `parallel` 115 as though they were CRAN
#: packages, and its classifier lowercased candidates before comparing against a
#: mixed-case set, so `grDevices`, `Matrix` and `KernSmooth` were mislabelled
#: third-party. R package names are case-sensitive; matching here is too.
BASE_R = frozenset(
    {
        "base",
        "compiler",
        "datasets",
        "grDevices",
        "graphics",
        "grid",
        "methods",
        "parallel",
        "splines",
        "stats",
        "stats4",
        "tcltk",
        "tools",
        "utils",
    }
)

#: Shipped with R but separately maintained and genuinely citable. Kept distinct
#: from base so the paper can report either convention.
RECOMMENDED_R = frozenset(
    {
        "KernSmooth",
        "MASS",
        "Matrix",
        "boot",
        "class",
        "cluster",
        "codetools",
        "foreign",
        "lattice",
        "mgcv",
        "nlme",
        "nnet",
        "rpart",
        "spatial",
        "survival",
    }
)

#: Import name -> PyPI distribution, for cases where they differ. Curated
#: because no complete machine-readable mapping exists; the resolution basis is
#: recorded per mention so incompleteness is visible rather than assumed away.
PYPI_IMPORT_ALIASES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",
    "sqlalchemy": "SQLAlchemy",
    "OpenSSL": "pyOpenSSL",
    "Crypto": "pycryptodome",
    "serial": "pyserial",
    "usb": "pyusb",
    "win32com": "pywin32",
    "win32api": "pywin32",
    "win32con": "pywin32",
    "win32gui": "pywin32",
    "win32evtlog": "pywin32",
    "win32evtlogutil": "pywin32",
    "pythoncom": "pywin32",
    "pywintypes": "pywin32",
    "umap": "umap-learn",
    "pkg_resources": "setuptools",
    "setuptools": "setuptools",
    "google": "google-api-python-client",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "fitz": "PyMuPDF",
    "skimage": "scikit-image",
    "statsmodels": "statsmodels",
    "Bio": "biopython",
    "netCDF4": "netCDF4",
    "osgeo": "GDAL",
    "gdal": "GDAL",
    "shapely": "Shapely",
    "geopandas": "geopandas",
    "tables": "tables",
    "igraph": "python-igraph",
    "graph_tool": "graph-tool",
    "psycopg2": "psycopg2-binary",
    "MySQLdb": "mysqlclient",
    "pymc3": "pymc3",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "nltk": "nltk",
    "gensim": "gensim",
    "wordcloud": "wordcloud",
    "dotenv": "python-dotenv",
    "magic": "python-magic",
    "lxml": "lxml",
    "ruamel": "ruamel.yaml",
    "attr": "attrs",
    "zmq": "pyzmq",
    "jwt": "PyJWT",
    "Levenshtein": "python-Levenshtein",
    "unidecode": "Unidecode",
    "pandas_datareader": "pandas-datareader",
    "linearmodels": "linearmodels",
    "stargazer": "stargazer",
}


def python_stdlib() -> frozenset[str]:
    """Top-level standard-library module names, across every Python since 2.6.

    A static fact, not a question put to the analyst's machine: v1 asked
    ``importlib.util.find_spec`` and so labelled ``os``, ``sys`` and ``json``
    third-party wherever they happened not to resolve.

    The list is `stdlib-list`'s, maintained under the PyPI organisation. It
    replaced ``sys.stdlib_module_names`` plus a hand-kept list of 3.8-3.13
    removals, which knew nothing of Python 2: ``cPickle``, ``urllib2``,
    ``StringIO`` and ``Queue`` were reported as packages no registry lists, in
    a corpus whose deposits reach back to 2010.
    """
    names = set(sys.stdlib_module_names)
    for version in stdlib_list.short_versions:
        names |= {
            module.split(".", 1)[0] for module in stdlib_list.stdlib_list(version)
        }
    # Interpreter hooks rather than modules anyone distributes.
    names |= {"__main__", "sitecustomize", "usercustomize", "_subprocess"}
    return frozenset(names)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass
class Snapshot:
    """One fetched registry, with everything needed to verify it later."""

    registry: str
    names: list[str]
    raw: bytes
    url: str
    fetched_at: datetime
    extra: dict[str, object] | None = None

    @property
    def sha256(self) -> str:
        """Digest of the raw payload, which is what pins the snapshot."""
        return _sha256(self.raw)

    def write(self, root: Path) -> Path:
        """Write to ``root/<registry>/<date>/`` and return the directory."""
        day = self.fetched_at.date().isoformat()
        directory = root / self.registry / day
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "names.json").write_text(json.dumps(sorted(self.names)))
        (directory / "raw.bin").write_bytes(self.raw)
        (directory / "source.json").write_text(
            json.dumps(
                {
                    "registry": self.registry,
                    "url": self.url,
                    "fetched_at": self.fetched_at.isoformat(),
                    "sha256": self.sha256,
                    "n_names": len(self.names),
                    **(self.extra or {}),
                },
                indent=2,
            )
        )
        logger.info(
            "snapshot written",
            extra={
                "registry": self.registry,
                "n_names": len(self.names),
                "sha256": self.sha256[:12],
            },
        )
        return directory


def _get(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch_cran(client: httpx.Client) -> Snapshot:
    """Current CRAN packages, with first-release dates where available."""
    raw = _get(client, CRANDB_DESC)
    payload = json.loads(raw)
    return Snapshot(
        registry="cran",
        names=list(payload),
        raw=raw,
        url=CRANDB_DESC,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_cran_archive(client: httpx.Client) -> Snapshot:
    """Packages that once existed on CRAN and have since been removed."""
    raw = _get(client, CRAN_ARCHIVE)
    names = re.findall(r'href="([A-Za-z0-9._]+)/"', raw.decode("utf-8", "replace"))
    return Snapshot(
        registry="cran_archive",
        names=sorted(set(names)),
        raw=raw,
        url=CRAN_ARCHIVE,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_bioconductor(client: httpx.Client) -> Snapshot:
    """Snapshot the Bioconductor package names from its views index."""
    raw = _get(client, BIOC_VIEWS)
    names = re.findall(
        r"^Package:\s*(\S+)", raw.decode("utf-8", "replace"), re.MULTILINE
    )
    return Snapshot(
        registry="bioconductor",
        names=sorted(set(names)),
        raw=raw,
        url=BIOC_VIEWS,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_pypi(client: httpx.Client) -> Snapshot:
    """Every PyPI distribution name, via the PEP 691 simple index."""
    response = client.get(
        PYPI_SIMPLE, headers={"Accept": "application/vnd.pypi.simple.v1+json"}
    )
    response.raise_for_status()
    raw = response.content
    # The PEP 691 content type is `application/vnd.pypi.simple.v1+json`, not
    # `application/json`. Matching the latter silently fell through to the HTML
    # branch, which matched nothing -- caught only because MINIMUM_EXPECTED
    # refused a snapshot of zero names.
    if "json" in response.headers.get("content-type", ""):
        names = [p["name"] for p in json.loads(raw)["projects"]]
    else:
        names = re.findall(r">([^<]+)</a>", raw.decode("utf-8", "replace"))
    return Snapshot(
        registry="pypi",
        names=sorted(set(names)),
        raw=raw,
        url=PYPI_SIMPLE,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_julia(client: httpx.Client) -> Snapshot:
    """Julia General registry.

    Entries live under ``[packages]`` as inline tables keyed by UUID::

        0004c1f4-... = { name = "TuringGLM", path = "T/TuringGLM" }

    A line-anchored ``^name =`` matches only the registry's own name field, so
    it returns exactly one result -- caught by MINIMUM_EXPECTED rather than
    shipped as a registry of one.
    """
    raw = _get(client, JULIA_REGISTRY)
    text = raw.decode("utf-8", "replace")
    _, _, packages = text.partition("[packages]")
    names = re.findall(r'\bname\s*=\s*"([^"]+)"', packages)
    return Snapshot(
        registry="julia_general",
        names=sorted(set(names)),
        raw=raw,
        url=JULIA_REGISTRY,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_pypi_import_map(client: httpx.Client) -> Snapshot:
    """Import name -> distribution, from pipreqs' maintained mapping.

    ``import shapefile`` installs as ``pyshp``. No registry publishes that
    relation, and the curated table below knows fifty-five of them; pipreqs
    keeps about eleven hundred, contributed by the people who hit each one.
    Entries are kept as ``import:distribution`` so the snapshot has the same
    shape as every other.
    """
    raw = _get(client, PIPREQS_MAPPING)
    names = [
        line.strip()
        for line in raw.decode("utf-8").splitlines()
        if line.count(":") == 1 and not line.startswith("#")
    ]
    return Snapshot(
        registry="pypi_import_map",
        names=names,
        raw=raw,
        url=PIPREQS_MAPPING,
        fetched_at=datetime.now(tz=UTC),
    )


_DEPENDENCY_FIELDS = ("Depends", "Imports", "LinkingTo")


def parse_cran_packages(text: str) -> dict[str, list[str]]:
    """Each CRAN package's hard dependencies, from the `PACKAGES` index.

    `Depends`, `Imports` and `LinkingTo` are what gets installed along with a
    package; `Suggests` is not. Version constraints and `R` itself are
    dropped, and a field may run over several lines.
    """
    out: dict[str, list[str]] = {}
    for stanza in re.split(r"\n\s*\n", text.strip()):
        fields: dict[str, str] = {}
        key = None
        for line in stanza.splitlines():
            if line[:1].isspace() and key:
                fields[key] += " " + line.strip()
            elif ":" in line:
                key, _, value = line.partition(":")
                fields[key] = value.strip()
        if "Package" not in fields:
            continue
        needs = {
            re.sub(r"\s*\(.*?\)", "", item).strip()
            for name in _DEPENDENCY_FIELDS
            for item in fields.get(name, "").split(",")
        }
        out[fields["Package"]] = sorted(needs - {"", "R"})
    return out


def fetch_cran_dependencies(client: httpx.Client) -> Snapshot:
    """What each CRAN package installs with it, as `package:dep,dep` entries."""
    raw = _get(client, CRAN_PACKAGES)
    graph = parse_cran_packages(gzip.decompress(raw).decode("utf-8", "replace"))
    return Snapshot(
        registry="cran_dependencies",
        names=[
            f"{package}:{','.join(needs)}" for package, needs in sorted(graph.items())
        ],
        raw=raw,
        url=CRAN_PACKAGES,
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_cran_task_views(client: httpx.Client) -> Snapshot:
    """Packages CRAN's task views list for the fields this corpus is drawn from.

    A task view is a maintained, signed list of the packages aimed at one
    kind of work. *Econometrics* and *CausalInference* are the two that name
    what economics and political science estimate; they were fixed before any
    result that uses them was computed. Entries are `view:package`.
    """
    names, raws = [], []
    for view in TASK_VIEWS:
        raw = _get(client, TASK_VIEW_SOURCE.format(view=view))
        raws.append(raw)
        text = raw.decode("utf-8", "replace")
        names += [
            f"{view}:{p}"
            for p in sorted(set(re.findall(r'pkg\("([A-Za-z0-9.]+)"', text)))
        ]
    return Snapshot(
        registry="cran_task_views",
        names=names,
        raw=b"\n".join(raws),
        url=TASK_VIEW_SOURCE.format(view="{" + ",".join(TASK_VIEWS) + "}"),
        fetched_at=datetime.now(tz=UTC),
    )


def fetch_stata_base_ado(client: httpx.Client) -> Snapshot:
    """The commands official Stata ships as `.ado` files, by listing them.

    StataCorp's help server answers for documented commands only. `_dots`,
    `tset` and `gr_setscheme` ship with Stata and have no help page, so asking
    it left them reported as software in no archive. No list is published, but
    a Stata installation's `ado/base` directory is one, and a public container
    build carries a copy. Only the file names are read.
    """
    raw = _get(client, STATA_BASE_ADO)
    names = {
        entry["path"].rsplit("/", 1)[-1][:-4].lower()
        for entry in json.loads(raw)["tree"]
        if re.fullmatch(r"bin/ado/base/[^/]+/[^/]+[.]ado", entry["path"])
    }
    return Snapshot(
        registry="stata_base_ado",
        names=sorted(names),
        raw=raw,
        url=STATA_BASE_ADO,
        fetched_at=datetime.now(tz=UTC),
    )


FETCHERS = {
    "cran_dependencies": fetch_cran_dependencies,
    "cran_task_views": fetch_cran_task_views,
    "stata_base_ado": fetch_stata_base_ado,
    "pypi_import_map": fetch_pypi_import_map,
    "cran": fetch_cran,
    "cran_archive": fetch_cran_archive,
    "bioconductor": fetch_bioconductor,
    "pypi": fetch_pypi,
    "julia_general": fetch_julia,
}

#: Sanity floors. A registry that returns far too few names has failed in a way
#: HTTP 200 will not reveal -- an error page, a truncated response, a changed
#: format. Writing that snapshot would quietly mark thousands of real packages
#: unknown, so the fetch fails loudly instead.
MINIMUM_EXPECTED = {
    "cran_dependencies": 20_000,
    "cran_task_views": 200,
    "stata_base_ado": 3_000,
    "pypi_import_map": 1_000,
    "cran": 20_000,
    "cran_archive": 20_000,
    "bioconductor": 1_500,
    "pypi": 500_000,
    "julia_general": 8_000,
}


def fetch_all(root: Path, registries: list[str] | None = None) -> dict[str, str]:
    """Fetch registries and write snapshots. Returns ``{registry: sha256}``."""
    wanted = registries or list(FETCHERS)
    digests: dict[str, str] = {}
    with (
        stage("registry-fetch", logger) as stats,
        httpx.Client(
            timeout=180.0,
            follow_redirects=True,
            headers={"User-Agent": "softverse (research; github.com/recite/softverse)"},
        ) as client,
    ):
        for name in wanted:
            snapshot = FETCHERS[name](client)
            floor = MINIMUM_EXPECTED.get(name, 0)
            if len(snapshot.names) < floor:
                raise RuntimeError(
                    f"{name}: got {len(snapshot.names):,} names, expected at least "
                    f"{floor:,}. Refusing to write a snapshot that would silently "
                    f"mark real packages unknown."
                )
            snapshot.write(root)
            digests[name] = snapshot.sha256
            stats.incr(f"{name}_names", len(snapshot.names))
    return digests

"""Configuration file for the Sphinx documentation builder."""

import importlib.metadata
import sys
from pathlib import Path

# Relative to this file, not to the working directory. `abspath("..")` is the
# parent of wherever sphinx-build happened to be invoked from, which is the
# repo's parent when it runs from the root, and the import below has to find
# `scripts_build_site.py` at the repo root every time.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# The site is generated, not checked in: `docs/index.md`, the data directory
# and the lookup page are all built from the released tables. The fleet's
# docs workflow runs `sphinx-build` with no pre-build hook, so this runs here,
# which Sphinx reads before it discovers sources.
from scripts_build_site import generate  # noqa: E402

generate()

project = "Softverse"
copyright = "2026, Gaurav Sood"
author = "Gaurav Sood"

# Bound through the module, not imported by name: a bare `version` at module
# scope in conf.py *is* the Sphinx `version` setting, so importing the
# function under that name hands Sphinx a callable and the build dies in
# inventory dumping with no mention of conf.py.
try:
    release = importlib.metadata.version("softverse")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build", "_extra", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Softverse"
# No `html_static_path`: there are no custom assets, and the empty `_static`
# it used to point at does not survive a clone, because git does not track
# empty directories. That is a warning locally and a failed build in CI.

# Copied into the output root untouched. The lookup page is a complete
# document with its own design and its own dark mode, so wrapping it in the
# Furo chrome would fight both.
html_extra_path = ["_extra"]

myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

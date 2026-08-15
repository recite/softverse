"""Configuration file for the Sphinx documentation builder."""

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

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

templates_path = ["_templates"]
exclude_patterns = ["_build", "_extra", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Softverse"
html_static_path = ["_static"]

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

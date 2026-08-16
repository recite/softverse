"""Sphinx configuration — fleet standard via py-canon, plus a generated site.

`preen adopt` overwrites this file with the four-line stub, so the additions
below have to be re-applied after an adopt. They are additions rather than
replacements: `configure()` sets the theme, the extensions and the project
metadata, and everything here is either something it does not know about or
something this repo needs it not to do.
"""

import sys
from pathlib import Path

from py_canon.sphinx import configure

# Relative to this file, not to the working directory: `sphinx-build` runs
# from the repo root, so `abspath("..")` would point at the repo's parent and
# the import below would not find `scripts_build_site.py`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts_build_site import generate  # noqa: E402

# The site is generated rather than checked in: `docs/index.md`, the released
# tables under `_extra/data/`, the lookup page and the paper are all built
# from `build/release/tally/`. The fleet's docs workflow runs `sphinx-build`
# directly and offers no pre-build hook, so this happens here, which Sphinx
# reads before it goes looking for sources. sharepack does the same thing for
# the same reason.
generate()

configure(
    globals(),
    html_title="Softverse",
    # Copied into the output root untouched. The lookup page is a complete
    # document with its own design and its own dark mode, so wrapping it in
    # the Furo chrome would fight both.
    html_extra_path=["_extra"],
    # `_extra` must be excluded as *source*, or Sphinx parses the release
    # descriptor that `stage()` copies to `_extra/data/README.md`, finds a
    # document in no toctree, and `-W` turns that into a failed build.
    exclude_patterns=["_build", "_extra", "Thumbs.db", ".DS_Store"],
    myst_heading_anchors=3,
    intersphinx_mapping={
        "python": ("https://docs.python.org/3", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
    },
)

"""Softverse: measuring software referenced in social science replication code.

The estimand, stated exactly: among replication datasets deposited to a journal
collection in a given year that contain at least one analyzable script in a
given language, the share that *statically reference* a given package.

Static reference is not runtime use. See ``docs/estimand.md``.
"""

__version__ = "2.0.0.dev0"
__author__ = "Gaurav Sood and Daniel Weitzel"

#: Bumped whenever extraction logic changes. Stamped on every mention row so a
#: number can always be attributed to the instrument that produced it.
EXTRACTOR_VERSION = "2.0.0.dev0"

"""Softverse: measuring software referenced in social science replication code.

The estimand, stated exactly: among replication datasets deposited to a journal
collection in a given year that contain at least one analyzable script in a
given language, the share that *statically reference* a given package.

Static reference is not runtime use. The paper states the estimand exactly
and measures the gap: https://recite.github.io/softverse/paper/softverse.pdf
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__author__ = "Gaurav Sood"

try:
    __version__ = _version("softverse")
except PackageNotFoundError:  # running from a source tree with nothing installed
    __version__ = "0.0.0"

#: Bumped whenever extraction logic changes, and deliberately *not* the package
#: version. It is stamped on every mention row so a count can be attributed to
#: the instrument that produced it, so it must change when the extractor
#: changes and hold still when the release number moves for any other reason.
#: Tying the two would silently restamp a corpus that was never re-parsed.
#: 2.1.0 records the called function and reads Python attribute calls through
#: the file's alias map, so it emits a column and 100,347 mentions that 2.0.0
#: never produced. Leaving the stamp at 2.0.0 would have given two different
#: extractors one name across two published releases, which is the single
#: thing this constant exists to prevent.
EXTRACTOR_VERSION = "2.1.0"

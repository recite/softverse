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
#: 2.2.0 records the version an install call asks for (`pinned_version`) and
#: stops recording `scikit-learn=1.2.2` and `dask[complete]` as package names.
#: 2.3.0 records where an install fetches from (`remote`), reads `ssc install
#: x, replace` and the rest of the install lines a comma or an `if` had hidden,
#: and lexes Stata's comment continuations, spaced `# delimit` and brace-form
#: Mata, each of which had been reporting option words as commands.
#: 2.4.0 reads the function an `egen` call asks for, which is how `egenmore`
#: and its kind are used and the only way they are, and gives every mention a
#: key of its own: Stata mentions had been sharing one per command per file.
EXTRACTOR_VERSION = "2.4.0"

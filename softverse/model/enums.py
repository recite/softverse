"""Controlled vocabularies.

Every categorical column in the data model draws from one of these. They are
``StrEnum`` so a value round-trips through Parquet as a plain string while still
failing loudly in Python if a typo is introduced.

The distinctions encoded here are the ones v1 collapsed. In particular:

- ``ParseStatus`` separates "no packages referenced" from "could not parse",
  "could not decode" and "never downloaded". v1 returned ``None`` for all four,
  which made the denominator unreconstructable.
- ``Resolution`` separates "resolved to a known package" from "base/stdlib",
  "dynamically loaded, unresolvable" and "unknown". v1 emitted a bare name and
  left the reader to guess.
"""

from enum import StrEnum


class Source(StrEnum):
    """Repository a deposit came from.

    Dataverse journal collections are the paper's sampling frame, but the model
    is source-general so Zenodo, ICPSR, OSF and ResearchBox can be added without
    a schema migration. What differs between them is the *grouping* a deposit
    belongs to -- a journal dataverse, a Zenodo community, an ICPSR series -- so
    that grouping is `collection_id`, and `journal_id` is set only when the
    collection is in fact a journal.
    """

    DATAVERSE = "dataverse"
    ZENODO = "zenodo"
    ICPSR = "icpsr"
    OSF = "osf"
    RESEARCHBOX = "researchbox"


class CollectionKind(StrEnum):
    """What kind of grouping a collection is."""

    JOURNAL = "journal"  # a journal's Dataverse collection
    COMMUNITY = "community"  # a Zenodo community
    SERIES = "series"  # an ICPSR series
    PROJECT_GROUP = "project_group"  # an OSF collection
    OTHER = "other"


class Language(StrEnum):
    """Source language of a file, after dispatch."""

    R = "r"
    PYTHON = "python"
    STATA = "stata"
    JULIA = "julia"
    MATLAB = "matlab"
    SAS = "sas"
    SPSS = "spss"
    SHELL = "shell"
    SQL = "sql"
    RMARKDOWN = "rmarkdown"
    NOTEBOOK = "notebook"
    MATA = "mata"
    UNKNOWN = "unknown"


#: Languages for which we make package-level claims. Everything else is
#: reported as language presence only -- MATLAB has no import statement and no
#: resolvable registry, so attributing toolboxes would be unbounded error.
FIRST_CLASS_LANGUAGES = frozenset({Language.R, Language.PYTHON, Language.STATA})


class Construct(StrEnum):
    """The syntactic form that produced a mention.

    Kept fine-grained because precision differs sharply by construct: an
    ``import`` statement is near-certain, whereas ``findit`` is a search and
    ``ssc_install`` is provisioning, neither of which implies use.
    """

    # R
    LIBRARY = "library"
    REQUIRE = "require"
    NAMESPACE_CHECK = "namespace_check"  # requireNamespace / loadNamespace
    NAMESPACE_OP = "namespace_op"  # pkg::fun
    NAMESPACE_INTERNAL = "namespace_internal"  # pkg:::fun
    P_LOAD = "p_load"  # pacman
    PACKAGE_ARG = "package_arg"  # data(package=), citation(), etc.
    INSTALL = "install"  # install.packages / install_github
    # Python
    IMPORT = "import"
    IMPORT_FROM = "import_from"
    DYNAMIC_IMPORT = "dynamic_import"  # importlib with a literal
    LOCAL_RELATIVE = "local_relative"  # from .x import y
    # Stata
    STATA_COMMAND = "stata_command"
    STATA_INSTALL = "stata_install"  # ssc install / net install
    STATA_WHICH = "stata_which"  # which / findit -- inquiry, not use
    STATA_REQUIRE = "stata_require"  # Correia's `require` manifest
    VENDORED_ADO = "vendored_ado"  # .ado shipped in the deposit
    LOCAL_PROGRAM = "local_program"  # program define in the same deposit
    # Other
    JULIA_USING = "julia_using"
    SHELL_INVOCATION = "shell_invocation"
    SHELL_INSTALL = "shell_install"
    YAML_OUTPUT = "yaml_output"  # Rmd front matter, e.g. bookdown::pdf_document2
    DYNAMIC_UNRESOLVED = "dynamic_unresolved"  # character.only=TRUE loop var


#: Constructs that indicate provisioning or inquiry rather than use. Reported,
#: but excluded from the headline tally and broken out separately.
NON_USE_CONSTRUCTS = frozenset(
    {
        Construct.INSTALL,
        Construct.STATA_INSTALL,
        Construct.STATA_WHICH,
        Construct.SHELL_INSTALL,
    }
)


class Resolution(StrEnum):
    """Outcome of resolving a raw name against the registry snapshots."""

    KNOWN_CURRENT = "known_current"
    KNOWN_ARCHIVED = "known_archived"  # was real, since removed from CRAN
    BASE_OR_STDLIB = "base_or_stdlib"
    BUILTIN = "builtin"  # Stata official command
    LOCAL_RELATIVE = "local_relative"  # Python relative import
    LOCAL_PROGRAM = "local_program"  # Stata program define
    AMBIGUOUS = "ambiguous"  # >1 candidate; see resolution_candidates
    DYNAMIC = "dynamic"  # unresolvable by construction
    UNKNOWN = "unknown"


class Ecosystem(StrEnum):
    """Registry a package was resolved against."""

    CRAN = "cran"
    CRAN_ARCHIVE = "cran_archive"
    BIOCONDUCTOR = "bioconductor"
    PYPI = "pypi"
    JULIA_GENERAL = "julia_general"
    SSC = "ssc"
    STATA_JOURNAL = "stata_journal"
    BASE_R = "base_r"
    PYTHON_STDLIB = "python_stdlib"
    STATA_BUILTIN = "stata_builtin"
    GITHUB = "github"


class ParseStatus(StrEnum):
    """Why a file did or did not yield mentions.

    No extractor may return an empty list without one of these. "Zero mentions"
    and "could not read the file" must never be the same value again.
    """

    OK = "ok"
    OK_WITH_ERRORS = "ok_with_errors"  # parsed, but ERROR nodes present
    FALLBACK = "fallback"  # e.g. Python tokenize path after SyntaxError
    DECODE_FALLBACK = "decode_fallback"  # encoding detection was low-confidence
    SYNTAX_ERROR = "syntax_error"
    DECODE_ERROR = "decode_error"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_BINARY = "skipped_binary"
    SKIPPED_VENDORED = "skipped_vendored"
    SKIPPED_DUPLICATE = "skipped_duplicate"
    NOT_ANALYZED = "not_analyzed"


class FileRole(StrEnum):
    """Role of a file within its deposit, used for the reachability analysis."""

    MASTER = "master"  # entry point: master.do, run_all.R, 00_*.R
    REACHABLE = "reachable"  # reached from a master via source()/do/import
    ORPHAN = "orphan"  # analyzable but not reachable from any master
    TEST = "test"
    EXAMPLE = "example"
    VENDORED = "vendored"
    GENERATED = "generated"
    UNKNOWN = "unknown"


class VendorRule(StrEnum):
    """Which rule marked a file as vendored. Recorded so exclusions are auditable."""

    V1_PATH = "v1_path"  # renv/, packrat/, site-packages/, .checkpoint/
    #: Stata's library layout: `ado/base`, `ado/plus`, or the bare `ado/` a
    #: replication package uses to ship its dependencies. Recorded apart from
    #: V1_PATH because it was added late and its exclusions are worth being
    #: able to count on their own.
    V1_STATA_ADO = "v1_stata_ado"
    V2_MARKER = "v2_marker"  # DESCRIPTION+NAMESPACE, pyproject+__init__
    V3_CROSS_DATASET = "v3_cross_dataset"  # same sha256 in >=5 deposits
    V4_NAME_SHAPE = "v4_name_shape"  # basename listed in an SSC .pkg
    V5_MANUAL = "v5_manual"  # outlier triage
    APPLEDOUBLE = "appledouble"  # __MACOSX/._* resource forks


class CollectionState(StrEnum):
    """Terminal state of a dataset's collection attempt.

    Written *after* work completes, never before. v1 wrote "success" at metadata
    time and so recorded success for 8,953 deposits it never downloaded.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_CANDIDATE_FILES = "no_candidate_files"
    RESTRICTED = "restricted"
    DEACCESSIONED = "deaccessioned"
    NOT_FOUND = "not_found"


class VersionState(StrEnum):
    """Dataverse dataset version state."""

    RELEASED = "released"
    DEACCESSIONED = "deaccessioned"
    DRAFT = "draft"


#: Statuses that mean the file contributed a real, trustworthy observation.
ANALYZABLE_STATUSES = frozenset(
    {ParseStatus.OK, ParseStatus.OK_WITH_ERRORS, ParseStatus.FALLBACK}
)

#: Collection states that must be retried on the next run rather than skipped.
RETRYABLE_STATES = frozenset({CollectionState.PARTIAL, CollectionState.FAILED})

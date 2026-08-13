"""Stata official commands, and why they are not load-bearing.

**Design note, and it matters for the error budget.**

The obvious architecture is "if a command is not a builtin, it must be a user
package". That makes the builtin list load-bearing: every omission becomes a
false positive, and every over-inclusion silently deletes a real package.

We do the opposite. Resolution is *inclusive*: a command counts as a
user-written package only if it appears in the SSC / Stata Journal index. A
command absent from that index is simply not attributed, whether it is a
builtin, a locally defined program, or something we have never seen. The builtin
list then serves one narrow purpose -- flagging the handful of commands claimed
by *both* an SSC package and official Stata -- and an error in it costs
precision on a small, enumerable set rather than corrupting the whole tally.

This inversion was not the original plan. It was forced by measurement: the
obvious machine-readable source for Stata's command list
(``kylebarron/language-stata``'s TextMate grammar) turned out to list
``reghdfe`` -- a user-written package, and the third most-used one in the corpus
-- as a builtin, while omitting ``regress``, ``summarize``, ``ttest`` and
``tabulate``. Building an exclusion filter on that would have deleted the single
most important Stata result in the paper. Probing the source before trusting it
is the only reason that was caught.

The list below is curated, and its limits are stated rather than hidden. The
authoritative list comes from a Stata installation::

    . local base : dir "`c(sysdir_base)'" files "*.ado"

If one is available, :func:`load_official` reads that export and takes
precedence. Until then the collision set is reported as a known limitation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from softverse.logging_setup import get_logger

logger = get_logger(__name__)

#: Curated core of official Stata commands, as ``(canonical, minimum_abbrev)``.
#: Stata lets any prefix at or beyond the minimum stand for the full command:
#: ``reg``, ``regr``, ``regre``, ``regres`` and ``regress`` are all ``regress``.
#: Scoped to commands plausibly appearing as the leading token of a statement in
#: replication code; it is not, and does not claim to be, exhaustive.
_OFFICIAL: tuple[tuple[str, str], ...] = (
    # data management
    ("use", "u"),
    ("save", "sa"),
    ("saveold", "saveold"),
    ("clear", "clear"),
    ("append", "app"),
    ("merge", "mer"),
    ("joinby", "joinby"),
    ("sort", "so"),
    ("gsort", "gsort"),
    ("generate", "g"),
    ("egen", "egen"),
    ("replace", "replace"),
    ("drop", "dr"),
    ("keep", "keep"),
    ("rename", "ren"),
    ("recode", "recode"),
    ("label", "la"),
    ("encode", "encode"),
    ("decode", "decode"),
    ("destring", "destring"),
    ("tostring", "tostring"),
    ("collapse", "collapse"),
    ("reshape", "reshape"),
    ("expand", "expand"),
    ("contract", "contract"),
    ("duplicates", "duplicates"),
    ("order", "order"),
    ("compress", "compress"),
    ("split", "split"),
    ("separate", "separate"),
    ("stack", "stack"),
    ("fillin", "fillin"),
    ("cross", "cross"),
    ("xpose", "xpose"),
    ("bcuse", "bcuse"),
    ("webuse", "webuse"),
    ("sysuse", "sysuse"),
    # Added after measuring the corpus rather than from memory: these official
    # commands sat at the top of the unresolved list (svmat 1,235 mentions,
    # levelsof 815, xtile 674, mkmat 442) purely because the curated core was
    # incomplete. Under the inclusive design they were never miscounted as
    # packages, so this raises the resolution rate rather than fixing a false
    # positive -- but an unresolved bucket full of ordinary Stata is noise that
    # hides the names actually worth looking at.
    ("levelsof", "levelsof"),
    ("xtile", "xtile"),
    ("clonevar", "clonevar"),
    ("matlist", "matlist"),
    ("svmat", "svmat"),
    ("mkmat", "mkmat"),
    ("tab1", "tab1"),
    ("tab2", "tab2"),
    ("tabi", "tabi"),
    ("unab", "unab"),
    ("c_local", "c_local"),
    ("_pctile", "_pctile"),
    ("cmclogit", "cmclogit"),
    ("post", "post"),
    ("postfile", "postfile"),
    ("postclose", "postclose"),
    ("pause", "pause"),
    ("ml", "ml"),
    ("nlsur", "nlsur"),
    ("tsfill", "tsfill"),
    ("tsappend", "tsappend"),
    ("recast", "recast"),
    ("mvencode", "mvencode"),
    ("mvdecode", "mvdecode"),
    ("cf", "cf"),
    ("compare", "compare"),
    ("lookfor", "lookfor"),
    ("ipolate", "ipolate"),
    ("range", "range"),
    ("eivreg", "eivreg"),
    ("vwls", "vwls"),
    ("fracreg", "fracreg"),
    ("etregress", "etregress"),
    ("pwmean", "pwmean"),
    ("pwcompare", "pwcompare"),
    ("total", "total"),
    ("ratio", "ratio"),
    ("twoway", "tw"),
    ("area", "area"),
    ("dotplot", "dotplot"),
    ("spikeplot", "spikeplot"),
    ("gr_edit", "gr_edit"),
    ("rm", "rm"),
    ("mkdir", "mkdir"),
    ("nulldata", "nulldata"),
    ("obs", "obs"),
    ("memory", "memory"),
    ("sleep", "sleep"),
    ("format", "form"),
    ("insheet", "insheet"),
    ("infile", "infile"),
    ("import", "import"),
    ("export", "export"),
    ("outsheet", "outsheet"),
    ("preserve", "preserve"),
    ("restore", "restore"),
    ("tsset", "tsset"),
    ("xtset", "xtset"),
    ("svyset", "svyset"),
    ("stset", "stset"),
    ("describe", "d"),
    ("codebook", "codebook"),
    ("inspect", "inspect"),
    ("assert", "assert"),
    ("count", "count"),
    ("isid", "isid"),
    # descriptives
    ("summarize", "su"),
    ("tabulate", "ta"),
    ("tabstat", "tabstat"),
    ("table", "table"),
    ("correlate", "corr"),
    ("pwcorr", "pwcorr"),
    ("list", "l"),
    ("display", "di"),
    ("mean", "mean"),
    ("proportion", "prop"),
    ("ci", "ci"),
    ("centile", "centile"),
    ("pctile", "pctile"),
    # estimation
    ("regress", "reg"),
    ("logit", "logit"),
    ("probit", "probit"),
    ("logistic", "logistic"),
    ("ologit", "ologit"),
    ("oprobit", "oprobit"),
    ("mlogit", "mlogit"),
    ("mprobit", "mprobit"),
    ("clogit", "clogit"),
    ("poisson", "poisson"),
    ("nbreg", "nbreg"),
    ("tobit", "tobit"),
    ("heckman", "heckman"),
    ("ivregress", "ivregress"),
    ("areg", "areg"),
    ("xtreg", "xtreg"),
    ("xtlogit", "xtlogit"),
    ("xtprobit", "xtprobit"),
    ("xtpoisson", "xtpoisson"),
    ("xtabond", "xtabond"),
    ("xtivreg", "xtivreg"),
    ("mixed", "mixed"),
    ("melogit", "melogit"),
    ("meqrlogit", "meqrlogit"),
    ("nl", "nl"),
    ("gmm", "gmm"),
    ("sem", "sem"),
    ("gsem", "gsem"),
    ("anova", "an"),
    ("manova", "manova"),
    ("oneway", "oneway"),
    ("newey", "newey"),
    ("prais", "prais"),
    ("arima", "arima"),
    ("var", "var"),
    ("vec", "vec"),
    ("arch", "arch"),
    ("stcox", "stcox"),
    ("streg", "streg"),
    ("teffects", "teffects"),
    ("didregress", "didregress"),
    ("xthdidregress", "xthdidregress"),
    ("eteffects", "eteffects"),
    ("ivprobit", "ivprobit"),
    ("ivtobit", "ivtobit"),
    ("boxcox", "boxcox"),
    ("quantile", "quantile"),
    ("qreg", "qreg"),
    ("rreg", "rreg"),
    ("cnsreg", "cnsreg"),
    ("sureg", "sureg"),
    ("reg3", "reg3"),
    ("biprobit", "biprobit"),
    ("truncreg", "truncreg"),
    ("intreg", "intreg"),
    ("zip", "zip"),
    ("zinb", "zinb"),
    ("betareg", "betareg"),
    # postestimation / inference
    ("predict", "predict"),
    ("margins", "margins"),
    ("marginsplot", "marginsplot"),
    ("test", "test"),
    ("testnl", "testnl"),
    ("lincom", "lincom"),
    ("nlcom", "nlcom"),
    ("contrast", "contrast"),
    ("estat", "estat"),
    ("estimates", "est"),
    ("estimates store", "est"),
    ("lrtest", "lrtest"),
    ("hausman", "hausman"),
    ("suest", "suest"),
    ("bootstrap", "bootstrap"),
    ("jackknife", "jackknife"),
    ("permute", "permute"),
    ("simulate", "simulate"),
    ("ttest", "ttest"),
    ("ranksum", "ranksum"),
    ("signrank", "signrank"),
    ("sdtest", "sdtest"),
    ("prtest", "prtest"),
    ("ksmirnov", "ksmirnov"),
    ("kwallis", "kwallis"),
    ("spearman", "spearman"),
    ("ttesti", "ttesti"),
    # graphics
    ("graph", "gr"),
    ("twoway", "twoway"),
    ("scatter", "sc"),
    ("line", "line"),
    ("histogram", "hist"),
    ("kdensity", "kdensity"),
    ("hist", "hist"),
    ("boxplot", "boxplot"),
    ("tsline", "tsline"),
    # programming / flow
    ("local", "loc"),
    ("global", "gl"),
    ("scalar", "sca"),
    ("matrix", "mat"),
    ("macro", "macro"),
    ("tempfile", "tempfile"),
    ("tempvar", "tempvar"),
    ("tempname", "tempname"),
    ("foreach", "foreach"),
    ("forvalues", "forv"),
    ("while", "while"),
    ("if", "if"),
    ("else", "else"),
    ("continue", "continue"),
    ("break", "break"),
    ("exit", "exit"),
    ("program", "pr"),
    ("syntax", "syntax"),
    ("args", "args"),
    ("return", "return"),
    ("ereturn", "ereturn"),
    ("sreturn", "sreturn"),
    ("capture", "cap"),
    ("quietly", "qui"),
    ("noisily", "noi"),
    ("set", "set"),
    ("version", "version"),
    ("do", "do"),
    ("run", "run"),
    ("include", "include"),
    ("which", "which"),
    ("help", "help"),
    ("log", "log"),
    ("cmdlog", "cmdlog"),
    ("cd", "cd"),
    ("mkdir", "mkdir"),
    ("dir", "dir"),
    ("erase", "erase"),
    ("copy", "copy"),
    ("type", "type"),
    ("shell", "shell"),
    ("winexec", "winexec"),
    ("sleep", "sleep"),
    ("timer", "timer"),
    ("mata", "mata"),
    ("end", "end"),
    ("discard", "discard"),
    ("clear all", "clear"),
    ("net", "net"),
    ("ssc", "ssc"),
    ("adopath", "adopath"),
    ("update", "update"),
    ("query", "query"),
    ("about", "about"),
    ("findit", "findit"),
    ("search", "search"),
    ("view", "view"),
    ("file", "file"),
    ("outfile", "outfile"),
    ("infix", "infix"),
    ("putexcel", "putexcel"),
    ("putdocx", "putdocx"),
    ("putpdf", "putpdf"),
    ("frame", "frame"),
    ("frames", "frames"),
    ("nulls", "nulls"),
    ("by", "by"),
    ("bysort", "bys"),
    ("xi", "xi"),
    ("svy", "svy"),
    ("char", "char"),
    ("notes", "notes"),
    ("numlist", "numlist"),
    ("confirm", "confirm"),
    ("marksample", "marksample"),
    ("markout", "markout"),
    ("gettoken", "gettoken"),
    ("tokenize", "tokenize"),
    ("nobreak", "nobreak"),
    ("plugin", "plugin"),
    ("python", "python"),
    ("javacall", "javacall"),
)


def _expand_abbreviations(canonical: str, minimum: str) -> set[str]:
    """Every legal abbreviation of ``canonical``, given its minimum form.

    ``regress`` with minimum ``reg`` yields reg, regr, regre, regres, regress.
    Multi-word entries (``estimates store``) contribute their first word only,
    since the lexer sees one leading token.
    """
    canonical = canonical.split()[0]
    minimum = minimum.split()[0]
    if not canonical.startswith(minimum):
        return {canonical, minimum}
    return {canonical[:i] for i in range(len(minimum), len(canonical) + 1)}


@dataclass(frozen=True)
class BuiltinSet:
    """Official Stata commands, including every legal abbreviation."""

    forms: frozenset[str]
    canonical: frozenset[str]
    source: str

    def __contains__(self, command: str) -> bool:
        return command.lower() in self.forms

    def collisions(self, package_commands: set[str]) -> set[str]:
        """Commands claimed by both official Stata and a user package.

        These are the only cases where the builtin list changes a number. They
        resolve to ``builtin``, which biases toward *under*-counting user
        packages -- the safer direction, and one the paper states explicitly.
        """
        return {c for c in package_commands if c.lower() in self.forms}


def curated() -> BuiltinSet:
    """The curated core list. Not exhaustive; see the module docstring."""
    forms: set[str] = set()
    canonical: set[str] = set()
    for full, minimum in _OFFICIAL:
        canonical.add(full.split()[0])
        forms |= _expand_abbreviations(full, minimum)
    return BuiltinSet(frozenset(forms), frozenset(canonical), source="curated")


def load_official(path: Path) -> BuiltinSet:
    """Load an authoritative list exported from a Stata installation.

    Expects JSON: ``{"stata_version": "18.0", "commands": ["regress", ...]}``,
    produced by listing ``*.ado`` under ``c(sysdir_base)``. Takes precedence
    over :func:`curated` when present, because it is a fact about Stata rather
    than a judgement about it.
    """
    payload = json.loads(Path(path).read_text())
    commands = {c.lower() for c in payload["commands"]}
    version = payload.get("stata_version", "unknown")
    # Without documented minimum abbreviations, exact names only. Abbreviated
    # builtins then fall through to "not a package", which is still correct --
    # they are absent from the SSC index.
    logger.info(
        "loaded official Stata command list",
        extra={"n_commands": len(commands), "stata_version": version},
    )
    return BuiltinSet(
        frozenset(commands), frozenset(commands), source=f"stata-{version}"
    )


def load_verified(path: Path) -> BuiltinSet:
    """Load names checked one by one against StataCorp's help server.

    Written by :mod:`softverse.stata.official`. This is the same question
    ``load_official`` answers from a Stata installation, asked of a public
    server instead, so it needs no licence -- and unlike the curated list it is
    a measurement rather than a recollection.

    Exact names only: the help server confirms that ``summarize`` exists but
    does not say that ``su`` is a legal abbreviation of it, and inventing
    abbreviations would be exactly the guessing this replaces. Abbreviations
    keep coming from the curated list, which documents its minimums.
    """
    payload = json.loads(Path(path).read_text())
    commands = {name for name, ok in payload["commands"].items() if ok}
    logger.info(
        "loaded verified Stata command list",
        extra={
            "n_official": len(commands),
            "n_checked": len(payload["commands"]),
            "snapshot": payload.get("snapshot_date"),
        },
    )
    return BuiltinSet(
        frozenset(commands),
        frozenset(commands),
        source=f"stata.com-help-{payload.get('snapshot_date', 'unknown')}",
    )


def merge(*sets: BuiltinSet) -> BuiltinSet:
    """Union several builtin sets, keeping every source in the provenance.

    Union rather than precedence because the sources answer the same question
    with different coverage: the curated list carries abbreviations the help
    server cannot confirm, and the help server carries hundreds of commands
    nobody thought to curate. Taking one over the other discards real
    information either way.
    """
    return BuiltinSet(
        frozenset().union(*(s.forms for s in sets)),
        frozenset().union(*(s.canonical for s in sets)),
        source="+".join(s.source for s in sets),
    )


def builtins(
    official_export: Path | None = None,
    verified_snapshot: Path | None = None,
) -> BuiltinSet:
    """The builtin set to use, widest available.

    A Stata installation's own listing is authoritative and wins outright.
    Otherwise the curated list is widened by whatever the help server
    confirmed, which on this corpus adds 165 commands that were previously
    reported as unresolved -- `ds`, `pca`, `bsample`, `testparm` and the like,
    official commands that simply were not in anyone's curated list.
    """
    if official_export is not None and Path(official_export).exists():
        return load_official(official_export)
    if verified_snapshot is not None and Path(verified_snapshot).exists():
        return merge(curated(), load_verified(verified_snapshot))
    return curated()

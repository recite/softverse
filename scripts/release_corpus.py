"""Build the corpus release for Hugging Face, and refuse it if it does not check.

    uv run python scripts/release_corpus.py

Reads the tally in `build/tally/`, the frame in `data/frame/` and the corpus
under `corpus/`, writes `build/release/corpus/`, then re-reads what it wrote
and recomputes its claims (`softverse.release.corpus.check`). A release that
fails is moved aside to `build/release/corpus.rejected/` rather than left where
an upload would find it.
"""

from __future__ import annotations

import shutil

from softverse.config import PATHS
from softverse.logging_setup import setup_logging
from softverse.release.corpus import Inputs, build, check

OUT = PATHS.root / "build" / "release" / "corpus"
REJECTED = OUT.with_name("corpus.rejected")


def main() -> int:
    setup_logging("INFO", log_dir=PATHS.logs, stage="release-corpus")
    inputs = Inputs(
        tally=PATHS.root / "build" / "tally",
        frame=PATHS.frame,
        dataverse_corpus=PATHS.root / "corpus" / "dataverse",
        zenodo_corpus=PATHS.root / "corpus" / "zenodo",
    )
    counts = build(inputs, OUT)
    for name, n in counts.items():
        print(f"  {name:<22}{n:>12,}")
    problems = check(inputs, OUT)
    if problems:
        shutil.rmtree(REJECTED, ignore_errors=True)
        OUT.rename(REJECTED)
        print("\nRELEASE REJECTED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nall checks passed; release in {OUT.relative_to(PATHS.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

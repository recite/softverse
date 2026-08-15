"""Expose the corpus in the layout sibling projects expect.

    uv run python scripts_corpus_view.py

Writes `outputs/scripts/<source>/<deposit>/<script>` as symlinks to the files
under `corpus/`, and copies the Dataverse journal metadata beside them.

`kasauti` reads a corpus of replication scripts and re-derives call sites from
it. Its loader expects `outputs/scripts/<source>/<id>/<script>`, which is what
softverse v1 produced. v2 moved the corpus to `corpus/<source>/files/` and
renamed the Dataverse source, so `kasauti frame extract --corpus <softverse>`
now finds nothing at all. This is the adapter, kept here rather than in
kasauti so the project that changed the layout is the one that absorbs it.

Symlinks, not copies: the Zenodo half alone is 20 GB, and the point is a view
rather than a second corpus that can drift from the first.

Per *file*, not per directory, which is the whole reason this script has a
self-check at the end. A symlinked deposit directory is invisible to
`pathlib.Path.rglob`, which declines to descend into it, while `os.walk`
with `followlinks=True` and `glob.glob(recursive=True)` both walk straight
through. Linking directories therefore produces a view that looks right,
costs nothing, and is empty to roughly half the ways a consumer might read
it. Linking the files themselves works for all of them.
"""

from __future__ import annotations

import shutil

from softverse.config import PATHS
from softverse.corpus.loaders import full_corpus

OUT = PATHS.root / "outputs" / "scripts"
METADATA = PATHS.root / "corpus" / "dataverse_legacy" / "metadata"

#: What each source is called in the view. `dataverse_legacy` records which
#: collection run the files came from, which matters inside softverse and is
#: noise to a consumer that only wants to know the deposit is from Dataverse.
VIEW_NAME = {"zenodo": "zenodo", "dataverse_legacy": "dataverse"}


def deposit_id(doi: str) -> str:
    """The identifier a consumer keys on, from the DOI.

    `zenodo:10012820` gives `10012820` and `doi:10.7910/DVN/03MTMJ` gives
    `03MTMJ`, which are the directory names the corpus already uses and the
    values kasauti's own path tests assert on.
    """
    return doi.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def source_of(doi: str) -> str:
    return "dataverse_legacy" if doi.startswith("doi:") else "zenodo"


def build() -> tuple[int, int]:
    """Link every corpus file into the view. Returns (files, deposits)."""
    if OUT.exists():
        shutil.rmtree(OUT)

    deposits: set[str] = set()
    linked = 0
    for item in full_corpus():
        target = (
            OUT
            / VIEW_NAME[source_of(item.dataset_doi)]
            / deposit_id(item.dataset_doi)
            / item.relative_path
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(item.path)
            linked += 1
        deposits.add(item.dataset_doi)
    return linked, len(deposits)


def copy_metadata() -> int:
    """The per-journal dataset listings, which date a deposit's paper.

    Copied rather than linked because they are 75 small CSVs and a consumer
    that walks them should not have to care that they came from elsewhere.
    """
    if not METADATA.is_dir():
        return 0
    destination = PATHS.root / "outputs" / "metadata"
    destination.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(METADATA.glob("*_datasets.csv")):
        if ".ipynb_checkpoints" in str(path):
            continue
        shutil.copyfile(path, destination / path.name)
        n += 1
    return n


def main() -> int:
    linked, deposits = build()
    if not linked:
        print("nothing to link; is the corpus present?")
        return 1
    n_metadata = copy_metadata()

    by_source = {
        name: len(list((OUT / name).iterdir()))
        for name in sorted(VIEW_NAME.values())
        if (OUT / name).is_dir()
    }
    print(
        f"linked {linked:,} files from {deposits:,} deposits "
        f"into {OUT.relative_to(PATHS.root)}"
    )
    for name, n in by_source.items():
        print(f"  {name:<12}{n:>7,} deposits")
    print(f"copied {n_metadata} journal metadata files")

    # Prove the view is readable rather than merely present: a consumer globs
    # it, so glob it. A symlinked directory that a recursive walk declines to
    # follow would otherwise look fine here and be empty to the reader.
    scripts = sum(1 for p in OUT.rglob("*.R") if p.is_file())
    scripts += sum(1 for p in OUT.rglob("*.r") if p.is_file())
    scripts += sum(1 for p in OUT.rglob("*.py") if p.is_file())
    print(f"visible through the view: {scripts:,} R and Python files")
    return 0 if scripts else 1


if __name__ == "__main__":
    raise SystemExit(main())

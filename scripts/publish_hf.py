"""Publish the corpus release to a Hugging Face dataset repository.

    uv run python scripts/publish_hf.py --repo ORG/NAME --dry-run
    uv run python scripts/publish_hf.py --repo ORG/NAME

Uploads `build/release/corpus/` as it stands -- the tables, the `contents`
shards and the dataset card -- with `HfApi.upload_folder`, which the
huggingface_hub documentation now recommends for large folders: it commits in
batches and, re-run after an interruption, skips what is already on the Hub.

Refuses to upload a release its own checks did not pass: `release_corpus.py`
moves a failing release to `corpus.rejected/`, so a missing `corpus/` or a
missing dataset card is a reason to stop, not to upload what is there.

After the upload it reads every table back from the Hub and compares row
counts with the local files, so "uploaded" means readable, not sent.

The token comes from `HF_TOKEN`, in the environment or `.env`.
"""

from __future__ import annotations

import sys

import pyarrow.parquet as pq
from huggingface_hub import HfApi

from softverse.config import PATHS, credential

RELEASE = PATHS.root / "build" / "release" / "corpus"


def local_counts() -> dict[str, int]:
    """Rows per table in the local release.

    Returns:
        Table name -> rows, with the `contents` shards summed.
    """
    counts = {
        path.stem: pq.ParquetFile(path).metadata.num_rows
        for path in RELEASE.glob("*.parquet")
    }
    counts["contents"] = sum(
        pq.ParquetFile(path).metadata.num_rows
        for path in (RELEASE / "contents").glob("*.parquet")
    )
    return counts


def main() -> int:
    if "--repo" not in sys.argv:
        print("pass --repo ORG/NAME")
        return 1
    repo = sys.argv[sys.argv.index("--repo") + 1]
    if not (RELEASE / "README.md").exists():
        print(f"no checked release at {RELEASE}; run scripts/release_corpus.py")
        return 1

    counts = local_counts()
    size = sum(p.stat().st_size for p in RELEASE.rglob("*") if p.is_file())
    print(f"{repo}: {len(counts)} tables, {size / 1e9:.2f} GB")
    for name, n in sorted(counts.items()):
        print(f"  {name:<22}{n:>12,}")
    if "--dry-run" in sys.argv:
        return 0

    token = credential("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set")
        return 1
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=RELEASE,
        commit_message="softverse 2026 corpus release",
    )

    from datasets import load_dataset

    problems = []
    for name, expected in counts.items():
        got = load_dataset(repo, name, split="train", token=token).num_rows
        if got != expected:
            problems.append(f"{name}: {got:,} on the Hub, {expected:,} locally")
    if problems:
        print("READ-BACK FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"read back every table from https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download one daily snapshot of the edwarddgao/open-apply-jobs dataset.

The dataset is Hive-partitioned as data/date=YYYY-MM-DD/source={ashby,greenhouse,lever}/*.parquet.
Every date folder is a FULL snapshot of active postings that day (not a delta), so for
analysis we only ever need one date.

Usage:
    python scripts/download_snapshot.py            # latest date
    python scripts/download_snapshot.py 2026-09-17 # specific date
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "edwarddgao/open-apply-jobs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def latest_date() -> str:
    api = HfApi()
    entries = api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data")
    dates = sorted(e.path.split("date=")[1] for e in entries if "date=" in e.path)
    return dates[-1]


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else latest_date()
    print(f"Downloading snapshot date={date} -> {RAW_DIR}")
    snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=[f"data/date={date}/*/*.parquet"],
        local_dir=RAW_DIR,
    )
    print("done")


if __name__ == "__main__":
    main()

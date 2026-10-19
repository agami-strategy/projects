"""Download one daily snapshot of the edwarddgao/open-apply-jobs dataset.

The dataset is Hive-partitioned as data/date=YYYY-MM-DD/source={ashby,greenhouse,lever}/*.parquet.
Every date folder is a FULL snapshot of active postings that day (not a delta), so for
analysis we only ever need one date.

Usage:
    python scripts/download_snapshot.py            # latest date
    python scripts/download_snapshot.py 2026-09-17 # specific date
"""
# %% Imports
# Only the standard library (argv, paths) and huggingface_hub. HfApi lists the dataset's folder
# tree; snapshot_download fetches files matching a glob. No DuckDB here: this script only moves
# parquet files onto disk, the reading happens in sample_jobs.py and the collectors.
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# %% Constants
# REPO is the Hugging Face dataset id (daily snapshots scraped from Greenhouse / Lever / Ashby ATS
# APIs, roughly 263k ads per day). RAW_DIR is <project>/data/raw. snapshot_download mirrors the repo
# layout underneath it, so files land at data/raw/data/date=YYYY-MM-DD/source=<ats>/<file>.parquet.
# sample_jobs.py globs that exact path, so the two must be changed together if this moves.
REPO = "edwarddgao/open-apply-jobs"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# %% Find the newest snapshot date on the hub
# Lists the top-level "data" folder of the dataset (one entry per date=YYYY-MM-DD partition) and
# returns the largest date. ISO dates sort correctly as plain strings, so no date parsing is needed.
# Needs network access. Raises IndexError if the listing contains no date folders.
def latest_date() -> str:
    api = HfApi()
    entries = api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data")
    # e.path looks like "data/date=2026-09-17": split on "date=" and keep the right-hand part.
    # Entries without "date=" (README, .gitattributes) are skipped by the filter.
    dates = sorted(e.path.split("date=")[1] for e in entries if "date=" in e.path)
    return dates[-1]


# %% Download one full day
# Optional argv[1] is the date; the default is the newest date on the hub. allow_patterns limits the
# download to that day's parquet files across all sources (the single "*" is the source=<ats> folder).
# One day is about 0.23 GB. Because every date is a complete snapshot, one day is enough for a
# cross-sectional sample; the multi-day scripts exist only because the Australian subset is small.
def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else latest_date()
    print(f"Downloading snapshot date={date} -> {RAW_DIR}")
    # snapshot_download skips files that are already present and unchanged, so re-running is cheap.
    # Reviewer: if a day on the hub carries duplicated parquet files, both copies are downloaded and
    # later read twice by the glob in sample_jobs.py (that script does not dedupe by id).
    snapshot_download(
        REPO,
        repo_type="dataset",
        allow_patterns=[f"data/date={date}/*/*.parquet"],
        local_dir=RAW_DIR,
    )
    print("done")


if __name__ == "__main__":
    main()

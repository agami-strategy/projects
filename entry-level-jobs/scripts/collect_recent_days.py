"""Collect Australian ads with full text from the N most recent daily snapshots.

Downloads each day's parquet files (newest first), extracts the Australian ads, drops ids
already collected, and appends to one JSONL. Parquet files are deleted after use unless
they were already present locally or --keep is given. Resumable.

Usage:
    python scripts/collect_recent_days.py --days 10
    python scripts/collect_recent_days.py --days 10 --tag aus_recent10

Output: data/samples/sample_<tag>.jsonl, same shape as sample_jobs.py output plus snapshot_date.
"""
# %% Imports
# shutil.rmtree deletes a day's parquet folder after use; snapshot_download fetches it; duckdb reads
# it; HfApi lists the dates on the hub. json writes the JSONL output line by line.
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, snapshot_download

# %% Shared helpers from sibling scripts
# html_to_text (sample_jobs.py) is the same converter used for every sample, and ads_condition
# (scan_australian_ads.py) is the same location filter + location cap used by the scan, so the ads
# collected here are exactly the population the scan counted. REPO is the dataset id.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import html_to_text  # noqa: E402
from scan_australian_ads import REPO, ads_condition  # noqa: E402

# %% Paths
# RAW_DIR is where snapshot_download mirrors the repo (parquet ends up under RAW_DIR/data/date=...).
# SAMPLE_DIR holds the JSONL output and the small .dates_done resume file next to it.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "samples"


# %% Main: CLI, resume state, per-day download / filter / append
def main() -> None:
    # --- CLI ---
    # --days: how many of the most recent snapshots to use (default 10). --dates: explicit list that
    # replaces --days. --country: preset or comma-separated terms (default australia; "us" was used
    # for the one-day US comparison via sample_jobs.py, not here). --tag: output suffix.
    # --keep: leave downloaded parquet files on disk.
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--country", default="australia", help="preset (australia, us) or comma-separated location terms")
    ap.add_argument("--dates", nargs="*", default=None, help="explicit snapshot dates instead of the N most recent")
    ap.add_argument("--tag", default=None, help="default <country>_recent<days>")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    # --- Output naming ---
    # "australia" is shortened to "aus"; any other country keeps its name with commas turned into "-".
    # Tag is "<short>_<dates joined by _>" when --dates is given, else "<short>_recent<days>".
    # done_file is a plain-text list of dates already processed, one per line, used to resume.
    short = {"australia": "aus"}.get(args.country.lower(), args.country.lower().replace(",", "-"))
    tag = args.tag or (f"{short}_{'_'.join(args.dates)}" if args.dates else f"{short}_recent{args.days}")
    out = SAMPLE_DIR / f"sample_{tag}.jsonl"
    done_file = SAMPLE_DIR / f"sample_{tag}.dates_done"
    cond = ads_condition(args.country)

    # --- Which dates ---
    # Newest first, so an interrupted run has already captured the most recent days, and an ad live
    # on several of the chosen days is written from the newest one (its most recent version).
    # With --dates the given list is used as-is, also sorted newest first.
    api = HfApi()
    if args.dates:
        dates = sorted(args.dates, reverse=True)
    else:
        dates = sorted((e.path.split("date=")[1] for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data") if "date=" in e.path), reverse=True)[: args.days]
    # --- Resume state ---
    # `seen` = ids already in the output JSONL (rebuilt by reading the whole file). `done` = dates
    # already processed. Both are needed: seen dedupes across days, done avoids re-downloading.
    # If the process dies between writing the ads and appending to done_file, the day is processed
    # again next run but every id is already in `seen`, so nothing is duplicated.
    seen: set[str] = set()
    if out.exists():
        seen = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    print(f"dates: {dates[-1]} .. {dates[0]} | already collected {len(seen):,} ads from {len(done)} dates", flush=True)

    # --- Per-day loop: download, filter, append, clean up ---
    # `fresh` records whether the day's folder was absent before this run; only freshly downloaded
    # days are deleted afterwards (and only without --keep), so a day fetched by download_snapshot.py
    # for other work is never removed. Each day is about 0.23 GB.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for d in dates:
            if d in done:
                continue
            t0 = time.time()
            local = RAW_DIR / "data" / f"date={d}"
            fresh = not local.exists()
            snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"data/date={d}/*/*.parquet"], local_dir=RAW_DIR)
            # Forward slashes: DuckDB does not accept Windows backslashes in a glob.
            glob = str(local / "*" / "*.parquet").replace("\\", "/")
            # Same SELECT as sample_jobs.py, without sampling. union_by_name=true handles the
            # 2026-04-17 salary_period schema drift (VARCHAR there, NULL-typed elsewhere).
            # Days with duplicated parquet files return the same ad twice; the `seen` check below
            # makes the second copy a no-op. A fresh connection per day keeps memory bounded.
            con = duckdb.connect()
            rows = con.execute(f"""
                SELECT id, source, source_slug, title, apply_url, description_html, employment_type, department,
                       locations, remote, posted_at, salary_min, salary_max, salary_currency, salary_period
                FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
                WHERE {cond}
            """).fetchall()
            # Column names in SELECT order. The loop variable is d_ so it does not shadow the date d.
            cols = [d_[0] for d_ in con.description]
            new = 0
            for r in rows:
                rec = dict(zip(cols, r))
                if rec["id"] in seen:
                    continue
                # description_html is popped (not kept) and replaced by the plain-text version to keep
                # the JSONL small; snapshot_date records which day's copy of the ad this is.
                rec["description_text"] = html_to_text(rec.pop("description_html"))
                rec["snapshot_date"] = d
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                seen.add(rec["id"])
                new += 1
            # Flush the ads before recording the date as done (see resume note above).
            f.flush()
            with open(done_file, "a") as df:
                df.write(d + "\n")
            if fresh and not args.keep:
                shutil.rmtree(local, ignore_errors=True)
            # len(rows) counts every matching row on the day (including duplicates); `new` is the
            # number actually appended. Their ratio shows how much day-to-day overlap there is.
            print(f"{d}: {len(rows):,} {args.country} ads, {new:,} new -> total {len(seen):,} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()

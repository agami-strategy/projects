"""Scan every daily snapshot on Hugging Face for Australian ads, reading only small columns.

Writes data/aus_ads_all_dates.csv with one row per (date, ad) so turnover and unique counts can
be computed. Uses the same location filter as sample_jobs.py's 'australia' preset. Reads remotely
via DuckDB httpfs with column projection, so the large description column is never downloaded.
Resumable: dates already in the output file are skipped.

Usage:
    python scripts/scan_australian_ads.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import COUNTRY_EXCLUDE_REGEX, COUNTRY_PRESETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "aus_ads_all_dates.csv"
REPO = "edwarddgao/open-apply-jobs"
HEADER = ["date", "id", "source", "source_slug", "title", "employment_type", "department", "locations", "remote", "posted_at"]


def australia_condition() -> str:
    loc = "lower(array_to_string(locations, ' | '))"
    terms = " OR ".join(f"{loc} LIKE '%{t}%'" for t in COUNTRY_PRESETS["australia"])
    return f"({terms}) AND NOT regexp_matches({loc}, '{COUNTRY_EXCLUDE_REGEX['australia']}') AND len(locations) <= 3"


def main() -> None:
    api = HfApi()
    dates = sorted(e.path.split("date=")[1] for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data") if "date=" in e.path)
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {r["date"] for r in csv.DictReader(f)}
    todo = [d for d in dates if d not in done]
    print(f"{len(dates)} dates on the hub, {len(done)} already scanned, {len(todo)} to do", flush=True)

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    cond = australia_condition()
    new_file = not OUT.exists()
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        for d in todo:
            t0 = time.time()
            for attempt in range(3):
                try:
                    rows = con.execute(f"""
                        SELECT '{d}', id, source, source_slug, title, employment_type, department,
                               array_to_string(locations, ' | '), remote, posted_at
                        FROM read_parquet('hf://datasets/{REPO}/data/date={d}/*/*.parquet', hive_partitioning=1, union_by_name=true)
                        WHERE {cond}
                    """).fetchall()
                    break
                except Exception as e:  # transient network errors
                    if attempt == 2:
                        raise
                    print(f"  {d}: retry after error {type(e).__name__}", flush=True)
                    time.sleep(10)
            w.writerows(rows)
            f.flush()
            print(f"{d}: {len(rows)} australian ads ({time.time() - t0:.0f}s)", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()

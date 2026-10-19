"""Scan every daily snapshot on Hugging Face for Australian ads, reading only small columns.

Writes data/aus_ads_all_dates.csv with one row per (date, ad) so turnover and unique counts can
be computed. Uses the same location filter as sample_jobs.py's 'australia' preset. Reads remotely
via DuckDB httpfs with column projection, so the large description column is never downloaded.
Resumable: dates already in the output file are skipped.

Usage:
    python scripts/scan_australian_ads.py
"""
# %% Imports
# csv for the append-only output; time for the retry pause and per-date timing; duckdb with the
# httpfs extension reads parquet straight from the hub over HTTP; HfApi lists the available dates.
# `from __future__ import annotations` only affects type hints (allows dict[str, ...] on older Pythons).
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi

# %% Shared location filter
# The scripts folder is put on sys.path so country_condition can be imported from sample_jobs.py.
# This keeps the Australia definition (country name or capital city, minus the US/UK name clashes) in
# one place; the noqa silences the "import not at top of file" lint that the sys.path line causes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import country_condition  # noqa: E402

# %% Paths and output schema
# OUT is one row per (date, ad) with only small columns. description_html is deliberately absent so
# the remote read stays cheap: parquet is columnar, so DuckDB fetches only the byte ranges of the
# projected columns. HEADER order must match the SELECT list in main(); it is written once, when the
# file is first created. collate_australian_ads.py imports OUT under the name SCAN_CSV.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "aus_ads_all_dates.csv"
REPO = "edwarddgao/open-apply-jobs"
HEADER = ["date", "id", "source", "source_slug", "title", "employment_type", "department", "locations", "remote", "posted_at"]


# %% Filter definitions
# ads_condition = country preset AND at most max_locations listed locations. The cap (default 3)
# drops "Sydney | London | New York | Remote" style multi-country remote ads that would otherwise
# pass the Australia filter because one of their locations is Australian. Ads whose only location is
# "Remote" match no preset term and are excluded as well. collect_recent_days.py and
# collate_australian_ads.py import these, so all multi-day collections use one definition.
def ads_condition(country: str = "australia", max_locations: int = 3) -> str:
    """Country preset plus a cap on the number of listed locations (drops multi-country remote ads)."""
    return f"{country_condition(country)} AND len(locations) <= {max_locations}"


# Convenience wrapper: the fixed Australian population used by the scan and the collation.
def australia_condition() -> str:
    return ads_condition("australia")


# %% Main: list dates, resume, scan each date remotely
def main() -> None:
    # --- Which dates to scan ---
    # Same folder listing as download_snapshot.latest_date(); "data/date=YYYY-MM-DD" -> "YYYY-MM-DD".
    # `done` is rebuilt from the output CSV so an interrupted run resumes. A date counts as done as
    # soon as one row for it exists; a date with zero Australian ads writes no rows and would be
    # re-scanned on every run (harmless, just slow). 127 dates existed when the study was run.
    api = HfApi()
    dates = sorted(e.path.split("date=")[1] for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data") if "date=" in e.path)
    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            done = {r["date"] for r in csv.DictReader(f)}
    todo = [d for d in dates if d not in done]
    print(f"{len(dates)} dates on the hub, {len(done)} already scanned, {len(todo)} to do", flush=True)

    # --- Remote scan, one date at a time ---
    # httpfs lets read_parquet open hf:// URLs directly. For each date DuckDB reads only the projected
    # columns of that day's ~263k ads and applies the filter locally. The output is opened in append
    # mode and flushed after every date so progress survives a crash or Ctrl-C.
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
            # Three attempts with a 10 s pause cover transient HTTP/timeout errors from the hub. The
            # third failure is re-raised so the run stops rather than silently skipping a day.
            for attempt in range(3):
                try:
                    # '{d}' is a string literal in the SQL: the date column comes from the loop, not
                    # from the parquet. locations (a list) is flattened with ' | ' for the CSV.
                    # union_by_name=true is needed because 2026-04-17 stores salary_period as VARCHAR
                    # (the column is not selected here, but the schema merge still has to succeed).
                    # Days with duplicated parquet files yield duplicate (date, id) rows; downstream
                    # code builds sets from this file, so the duplicates are harmless.
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

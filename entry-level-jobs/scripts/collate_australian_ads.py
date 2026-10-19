"""Collate every unique Australian ad since April 2026 with full text.

Step 1 (scan_australian_ads.py) records which ad ids are live on which dates, without text.
This script uses that to choose the fewest snapshot dates that together contain every unique
ad (greedy set cover: an ad live for three weeks appears on 21 dates, so one download of a
date inside that window captures it). It then downloads only those dates' parquet files,
extracts the Australian ads not yet collected, converts the HTML description to text, and
appends them to the output. Downloaded parquet files are deleted after use unless --keep.

Resumable: ads already in the output JSONL are skipped, and a date is skipped once every ad
it would contribute has been collected.

Usage:
    python scripts/collate_australian_ads.py              # plan + download + extract
    python scripts/collate_australian_ads.py --plan-only  # just show which dates would be fetched
    python scripts/collate_australian_ads.py --keep       # keep downloaded parquet files in data/raw
    python scripts/collate_australian_ads.py --exclude data/samples/sample_aus_recent10.jsonl --min-gain 20
        # skip ads already in the recent-10-day file; stop once a day adds fewer than 20 ads

Output: data/samples/sample_aus_all.jsonl (same shape as sample_jobs.py output, plus first_seen /
last_seen / days_live), usable directly by label_osca.py and the rest of the pipeline.
"""
# %% Imports
# csv reads the scan table; defaultdict builds the date -> ids index; shutil removes downloaded
# parquet folders; snapshot_download fetches them; duckdb filters them. No HfApi here: the list of
# dates comes from the scan CSV, not from the hub.
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
from huggingface_hub import snapshot_download

# %% Shared helpers from sibling scripts
# html_to_text is the common HTML -> text converter. From scan_australian_ads.py: OUT (renamed
# SCAN_CSV here) is the per-(date, id) table produced by the scan, REPO the dataset id, and
# australia_condition the identical filter the scan used, so the ids found on disk match the plan.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import html_to_text  # noqa: E402
from scan_australian_ads import OUT as SCAN_CSV, REPO, australia_condition  # noqa: E402

# %% Paths
# RAW_DIR is the snapshot_download mirror root; OUT is the collated JSONL that the labelling
# scripts consume (one line per unique Australian ad).
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "samples" / "sample_aus_all.jsonl"


# %% Load the scan into two lookups
# One pass over the scan CSV builds: by_date = {date: set of ids live that day} and span = {id:
# (first_seen, last_seen)}. Dates are ISO strings, so min()/max() on them is chronological.
# Duplicate (date, id) rows from days with duplicated parquet files collapse inside the sets.
# by_date keys are in CSV order (chronological for an uninterrupted scan), which matters for
# tie-breaking in plan_dates(). Memory: a few hundred thousand short strings, fine.
def load_scan() -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    """Return ({date: {ids live that day}}, {id: (first_seen, last_seen)})."""
    by_date: dict[str, set[str]] = defaultdict(set)
    span: dict[str, list[str]] = {}
    with open(SCAN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_date[r["date"]].add(r["id"])
            # setdefault returns the existing [first, last] list on later rows, then min/max widen it.
            s = span.setdefault(r["id"], [r["date"], r["date"]])
            s[0], s[1] = min(s[0], r["date"]), max(s[1], r["date"])
    return by_date, {k: (v[0], v[1]) for k, v in span.items()}


# %% Greedy set cover over dates
# Input: by_date and the set of ids still needed. Output: ordered list of (date, gain) where gain is
# the number of previously uncovered ids that date adds. Each round picks the date with the largest
# overlap with what is still uncovered, then removes those ids. Greedy is not guaranteed optimal, but
# ads are live on contiguous runs of days, so a handful of well-spaced dates covers nearly all of
# the 127 scanned. Ties go to the first date in dict order. Loop ends when nothing is left or when
# no date adds anything (ids the scan saw but no date in by_date carries, which cannot happen unless
# the CSV was edited). Cost is O(dates) set intersections per chosen date.
def plan_dates(by_date: dict[str, set[str]], needed: set[str]) -> list[tuple[str, int]]:
    """Greedy set cover: repeatedly take the date that adds the most uncovered ids."""
    remaining, chosen = set(needed), []
    while remaining:
        best = max(by_date, key=lambda d: len(by_date[d] & remaining))
        gain = len(by_date[best] & remaining)
        if gain == 0:
            break
        chosen.append((best, gain))
        remaining -= by_date[best]
    return chosen


# %% Download one date and extract the wanted ads
# Same download / filter / convert steps as collect_recent_days.py, restricted to ids in `wanted`.
# `fresh` guards deletion: a day that already existed locally is never removed. Returns a list of
# dicts (all selected columns, description_html replaced by description_text, plus snapshot_date).
# Ids that match the Australia filter but are not in `wanted` (already collected, or assigned to
# another date by the plan) are dropped here. Duplicated parquet files on some days can make the
# same wanted id appear twice in the returned list; main() dedupes on write.
def extract_date(date: str, wanted: set[str], keep: bool) -> list[dict]:
    local = RAW_DIR / "data" / f"date={date}"
    fresh = not local.exists()
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"data/date={date}/*/*.parquet"], local_dir=RAW_DIR)
    # Forward slashes for DuckDB's glob on Windows.
    glob = str(local / "*" / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    # union_by_name=true: 2026-04-17 stores salary_period as VARCHAR, other days as a NULL-typed
    # column. The WHERE uses the exact scan filter so the ids line up with the plan.
    rows = con.execute(f"""
        SELECT id, source, source_slug, title, apply_url, description_html, employment_type, department,
               locations, remote, posted_at, salary_min, salary_max, salary_currency, salary_period
        FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
        WHERE {australia_condition()}
    """).fetchall()
    cols = [d[0] for d in con.description]
    out = []
    for r in rows:
        rec = dict(zip(cols, r))
        if rec["id"] in wanted:
            rec["description_text"] = html_to_text(rec.pop("description_html"))
            rec["snapshot_date"] = date
            out.append(rec)
    if fresh and not keep:
        shutil.rmtree(local, ignore_errors=True)
    return out


# %% Main: resume state, plan, download + extract
def main() -> None:
    # --- CLI ---
    # --plan-only prints the chosen dates and exits. --exclude takes JSONL files whose ads should
    # count as already collected (used with sample_aus_recent10.jsonl so the two files together cover
    # everything without overlap). --min-gain drops plan entries that add fewer than N ads.
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep downloaded parquet files")
    ap.add_argument("--exclude", nargs="*", default=[], help="JSONL files whose ads are already collected elsewhere")
    ap.add_argument("--min-gain", type=int, default=1, help="stop once a date would add fewer than this many ads")
    args = ap.parse_args()

    # --- Resume state: what is already collected ---
    # collected = ids already in OUT plus ids in every --exclude file. needed = every id the scan
    # ever saw, minus those. This is what makes the script resumable: on a re-run the plan is
    # recomputed only over ids that are still missing.
    if not SCAN_CSV.exists():
        sys.exit(f"{SCAN_CSV} not found; run scan_australian_ads.py first")
    by_date, span = load_scan()
    collected: set[str] = set()
    if OUT.exists():
        collected = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8")}
    for path in args.exclude:
        collected |= {json.loads(l)["id"] for l in open(path, encoding="utf-8")}
    needed = set(span) - collected
    print(f"scan covers {len(by_date)} dates, {len(span):,} unique Australian ads; {len(collected):,} already collected, {len(needed):,} to fetch")
    # --- Plan ---
    # The greedy plan's tail consists of dates that each add only a few ads (ads live for a single
    # day or two) at 0.23 GB per download. --min-gain trims that tail; the "reachable" line reports
    # how many of the needed ads the trimmed plan still captures, so the coverage loss is explicit.
    # 0.23 GB/day is an observed average used only for this estimate.
    plan = [(d, g) for d, g in plan_dates(by_date, needed) if g >= args.min_gain]
    print(f"with --min-gain {args.min_gain}: {sum(g for _, g in plan):,} of {len(needed):,} ads reachable")
    print(f"dates to download: {len(plan)} (about {len(plan) * 0.23:.1f} GB)")
    for d, gain in plan:
        print(f"  {d}: +{gain:,} ads")
    if args.plan_only or not plan:
        return

    # --- Download and extract, one planned date at a time ---
    # `wanted` is recomputed per date from the live `collected` set, so a date whose ads were all
    # picked up already (e.g. by an earlier date this run, or a previous run) is skipped without a
    # download. Rows are appended and flushed per date, so an interrupted run resumes cleanly.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        for d, _ in plan:
            wanted = (by_date[d] & needed) - collected
            if not wanted:
                continue
            t0 = time.time()
            recs = extract_date(d, wanted, args.keep)
            for rec in recs:
                if rec["id"] in collected:  # some days carry duplicate parquet files
                    continue
                # Longevity fields from the scan: first/last date the id was seen, and the number of
                # distinct scan dates it appeared on. days_live can be smaller than the span length
                # if an ad was pulled and re-listed, or if a scan date is missing from the CSV.
                # The days_live set comprehension walks every date per record (O(dates) each);
                # acceptable for a few thousand ads.
                rec["first_seen"], rec["last_seen"] = span[rec["id"]]
                rec["days_live"] = len({dd for dd, ids in by_date.items() if rec["id"] in ids})
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                collected.add(rec["id"])
            f.flush()
            # len(recs) counts extracted rows including any duplicate-file copies, so "total" can
            # grow by slightly less than the "+N" shown.
            print(f"{d}: +{len(recs):,} ads ({time.time() - t0:.0f}s) -> total {len(collected):,}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import html_to_text  # noqa: E402
from scan_australian_ads import OUT as SCAN_CSV, REPO, australia_condition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "samples" / "sample_aus_all.jsonl"


def load_scan() -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    """Return ({date: {ids live that day}}, {id: (first_seen, last_seen)})."""
    by_date: dict[str, set[str]] = defaultdict(set)
    span: dict[str, list[str]] = {}
    with open(SCAN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_date[r["date"]].add(r["id"])
            s = span.setdefault(r["id"], [r["date"], r["date"]])
            s[0], s[1] = min(s[0], r["date"]), max(s[1], r["date"])
    return by_date, {k: (v[0], v[1]) for k, v in span.items()}


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


def extract_date(date: str, wanted: set[str], keep: bool) -> list[dict]:
    local = RAW_DIR / "data" / f"date={date}"
    fresh = not local.exists()
    snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"data/date={date}/*/*.parquet"], local_dir=RAW_DIR)
    glob = str(local / "*" / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep downloaded parquet files")
    ap.add_argument("--exclude", nargs="*", default=[], help="JSONL files whose ads are already collected elsewhere")
    ap.add_argument("--min-gain", type=int, default=1, help="stop once a date would add fewer than this many ads")
    args = ap.parse_args()

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
    plan = [(d, g) for d, g in plan_dates(by_date, needed) if g >= args.min_gain]
    print(f"with --min-gain {args.min_gain}: {sum(g for _, g in plan):,} of {len(needed):,} ads reachable")
    print(f"dates to download: {len(plan)} (about {len(plan) * 0.23:.1f} GB)")
    for d, gain in plan:
        print(f"  {d}: +{gain:,} ads")
    if args.plan_only or not plan:
        return

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
                rec["first_seen"], rec["last_seen"] = span[rec["id"]]
                rec["days_live"] = len({dd for dd, ids in by_date.items() if rec["id"] in ids})
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                collected.add(rec["id"])
            f.flush()
            print(f"{d}: +{len(recs):,} ads ({time.time() - t0:.0f}s) -> total {len(collected):,}", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()

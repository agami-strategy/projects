"""Collect Australian ads with full text from the N most recent daily snapshots.

Downloads each day's parquet files (newest first), extracts the Australian ads, drops ids
already collected, and appends to one JSONL. Parquet files are deleted after use unless
they were already present locally or --keep is given. Resumable.

Usage:
    python scripts/collect_recent_days.py --days 10
    python scripts/collect_recent_days.py --days 10 --tag aus_recent10

Output: data/samples/sample_<tag>.jsonl, same shape as sample_jobs.py output plus snapshot_date.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_jobs import html_to_text  # noqa: E402
from scan_australian_ads import REPO, ads_condition  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_DIR = ROOT / "data" / "samples"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--country", default="australia", help="preset (australia, us) or comma-separated location terms")
    ap.add_argument("--dates", nargs="*", default=None, help="explicit snapshot dates instead of the N most recent")
    ap.add_argument("--tag", default=None, help="default <country>_recent<days>")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    short = {"australia": "aus"}.get(args.country.lower(), args.country.lower().replace(",", "-"))
    tag = args.tag or (f"{short}_{'_'.join(args.dates)}" if args.dates else f"{short}_recent{args.days}")
    out = SAMPLE_DIR / f"sample_{tag}.jsonl"
    done_file = SAMPLE_DIR / f"sample_{tag}.dates_done"
    cond = ads_condition(args.country)

    api = HfApi()
    if args.dates:
        dates = sorted(args.dates, reverse=True)
    else:
        dates = sorted((e.path.split("date=")[1] for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="data") if "date=" in e.path), reverse=True)[: args.days]
    seen: set[str] = set()
    if out.exists():
        seen = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    print(f"dates: {dates[-1]} .. {dates[0]} | already collected {len(seen):,} ads from {len(done)} dates", flush=True)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for d in dates:
            if d in done:
                continue
            t0 = time.time()
            local = RAW_DIR / "data" / f"date={d}"
            fresh = not local.exists()
            snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"data/date={d}/*/*.parquet"], local_dir=RAW_DIR)
            glob = str(local / "*" / "*.parquet").replace("\\", "/")
            con = duckdb.connect()
            rows = con.execute(f"""
                SELECT id, source, source_slug, title, apply_url, description_html, employment_type, department,
                       locations, remote, posted_at, salary_min, salary_max, salary_currency, salary_period
                FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
                WHERE {cond}
            """).fetchall()
            cols = [d_[0] for d_ in con.description]
            new = 0
            for r in rows:
                rec = dict(zip(cols, r))
                if rec["id"] in seen:
                    continue
                rec["description_text"] = html_to_text(rec.pop("description_html"))
                rec["snapshot_date"] = d
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                seen.add(rec["id"])
                new += 1
            f.flush()
            with open(done_file, "a") as df:
                df.write(d + "\n")
            if fresh and not args.keep:
                shutil.rmtree(local, ignore_errors=True)
            print(f"{d}: {len(rows):,} {args.country} ads, {new:,} new -> total {len(seen):,} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()

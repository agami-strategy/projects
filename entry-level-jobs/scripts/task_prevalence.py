"""Task prevalence by occupation and seniority: for every (OSCA unit group, task) pair, the share
of entry-level ads and of experienced ads in which the task is required (judged explicit or
implied), plus the same for explicit only, the difference, and the task's exposure attributes.

Usage:
    python scripts/task_prevalence.py --tag us_2026-09-17_selected_tasks --model flash

Output: data/task_prevalence_<tag>_<model>.csv and .xlsx (one row per occupation x task)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
TASK_FILE = ROOT / "data" / "osca_task_share.csv"
LEVELS = ("entry_level", "experienced")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="us_2026-09-17_selected_tasks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    summ = pd.read_csv(LABEL_DIR / f"match_summary_{args.tag}_{args.model}.csv", encoding="utf-8-sig", dtype={"osca_code": str})
    summ = summ[summ["error"].isna() & summ["level"].isin(LEVELS)]
    ad_info = {r.ad_id: (r.osca_code, r.level) for r in summ.itertuples()}

    tasks = {}
    with open(TASK_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ts = float(r["TIME_SHARE"])
            tasks[(r["OSCA_UNIT_CODE"], r["Task ID"])] = {
                "osca_title": r["OSCA_TITLE"], "task": r["Task"], "time_share": ts,
                "automation_frac": float(r["automation_time"]) / ts if ts else None,
                "augmentation_frac": float(r["augmentation_time"]) / ts if ts else None,
                "unaffected_frac": float(r["unaffected_time"]) / ts if ts else None,
                "gpt4_exposure": r["gpt4_exposure"], "gpt4_automation": r["gpt4_automation"],
            }

    n_ads = defaultdict(int)                                   # (code, level) -> ads
    req = defaultdict(lambda: defaultdict(int))                # (code, task) -> {level_required, level_explicit}
    for line in open(LABEL_DIR / f"match_labels_{args.tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "judgements" not in r or r["id"] not in ad_info:
            continue
        code, level = ad_info[r["id"]]
        n_ads[(code, level)] += 1
        for tid, j in r["judgements"].items():
            if j["judgement"] in ("explicit", "implied"):
                req[(code, tid)][f"{level}_required"] += 1
                if j["judgement"] == "explicit":
                    req[(code, tid)][f"{level}_explicit"] += 1

    rows = []
    for (code, tid), t in tasks.items():
        ne, nx = n_ads.get((code, "entry_level"), 0), n_ads.get((code, "experienced"), 0)
        if not ne and not nx:
            continue
        c = req.get((code, tid), {})
        pe = c.get("entry_level_required", 0) / ne if ne else None
        px = c.get("experienced_required", 0) / nx if nx else None
        rows.append({
            "osca_code": code, "osca_title": t["osca_title"], "task_id": tid, "task": t["task"],
            "n_entry_ads": ne, "n_experienced_ads": nx,
            "pct_entry_required": pe, "pct_experienced_required": px,
            "diff_entry_minus_experienced": (pe - px) if pe is not None and px is not None else None,
            "pct_entry_explicit": c.get("entry_level_explicit", 0) / ne if ne else None,
            "pct_experienced_explicit": c.get("experienced_explicit", 0) / nx if nx else None,
            "time_share": t["time_share"], "automation_frac": t["automation_frac"],
            "augmentation_frac": t["augmentation_frac"], "unaffected_frac": t["unaffected_frac"],
            "largely_automatable": int(t["automation_frac"] >= 0.5) if t["automation_frac"] is not None else None,
            "largely_augmentable": int(t["augmentation_frac"] >= 0.5) if t["augmentation_frac"] is not None else None,
            "largely_unaffected": int(t["unaffected_frac"] >= 0.5) if t["unaffected_frac"] is not None else None,
            "gpt4_exposure": t["gpt4_exposure"], "gpt4_automation": t["gpt4_automation"],
        })
    df = pd.DataFrame(rows).sort_values(["osca_code", "pct_entry_required"], ascending=[True, False])
    stem = ROOT / "data" / f"task_prevalence_{args.tag}_{args.model}"
    df.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(f"{stem}.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="task_prevalence", index=False)
        ws = xw.sheets["task_prevalence"]
        ws.freeze_panes = "E2"
        ws.auto_filter.ref = ws.dimensions
        for col, w in {"B": 34, "D": 70}.items():
            ws.column_dimensions[col].width = w
    print(f"rows: {len(df):,} (occupation x task) | occupations: {df['osca_code'].nunique()} | "
          f"ads: entry {sum(v for (c, l), v in n_ads.items() if l == 'entry_level'):,}, "
          f"experienced {sum(v for (c, l), v in n_ads.items() if l == 'experienced'):,}")
    print(f"wrote {stem}.csv and .xlsx")


if __name__ == "__main__":
    main()

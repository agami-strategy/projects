"""Task prevalence by occupation and seniority: for every (OSCA unit group, task) pair, the share
of entry-level ads and of experienced ads in which the task is required (judged explicit or
implied), plus the same for explicit only, the difference, and the task's exposure attributes.

Usage:
    python scripts/task_prevalence.py --tag us_2026-09-17_selected_tasks --model flash

Output: data/task_prevalence_<tag>_<model>.csv and .xlsx (one row per occupation x task)
"""
# %% Imports and paths
# Descriptive companion to the regressions: instead of collapsing each ad to a share, it asks, task by
# task, how often entry-level and experienced ads within the same occupation require it. This is the
# table to consult when a regression effect needs to be traced back to specific tasks.
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

# Inputs: match_summary / match_labels from match_tasks.py in data/labels/, and the OSCA task list
# (one row per occupation x task with time shares and exposure attributes) in data/.
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
TASK_FILE = ROOT / "data" / "osca_task_share.csv"
LEVELS = ("entry_level", "experienced")


# %% Command-line entry point
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="us_2026-09-17_selected_tasks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # %% Ads: occupation and level for each usable ad
    # Same sample rule as build_ad_table in regress_automation.py: no matching error, and level is one
    # of the two compared. ad_info maps ad_id -> (osca_code, level); osca_code is read as a string so
    # codes keep their exact form for joining to the task file.
    summ = pd.read_csv(LABEL_DIR / f"match_summary_{args.tag}_{args.model}.csv", encoding="utf-8-sig", dtype={"osca_code": str})
    summ = summ[summ["error"].isna() & summ["level"].isin(LEVELS)]
    ad_info = {r.ad_id: (r.osca_code, r.level) for r in summ.itertuples()}

    # %% Task attributes keyed by (occupation, task id)
    # Fractions are each component's share of the task's own time (automation + augmentation +
    # unaffected = 1 per task); None when the task has zero time share. gpt4_exposure / gpt4_automation
    # are the task-level labels carried through from the ANZSCO source workbook, kept as text.
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

    # %% Count ads and task requirements
    # n_ads[(code, level)] is the denominator: every usable ad with a judgements record counts, whether
    # or not any task was matched. req[(code, task)] holds four counters keyed "<level>_required" (explicit
    # or implied) and "<level>_explicit" (explicit only). Task ids in an ad's judgements that are not in
    # the task file are counted here but never emitted, because the output loop iterates over `tasks`.
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

    # %% One output row per occupation x task
    # Occupations with no ads at either level are skipped. Shares are None (blank in the CSV) when an
    # occupation has no ads at that level, and the entry-minus-experienced difference is only computed
    # when both shares exist. Tasks that no ad required still appear with a share of 0, which is
    # informative (the occupation's task list includes work that ads never mention).
    # largely_* flags use the same >= 50% rule as the regression outcomes so the two tables agree.
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
    # %% Write CSV and Excel
    # Sorted by occupation, then by how often entry-level ads require the task (most common first).
    # utf-8-sig adds a BOM so Excel opens the CSV with the right encoding.
    df = pd.DataFrame(rows).sort_values(["osca_code", "pct_entry_required"], ascending=[True, False])
    stem = ROOT / "data" / f"task_prevalence_{args.tag}_{args.model}"
    df.to_csv(f"{stem}.csv", index=False, encoding="utf-8-sig")
    # openpyxl touches for reading in Excel: freeze_panes "E2" keeps the header row and the first four
    # columns (code, title, task_id, task) fixed while scrolling; auto_filter over ws.dimensions (the used
    # range, e.g. "A1:V500") adds filter dropdowns; column B (title) and D (task text) are widened.
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

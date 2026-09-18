"""For each labelled ad, judge which tasks of its OSCA unit group the ad calls for.

Reads the merged label table (run merge_labels.py first, so gold codes and levels are respected)
and the converted task file (convert_tasks_to_osca.py). One Claude call per ad: the ad text plus
the unit group's task list go in, and a judgement per task comes back:
    explicit      the ad describes this task or a clear equivalent
    implied       not stated, but the role as described could not be done without it
    not_required  nothing in the ad calls for it

Outputs (data/labels/):
    match_labels_<tag>_<model>.jsonl   raw judgements per ad (resumable)
    match_tasks_<tag>_<model>.xlsx     sheet `tasks`: one row per ad x task with time shares
                                       sheet `summary`: one row per ad with coverage and automation shares
    match_summary_<tag>_<model>.csv    the summary sheet as CSV

Usage:
    python scripts/match_tasks.py data/samples/sample_100_australia.jsonl --dry-run
    python scripts/match_tasks.py data/samples/sample_100_australia.jsonl --limit 5
    python scripts/match_tasks.py data/samples/sample_100_australia.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import anthropic
import openpyxl
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env  # noqa: E402
from label_osca import DEFAULT_MODEL, MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
MERGED_DIR = ROOT / "data" / "merged"
TASK_FILE = ROOT / "data" / "osca_task_share.csv"
MAX_AD_CHARS = 6000

SYSTEM_PROMPT = """You compare a job advertisement against the standard task list for its occupation and decide, \
for every task, whether the advertised job calls for it.

Judgements:
- explicit: the ad describes this task, or something clearly equivalent, as part of the job. Quote the phrase.
- implied: the ad does not mention it, but the role as described could not be done without it. Use this sparingly; \
generic statements like "other duties as required" do not make every task implied.
- not_required: nothing in the ad calls for this task, or the ad describes a job scope that excludes it.

Judge from the ad's duties, responsibilities and requirements. Company descriptions, benefits and equal-opportunity \
boilerplate carry no information about tasks. Return exactly one judgement for every task ID in the list, in the same \
order. Evidence is a short phrase quoted from the ad for explicit judgements only; leave it empty otherwise."""


class TaskJudgement(BaseModel):
    task_id: str
    judgement: Literal["explicit", "implied", "not_required"]
    evidence: str = Field(description="short quote from the ad for explicit judgements; empty string otherwise")


class Matches(BaseModel):
    judgements: list[TaskJudgement]


def load_tasks() -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = defaultdict(list)
    with open(TASK_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tasks[r["OSCA_UNIT_CODE"]].append({
                "task_id": r["Task ID"], "task": r["Task"],
                "time_share": float(r["TIME_SHARE"]), "automation_time": float(r["automation_time"]),
                "augmentation_time": float(r["augmentation_time"]), "unaffected_time": float(r["unaffected_time"]),
                "gpt4_exposure": r["gpt4_exposure"], "gpt4_automation": r["gpt4_automation"],
            })
    return tasks


def load_merged(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def ad_to_message(ad: dict, tasks: list[dict], osca_title: str) -> str:
    text = (ad.get("description_text") or "").strip()
    if len(text) > MAX_AD_CHARS:
        text = text[:MAX_AD_CHARS] + "\n[... truncated ...]"
    task_lines = "\n".join(f"{t['task_id']}: {t['task']}" for t in tasks)
    return (
        f"JOB TITLE: {ad.get('title')}\nDEPARTMENT: {ad.get('department') or '(none)'}\n\n"
        f"AD TEXT:\n{text}\n\n"
        f"OCCUPATION: {osca_title}\nTASK LIST ({len(tasks)} tasks, format 'task_id: task'):\n{task_lines}\n\n"
        "Return one judgement per task ID."
    )


def match_one(client: anthropic.Anthropic, ad: dict, tasks: list[dict], osca_title: str, model: dict, effort: str) -> dict:
    extra = {"output_config": {"effort": effort}} if model["effort"] else {}
    resp = client.messages.parse(
        model=model["id"],
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ad_to_message(ad, tasks, osca_title)}],
        output_format=Matches,
        **extra,
    )
    if resp.stop_reason == "refusal":
        return {"id": ad["id"], "error": "refusal", "stop_details": str(resp.stop_details)}
    if resp.stop_reason == "max_tokens":
        return {"id": ad["id"], "error": "max_tokens: task list too long for one response"}
    judgements = {j.task_id: {"judgement": j.judgement, "evidence": j.evidence} for j in resp.parsed_output.judgements}
    missing = [t["task_id"] for t in tasks if t["task_id"] not in judgements]
    u = resp.usage
    return {
        "id": ad["id"],
        "judgements": judgements,
        "missing_task_ids": missing,
        "usage": {"input": u.input_tokens, "cache_write": u.cache_creation_input_tokens or 0,
                  "cache_read": u.cache_read_input_tokens or 0, "output": u.output_tokens},
        "model": resp.model,
    }


def summarise(ad_row: dict, tasks: list[dict], result: dict | None) -> tuple[list[list], list]:
    """Return (task rows, summary row) for one ad."""
    j = (result or {}).get("judgements", {})
    task_rows, acc = [], defaultdict(float)
    for t in tasks:
        jj = j.get(t["task_id"], {"judgement": "missing", "evidence": ""})
        task_rows.append([ad_row["id"], ad_row["title"], ad_row["level"], ad_row["osca_code"], ad_row["osca_title"],
                          t["task_id"], t["task"], jj["judgement"], jj["evidence"],
                          t["time_share"], t["automation_time"], t["augmentation_time"], t["unaffected_time"],
                          t["gpt4_exposure"], t["gpt4_automation"]])
        k = jj["judgement"]
        acc[f"n_{k}"] += 1
        acc[f"time_{k}"] += t["time_share"]
        acc[f"auto_{k}"] += t["automation_time"]
        acc[f"aug_{k}"] += t["augmentation_time"]
    time_ex, time_exim = acc["time_explicit"], acc["time_explicit"] + acc["time_implied"]
    auto_ex, auto_exim = acc["auto_explicit"], acc["auto_explicit"] + acc["auto_implied"]
    aug_ex, aug_exim = acc["aug_explicit"], acc["aug_explicit"] + acc["aug_implied"]
    occ_time = sum(t["time_share"] for t in tasks)
    occ_auto = sum(t["automation_time"] for t in tasks)
    occ_aug = sum(t["augmentation_time"] for t in tasks)
    summary = [
        ad_row["id"], ad_row["title"], ad_row["company"], ad_row["level"], ad_row["level_source"],
        ad_row["osca_code"], ad_row["osca_title"], ad_row["osca_source"],
        len(tasks), int(acc["n_explicit"]), int(acc["n_implied"]), int(acc["n_not_required"]), int(acc["n_missing"]),
        round(time_ex, 4), round(time_exim, 4),
        round(auto_ex / time_ex, 4) if time_ex else None,
        round(auto_exim / time_exim, 4) if time_exim else None,
        round(occ_auto / occ_time, 4) if occ_time else None,
        round(aug_ex / time_ex, 4) if time_ex else None,
        round(aug_exim / time_exim, 4) if time_exim else None,
        round(occ_aug / occ_time, 4) if occ_time else None,
        (result or {}).get("error"),
    ]
    return task_rows, summary


TASK_HEADER = ["ad_id", "ad_title", "level", "osca_code", "osca_title", "task_id", "task", "judgement", "evidence",
               "time_share", "automation_time", "augmentation_time", "unaffected_time", "gpt4_exposure", "gpt4_automation"]
SUMMARY_HEADER = ["ad_id", "ad_title", "company", "level", "level_source", "osca_code", "osca_title", "osca_source",
                  "n_tasks", "n_explicit", "n_implied", "n_not_required", "n_missing",
                  "time_share_explicit", "time_share_explicit_or_implied",
                  "automation_share_explicit", "automation_share_explicit_or_implied", "automation_share_occupation",
                  "augmentation_share_explicit", "augmentation_share_explicit_or_implied", "augmentation_share_occupation",
                  "error"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of ads from sample_jobs.py")
    ap.add_argument("--tag", default=None, help="default = input file stem; must match the merged file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    model = MODELS[args.model]
    PRICE = model["price"]
    base_tag = args.tag or Path(args.input).stem.replace("sample_", "")
    tag = base_tag + "_" + args.model
    merged_path = MERGED_DIR / f"merged_{base_tag}.csv"
    if not merged_path.exists():
        sys.exit(f"{merged_path} not found; run merge_labels.py on this input first")
    merged = load_merged(merged_path)
    tasks_by_code = load_tasks()

    ads = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        ads = ads[: args.limit]
    work_items, skipped = [], defaultdict(int)
    for ad in ads:
        row = merged.get(ad["id"])
        if not row or not row["osca_code"]:
            skipped["no OSCA code"] += 1
            continue
        if not row["level"]:
            skipped["no seniority label"] += 1
            continue
        tasks = tasks_by_code.get(row["osca_code"])
        if not tasks:
            skipped["no task data for OSCA code"] += 1
            continue
        work_items.append((ad, row, tasks))
    print(f"ads: {len(ads)} | usable: {len(work_items)} | skipped: {dict(skipped)} | model {model['id']}")
    if not work_items:
        sys.exit("nothing to do: every ad is missing an OSCA code, a seniority label or task data")

    jsonl_path = LABEL_DIR / f"match_labels_{tag}.jsonl"
    xlsx_path = LABEL_DIR / f"match_tasks_{tag}.xlsx"
    csv_path = LABEL_DIR / f"match_summary_{tag}.csv"

    if args.dry_run:
        ad, row, tasks = work_items[0]
        msg = ad_to_message(ad, tasks, row["osca_title"])
        print(f"\n--- first ad message ({len(tasks)} tasks) ---\n{msg[:600]}\n...\n{msg[-500:]}")
        est_in = sum(len(SYSTEM_PROMPT) // 4 + len(ad_to_message(a, t, r["osca_title"])) // 4 for a, r, t in work_items)
        est_out = sum(len(t) * 30 + 50 for _, _, t in work_items)
        print(f"\nestimated cost at {model['id']} list price: ~${(est_in * PRICE['input'] + est_out * PRICE['output']) / 1e6:.2f} "
              f"(~{est_in:,} input, ~{est_out:,} output tokens; tasks per ad max {max(len(t) for _, _, t in work_items)})")
        return

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = anthropic.Anthropic(max_retries=4)

    results: dict[str, dict] = {}
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and jsonl_path.exists():
        jsonl_path.unlink()
        print("--fresh: discarded previous output")
    if jsonl_path.exists():
        for line in open(jsonl_path, encoding="utf-8"):
            r = json.loads(line)
            if "judgements" in r:
                results[r["id"]] = r
        print(f"resuming: {len(results)} already matched")
    todo = [w for w in work_items if w[0]["id"] not in results]

    lock = threading.Lock()
    out = open(jsonl_path, "a", encoding="utf-8")
    t0 = time.time()

    def work(item):
        ad, row, tasks = item
        try:
            return match_one(client, ad, tasks, row["osca_title"], model, args.effort)
        except anthropic.APIError as e:
            return {"id": ad["id"], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, w): w for w in todo}
        for i, fut in enumerate(as_completed(futs), start=len(results) + 1):
            ad, row, tasks = futs[fut]
            r = fut.result()
            with lock:
                results[ad["id"]] = r
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
            if "judgements" in r:
                c = defaultdict(int)
                for j in r["judgements"].values():
                    c[j["judgement"]] += 1
                shown = f"{len(tasks)} tasks: {c['explicit']} explicit, {c['implied']} implied" + (f", {len(r['missing_task_ids'])} missing" if r["missing_task_ids"] else "")
            else:
                shown = r.get("error")
            print(f"[{i}/{len(work_items)}] {ad['title'][:50]!r} [{row['osca_code']}] -> {shown}")
    out.close()

    ok = [r for r in results.values() if "judgements" in r]
    usage = {k: sum(r["usage"][k] for r in ok) for k in PRICE}
    cost = sum(usage[k] * PRICE[k] / 1e6 for k in PRICE)
    print(f"\nmatched {len(ok)}, errors {len(results) - len(ok)}, {time.time() - t0:.0f}s | tokens {usage} | cost ~${cost:.2f}")

    task_rows, summary_rows = [], []
    for ad, row, tasks in work_items:
        tr, sr = summarise(row, tasks, results.get(ad["id"]))
        task_rows.extend(tr)
        summary_rows.append(sr)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(SUMMARY_HEADER)
    for r in summary_rows:
        ws.append(r)
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    ws2 = wb.create_sheet("tasks")
    ws2.append(TASK_HEADER)
    for r in task_rows:
        ws2.append(r)
    ws2.freeze_panes = "C2"
    ws2.auto_filter.ref = ws2.dimensions
    wb.save(xlsx_path)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        w.writerows(summary_rows)
    print(f"wrote {jsonl_path}\nwrote {xlsx_path}\nwrote {csv_path}")


if __name__ == "__main__":
    main()

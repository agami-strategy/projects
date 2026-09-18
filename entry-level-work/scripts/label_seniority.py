"""Classify job ads as entry-level vs experienced hire using Claude.

The system prompt defines entry-level relative to the occupation and lists the signals to
look for; the model returns one label with confidence, the evidence phrase it relied on and
a one-line rationale. Output is a JSONL keyed by ad id (joinable with the OSCA labels) and an
Excel review sheet with blank gold_level / reviewer_notes columns.

Usage:
    python scripts/label_seniority.py data/samples/sample_100_australia.jsonl --dry-run
    python scripts/label_seniority.py data/samples/sample_100_australia.jsonl --limit 5
    python scripts/label_seniority.py data/samples/sample_100_australia.jsonl
    python scripts/label_seniority.py data/samples/sample_100_australia.jsonl --model haiku-4-5 --fresh

Output: data/labels/seniority_labels_<tag>_<model>.jsonl and seniority_review_<tag>_<model>.xlsx.
Re-running resumes; --fresh discards previous output for that input and model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import anthropic
import openpyxl
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_env  # noqa: E402
from label_osca import DEFAULT_MODEL, MODELS  # noqa: E402  (one price table for all scripts)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "labels"
MAX_AD_CHARS = 6000

SYSTEM_PROMPT = """You classify job advertisements as ENTRY-LEVEL or EXPERIENCED hires.

Definition: entry-level means the lowest rung of that occupation that a new entrant to the occupation \
could be hired into. Judge entry relative to the occupation, not the labour market: a graduate solicitor \
or a newly registered nurse is entry-level for their occupation even though they hold a qualification. \
An experienced hire is a role that expects the person to have already worked in the occupation or a \
closely related one.

Signals to look for, in rough order of weight:
- Minimum years of experience stated in the requirements. Two years or fewer, or none required, points to \
entry-level; three or more points to experienced. Stated years outrank the job title when they conflict.
- Whether prior experience in this kind of work is required, merely preferred, or not needed.
- Seniority words in the title and text: intern, graduate, trainee, junior, associate, entry-level, early \
career versus senior, lead, principal, staff, head of, director, manager, executive. Treat "associate" \
with care: it means junior in US finance, law and consulting but mid-level elsewhere.
- Explicit early-career framing: graduate programs, "no experience necessary", "we will train you".
- Whether the role supervises or manages others, owns a function, or is accountable for a team or budget.
- The gap between the education required and the experience required. A degree plus no experience is \
typical of entry-level professional roles.

Internships, placements, cadetships, apprenticeships, traineeships and vacation programs are always \
entry_level; also set is_internship so they can be identified. \
Return "unclear" when the ad gives no usable requirements, describes several roles at different levels, \
or is not a real single job. Quote the single phrase from the ad that most drove your decision as evidence."""


class Seniority(BaseModel):
    level: Literal["entry_level", "experienced", "unclear"]
    is_internship: bool
    confidence: float = Field(description="probability between 0 and 1 that the level is correct")
    evidence: str = Field(description="short phrase quoted from the ad that most drove the decision")
    rationale: str = Field(description="one sentence")


def ad_to_message(ad: dict) -> str:
    text = (ad.get("description_text") or "").strip()
    if len(text) > MAX_AD_CHARS:
        text = text[:MAX_AD_CHARS] + "\n[... truncated ...]"
    return (
        f"JOB TITLE: {ad.get('title')}\n"
        f"DEPARTMENT: {ad.get('department') or '(none)'}\n"
        f"EMPLOYMENT TYPE: {ad.get('employment_type') or '(none)'}\n\n"
        f"AD TEXT:\n{text}\n\n"
        "Classify this ad as entry_level, experienced or unclear."
    )


def label_one(client: anthropic.Anthropic, ad: dict, model: dict, effort: str) -> dict:
    extra = {"output_config": {"effort": effort}} if model["effort"] else {}
    resp = client.messages.parse(
        model=model["id"],
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ad_to_message(ad)}],
        output_format=Seniority,
        **extra,
    )
    if resp.stop_reason == "refusal":
        return {"id": ad["id"], "error": "refusal", "stop_details": str(resp.stop_details)}
    u = resp.usage
    label = resp.parsed_output.model_dump()
    if label["is_internship"]:
        label["level"] = "entry_level"  # internships are entry-level by definition, whatever the model chose
    return {
        "id": ad["id"],
        "label": label,
        "usage": {
            "input": u.input_tokens,
            "cache_write": u.cache_creation_input_tokens or 0,
            "cache_read": u.cache_read_input_tokens or 0,
            "output": u.output_tokens,
        },
        "model": resp.model,
    }


def write_review_xlsx(ads: list[dict], results: dict[str, dict], path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(["id", "title", "company", "location", "url",
               "level", "is_internship", "confidence", "evidence", "rationale",
               "gold_level", "reviewer_notes", "error"])
    for ad in ads:
        r = results.get(ad["id"])
        row = [ad["id"], ad.get("title"), ad.get("source_slug"), ", ".join(ad.get("locations") or []), ad.get("apply_url")]
        if r and "label" in r:
            lab = r["label"]
            row += [lab["level"], "Y" if lab["is_internship"] else "", round(lab["confidence"], 2), lab["evidence"], lab["rationale"], None, None, None]
        else:
            row += [None] * 7 + [(r or {}).get("error", "not labelled")]
        ws.append(row)
    for col, w in {"B": 40, "C": 18, "D": 28, "F": 13, "I": 45, "J": 60, "K": 12, "L": 30}.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of ads from sample_jobs.py")
    ap.add_argument("--tag", default=None, help="output name suffix; default = input file stem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                    help="ignored for models that do not support effort")
    ap.add_argument("--dry-run", action="store_true", help="show the prompt and cost estimate; no API calls")
    ap.add_argument("--fresh", action="store_true", help="discard existing output for this input+model")
    args = ap.parse_args()

    model = MODELS[args.model]
    PRICE = model["price"]
    ads = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        ads = ads[: args.limit]
    tag = (args.tag or Path(args.input).stem.replace("sample_", "")) + "_" + args.model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"seniority_labels_{tag}.jsonl"
    xlsx_path = OUT_DIR / f"seniority_review_{tag}.xlsx"
    print(f"ads: {len(ads)} from {args.input}; model {model['id']}")

    if args.dry_run:
        print("\n--- system prompt ---\n" + SYSTEM_PROMPT)
        print("\n--- first ad message ---\n" + ad_to_message(ads[0])[:800] + "\n...")
        sys_tokens = len(SYSTEM_PROMPT) // 4
        est_in = len(ads) * (sys_tokens + len(ad_to_message(ads[0])) // 4 + 200)
        est_out = len(ads) * 120
        print(f"\nestimated cost at {model['id']} list price: ~${(est_in * PRICE['input'] + est_out * PRICE['output']) / 1e6:.2f} "
              f"(~{est_in:,} input tokens, ~{est_out:,} output tokens; no caching, the prompt is small)")
        return

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = anthropic.Anthropic(max_retries=4)

    results: dict[str, dict] = {}
    if args.fresh and jsonl_path.exists():
        jsonl_path.unlink()
        print("--fresh: discarded previous labels for this input and model")
    if jsonl_path.exists():
        for line in open(jsonl_path, encoding="utf-8"):
            r = json.loads(line)
            if "label" in r:
                results[r["id"]] = r
        print(f"resuming: {len(results)} already labelled")
    todo = [a for a in ads if a["id"] not in results]

    lock = threading.Lock()
    out = open(jsonl_path, "a", encoding="utf-8")
    t0 = time.time()

    def work(ad: dict) -> dict:
        try:
            return label_one(client, ad, model, args.effort)
        except anthropic.APIError as e:
            return {"id": ad["id"], "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, ad): ad for ad in todo}
        for i, fut in enumerate(as_completed(futs), start=len(results) + 1):
            ad = futs[fut]
            r = fut.result()
            with lock:
                results[ad["id"]] = r
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
            lab = r.get("label")
            shown = f"{lab['level']} ({lab['confidence']:.2f}){' intern' if lab['is_internship'] else ''}" if lab else r.get("error")
            print(f"[{i}/{len(ads)}] {ad['title'][:60]!r} -> {shown}")
    out.close()

    ok = [r for r in results.values() if "label" in r]
    errs = [r for r in results.values() if "label" not in r]
    usage = {k: sum(r["usage"][k] for r in ok) for k in PRICE}
    cost = sum(usage[k] * PRICE[k] / 1e6 for k in PRICE)
    counts: dict[str, int] = {}
    for r in ok:
        counts[r["label"]["level"]] = counts.get(r["label"]["level"], 0) + 1
    print(f"\nlabelled {len(ok)}, errors {len(errs)}, {time.time() - t0:.0f}s")
    print(f"tokens: {usage}  cost ~${cost:.2f}")
    print(f"levels: {counts}  internships: {sum(1 for r in ok if r['label']['is_internship'])}")

    write_review_xlsx(ads, results, xlsx_path)
    print(f"wrote {jsonl_path}\nwrote {xlsx_path}")


if __name__ == "__main__":
    main()

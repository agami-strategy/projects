"""Label job ads with OSCA unit group codes (4-digit) using DeepSeek.

Approach: the full list of OSCA unit groups (code, title, lead statement, skill level,
occupation titles and alternative titles) goes in the system prompt, which DeepSeek caches
automatically by prefix. For each ad the model returns its top 3 candidate unit groups with
confidence and rationale, plus flags for ads that are not a single real occupation. Output is
a JSONL of raw results and an Excel review sheet with blank gold_code / reviewer_notes columns.

Usage:
    python scripts/label_osca.py data/samples/sample_100.jsonl --dry-run     # no API calls
    python scripts/label_osca.py data/samples/sample_100.jsonl               # label all
    python scripts/label_osca.py data/samples/sample_100.jsonl --limit 5     # try a few first
    python scripts/label_osca.py data/samples/sample_100.jsonl --model pro --fresh
    python scripts/label_osca.py data/samples/sample_100.jsonl --thinking    # slower, reasons before answering

Output files carry the model name: data/labels/osca_labels_<tag>_<model>.jsonl and osca_review_<tag>_<model>.xlsx.
Needs DEEPSEEK_API_KEY (environment or .env). Re-running resumes: ads already in the output JSONL are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import openpyxl
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL, MODELS, USAGE_KEYS, Refusal, complete_json, cost, estimate_cost  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSCA = ROOT / "data" / "osca-descriptions.xlsx"
OUT_DIR = ROOT / "data" / "labels"
MAX_AD_CHARS = 6000  # ~1.5k tokens; responsibilities/qualifications are almost always within this


# --------------------------------------------------------------------------- OSCA loading
def _read_bytes_shared(path: Path) -> bytes:
    """Read a file that Excel has open. Excel's lock rejects Python's normal open(), but a
    Win32 CreateFile with read+write+delete sharing is allowed."""
    import ctypes
    import msvcrt

    GENERIC_READ, SHARE_ALL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL = 0x80000000, 0x7, 3, 0x80
    h = ctypes.windll.kernel32.CreateFileW(str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if h == -1:
        raise PermissionError(f"cannot open {path} even with shared access (Win32 error {ctypes.GetLastError()})")
    with os.fdopen(msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY), "rb") as f:
        return f.read()


def load_workbook_safely(path: Path) -> openpyxl.Workbook:
    """openpyxl gets PermissionError while the workbook is open in Excel; fall back to a shared read."""
    try:
        return openpyxl.load_workbook(path, read_only=True)
    except PermissionError:
        if sys.platform != "win32":
            raise
        return openpyxl.load_workbook(io.BytesIO(_read_bytes_shared(path)), read_only=True)


def load_osca(path: Path) -> dict[str, dict]:
    """Return {unit_group_code: {title, lead, skill_level, occupations:[{code,title,alt,spec}]}}."""
    wb = load_workbook_safely(path)
    t2 = [r for r in wb["Table 2"].iter_rows(values_only=True) if r[0] and str(r[0]).strip().isdigit()]
    t1 = [r for r in wb["Table 1"].iter_rows(values_only=True) if r[0] and str(r[0]).strip().isdigit()]

    groups: dict[str, dict] = {}
    for r in t2:
        code = str(r[0]).strip()
        if len(code) != 4:
            continue
        groups[code] = {
            "code": code,
            "title": str(r[1]).strip(),
            "lead": str(r[2] or "").strip(),
            "skill_level": str(r[4] or "").strip(),
            "occupations": [],
        }
    for r in t1:
        code = str(r[0]).strip()
        ug = code[:4]
        if ug in groups:
            groups[ug]["occupations"].append({
                "code": code,
                "title": str(r[1] or "").strip(),
                "alt": str(r[2] or "").strip(),
                "spec": str(r[9] or "").strip(),
            })
    return groups


def format_group(g: dict) -> str:
    occ_titles = "; ".join(o["title"] for o in g["occupations"] if o["title"])
    alts = "; ".join(x for o in g["occupations"] for x in (o["alt"], o["spec"]) if x)
    line = f"{g['code']} {g['title']} (skill level {g['skill_level']}): {g['lead']}"
    if occ_titles:
        line += f" Occupations: {occ_titles}."
    if alts:
        line += f" Also known as: {alts}."
    return line


def build_system_prompt(groups: dict[str, dict]) -> str:
    instructions = """You are an expert occupation coder for the Australian Bureau of Statistics' OSCA \
(Occupation Standard Classification for Australia, 2024). You classify job advertisements to the \
4-digit OSCA unit group that best matches the work actually performed.

Coding rules:
- Code the occupation, not the industry or the employer. A software engineer at a bank is still a software engineer.
- Code the substantive work described in the responsibilities, not the title alone. Titles are inflated and vary by country: \
"Account Executive" in a US tech company is usually a sales representative, not a manager; a "Manager" who has no reports and \
does hands-on work is coded to the hands-on unit group; "Head of" / "Director" / "VP" with genuine control over a function is a manager.
- OSCA skill levels: 1 = bachelor degree or higher / 5+ years experience; 2 = diploma; 3 = certificate III/IV or trade; \
4 = certificate II; 5 = certificate I or short on-the-job training. Use the ad's stated requirements to judge the ad's skill level \
and let that inform the choice between adjacent unit groups.
- Most ads in this dataset come from technology, startup and knowledge-work employers in many countries. Map foreign titles to \
the closest Australian occupation.
- Give up to 3 candidates ordered best first. Confidence is your probability that the candidate is the correct unit group; \
the confidences do not need to sum to 1, but should sum to at most 1.
- Set issue when the ad is not a single, real, identifiable job: talent-community / expression-of-interest pages, ads covering \
several unrelated roles, placeholder or test postings, ads not in English (still attempt a code if you can read it), \
or ads with too little text to classify.

Answer in JSON. Example of the shape (values illustrative):
{"issue": "none", "ad_skill_level": 1, "candidates": [{"code": "2733", "title": "Software and Applications Engineers", \
"confidence": 0.85, "rationale": "designs and builds backend services"}, {"code": "2732", "title": "ICT Business and Systems \
Analysts and Architects", "confidence": 0.1, "rationale": "some architecture duties"}]}

The full OSCA unit group list follows. Return codes only from this list.

OSCA UNIT GROUPS
"""
    listing = "\n".join(format_group(g) for g in sorted(groups.values(), key=lambda g: g["code"]))
    return instructions + listing


# --------------------------------------------------------------------------- output schema
class Candidate(BaseModel):
    code: str = Field(description="4-digit OSCA unit group code")
    title: str = Field(description="unit group title exactly as listed")
    confidence: float = Field(description="probability between 0 and 1 that this is the correct unit group")
    rationale: str = Field(description="one sentence: which duties / requirements drove this choice")


class Label(BaseModel):
    issue: Literal["none", "talent_community", "multiple_roles", "not_a_job", "non_english", "insufficient_text"]
    ad_skill_level: int = Field(description="OSCA skill level 1-5 implied by the ad's stated requirements")
    candidates: list[Candidate] = Field(description="best first, 1 to 3 items; may be empty only if issue is not_a_job or insufficient_text")


# --------------------------------------------------------------------------- per-ad call
def ad_to_message(ad: dict) -> str:
    text = (ad.get("description_text") or "").strip()
    if len(text) > MAX_AD_CHARS:
        text = text[:MAX_AD_CHARS] + "\n[... truncated ...]"
    locs = ", ".join(ad.get("locations") or []) or "(none given)"
    return (
        f"JOB TITLE: {ad.get('title')}\n"
        f"DEPARTMENT: {ad.get('department') or '(none)'}\n"
        f"EMPLOYMENT TYPE: {ad.get('employment_type') or '(none)'}\n"
        f"LOCATION: {locs}\n"
        f"EMPLOYER SLUG: {ad.get('source_slug')}\n\n"
        f"AD TEXT:\n{text}\n\n"
        "Classify this ad to OSCA unit groups. Answer in JSON."
    )


def label_one(system: str, ad: dict, model_key: str, thinking: bool) -> dict:
    try:
        parsed, usage, model_id = complete_json(system, ad_to_message(ad), Label, model_key, max_tokens=2000, thinking=thinking)
    except Refusal as e:
        return {"id": ad["id"], "error": f"no valid response: {e}"}
    return {"id": ad["id"], "label": parsed.model_dump(), "usage": usage, "model": model_id}


# --------------------------------------------------------------------------- review sheet
def write_review_xlsx(ads: list[dict], results: dict[str, dict], groups: dict[str, dict], path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "review"
    header = [
        "id", "title", "company", "department", "location", "url",
        "issue", "ad_skill_level",
        "code_1", "title_1", "conf_1", "rationale_1", "ug_skill_1", "skill_mismatch_1",
        "code_2", "title_2", "conf_2", "rationale_2",
        "code_3", "title_3", "conf_3", "rationale_3",
        "gold_code", "gold_title", "reviewer_notes", "error",
    ]
    ws.append(header)
    for ad in ads:
        r = results.get(ad["id"])
        row = [
            ad["id"], ad.get("title"), ad.get("source_slug"), ad.get("department"),
            ", ".join(ad.get("locations") or []), ad.get("apply_url"),
        ]
        if not r or "label" not in r:
            row += [None] * 16 + [None, None, None, (r or {}).get("error", "not labelled")]
            ws.append(row)
            continue
        lab = r["label"]
        row += [lab["issue"], lab["ad_skill_level"]]
        cands = lab["candidates"][:3]
        for i in range(3):
            if i < len(cands):
                c = cands[i]
                valid = c["code"] in groups
                title = groups[c["code"]]["title"] if valid else f"INVALID CODE: {c['title']}"
                row += [c["code"], title, round(c["confidence"], 2), c["rationale"]]
                if i == 0:
                    ug_sl = groups[c["code"]]["skill_level"] if valid else ""
                    mismatch = valid and ug_sl and str(lab["ad_skill_level"]) not in ug_sl.replace(" ", "").split(";")
                    row += [ug_sl, "Y" if mismatch else ""]
            else:
                row += [None, None, None, None] + ([None, None] if i == 0 else [])
        row += [None, None, None, None]
        ws.append(row)

    ws2 = wb.create_sheet("osca_unit_groups")
    ws2.append(["code", "title", "skill_level", "lead_statement", "occupations"])
    for g in sorted(groups.values(), key=lambda g: g["code"]):
        ws2.append([g["code"], g["title"], g["skill_level"], g["lead"],
                    "; ".join(o["title"] for o in g["occupations"])])

    widths = {"B": 40, "C": 18, "E": 28, "J": 34, "L": 60, "P": 34, "R": 60, "T": 34, "V": 60, "X": 30, "Y": 40}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of ads from sample_jobs.py")
    ap.add_argument("--osca", default=str(DEFAULT_OSCA))
    ap.add_argument("--tag", default=None, help="output name suffix; default = input file stem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS), help="short model name; output files carry it")
    ap.add_argument("--thinking", action="store_true", help="enable DeepSeek thinking mode (slower, more output tokens)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and report sizes; no API calls")
    ap.add_argument("--fresh", action="store_true", help="discard existing output for this input+model and relabel everything")
    args = ap.parse_args()

    groups = load_osca(Path(args.osca))
    system = build_system_prompt(groups)
    ads = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    if args.limit:
        ads = ads[: args.limit]

    tag = (args.tag or Path(args.input).stem.replace("sample_", "")) + "_" + args.model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"osca_labels_{tag}.jsonl"
    xlsx_path = OUT_DIR / f"osca_review_{tag}.xlsx"

    n_occ = sum(len(g["occupations"]) for g in groups.values())
    sys_tokens = len(system) // 4
    print(f"OSCA: {len(groups)} unit groups, {n_occ} occupations; system prompt ~{sys_tokens:,} tokens")
    print(f"ads: {len(ads)} from {args.input}; model {MODELS[args.model]['id']}")

    if args.dry_run:
        print("\n--- first ad message ---\n" + ad_to_message(ads[0])[:1500] + "\n...")
        per_in = len(ad_to_message(ads[0])) // 4 + 300
        est = estimate_cost(len(ads), sys_tokens, per_in, 300, args.model)
        print(f"\nestimated cost at {MODELS[args.model]['id']} PEAK list price: ~${est:.2f} (off-peak is half); "
              f"assumes the ~{sys_tokens:,}-token OSCA prefix is served from cache after the first call")
        return

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

    # first call alone so the prefix cache is populated before the workers fan out
    if todo:
        first = todo.pop(0)
        r = label_one(system, first, args.model, args.thinking)
        results[first["id"]] = r
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        print(f"[1/{len(ads)}] {first['title'][:60]!r} -> "
              f"{[c['code'] for c in r.get('label', {}).get('candidates', [])] or r.get('error')}  usage={r.get('usage')}")

    def work(ad: dict) -> dict:
        try:
            return label_one(system, ad, args.model, args.thinking)
        except Exception as e:  # network / API errors are recorded, not fatal
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
            codes = [c["code"] for c in r.get("label", {}).get("candidates", [])] if "label" in r else r.get("error")
            print(f"[{i}/{len(ads)}] {ad['title'][:60]!r} -> {codes}")
    out.close()

    ok = [r for r in results.values() if "label" in r]
    errs = [r for r in results.values() if "label" not in r]
    usage = {k: sum(r["usage"][k] for r in ok) for k in USAGE_KEYS}
    invalid = sum(1 for r in ok for c in r["label"]["candidates"] if c["code"] not in groups)
    issues: dict[str, int] = {}
    for r in ok:
        issues[r["label"]["issue"]] = issues.get(r["label"]["issue"], 0) + 1
    print(f"\nlabelled {len(ok)}, errors {len(errs)}, invalid codes returned {invalid}, {time.time()-t0:.0f}s")
    print(f"tokens: {usage}  cost at peak rates ~${cost(usage, args.model):.2f} (off-peak half)")
    print(f"issues: {issues}")

    write_review_xlsx(ads, results, groups, xlsx_path)
    print(f"wrote {jsonl_path}\nwrote {xlsx_path}")


if __name__ == "__main__":
    main()

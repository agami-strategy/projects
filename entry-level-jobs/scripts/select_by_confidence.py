"""Per OSCA unit group, pick the highest OSCA-confidence cutoff that still leaves at least
--min-ads ads (falling back to --floor when none does), and optionally write the selected ads
as a new sample file for the seniority / task stages.

Rule: for each group, try the thresholds from highest to lowest; the first with >= min-ads
ads is chosen. If none qualifies, the floor threshold is used and all ads at or above it are
kept. Only ads with no issue flag and a valid top-1 code are considered.

Usage:
    python scripts/select_by_confidence.py data/samples/sample_us_2026-09-17.jsonl                # table only
    python scripts/select_by_confidence.py data/samples/sample_us_2026-09-17.jsonl --write        # + sample file
    python scripts/select_by_confidence.py ... --target data/osca_anzsco_target.xlsx --u25-only  # restrict groups
    python scripts/select_by_confidence.py ... --min-ads 1000 --thresholds 0.9 0.8 0.7 --floor 0.5 --cap 1500

Outputs:
    data/selection_<tag>_<model>.xlsx                     the table (one row per group)
    with --write: data/samples/sample_<tag>_selected.jsonl and data/labels/osca_labels_<tag>_selected_<model>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL, MODELS  # noqa: E402
from label_osca import DEFAULT_OSCA, load_osca, load_workbook_safely  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
SAMPLE_DIR = ROOT / "data" / "samples"


def u25_osca_groups(target_path: Path) -> set[str]:
    """OSCA unit groups linked (via the workbook's Table 1) to any ANZSCO unit with U25 = 1."""
    wb = load_workbook_safely(target_path)
    targets = {str(r[0]).strip() for r in list(wb["Sheet1"].iter_rows(values_only=True))[1:] if r[0] and str(r[3]) == "1"}
    out, src = set(), None
    for r in list(wb["Table 1"].iter_rows(values_only=True))[5:]:
        if r[1]:
            src = str(r[1]).strip()
        if src in targets and r[5]:
            out.add(str(r[5]).strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="ads JSONL that label_osca.py was run on")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--min-ads", type=int, default=1000)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.9, 0.8, 0.7])
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=None, help="random cap per group after selection (seeded)")
    ap.add_argument("--target", default=str(ROOT / "data" / "osca_anzsco_target.xlsx"))
    ap.add_argument("--u25-only", action="store_true", help="only groups linked to U25 = 1 ANZSCO units")
    ap.add_argument("--write", action="store_true", help="write the selected ads + their OSCA labels as a new sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = Path(args.input).stem.replace("sample_", "")
    groups = load_osca(Path(DEFAULT_OSCA))
    allowed = u25_osca_groups(Path(args.target)) if args.u25_only else set(groups)

    labels: dict[str, dict] = {}
    for line in open(LABEL_DIR / f"osca_labels_{tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        lab = r.get("label")
        if lab and lab["issue"] == "none" and lab["candidates"] and lab["candidates"][0]["code"] in allowed:
            labels[r["id"]] = r

    by_group: dict[str, list[tuple[float, str]]] = defaultdict(list)  # code -> [(conf, id)]
    for id_, r in labels.items():
        c = r["label"]["candidates"][0]
        by_group[c["code"]].append((c["confidence"], id_))

    ladder = sorted(set(args.thresholds), reverse=True)
    rows, selected_ids = [], set()
    rng = random.Random(args.seed)
    for code in sorted(allowed):
        items = by_group.get(code, [])
        counts = {t: sum(1 for conf, _ in items if conf >= t) for t in ladder + [args.floor]}
        chosen = next((t for t in ladder if counts[t] >= args.min_ads), None)
        rule = "min-ads met" if chosen is not None else "floor"
        chosen = chosen if chosen is not None else args.floor
        ids = [i for conf, i in items if conf >= chosen]
        if args.cap and len(ids) > args.cap:
            ids = rng.sample(ids, args.cap)
        selected_ids.update(ids)
        rows.append([code, groups[code]["title"], len(items), *(counts[t] for t in ladder), counts[args.floor],
                     chosen, rule, len(ids)])

    header = ["osca_code", "osca_title", "ads_any_conf", *(f"ads_conf>={t}" for t in ladder), f"ads_conf>={args.floor}",
              "threshold_used", "rule", "ads_selected"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "selection"
    ws.append(header)
    for r in sorted(rows, key=lambda r: -r[-1]):
        ws.append(r)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    out_xlsx = ROOT / "data" / f"selection_{tag}_{args.model}.xlsx"
    wb.save(out_xlsx)

    n_groups = sum(1 for r in rows if r[-1] > 0)
    print(f"groups considered: {len(rows)} | with ads: {n_groups} | rule: highest of {ladder} with >= {args.min_ads} ads, else floor {args.floor}")
    print(f"threshold used: {dict(Counter(r[-3] for r in rows if r[-1] > 0))} | ads selected: {len(selected_ids):,} of {len(labels):,} eligible")
    print(f"wrote {out_xlsx}")

    if args.write:
        out_sample = SAMPLE_DIR / f"sample_{tag}_selected.jsonl"
        out_labels = LABEL_DIR / f"osca_labels_{tag}_selected_{args.model}.jsonl"
        n = 0
        with open(out_sample, "w", encoding="utf-8") as fs, open(out_labels, "w", encoding="utf-8") as fl:
            for line in open(args.input, encoding="utf-8"):
                ad = json.loads(line)
                if ad["id"] in selected_ids:
                    fs.write(line if line.endswith("\n") else line + "\n")
                    fl.write(json.dumps(labels[ad["id"]], ensure_ascii=False) + "\n")
                    n += 1
        print(f"wrote {n:,} ads -> {out_sample}\nwrote matching OSCA labels -> {out_labels}")
        print(f"next: python scripts/label_seniority.py {out_sample.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

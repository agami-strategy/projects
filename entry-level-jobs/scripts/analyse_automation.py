"""Summarise automation exposure of entry-level vs experienced ads, controlling for occupation.

Input: data/labels/match_summary_<tag>_<model>.csv from match_tasks.py.

Three views, from weakest to strongest control:
  1. Raw means by level.
  2. Excess exposure: each ad's automation share minus its occupation's overall share, by level.
     This nets out occupation composition ad by ad.
  3. Within-occupation: only OSCA unit groups that contain both levels; the entry-minus-experienced
     gap per group, then averaged. Also the equivalent occupation-fixed-effects regression coefficient.

Usage:
    python scripts/analyse_automation.py                                  # default tag/model
    python scripts/analyse_automation.py --tag 100_australia --model sonnet-5
    python scripts/analyse_automation.py --measure automation_share_explicit_or_implied
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
LEVELS = ["entry_level", "experienced"]


def fnum(x: str) -> float | None:
    return None if x in ("", "None") else float(x)


def mean_sd(v: list[float]) -> str:
    if not v:
        return "n/a"
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{statistics.mean(v):.3f} (sd {sd:.3f}, n={len(v)})"


def fe_coefficient(rows: list[dict], y: str) -> tuple[float, float, int]:
    """OLS of y on entry dummy with occupation fixed effects (within transformation).
    Returns (coef, standard error, n)."""
    by_occ = defaultdict(list)
    for r in rows:
        by_occ[r["osca_code"]].append(r)
    xs, ys = [], []
    for occ, rs in by_occ.items():
        if len(rs) < 2:
            continue
        mx = statistics.mean(1.0 if r["level"] == "entry_level" else 0.0 for r in rs)
        my = statistics.mean(r[y] for r in rs)
        for r in rs:
            xs.append((1.0 if r["level"] == "entry_level" else 0.0) - mx)
            ys.append(r[y] - my)
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), len(xs)
    b = sum(x * yy for x, yy in zip(xs, ys)) / sxx
    resid = [yy - b * x for x, yy in zip(xs, ys)]
    n_groups = sum(1 for rs in by_occ.values() if len(rs) >= 2)
    dof = max(len(xs) - n_groups - 1, 1)
    se = (sum(e * e for e in resid) / dof / sxx) ** 0.5
    return b, se, len(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="100_australia")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--measure", default="automation_share_explicit",
                    choices=["automation_share_explicit", "automation_share_explicit_or_implied",
                             "augmentation_share_explicit", "augmentation_share_explicit_or_implied"])
    args = ap.parse_args()
    occ_col = "automation_share_occupation" if args.measure.startswith("automation") else "augmentation_share_occupation"

    path = LABEL_DIR / f"match_summary_{args.tag}_{args.model}.csv"
    raw = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    rows = []
    for r in raw:
        y, occ = fnum(r[args.measure]), fnum(r[occ_col])
        if r["error"] or y is None or occ is None or r["level"] not in LEVELS:
            continue
        rows.append({**r, "y": y, "occ": occ, "excess": y - occ})
    dropped = len(raw) - len(rows)
    print(f"measure: {args.measure} | ads used {len(rows)} (dropped {dropped}: errors, unclear level, or no matched tasks)")
    print(f"levels: { {l: sum(1 for r in rows if r['level'] == l) for l in LEVELS} }\n")

    print("1. Raw mean of the measure by level")
    for l in LEVELS:
        print(f"   {l:12s} {mean_sd([r['y'] for r in rows if r['level'] == l])}")
    print(f"   occupation baseline (mean over ads): {mean_sd([r['occ'] for r in rows])}\n")

    print("2. Excess exposure = ad measure minus its occupation's overall share, by level")
    ex = {l: [r["excess"] for r in rows if r["level"] == l] for l in LEVELS}
    for l in LEVELS:
        print(f"   {l:12s} {mean_sd(ex[l])}")
    if all(len(ex[l]) > 1 for l in LEVELS):
        diff = statistics.mean(ex["entry_level"]) - statistics.mean(ex["experienced"])
        se = (statistics.variance(ex["entry_level"]) / len(ex["entry_level"]) + statistics.variance(ex["experienced"]) / len(ex["experienced"])) ** 0.5
        print(f"   entry minus experienced: {diff:+.3f} (se {se:.3f}, t={diff / se:.2f})\n")

    print("3. Within-occupation: OSCA unit groups containing both levels")
    by_occ = defaultdict(lambda: defaultdict(list))
    titles = {}
    for r in rows:
        by_occ[r["osca_code"]][r["level"]].append(r["y"])
        titles[r["osca_code"]] = r["osca_title"]
    gaps = []
    for occ, d in sorted(by_occ.items()):
        if all(d[l] for l in LEVELS):
            g = statistics.mean(d["entry_level"]) - statistics.mean(d["experienced"])
            gaps.append(g)
            print(f"   {occ} {titles[occ][:38]:38s} entry {statistics.mean(d['entry_level']):.3f} (n={len(d['entry_level'])})  "
                  f"exp {statistics.mean(d['experienced']):.3f} (n={len(d['experienced'])})  gap {g:+.3f}")
    if gaps:
        print(f"   mean gap across {len(gaps)} groups: {statistics.mean(gaps):+.3f}; groups with entry > experienced: {sum(1 for g in gaps if g > 0)}/{len(gaps)}")
    else:
        print("   no unit group has ads at both levels")
    b, se, n = fe_coefficient(rows, "y")
    print(f"   occupation fixed-effects coefficient on entry_level: {b:+.3f} (se {se:.3f}, n={n} ads in groups with 2+ ads)")


if __name__ == "__main__":
    main()

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
# %% Imports and constants
# This is the earlier pilot analysis, written for the small Australian sample (default tag
# "100_australia") before the regression scripts existed. It uses only the standard library (csv,
# statistics) and reads a single CSV, so it is a quick sanity check rather than the main analysis.
# regress_automation.py / regress_full.py supersede it for the reported results; the three views here
# are useful for seeing how the answer changes as occupation control is tightened.
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# DEFAULT_MODEL is taken from label_osca (the OSCA-labelling script) rather than llm.py here; both
# expose the same model name used in file tags.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
# the two seniority levels compared; ads with any other level (e.g. unclear) are dropped
LEVELS = ["entry_level", "experienced"]


# %% Small helpers
# fnum: the CSV stores missing numbers as "" or "None" (str(None) from an earlier writer); both -> None.
def fnum(x: str) -> float | None:
    return None if x in ("", "None") else float(x)


# mean_sd: formats mean, sample SD and n for printing; SD is 0 for a single observation because
# statistics.stdev needs at least two values.
def mean_sd(v: list[float]) -> str:
    if not v:
        return "n/a"
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{statistics.mean(v):.3f} (sd {sd:.3f}, n={len(v)})"


# %% Hand-rolled occupation fixed-effects coefficient
# This reproduces the OLS coefficient on `entry` from y ~ entry + C(osca_code) without statsmodels,
# using the within (demeaning) transformation: subtract each occupation's mean from both x (the entry
# dummy) and y, then regress demeaned y on demeaned x with no intercept. By the Frisch-Waugh-Lovell
# theorem this gives exactly the fixed-effects coefficient. Occupations with a single ad are skipped
# (their demeaned values would be zero and contribute nothing). The SE is the classical homoskedastic
# one, with degrees of freedom n - (number of groups) - 1 to account for the absorbed group means;
# regress_automation.py uses heteroskedasticity-robust SEs instead, so its SEs will differ.
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
        # mx is the occupation's entry share; my its mean outcome
        mx = statistics.mean(1.0 if r["level"] == "entry_level" else 0.0 for r in rs)
        my = statistics.mean(r[y] for r in rs)
        for r in rs:
            xs.append((1.0 if r["level"] == "entry_level" else 0.0) - mx)
            ys.append(r[y] - my)
    # sxx = 0 means no occupation has both levels, so the entry effect is not identified
    sxx = sum(x * x for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), len(xs)
    # b = sum(x~ y~) / sum(x~^2): the no-intercept OLS slope on the demeaned data
    b = sum(x * yy for x, yy in zip(xs, ys)) / sxx
    resid = [yy - b * x for x, yy in zip(xs, ys)]
    n_groups = sum(1 for rs in by_occ.values() if len(rs) >= 2)
    dof = max(len(xs) - n_groups - 1, 1)
    # var(b) = sigma^2 / sxx with sigma^2 = RSS / dof
    se = (sum(e * e for e in resid) / dof / sxx) ** 0.5
    return b, se, len(xs)


# %% Command-line entry point
def main() -> None:
    # %% Arguments and input
    # --measure selects the per-ad column to analyse; the matching occupation baseline column is chosen
    # by prefix (automation_* -> automation_share_occupation; augmentation_* -> the augmentation one).
    # These are time-weighted shares computed by match_tasks.py, not the count shares of the later scripts.
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="100_australia")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--measure", default="automation_share_explicit",
                    choices=["automation_share_explicit", "automation_share_explicit_or_implied",
                             "augmentation_share_explicit", "augmentation_share_explicit_or_implied"])
    args = ap.parse_args()
    occ_col = "automation_share_occupation" if args.measure.startswith("automation") else "augmentation_share_occupation"

    # Each usable row gets y (the ad's measure), occ (the occupation's overall share, i.e. the
    # time-weighted automation share over ALL of the occupation's tasks) and excess = y - occ.
    # Ads are dropped if match_tasks recorded an error, the measure or baseline is missing (no matched
    # tasks), or the level is not one of the two compared.
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

    # %% View 1: raw means by level
    # No occupation control: any difference here mixes "entry ads describe different tasks within an
    # occupation" with "entry ads are concentrated in different occupations".
    print("1. Raw mean of the measure by level")
    for l in LEVELS:
        print(f"   {l:12s} {mean_sd([r['y'] for r in rows if r['level'] == l])}")
    print(f"   occupation baseline (mean over ads): {mean_sd([r['occ'] for r in rows])}\n")

    # %% View 2: excess exposure over the occupation baseline
    # Subtracting the occupation's overall share from each ad removes the between-occupation level but
    # not the within-occupation composition. The entry-minus-experienced difference of mean excess is
    # tested with a Welch-style SE (sum of the two sample variances over n), treating ads as independent.
    print("2. Excess exposure = ad measure minus its occupation's overall share, by level")
    ex = {l: [r["excess"] for r in rows if r["level"] == l] for l in LEVELS}
    for l in LEVELS:
        print(f"   {l:12s} {mean_sd(ex[l])}")
    if all(len(ex[l]) > 1 for l in LEVELS):
        diff = statistics.mean(ex["entry_level"]) - statistics.mean(ex["experienced"])
        se = (statistics.variance(ex["entry_level"]) / len(ex["entry_level"]) + statistics.variance(ex["experienced"]) / len(ex["experienced"])) ** 0.5
        print(f"   entry minus experienced: {diff:+.3f} (se {se:.3f}, t={diff / se:.2f})\n")

    # %% View 3: within-occupation gaps and the fixed-effects coefficient
    # Only occupations with ads at both levels can speak to the within-occupation question. The per-group
    # gap is the difference in mean y; the "mean gap across groups" weights every occupation equally,
    # whereas the FE coefficient below weights occupations by n_entry * n_exp / n. The count of groups
    # with a positive gap is a crude sign test.
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
    # the FE coefficient uses groups with 2+ ads (not necessarily both levels); single-level groups
    # simply contribute zero to sxx, so this matches the pooled model of regress_automation.py
    b, se, n = fe_coefficient(rows, "y")
    print(f"   occupation fixed-effects coefficient on entry_level: {b:+.3f} (se {se:.3f}, n={n} ads in groups with 2+ ads)")


if __name__ == "__main__":
    main()

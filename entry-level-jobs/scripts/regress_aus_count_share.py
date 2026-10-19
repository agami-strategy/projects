"""Replicate the Australian count-share regressions shown on slide 12 of the job-ads methodology deck.

The Australian match summary (match_summary_aus_all_flash.csv) was written by an earlier version of
match_tasks.py and has no count-share columns. This script rebuilds them from the raw task judgements
using the same rule as summarise() in match_tasks.py, then fits the same models used for the US:

  count shares   a task is "largely automatable" if automation_time / TIME_SHARE >= 0.5
                 (likewise augmentable); share = largely-X matched tasks / matched tasks,
                 matched = explicit + implied
  pooled model   y ~ C(osca_code) + entry                   OLS, HC1 standard errors
  interaction    y ~ C(osca_code) + entry:C(osca_code)      OLS, HC1, occupations with >= 10 ads per level

Usage (from the entry-level-jobs folder):
    python scripts/regress_aus_count_share.py
    python scripts/regress_aus_count_share.py --tag aus_all --model flash --min-per-level 10

Reads only; writes nothing unless --out is given (a CSV of the per-ad count shares).
Needs pandas and statsmodels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_task_flags(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """(OSCA code, task id) -> (largely automatable, largely augmentable)."""
    t = pd.read_csv(path, encoding="utf-8-sig", dtype={"OSCA_UNIT_CODE": str, "Task ID": str})
    ts = t["TIME_SHARE"]
    auto = np.where(ts > 0, (t["automation_time"] / ts >= 0.5).astype(int), 0)
    aug = np.where(ts > 0, (t["augmentation_time"] / ts >= 0.5).astype(int), 0)
    return {(c, k): (int(a), int(g)) for c, k, a, g in zip(t["OSCA_UNIT_CODE"], t["Task ID"], auto, aug)}


def count_shares(labels: Path, summary: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Per-ad count shares over explicit+implied tasks (and automation over explicit only)."""
    code = dict(zip(summary["ad_id"], summary["osca_code"]))
    rows = {}
    with open(labels, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "judgements" not in r or r["id"] not in code:
                continue
            c = code[r["id"]]
            n_m = n_e = a_m = g_m = a_e = 0
            for tid, j in r["judgements"].items():
                if (c, tid) not in flags:  # ids outside the occupation's task list are ignored, as in summarise()
                    continue
                a, g = flags[(c, tid)]
                if j["judgement"] in ("explicit", "implied"):
                    n_m += 1; a_m += a; g_m += g
                    if j["judgement"] == "explicit":
                        n_e += 1; a_e += a
            rows[r["id"]] = {  # last record per id wins (resumed runs can repeat an id)
                "ad_id": r["id"],
                "automation_count_share": a_m / n_m if n_m else np.nan,
                "augmentation_count_share": g_m / n_m if n_m else np.nan,
                "automation_count_share_explicit": a_e / n_e if n_e else np.nan,
            }
    return summary.merge(pd.DataFrame(rows.values()), on="ad_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="aus_all")
    ap.add_argument("--model", default="flash")
    ap.add_argument("--min-per-level", type=int, default=10, help="interaction model: min ads per level per occupation")
    ap.add_argument("--out", default=None, help="optional CSV path for per-ad count shares")
    args = ap.parse_args()

    tag = f"{args.tag}_{args.model}"
    summary = pd.read_csv(DATA / "labels" / f"match_summary_{tag}.csv", encoding="utf-8-sig", dtype={"osca_code": str})
    flags = load_task_flags(DATA / "osca_task_share.csv")
    d = count_shares(DATA / "labels" / f"match_labels_{tag}.jsonl", summary, flags)
    if args.out:
        d.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"ads in match summary: {len(d):,} | occupations: {d['osca_code'].nunique()} | "
          f"no matched task: {d['automation_count_share'].isna().sum()}")

    # raw means by level
    cols = ["automation_count_share", "augmentation_count_share", "automation_share_explicit_or_implied"]
    print("\nRaw means by level")
    print(d.groupby("level")[cols].mean().round(3).T.to_string())

    # pooled within-occupation models
    dd = d.dropna(subset=["automation_count_share"]).copy()
    dd["entry"] = (dd["level"] == "entry_level").astype(int)
    print(f"\nPooled model y ~ C(osca_code) + entry, HC1 (n = {len(dd):,}, occupations = {dd['osca_code'].nunique()})")
    for y in cols:
        m = smf.ols(f"{y} ~ C(osca_code) + entry", dd).fit(cov_type="HC1")
        print(f"  {y:40s} b = {m.params['entry']:+.4f}  se = {m.bse['entry']:.4f}  p = {m.pvalues['entry']:.4f}")

    # occupation-specific entry effects
    cnt = dd.groupby(["osca_code", "level"]).size().unstack(fill_value=0)
    keep = cnt[(cnt.get("entry_level", 0) >= args.min_per_level) & (cnt.get("experienced", 0) >= args.min_per_level)].index
    e = dd[dd["osca_code"].isin(keep)]
    m = smf.ols("automation_count_share ~ C(osca_code) + entry:C(osca_code)", e).fit(cov_type="HC1")
    b = m.params.filter(like="entry:")
    p = m.pvalues.filter(like="entry:")
    ci = m.conf_int().loc[b.index]
    titles = e.groupby("osca_code")["osca_title"].first()
    print(f"\nInteraction model, automation_count_share, HC1 (occupations with >= {args.min_per_level} ads per level: "
          f"{len(keep)}, n = {len(e):,})")
    print(f"  positive and significant at 5%: {((b > 0) & (p < 0.05)).sum()}")
    print(f"  negative and significant at 5%: {((b < 0) & (p < 0.05)).sum()}")
    print(f"  mean effect (unweighted): {b.mean():+.4f}")
    out = pd.DataFrame({
        "osca_code": [k.split("[")[1].rstrip("]") for k in b.index],
        "b": b.values.round(4), "ci_low": ci[0].values.round(4), "ci_high": ci[1].values.round(4), "p": p.values.round(4),
    })
    out["title"] = out["osca_code"].map(titles)
    out["n_entry"] = out["osca_code"].map(cnt["entry_level"])
    out["n_experienced"] = out["osca_code"].map(cnt["experienced"])
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

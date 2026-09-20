"""Occupation fixed-effects regressions of advertised automation exposure on entry-level status.
Standard errors are heteroskedasticity-robust by default (--se classical / --se cluster to change).

    y_i = a + b * entry_level_i + occupation FE + e_i

Outcomes (per ad, from match_tasks.py output joined to the task file):
  Count-based, "largely X" tasks (a task is largely automatable if automation time is >= 50% of its time):
    automation_count_share       share of explicit+implied tasks that are largely automatable
    augmentation_count_share     ... largely augmentable
    unaffected_count_share       ... largely unaffected
    automation_count_share_explicit   same as the first, over explicit tasks only
    aug_minus_auto_count_share        augmentation_count_share minus automation_count_share (net tilt to augmentation)
    aug_minus_auto_count_share_explicit   the same over explicit tasks only
  Time-weighted (main):
    auto_share_explicit         automation time / total time, over tasks the ad explicitly describes
    auto_share_expl_impl        same, over explicit + implied tasks
    auto_time_explicit          automation time in explicit tasks (level, share of occupation's working time)
    time_share_explicit         total time in explicit tasks (coverage; how much of the job the ad describes)
  Count-based (robustness):
    auto_frac_mean_explicit     unweighted mean over explicit tasks of each task's automation fraction
    auto_flag_share_explicit    share of explicit tasks flagged automatable (ai_dummy = 1)
    n_explicit                  number of explicit tasks
Robustness rows: excluding internships; occupations with >= 20 ads per level.

Usage:
    python scripts/regress_automation.py --tag us_2026-09-17_selected_tasks --model flash
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
TASK_FILE = ROOT / "data" / "osca_task_share.csv"

SE_NOTE = {"hc1": "heteroskedasticity-robust (HC1)", "classical": "classical OLS", "cluster": "clustered on occupation",
           "employer": "clustered on employer (source_slug)"}


def fit_kwargs(se: str, d: pd.DataFrame) -> dict:
    if se == "cluster":
        return {"cov_type": "cluster", "cov_kwds": {"groups": d["osca_code"]}}
    if se == "employer":
        return {"cov_type": "cluster", "cov_kwds": {"groups": d["company"].fillna("(unknown)")}}
    if se == "classical":
        return {}
    return {"cov_type": "HC1"}


SCALE_NOTE = {
    "level": "level difference (entry minus experienced, in units of the outcome)",
    "ratio": "proportional difference: Poisson quasi-likelihood with log link, exp(coef) = entry / experienced ratio of means",
}


def fit_model(formula: str, d: pd.DataFrame, se: str, scale: str):
    """OLS for level differences; Poisson QML (log link) for proportional differences. The Poisson
    estimator is valid for non-negative fractional outcomes and copes with zeros, unlike log(y). With
    a saturated fixed-effects model its coefficients equal the log ratio of cell means."""
    if scale == "ratio":
        y = formula.split("~")[0].strip()
        if (d[y] < 0).any():
            raise ValueError(f"{y} takes negative values; a proportional (log-link) model is not defined for it")
        return smf.glm(formula, data=d, family=sm.families.Poisson()).fit(**fit_kwargs(se, d))
    return smf.ols(formula, data=d).fit(**fit_kwargs(se, d))


def build_ad_table(tag: str, model: str) -> pd.DataFrame:
    summ = pd.read_csv(LABEL_DIR / f"match_summary_{tag}_{model}.csv", encoding="utf-8-sig", dtype={"osca_code": str})
    summ = summ[summ["error"].isna() & summ["level"].isin(["entry_level", "experienced"])].copy()

    tasks = {}
    with open(TASK_FILE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ts = float(r["TIME_SHARE"])
            tasks[(r["OSCA_UNIT_CODE"], r["Task ID"])] = {
                "time": ts, "auto": float(r["automation_time"]),
                "frac": float(r["automation_time"]) / ts if ts else np.nan,
                "flag": int(r["ai_dummy"]) if r["ai_dummy"] not in ("", None) else np.nan,
                # "largely X" = X accounts for at least half of the task's time (same rule as the workbook columns)
                "largely_auto": float(r["automation_time"]) / ts >= 0.5 if ts else np.nan,
                "largely_aug": float(r["augmentation_time"]) / ts >= 0.5 if ts else np.nan,
                "largely_unaff": float(r["unaffected_time"]) / ts >= 0.5 if ts else np.nan,
            }
    intern = {}
    for line in open(LABEL_DIR / f"seniority_labels_{tag}_{model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r:
            intern[r["id"]] = bool(r["label"]["is_internship"])
    code_of = dict(zip(summ["ad_id"], summ["osca_code"]))

    count_rows = {}
    for line in open(LABEL_DIR / f"match_labels_{tag}_{model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "judgements" not in r or r["id"] not in code_of:
            continue
        code = code_of[r["id"]]
        fr, fl = [], []                       # explicit tasks: automation fraction, automatable flag
        ex = {"largely_auto": [], "largely_aug": [], "largely_unaff": []}      # explicit only
        ei = {"largely_auto": [], "largely_aug": [], "largely_unaff": []}      # explicit or implied
        for tid, j in r["judgements"].items():
            if j["judgement"] not in ("explicit", "implied"):
                continue
            t = tasks.get((code, tid))
            if not t:
                continue
            for k in ei:
                ei[k].append(t[k])
            if j["judgement"] == "explicit":
                fr.append(t["frac"]); fl.append(t["flag"])
                for k in ex:
                    ex[k].append(t[k])
        mean = lambda v: float(np.nanmean(v)) if v else np.nan  # noqa: E731
        count_rows[r["id"]] = {
            "auto_frac_mean_explicit": mean(fr),
            "auto_flag_share_explicit": mean(fl),
            # share of matched tasks that are largely automatable / augmentable / unaffected
            "automation_count_share": mean(ei["largely_auto"]),          # explicit + implied (workbook definition)
            "augmentation_count_share": mean(ei["largely_aug"]),
            "unaffected_count_share": mean(ei["largely_unaff"]),
            "automation_count_share_explicit": mean(ex["largely_auto"]),  # explicit only
            "augmentation_count_share_explicit": mean(ex["largely_aug"]),
            "unaffected_count_share_explicit": mean(ex["largely_unaff"]),
        }
    cr = pd.DataFrame.from_dict(count_rows, orient="index")
    # newer match_tasks.py output already carries the count-share columns; recompute here regardless so
    # older outputs work too, and drop the CSV copies to avoid a join collision
    summ = summ.drop(columns=[c for c in cr.columns if c in summ.columns])
    df = summ.set_index("ad_id").join(cr)
    df["entry"] = (df["level"] == "entry_level").astype(int)
    df["internship"] = df.index.map(lambda i: intern.get(i, False)).astype(int)
    df["auto_share_explicit"] = df["automation_share_explicit"]
    df["auto_share_expl_impl"] = df["automation_share_explicit_or_implied"]
    df["auto_time_explicit"] = df["automation_share_explicit"] * df["time_share_explicit"]
    # net tilt of the described tasks towards augmentation over automation (positive = more augmentable)
    df["aug_minus_auto_count_share"] = df["augmentation_count_share"] - df["automation_count_share"]
    df["aug_minus_auto_count_share_explicit"] = df["augmentation_count_share_explicit"] - df["automation_count_share_explicit"]
    return df


def fit_fe(df: pd.DataFrame, y: str, se: str = "hc1", scale: str = "level") -> dict:
    d = df.dropna(subset=[y])
    # keep occupations with variation in entry status; others contribute nothing to b
    keep = d.groupby("osca_code")["entry"].nunique()
    d = d[d["osca_code"].isin(keep[keep > 1].index)]
    base = {"outcome": y, "n_ads": len(d), "n_occupations": d["osca_code"].nunique(),
            "mean_experienced": d.loc[d["entry"] == 0, y].mean(), "mean_entry": d.loc[d["entry"] == 1, y].mean()}
    try:
        m = fit_model(f"{y} ~ entry + C(osca_code)", d, se, scale)
    except ValueError:
        return {**base, "coef_entry": np.nan, "pct_diff": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}
    b = m.params["entry"]
    return {**base, "coef_entry": b, "pct_diff": (np.exp(b) - 1) if scale == "ratio" else np.nan,
            "se": m.bse["entry"], "t": m.tvalues["entry"], "p": m.pvalues["entry"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="us_2026-09-17_selected_tasks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--min-per-level", type=int, default=20, help="threshold for the 'deep occupations' robustness row")
    ap.add_argument("--se", default="hc1", choices=["hc1", "classical", "cluster", "employer"],
                    help="standard errors: heteroskedasticity-robust (default), classical OLS, clustered on occupation, or clustered on employer")
    ap.add_argument("--scale", default="level", choices=["level", "ratio"],
                    help="level: coefficient is the difference in the outcome (default); ratio: proportional difference, "
                         "reported as pct_diff = entry/experienced - 1 (Poisson QML, log link)")
    args = ap.parse_args()

    df = build_ad_table(args.tag, args.model)
    print(f"ads: {len(df):,} | occupations: {df['osca_code'].nunique()} | entry: {int(df['entry'].sum()):,} | "
          f"experienced: {int((df['entry']==0).sum()):,} | internships: {int(df['internship'].sum()):,}")
    print(f"ads with no explicit task (dropped from explicit-share outcomes): {int(df['auto_share_explicit'].isna().sum()):,}\n")

    outcomes = ["automation_count_share", "augmentation_count_share", "unaffected_count_share",
                "aug_minus_auto_count_share", "automation_count_share_explicit", "aug_minus_auto_count_share_explicit",
                "auto_share_explicit", "auto_share_expl_impl", "auto_time_explicit", "time_share_explicit",
                "auto_frac_mean_explicit", "auto_flag_share_explicit", "n_explicit"]
    per_level = df.groupby(["osca_code", "entry"]).size().unstack(fill_value=0)
    deep = per_level[(per_level[0] >= args.min_per_level) & (per_level[1] >= args.min_per_level)].index
    samples = {"all": df, "excluding internships": df[df["internship"] == 0],
               f"occupations with >= {args.min_per_level} per level": df[df["osca_code"].isin(deep)]}

    rows = []
    for sname, d in samples.items():
        for y in outcomes:
            r = fit_fe(d, y, args.se, args.scale); r["sample"] = sname; rows.append(r)
    cols = ["sample", "outcome", "coef_entry", "pct_diff", "se", "t", "p", "n_ads", "n_occupations", "mean_entry", "mean_experienced"]
    if args.scale == "level":
        cols.remove("pct_diff")
    res = pd.DataFrame(rows)[cols]
    print(f"standard errors: {SE_NOTE[args.se]}\nscale: {SCALE_NOTE[args.scale]}"
          + ("\n(coef_entry is the log ratio; outcomes with negative values are skipped)\n" if args.scale == "ratio" else "\n"))
    out = ROOT / "data" / f"regression_{args.tag}_{args.model}{'_ratio' if args.scale == 'ratio' else ''}.csv"
    res.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    for sname in samples:
        print(f"=== {sname} ===")
        sub = res[res["sample"] == sname].drop(columns="sample")
        print(sub.to_string(index=False, formatters={
            "coef_entry": "{:+.4f}".format, "pct_diff": "{:+.1%}".format, "se": "{:.4f}".format, "t": "{:+.2f}".format, "p": "{:.3f}".format,
            "mean_entry": "{:.4f}".format, "mean_experienced": "{:.4f}".format}))
        print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

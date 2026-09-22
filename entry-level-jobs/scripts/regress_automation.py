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
# %% Imports and paths
# This script is the pooled ("one entry coefficient") regression stage of the project. It turns the
# per-ad task-matching output into an analysis table and fits y ~ entry + C(osca_code) for every outcome.
# It is also imported by regress_full.py (build_ad_table, fit_model, SE_NOTE, SCALE_NOTE), so changes to
# the table construction here propagate to the main interaction-model script.
# Dependencies: pandas for the table, statsmodels for OLS / GLM via the R-style formula API (smf).
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

# The sibling module llm.py holds DEFAULT_MODEL (the LLM name used in output file tags, e.g. "flash").
# Adding scripts/ to sys.path lets the script run from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL  # noqa: E402

# ROOT is the repository root (scripts/ is one level below it). Inputs live in data/labels/ (match_tasks
# output) and data/osca_task_share.csv (per-occupation task list with time shares; convert_tasks_to_osca.py).
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
TASK_FILE = ROOT / "data" / "osca_task_share.csv"

# Human-readable label for each --se choice; printed in the output header so the reader knows which
# standard errors the table reports.
SE_NOTE = {"hc1": "heteroskedasticity-robust (HC1)", "classical": "classical OLS", "cluster": "clustered on occupation",
           "employer": "clustered on employer (source_slug)"}


# %% Standard-error options
# Maps the --se flag to the keyword arguments statsmodels' .fit() needs. Each is a different assumption
# about the error structure:
#   hc1       (default) heteroskedasticity-robust; allows the error variance to differ across ads but
#             treats ads as independent. Sensible for shares in [0, 1] whose variance depends on the mean.
#   classical homoskedastic OLS; the smallest SEs, used only as a comparison.
#   cluster   one cluster per occupation; allows arbitrary correlation among ads in the same occupation.
#             With few occupations the cluster-robust asymptotics can be poor (few-clusters problem).
#   employer  one cluster per company; ads from the same employer are often near-duplicates, so this is
#             the most conservative option. Missing company names are pooled into one "(unknown)" cluster,
#             which understates their independence slightly; a reviewer could instead drop them.
# Note: the returned dict is unpacked into .fit(**kwargs), so for classical SEs it is simply empty.
def fit_kwargs(se: str, d: pd.DataFrame) -> dict:
    if se == "cluster":
        return {"cov_type": "cluster", "cov_kwds": {"groups": d["osca_code"]}}
    if se == "employer":
        return {"cov_type": "cluster", "cov_kwds": {"groups": d["company"].fillna("(unknown)")}}
    if se == "classical":
        return {}
    return {"cov_type": "HC1"}


# Printed alongside SE_NOTE so the reader knows whether coefficients are level differences or log ratios.
SCALE_NOTE = {
    "level": "level difference (entry minus experienced, in units of the outcome)",
    "ratio": "proportional difference: Poisson quasi-likelihood with log link, exp(coef) = entry / experienced ratio of means",
}


# %% Model fitting: OLS (level) or Poisson QML (ratio)
# One entry point for both scales so the callers (fit_fe below and regress_full.py) do not repeat the
# choice. `formula` is a patsy/R-style string such as "auto_share_explicit ~ entry + C(osca_code)":
# the left-hand side is the outcome column, C(...) turns a column into a set of dummies (one occupation
# is the omitted reference), and `entry` is 0/1, so its coefficient is the entry-minus-experienced gap.
#
# The "ratio" scale uses the Poisson pseudo-maximum-likelihood trick: with a log link, E[y|x] = exp(x'b),
# and the Poisson score equations are consistent for b whenever that conditional-mean model is right,
# regardless of whether y is integer or Poisson-distributed (Gourieroux, Monfort & Trognon 1984;
# Santos Silva & Tenreyro 2006). So fractional outcomes with exact zeros are fine, unlike log(y).
# With a saturated fixed-effects specification the fitted cell means equal the sample cell means, so
# exp(coef on entry) is literally the ratio of the entry mean to the experienced mean within occupation.
# The SE options apply unchanged (statsmodels GLM accepts the same cov_type arguments), and robust SEs
# are needed because the Poisson variance assumption (var = mean) is not meant to hold.
# Negative outcomes (e.g. aug_minus_auto_count_share) have no ratio interpretation, hence the ValueError,
# which fit_fe catches and reports as NaN.
def fit_model(formula: str, d: pd.DataFrame, se: str, scale: str):
    """OLS for level differences; Poisson QML (log link) for proportional differences. The Poisson
    estimator is valid for non-negative fractional outcomes and copes with zeros, unlike log(y). With
    a saturated fixed-effects model its coefficients equal the log ratio of cell means."""
    if scale == "ratio":
        # the outcome name is the text left of "~"; used to check the sign restriction
        y = formula.split("~")[0].strip()
        if (d[y] < 0).any():
            raise ValueError(f"{y} takes negative values; a proportional (log-link) model is not defined for it")
        return smf.glm(formula, data=d, family=sm.families.Poisson()).fit(**fit_kwargs(se, d))
    return smf.ols(formula, data=d).fit(**fit_kwargs(se, d))


# %% Build the per-ad analysis table
# Inputs (all keyed by the run tag and LLM model name):
#   match_summary_<tag>_<model>.csv    one row per ad from match_tasks.py: osca_code, level, company,
#                                      time-weighted automation shares, n_explicit, time_share_explicit, error
#   match_labels_<tag>_<model>.jsonl   one JSON object per ad with "judgements": {task_id: {"judgement": ...}}
#                                      where judgement is explicit / implied / (something else = not required)
#   seniority_labels_<tag>_<model>.jsonl  per-ad LLM seniority label incl. is_internship
#   data/osca_task_share.csv           one row per (OSCA unit group, task) with TIME_SHARE and its split into
#                                      automation_time / augmentation_time / unaffected_time, plus ai_dummy
# Output: a DataFrame indexed by ad_id with one row per usable ad and all outcome columns listed in the
# module docstring, plus `entry` (1 = entry-level) and `internship` (1 = flagged as internship).
def build_ad_table(tag: str, model: str) -> pd.DataFrame:
    # osca_code must stay a string: codes have leading structure (4-digit unit groups) and pandas would
    # otherwise parse them as integers. utf-8-sig strips a BOM if the CSV was saved from Excel.
    summ = pd.read_csv(LABEL_DIR / f"match_summary_{tag}_{model}.csv", encoding="utf-8-sig", dtype={"osca_code": str})
    # Keep ads that matched without error and have a definite level. Ads labelled "unclear" (or any
    # other level) are dropped here, so the comparison is strictly entry_level vs experienced.
    summ = summ[summ["error"].isna() & summ["level"].isin(["entry_level", "experienced"])].copy()

    # %% Task attributes keyed by (occupation, task id)
    # Every task has a TIME_SHARE (share of the occupation's working time) split into three components.
    # "largely X" means component X is at least half of the task's time, which is the same rule the
    # source workbook uses for its own automatable / augmentable / unaffected columns. The 50% cut-off is
    # a modelling choice: because the three components sum to the task's time, at most one of the three
    # "largely" flags can be true for a task, but a task can be "largely nothing" (e.g. 40/30/30).
    # A zero time share gives NaN for the fractions so such tasks drop out of the means below.
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
    # %% Internship flag and occupation lookup
    # The seniority labels file may contain error records without a "label" key; those ads simply get
    # internship = False below (via intern.get(i, False)). Internships are a robustness exclusion because
    # they are entry-level by construction but may describe atypical task bundles.
    intern = {}
    for line in open(LABEL_DIR / f"seniority_labels_{tag}_{model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r:
            intern[r["id"]] = bool(r["label"]["is_internship"])
    # ad_id -> OSCA code, needed to look up the right occupation's task list for each ad's judgements
    code_of = dict(zip(summ["ad_id"], summ["osca_code"]))

    # %% Count-based outcomes from the task judgements
    # For each ad, walk its task judgements and collect the attributes of the tasks it requires. Two
    # task sets are tracked: explicit-or-implied ("ei", the workbook definition) and explicit-only ("ex").
    # Each outcome is then the mean of a boolean/fraction over that set, i.e. a share of matched tasks.
    # Tasks are counted equally here regardless of their time share (contrast with the time-weighted
    # shares that come from match_summary). Ads with no required task in a set get NaN, not 0.
    count_rows = {}
    for line in open(LABEL_DIR / f"match_labels_{tag}_{model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        # skip error records and ads that were filtered out of `summ` (errors / unclear level)
        if "judgements" not in r or r["id"] not in code_of:
            continue
        code = code_of[r["id"]]
        fr, fl = [], []                       # explicit tasks: automation fraction, automatable flag
        ex = {"largely_auto": [], "largely_aug": [], "largely_unaff": []}      # explicit only
        ei = {"largely_auto": [], "largely_aug": [], "largely_unaff": []}      # explicit or implied
        for tid, j in r["judgements"].items():
            if j["judgement"] not in ("explicit", "implied"):
                continue
            # a judgement can reference a task id that is not in the occupation's task list (e.g. the
            # task file was regenerated); such tasks are silently ignored
            t = tasks.get((code, tid))
            if not t:
                continue
            for k in ei:
                ei[k].append(t[k])
            if j["judgement"] == "explicit":
                fr.append(t["frac"]); fl.append(t["flag"])
                for k in ex:
                    ex[k].append(t[k])
        # nanmean ignores tasks whose attribute is NaN (zero time share or missing ai_dummy);
        # an empty list yields NaN so the ad is excluded from that outcome's regression
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
    # %% Join and derive the remaining outcome columns
    # count_rows (dict of dicts) becomes a DataFrame indexed by ad_id; the left join onto summ keeps every
    # ad in summ, with NaN count shares for ads that had no judgements record.
    cr = pd.DataFrame.from_dict(count_rows, orient="index")
    # newer match_tasks.py output already carries the count-share columns; recompute here regardless so
    # older outputs work too, and drop the CSV copies to avoid a join collision
    summ = summ.drop(columns=[c for c in cr.columns if c in summ.columns])
    df = summ.set_index("ad_id").join(cr)
    # the regressor of interest: 1 for entry-level ads, 0 for experienced
    df["entry"] = (df["level"] == "entry_level").astype(int)
    df["internship"] = df.index.map(lambda i: intern.get(i, False)).astype(int)
    # Time-weighted shares come straight from match_summary (computed by match_tasks.py); they are renamed
    # here to the shorter outcome names used in the results tables.
    df["auto_share_explicit"] = df["automation_share_explicit"]
    df["auto_share_expl_impl"] = df["automation_share_explicit_or_implied"]
    # share x coverage = automation time in explicit tasks as a share of the occupation's total working
    # time (a level, not a within-described-tasks share); it rises with both automatability and coverage
    df["auto_time_explicit"] = df["automation_share_explicit"] * df["time_share_explicit"]
    # net tilt of the described tasks towards augmentation over automation (positive = more augmentable)
    df["aug_minus_auto_count_share"] = df["augmentation_count_share"] - df["automation_count_share"]
    df["aug_minus_auto_count_share_explicit"] = df["augmentation_count_share_explicit"] - df["automation_count_share_explicit"]
    return df


# %% Fit one pooled fixed-effects regression
# Fits y ~ entry + C(osca_code) on the ads with a non-missing outcome and returns one summary row.
# Assumption: a single entry effect common to all occupations, with occupation dummies absorbing
# level differences between occupations. The coefficient is a weighted average of within-occupation
# entry-minus-experienced gaps, with weights proportional to n_entry * n_exp / n within each occupation
# (occupations with balanced and large samples count most). regress_full.py relaxes this to one effect
# per occupation. Sample sizes and raw means by level are reported alongside so the coefficient can be
# compared with the naive (unadjusted) difference mean_entry - mean_experienced.
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
        # ratio scale requested for an outcome that can be negative: leave the row blank rather than fail
        return {**base, "coef_entry": np.nan, "pct_diff": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}
    b = m.params["entry"]
    # on the ratio scale b is a log ratio, so exp(b) - 1 is the proportional difference (entry/exp - 1);
    # t and p are on the log scale, as reported by statsmodels
    return {**base, "coef_entry": b, "pct_diff": (np.exp(b) - 1) if scale == "ratio" else np.nan,
            "se": m.bse["entry"], "t": m.tvalues["entry"], "p": m.pvalues["entry"]}


# %% Command-line entry point
def main() -> None:
    # %% Arguments
    # --tag / --model identify which match_tasks run to analyse (they form the input file names).
    # --min-per-level sets the depth threshold for the third sample; 20 per level is a judgement about
    # when an occupation has enough ads on both sides for its within-gap to be informative.
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

    # %% Load the analysis table and print sample counts
    # The "no explicit task" count matters: those ads have NaN for every *_explicit outcome and are
    # dropped from those regressions only, so n_ads differs across outcome rows in the table.
    df = build_ad_table(args.tag, args.model)
    print(f"ads: {len(df):,} | occupations: {df['osca_code'].nunique()} | entry: {int(df['entry'].sum()):,} | "
          f"experienced: {int((df['entry']==0).sum()):,} | internships: {int(df['internship'].sum()):,}")
    print(f"ads with no explicit task (dropped from explicit-share outcomes): {int(df['auto_share_explicit'].isna().sum()):,}\n")

    # %% Outcomes and samples
    # Every outcome is regressed in every sample, giving 13 x 3 rows. The order below is the order of
    # the printed table: count shares first (headline), then time-weighted shares, then robustness measures.
    outcomes = ["automation_count_share", "augmentation_count_share", "unaffected_count_share",
                "aug_minus_auto_count_share", "automation_count_share_explicit", "aug_minus_auto_count_share_explicit",
                "auto_share_explicit", "auto_share_expl_impl", "auto_time_explicit", "time_share_explicit",
                "auto_frac_mean_explicit", "auto_flag_share_explicit", "n_explicit"]
    # per_level: occupations x {0, 1} table of ad counts (columns 0 = experienced, 1 = entry)
    per_level = df.groupby(["osca_code", "entry"]).size().unstack(fill_value=0)
    # "deep" occupations have at least --min-per-level ads on both sides
    deep = per_level[(per_level[0] >= args.min_per_level) & (per_level[1] >= args.min_per_level)].index
    # Three samples: full; without internships (are the results driven by intern ads?); deep occupations
    # only (are they driven by thinly-populated occupations where the FE is estimated from a handful of ads?).
    samples = {"all": df, "excluding internships": df[df["internship"] == 0],
               f"occupations with >= {args.min_per_level} per level": df[df["osca_code"].isin(deep)]}

    # %% Fit all models and assemble the results table
    rows = []
    for sname, d in samples.items():
        for y in outcomes:
            r = fit_fe(d, y, args.se, args.scale); r["sample"] = sname; rows.append(r)
    cols = ["sample", "outcome", "coef_entry", "pct_diff", "se", "t", "p", "n_ads", "n_occupations", "mean_entry", "mean_experienced"]
    # pct_diff is only meaningful on the ratio scale
    if args.scale == "level":
        cols.remove("pct_diff")
    res = pd.DataFrame(rows)[cols]
    print(f"standard errors: {SE_NOTE[args.se]}\nscale: {SCALE_NOTE[args.scale]}"
          + ("\n(coef_entry is the log ratio; outcomes with negative values are skipped)\n" if args.scale == "ratio" else "\n"))
    # the CSV name carries the tag, model and a _ratio suffix so level and ratio runs do not overwrite
    out = ROOT / "data" / f"regression_{args.tag}_{args.model}{'_ratio' if args.scale == 'ratio' else ''}.csv"
    res.to_csv(out, index=False)

    # %% Print one table per sample
    # Formatting only: signed coefficients, percentages for pct_diff, three decimals for p-values.
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

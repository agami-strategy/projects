"""Proportional version of the net automation-vs-augmentation measure.

The level model uses aug_minus_auto_count_share, which can be negative, so it has no log-link
analogue. The proportional counterpart is the RATIO OF RATIOS: within an occupation, how much does
the automation-to-augmentation ratio of the tasks an ad describes change for entry-level ads?

    R_j = (auto_entry_j / auto_exp_j) / (aug_entry_j / aug_exp_j)
        = exp(b_auto_j - b_aug_j)

where b_auto_j and b_aug_j are the per-occupation log entry effects on the automation count share
and the augmentation count share. Reported as R_j - 1 (percent change in the ratio).

Estimation: each ad contributes two rows (category = automation / augmentation, y = that count
share) to one Poisson quasi-likelihood model with a parameter per (occupation x category) cell and
an entry effect per cell. b_auto_j - b_aug_j and its variance come from the fitted coefficients and
their covariance. Standard errors are clustered on ad id by default, because the two rows of an ad
are not independent; --se hc1 is available for comparison.

Occupations where any of the four (level x category) cell means is zero are dropped: a log-link
effect for a zero-mean cell is not finite.

Usage:
    python scripts/regress_ratio_of_ratios.py
    python scripts/regress_ratio_of_ratios.py --explicit-only --exclude-internships
    python scripts/regress_ratio_of_ratios.py --pooled

Outputs: data/regression_ratio_of_ratios_<tag>_<model>[_explicit].txt / .csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_OSCA, load_osca  # noqa: E402
from llm import DEFAULT_MODEL  # noqa: E402
from regress_automation import build_ad_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def stack(df: pd.DataFrame, auto_col: str, aug_col: str) -> pd.DataFrame:
    """Two rows per ad: (cat='auto', y=automation share) and (cat='aug', y=augmentation share)."""
    base = df[["osca_code", "entry", auto_col, aug_col]].dropna().reset_index().rename(columns={"index": "ad_id"})
    a = base[["ad_id", "osca_code", "entry"]].assign(cat="auto", y=base[auto_col].values)
    g = base[["ad_id", "osca_code", "entry"]].assign(cat="aug", y=base[aug_col].values)
    return pd.concat([a, g], ignore_index=True)


def fit(long: pd.DataFrame, se: str, pooled: bool):
    # no intercept: one parameter per occupation x category cell, plus entry effects
    if pooled:
        formula = "y ~ 0 + C(osca_code):C(cat) + entry:C(cat)"
    else:
        formula = "y ~ 0 + C(osca_code):C(cat) + entry:C(osca_code):C(cat)"
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": long["ad_id"]}} if se == "cluster_ad" else {"cov_type": "HC1"}
    return formula, smf.glm(formula, data=long, family=sm.families.Poisson()).fit(**kw)


def contrast(m, name_auto: str, name_aug: str) -> tuple[float, float]:
    """b_auto - b_aug and its standard error from the coefficient covariance."""
    V = m.cov_params()
    b = m.params[name_auto] - m.params[name_aug]
    var = V.loc[name_auto, name_auto] + V.loc[name_aug, name_aug] - 2 * V.loc[name_auto, name_aug]
    return b, float(np.sqrt(var))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="us_2026-09-17_selected_tasks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--explicit-only", action="store_true", help="use the explicit-only count shares")
    ap.add_argument("--exclude-internships", action="store_true")
    ap.add_argument("--min-per-level", type=int, default=10)
    ap.add_argument("--se", default="cluster_ad", choices=["cluster_ad", "hc1"])
    ap.add_argument("--pooled", action="store_true", help="single ratio-of-ratios across occupations (with occupation cells)")
    args = ap.parse_args()

    groups = load_osca(Path(DEFAULT_OSCA))
    df = build_ad_table(args.tag, args.model)
    if args.exclude_internships:
        df = df[df["internship"] == 0]
    suffix = "_explicit" if args.explicit_only else ""
    auto_col, aug_col = f"automation_count_share{suffix}", f"augmentation_count_share{suffix}"
    long = stack(df, auto_col, aug_col)

    # eligibility: enough ads per level, and no zero-mean cell (log link needs a positive mean)
    per_level = long[long.cat == "auto"].groupby(["osca_code", "entry"]).size().unstack(fill_value=0)
    cell_means = long.groupby(["osca_code", "entry", "cat"])["y"].mean().unstack(["entry", "cat"])
    ok_n = per_level[(per_level.get(0, 0) >= args.min_per_level) & (per_level.get(1, 0) >= args.min_per_level)].index
    ok_mean = cell_means[(cell_means > 0).all(axis=1)].index
    dropped_zero = sorted(set(ok_n) - set(ok_mean))
    keep = sorted(set(ok_n) & set(ok_mean))
    long = long[long.osca_code.isin(keep)]

    formula, m = fit(long, args.se, args.pooled)
    lines = []
    out = lines.append
    out(f"Formula: {formula}   (Poisson quasi-likelihood, log link; two rows per ad)")
    out(f"Standard errors: {'clustered on ad id' if args.se == 'cluster_ad' else 'heteroskedasticity-robust (HC1)'}")
    out(f"Shares: {auto_col} vs {aug_col}{' | internships excluded' if args.exclude_internships else ''}")
    out(f"Sample: {long.ad_id.nunique():,} ads | occupations {len(keep)} (>= {args.min_per_level} ads per level and no zero-mean cell)"
        + (f" | dropped for a zero-mean cell: {', '.join(dropped_zero)}" if dropped_zero else ""))
    out("")

    if args.pooled:
        b, se = contrast(m, "entry:C(cat)[auto]", "entry:C(cat)[aug]")
        z = b / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        out("Pooled ratio of ratios: change in the automation-to-augmentation ratio for entry-level ads")
        out(f"  log effect            {b:+.4f}   SE {se:.4f}   z {z:+.2f}   p {p:.4f}")
        out(f"  ratio of ratios - 1   {np.exp(b) - 1:+.1%}   95% CI [{np.exp(b - 1.96 * se) - 1:+.1%}, {np.exp(b + 1.96 * se) - 1:+.1%}]")
        for cat, label in (("auto", "automation share"), ("aug", "augmentation share")):
            bb = m.params[f"entry:C(cat)[{cat}]"]
            out(f"  component, {label:19s} entry / experienced - 1 = {np.exp(bb) - 1:+.1%}")
        text = "\n".join(lines)
        print(text)
        stem = ROOT / "data" / f"regression_ratio_of_ratios_pooled_{args.tag}_{args.model}{suffix}"
        Path(f"{stem}.txt").write_text(text, encoding="utf-8")
        print(f"\nwrote {stem}.txt")
        return

    eff = []
    for code in keep:
        na, ng = f"entry:C(osca_code)[{code}]:C(cat)[auto]", f"entry:C(osca_code)[{code}]:C(cat)[aug]"
        b, se = contrast(m, na, ng)
        z = b / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        eff.append({"osca_code": code, "osca_title": groups[code]["title"],
                    "ratio_of_ratios_minus_1": np.exp(b) - 1, "log_effect": b, "se_log": se, "p": p,
                    "ci_low": np.exp(b - 1.96 * se) - 1, "ci_high": np.exp(b + 1.96 * se) - 1,
                    "automation_pct_diff": np.exp(m.params[na]) - 1, "augmentation_pct_diff": np.exp(m.params[ng]) - 1,
                    "n_entry": int(per_level.loc[code, 1]), "n_exp": int(per_level.loc[code, 0])})
    res = pd.DataFrame(eff).sort_values("ratio_of_ratios_minus_1", ascending=False)

    out("Entry-level change in the automation-to-augmentation ratio, by occupation (ratio of ratios - 1)")
    out(f"{'code':6s} {'title':44s} {'ratio-1':>9s} {'se_log':>7s} {'p':>7s} {'ci_low':>8s} {'ci_high':>8s} {'auto%':>8s} {'aug%':>8s} {'n_entry':>8s} {'n_exp':>6s}")
    for r in res.itertuples():
        out(f"{r.osca_code:6s} {r.osca_title[:44]:44s} {r.ratio_of_ratios_minus_1:+9.1%} {r.se_log:7.3f} {r.p:7.3f} "
            f"{r.ci_low:+8.1%} {r.ci_high:+8.1%} {r.automation_pct_diff:+8.1%} {r.augmentation_pct_diff:+8.1%} {r.n_entry:8d} {r.n_exp:6d}")

    # joint tests on the contrasts: all zero, all equal (restriction matrices over the parameter vector)
    names = list(m.params.index)
    k = len(names)
    R0 = np.zeros((len(keep), k))
    for i, code in enumerate(keep):
        R0[i, names.index(f"entry:C(osca_code)[{code}]:C(cat)[auto]")] = 1.0
        R0[i, names.index(f"entry:C(osca_code)[{code}]:C(cat)[aug]")] = -1.0
    R1 = R0[1:] - R0[0]  # contrast_i - contrast_0 = 0
    w0 = m.wald_test(R0, scalar=True)
    w1 = m.wald_test(R1, scalar=True)
    sig_pos = int(((res.p < 0.05) & (res.ratio_of_ratios_minus_1 > 0)).sum())
    sig_neg = int(((res.p < 0.05) & (res.ratio_of_ratios_minus_1 < 0)).sum())
    n_ads = res.n_entry + res.n_exp
    out("")
    out("Summary")
    out(f"  occupations                          {len(res)}")
    out(f"  positive and significant at 5%       {sig_pos}   (entry-level ads tilt further towards automation)")
    out(f"  negative and significant at 5%       {sig_neg}")
    out(f"  median ratio of ratios - 1           {res.ratio_of_ratios_minus_1.median():+.1%}")
    out(f"  ad-weighted mean, log scale, exp'd   {np.exp((res.log_effect * n_ads).sum() / n_ads.sum()) - 1:+.1%}")
    out(f"  Wald test, all contrasts = 0         chi2 = {float(w0.statistic):.1f}, p = {float(w0.pvalue):.4f}")
    out(f"  Wald test, all contrasts equal       chi2 = {float(w1.statistic):.1f}, p = {float(w1.pvalue):.4f}")
    text = "\n".join(lines)
    print(text)
    stem = ROOT / "data" / f"regression_ratio_of_ratios_{args.tag}_{args.model}{suffix}"
    Path(f"{stem}.txt").write_text(text, encoding="utf-8")
    res.to_csv(f"{stem}.csv", index=False)
    print(f"\nwrote {stem}.txt and .csv")


if __name__ == "__main__":
    main()

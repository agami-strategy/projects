"""Fit one occupation fixed-effects regression and print the full output: statsmodels summary
(all coefficients including every occupation dummy), then the occupation effects as a readable
table with unit group titles, then the model statistics.

    <outcome> ~ entry + C(osca_code)    (--se hc1 | classical | cluster; default hc1)

Usage:
    python scripts/regress_full.py                                    # auto_share_explicit, all ads
    python scripts/regress_full.py --outcome auto_time_explicit
    python scripts/regress_full.py --outcome auto_frac_mean_explicit --exclude-internships
    python scripts/regress_full.py --outcome auto_share_explicit --covariates time_share_explicit n_tasks

Outcomes: auto_share_explicit, auto_share_expl_impl, auto_time_explicit, time_share_explicit,
          auto_frac_mean_explicit, auto_flag_share_explicit, n_explicit
Writes data/regression_full_<outcome>_<tag>_<model>.txt (the printed output) and .csv (all coefficients).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_OSCA, load_osca  # noqa: E402
from llm import DEFAULT_MODEL  # noqa: E402
import numpy as np  # noqa: E402
from regress_automation import SCALE_NOTE, SE_NOTE, build_ad_table, fit_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="us_2026-09-17_selected_tasks")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--outcome", default="automation_count_share",
                    help="e.g. automation_count_share (default), augmentation_count_share, unaffected_count_share, "
                         "automation_count_share_explicit, auto_share_explicit, auto_time_explicit, n_explicit")
    ap.add_argument("--covariates", nargs="*", default=[], help="extra ad-level controls, e.g. time_share_explicit n_tasks")
    ap.add_argument("--exclude-internships", action="store_true")
    ap.add_argument("--min-per-level", type=int, default=0, help="keep occupations with at least this many ads at each level")
    ap.add_argument("--se", default="hc1", choices=["hc1", "classical", "cluster", "employer"],
                    help="standard errors: heteroskedasticity-robust (default), classical OLS, clustered on occupation, or clustered on employer "
                         "(employer clustering is the usable robustness check for the interaction model)")
    ap.add_argument("--pooled", action="store_true",
                    help="fit the pooled summary model (single entry coefficient) instead of the main interaction model")
    ap.add_argument("--scale", default="level", choices=["level", "ratio"],
                    help="level: effects are differences in the outcome (default); ratio: proportional differences, "
                         "entry/experienced - 1, from a Poisson quasi-likelihood model with log link")
    args = ap.parse_args()
    args.interact = not args.pooled  # the entry x occupation interaction model is the main specification

    groups = load_osca(Path(DEFAULT_OSCA))
    df = build_ad_table(args.tag, args.model)
    if args.exclude_internships:
        df = df[df["internship"] == 0]
    d = df.dropna(subset=[args.outcome] + args.covariates)
    per_level = d.groupby(["osca_code", "entry"]).size().unstack(fill_value=0)
    # the interaction model estimates one effect per occupation, so each side needs a few ads for a usable SE
    min_pl = max(args.min_per_level, 10 if args.interact else 1)
    ok = per_level[(per_level.get(0, 0) >= min_pl) & (per_level.get(1, 0) >= min_pl)].index
    d = d[d["osca_code"].isin(ok)]

    if args.interact:
        rhs = " + ".join(["C(osca_code)", "entry:C(osca_code)", *args.covariates])
        formula = f"{args.outcome} ~ {rhs}"
        m = fit_model(formula, d, "hc1" if args.se == "cluster" else args.se, args.scale)
        se_note = SE_NOTE["hc1" if args.se == "cluster" else args.se] + (" (clustering not usable with one entry effect per occupation)" if args.se == "cluster" else "")
    else:
        rhs = " + ".join(["entry", *args.covariates, "C(osca_code)"])
        formula = f"{args.outcome} ~ {rhs}"
        m = fit_model(formula, d, args.se, args.scale)
        se_note = SE_NOTE[args.se]
    ratio = args.scale == "ratio"

    lines = []
    out = lines.append
    out(f"Formula: {formula}")
    out(f"Standard errors: {se_note}")
    out(f"Scale: {SCALE_NOTE[args.scale]}")
    out(f"Sample: {len(d):,} ads | entry-level {int(d['entry'].sum()):,} | experienced {int((d['entry']==0).sum()):,}"
        f" | occupations {d['osca_code'].nunique()} (each with >= {min_pl} ads per level after dropping missing outcomes)"
        f"{' | internships excluded' if args.exclude_internships else ''}")
    out("")
    out(str(m.summary()))
    out("")

    if args.interact:
        # per-occupation entry effects
        eff = []
        for p in m.params.index:
            if p.startswith("entry:C(osca_code)"):
                code = p.split("[")[1].rstrip("]")
                eff.append((code, m.params[p], m.bse[p], m.pvalues[p], m.conf_int().loc[p, 0], m.conf_int().loc[p, 1]))
        if ratio:
            # report exp(coef) - 1: the proportional difference entry/experienced - 1, with CI on the same scale
            eff = [(c, np.exp(b) - 1, se, p, np.exp(lo) - 1, np.exp(hi) - 1) for c, b, se, p, lo, hi in eff]
            out("Entry-level effect by occupation (entry / experienced - 1, within occupation), sorted by size; se is on the log scale")
            out(f"{'code':6s} {'title':46s} {'pct_diff':>9s} {'se_log':>8s} {'p':>7s} {'ci_low':>8s} {'ci_high':>8s} {'n_entry':>8s} {'n_exp':>7s}")
            for code, coef, se, p, lo, hi in sorted(eff, key=lambda x: -x[1]):
                out(f"{code:6s} {groups[code]['title'][:46]:46s} {coef:+9.1%} {se:8.4f} {p:7.3f} {lo:+8.1%} {hi:+8.1%} "
                    f"{int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
        else:
            out("Entry-level effect by occupation (entry minus experienced, within occupation), sorted by size")
            out(f"{'code':6s} {'title':46s} {'effect':>8s} {'se':>8s} {'p':>7s} {'ci_low':>8s} {'ci_high':>8s} {'n_entry':>8s} {'n_exp':>7s}")
            for code, coef, se, p, lo, hi in sorted(eff, key=lambda x: -x[1]):
                out(f"{code:6s} {groups[code]['title'][:46]:46s} {coef:+8.4f} {se:8.4f} {p:7.3f} {lo:+8.4f} {hi:+8.4f} "
                    f"{int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
        sig_pos = sum(1 for e in eff if e[3] < 0.05 and e[1] > 0)
        sig_neg = sum(1 for e in eff if e[3] < 0.05 and e[1] < 0)
        # joint tests via explicit restriction matrices (string hypotheses are fragile with many terms)
        names = [p for p in m.params.index if p.startswith("entry:C(osca_code)")]
        idx = [list(m.params.index).index(n) for n in names]
        k = len(m.params)
        R0 = np.zeros((len(idx), k))
        for r, j in enumerate(idx):
            R0[r, j] = 1.0                       # each entry effect = 0
        R1 = np.zeros((len(idx) - 1, k))
        for r, j in enumerate(idx[1:]):
            R1[r, idx[0]] = 1.0                  # effect_0 - effect_r = 0
            R1[r, j] = -1.0
        all_zero = m.wald_test(R0, use_f=True, scalar=True)
        equal = m.wald_test(R1, use_f=True, scalar=True)
        out("")
        out("Summary of occupation-specific entry effects")
        out(f"  occupations                          {len(eff)}")
        out(f"  positive and significant at 5%       {sig_pos}")
        out(f"  negative and significant at 5%       {sig_neg}")
        fmt = "{:+.1%}" if ratio else "{:+.4f}"
        out(f"  mean effect (unweighted)             {fmt.format(sum(e[1] for e in eff) / len(eff))}")
        out(f"  ad-weighted mean effect              {fmt.format(sum(e[1] * (per_level.loc[e[0], 1] + per_level.loc[e[0], 0]) for e in eff) / len(d))}")
        out(f"  Wald test, all effects = 0           F = {float(all_zero.statistic):.2f}, p = {float(all_zero.pvalue):.4f}")
        out(f"  Wald test, all effects equal         F = {float(equal.statistic):.2f}, p = {float(equal.pvalue):.4f}")
        out(f"  {'pseudo R-squared' if ratio else 'R-squared':36s} {(m.pseudo_rsquared() if ratio else m.rsquared):.4f}   observations {int(m.nobs):,}")
        text = "\n".join(lines)
        print(text)
        stem = ROOT / "data" / f"regression_interact_{args.outcome}_{args.tag}_{args.model}{'_ratio' if ratio else ''}"
        Path(f"{stem}.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(eff, columns=["osca_code", "pct_diff" if ratio else "entry_effect", "se_log" if ratio else "se", "p", "ci_low", "ci_high"]).assign(
            osca_title=lambda x: x["osca_code"].map(lambda c: groups[c]["title"])).to_csv(f"{stem}.csv", index=False)
        print(f"\nwrote {stem}.txt and .csv")
        return

    out("Occupation fixed effects (intercept shift relative to the reference occupation), sorted by size")
    out(f"{'code':6s} {'title':50s} {'coef':>9s} {'se':>8s} {'p':>7s} {'n_entry':>8s} {'n_exp':>7s}")
    ref = sorted(d["osca_code"].unique())[0]
    out(f"{ref:6s} {groups[ref]['title'][:50]:50s} {'(ref)':>9s} {'':>8s} {'':>7s} {int(per_level.loc[ref, 1]):8d} {int(per_level.loc[ref, 0]):7d}")
    fe = [(p, m.params[p], m.bse[p], m.pvalues[p]) for p in m.params.index if p.startswith("C(osca_code)")]
    for name, coef, se, p in sorted(fe, key=lambda x: -x[1]):
        code = name.split("[T.")[1].rstrip("]")
        out(f"{code:6s} {groups[code]['title'][:50]:50s} {coef:+9.4f} {se:8.4f} {p:7.3f} {int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
    out("")
    out("Model statistics")
    out(f"  entry coefficient        {m.params['entry']:+.5f}   SE {m.bse['entry']:.5f}   t {m.tvalues['entry']:+.2f}   p {m.pvalues['entry']:.4f}")
    ci = m.conf_int().loc["entry"]
    out(f"  95% CI for entry         [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    if ratio:
        out(f"  proportional difference  {np.exp(m.params['entry']) - 1:+.1%}   95% CI [{np.exp(ci[0]) - 1:+.1%}, {np.exp(ci[1]) - 1:+.1%}]   (entry / experienced - 1)")
        out(f"  pseudo R-squared         {m.pseudo_rsquared():.4f}")
    else:
        out(f"  R-squared                {m.rsquared:.4f}   adjusted {m.rsquared_adj:.4f}")
    out(f"  observations             {int(m.nobs):,}   occupations {d['osca_code'].nunique()}   parameters {int(m.df_model)+1}")
    out(f"  outcome mean             entry {d.loc[d['entry']==1, args.outcome].mean():.4f}   experienced {d.loc[d['entry']==0, args.outcome].mean():.4f}")

    text = "\n".join(lines)
    print(text)
    stem = ROOT / "data" / f"regression_full_{args.outcome}_{args.tag}_{args.model}"
    Path(f"{stem}.txt").write_text(text, encoding="utf-8")
    coefs = pd.DataFrame({"term": m.params.index, "coef": m.params.values, "se": m.bse.values,
                          "t": m.tvalues.values, "p": m.pvalues.values,
                          "ci_low": m.conf_int()[0].values, "ci_high": m.conf_int()[1].values})
    coefs.to_csv(f"{stem}.csv", index=False)
    print(f"\nwrote {stem}.txt and .csv")


if __name__ == "__main__":
    main()

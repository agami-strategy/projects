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
# %% Imports and paths
# This is the main specification script. Unlike regress_automation.py (many outcomes, one line each), it
# fits ONE model for ONE outcome and prints everything: the full statsmodels summary, a per-occupation
# table with OSCA titles, joint tests and fit statistics. The default model is the entry x occupation
# interaction (one entry effect per occupation); --pooled gives the single-coefficient summary model.
# Note the docstring's formula and outcome list describe the pooled model and the older outcome set;
# the default --outcome is now automation_count_share and the default model is the interaction model.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

# Sibling modules: label_osca provides the OSCA unit-group list (code -> title, skill level) used to
# label rows; llm provides DEFAULT_MODEL; regress_automation provides the table builder and model fitter
# so both scripts share exactly the same sample construction and estimator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_OSCA, load_osca  # noqa: E402
from llm import DEFAULT_MODEL  # noqa: E402
import numpy as np  # noqa: E402
from regress_automation import SCALE_NOTE, SE_NOTE, build_ad_table, fit_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# %% Command-line entry point
def main() -> None:
    # %% Arguments
    # --covariates adds ad-level controls to the right-hand side (e.g. time_share_explicit to hold
    #   coverage fixed, n_tasks for ad length). Adding coverage changes the question from "do entry ads
    #   describe more automatable work?" to "... conditional on how much of the job they describe".
    # --min-per-level floors at 10 for the interaction model (see below) and at 1 for the pooled model.
    # --se cluster is accepted but replaced by hc1 for the interaction model (see the explanation below).
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

    # %% Sample construction
    # groups: {osca_code: {"title": ..., "skill_level": ...}} for labelling the output tables.
    # The ad table is the same one regress_automation.py uses (see build_ad_table there).
    groups = load_osca(Path(DEFAULT_OSCA))
    df = build_ad_table(args.tag, args.model)
    if args.exclude_internships:
        df = df[df["internship"] == 0]
    # drop ads missing the outcome or any covariate so all reported counts refer to the estimation sample
    d = df.dropna(subset=[args.outcome] + args.covariates)
    # per_level: occupations x {0 = experienced, 1 = entry} ad counts, reused for the n columns below
    per_level = d.groupby(["osca_code", "entry"]).size().unstack(fill_value=0)
    # the interaction model estimates one effect per occupation, so each side needs a few ads for a usable SE
    # Occupations below the floor are dropped entirely (not just their entry effect), so the sample for the
    # interaction model is smaller than for the pooled model. 10 per level is a judgement call: lower it and
    # more occupations enter with very wide CIs; raise it and the sample concentrates in large occupations.
    # per_level.get(0, 0) guards against a sample with no ads at one level (the column would be missing).
    min_pl = max(args.min_per_level, 10 if args.interact else 1)
    ok = per_level[(per_level.get(0, 0) >= min_pl) & (per_level.get(1, 0) >= min_pl)].index
    d = d[d["osca_code"].isin(ok)]

    # %% Model specification and fit
    # Interaction model:  y ~ C(osca_code) + entry:C(osca_code) [+ covariates]
    #   C(osca_code) gives an intercept per occupation (one reference level absorbed into the constant).
    #   entry:C(osca_code) is the interaction WITHOUT a main entry effect, so patsy creates one
    #   "entry:C(osca_code)[code]" slope per occupation (all occupations, no reference level dropped);
    #   each is the entry-minus-experienced gap within that occupation. The pooled model is nested in it.
    # Why clustering on occupation is degenerate here: each entry effect is identified entirely by the ads
    #   of one occupation, i.e. by a single cluster. Cluster-robust variance with G clusters is rank <= G,
    #   and an effect living in one cluster has no between-cluster variation to estimate its variance
    #   from, so the SEs are meaningless (statsmodels may return zeros or singular-matrix warnings). The
    #   code therefore silently substitutes HC1 and says so in the printed note. Employer clustering
    #   (--se employer) is fine because employers cut across occupations and there are many of them.
    # Pooled model:  y ~ entry [+ covariates] + C(osca_code): one common entry coefficient.
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

    # %% Output header and full statsmodels summary
    # Output is accumulated in `lines` so the same text is printed and written to the .txt file.
    # m.summary() lists every coefficient including each occupation dummy; for the interaction model with
    # many occupations this is long but is the auditable record of the fit.
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

    # %% Interaction model: per-occupation entry effects
    # Pull every "entry:C(osca_code)[<code>]" term into tuples (code, coef, se, p, ci_low, ci_high).
    # m.conf_int() returns a DataFrame with columns 0 (lower) and 1 (upper) at the default 95% level.
    if args.interact:
        # per-occupation entry effects
        eff = []
        for p in m.params.index:
            if p.startswith("entry:C(osca_code)"):
                # parameter names look like "entry:C(osca_code)[1234]"; extract the code between the brackets
                code = p.split("[")[1].rstrip("]")
                eff.append((code, m.params[p], m.bse[p], m.pvalues[p], m.conf_int().loc[p, 0], m.conf_int().loc[p, 1]))
        # On the ratio scale the coefficient and CI bounds are log ratios; exponentiating each bound gives a
        # CI on the proportional scale (the CI is asymmetric there). The SE stays on the log scale because
        # exp(se) has no direct meaning; the p-value is unchanged by the monotone transformation.
        if ratio:
            # report exp(coef) - 1: the proportional difference entry/experienced - 1, with CI on the same scale
            eff = [(c, np.exp(b) - 1, se, p, np.exp(lo) - 1, np.exp(hi) - 1) for c, b, se, p, lo, hi in eff]
            out("Entry-level effect by occupation (entry / experienced - 1, within occupation), sorted by size; se is on the log scale")
            out(f"{'code':6s} {'title':46s} {'pct_diff':>9s} {'se_log':>8s} {'p':>7s} {'ci_low':>8s} {'ci_high':>8s} {'n_entry':>8s} {'n_exp':>7s}")
            # sorted by effect size, largest positive first; n_entry / n_exp are per_level columns 1 / 0
            for code, coef, se, p, lo, hi in sorted(eff, key=lambda x: -x[1]):
                out(f"{code:6s} {groups[code]['title'][:46]:46s} {coef:+9.1%} {se:8.4f} {p:7.3f} {lo:+8.1%} {hi:+8.1%} "
                    f"{int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
        else:
            out("Entry-level effect by occupation (entry minus experienced, within occupation), sorted by size")
            out(f"{'code':6s} {'title':46s} {'effect':>8s} {'se':>8s} {'p':>7s} {'ci_low':>8s} {'ci_high':>8s} {'n_entry':>8s} {'n_exp':>7s}")
            for code, coef, se, p, lo, hi in sorted(eff, key=lambda x: -x[1]):
                out(f"{code:6s} {groups[code]['title'][:46]:46s} {coef:+8.4f} {se:8.4f} {p:7.3f} {lo:+8.4f} {hi:+8.4f} "
                    f"{int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
        # %% Counts of significant effects and joint Wald tests
        # sig_pos / sig_neg count occupations whose own effect is significant at 5% (e[3] is p, e[1] the
        # effect). With many occupations some will be significant by chance, so the joint tests below are
        # the better summary. No multiple-testing correction is applied to the per-occupation p-values.
        sig_pos = sum(1 for e in eff if e[3] < 0.05 and e[1] > 0)
        sig_neg = sum(1 for e in eff if e[3] < 0.05 and e[1] < 0)
        # joint tests via explicit restriction matrices (string hypotheses are fragile with many terms)
        # A Wald test of H0: R b = 0 where b is the full parameter vector (length k) and each row of R
        # picks out one linear restriction. `idx` holds the positions of the entry-effect parameters in b.
        names = [p for p in m.params.index if p.startswith("entry:C(osca_code)")]
        idx = [list(m.params.index).index(n) for n in names]
        k = len(m.params)
        # R0: one row per occupation with a 1 in that effect's column -> H0: every entry effect is zero
        # (no entry-level difference in any occupation). This is the omnibus test of the paper's question.
        R0 = np.zeros((len(idx), k))
        for r, j in enumerate(idx):
            R0[r, j] = 1.0                       # each entry effect = 0
        # R1: rows of the form (+1 on effect_0, -1 on effect_r) -> H0: all entry effects are equal to the
        # first one, i.e. equal to each other. Rejection means the pooled model's single coefficient hides
        # heterogeneity across occupations; non-rejection supports summarising with the pooled coefficient.
        R1 = np.zeros((len(idx) - 1, k))
        for r, j in enumerate(idx[1:]):
            R1[r, idx[0]] = 1.0                  # effect_0 - effect_r = 0
            R1[r, j] = -1.0
        # use_f=True reports an F statistic (chi2 / q) with the model's residual df; scalar=True returns
        # plain floats. The test uses the covariance chosen by --se, so it is robust when the SEs are.
        # For the Poisson GLM the F form is a convenience; the chi-square version is the textbook one.
        all_zero = m.wald_test(R0, use_f=True, scalar=True)
        equal = m.wald_test(R1, use_f=True, scalar=True)
        # %% Summary block, then write .txt and .csv
        # Two averages of the occupation effects: unweighted (each occupation counts once) and ad-weighted
        # (each occupation weighted by its number of ads in the sample, n_entry + n_exp, divided by the
        # total number of ads len(d)). The ad-weighted mean is close to, but not identical to, the pooled
        # model's coefficient, which weights by n_entry * n_exp / n within each occupation.
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
        # GLM results have no R-squared; statsmodels' pseudo R-squared (1 - deviance/null deviance by
        # default) is reported instead on the ratio scale
        out(f"  {'pseudo R-squared' if ratio else 'R-squared':36s} {(m.pseudo_rsquared() if ratio else m.rsquared):.4f}   observations {int(m.nobs):,}")
        text = "\n".join(lines)
        print(text)
        # file names: regression_interact_<outcome>_<tag>_<model>[_ratio].txt / .csv
        stem = ROOT / "data" / f"regression_interact_{args.outcome}_{args.tag}_{args.model}{'_ratio' if ratio else ''}"
        Path(f"{stem}.txt").write_text(text, encoding="utf-8")
        # the CSV has one row per occupation: effect, SE, p, CI and title (column names depend on scale)
        pd.DataFrame(eff, columns=["osca_code", "pct_diff" if ratio else "entry_effect", "se_log" if ratio else "se", "p", "ci_low", "ci_high"]).assign(
            osca_title=lambda x: x["osca_code"].map(lambda c: groups[c]["title"])).to_csv(f"{stem}.csv", index=False)
        print(f"\nwrote {stem}.txt and .csv")
        return

    # %% Pooled model: occupation fixed effects table
    # Only reached with --pooled. The occupation dummies are intercept shifts relative to the reference
    # occupation, which patsy picks as the alphabetically first code (hence sorted(...)[0]); their sizes are
    # descriptive (which occupations have more automatable task bundles overall) and not the object of study.
    out("Occupation fixed effects (intercept shift relative to the reference occupation), sorted by size")
    out(f"{'code':6s} {'title':50s} {'coef':>9s} {'se':>8s} {'p':>7s} {'n_entry':>8s} {'n_exp':>7s}")
    ref = sorted(d["osca_code"].unique())[0]
    out(f"{ref:6s} {groups[ref]['title'][:50]:50s} {'(ref)':>9s} {'':>8s} {'':>7s} {int(per_level.loc[ref, 1]):8d} {int(per_level.loc[ref, 0]):7d}")
    fe = [(p, m.params[p], m.bse[p], m.pvalues[p]) for p in m.params.index if p.startswith("C(osca_code)")]
    for name, coef, se, p in sorted(fe, key=lambda x: -x[1]):
        # dummy names look like "C(osca_code)[T.1234]" (T. marks a treatment-coded level)
        code = name.split("[T.")[1].rstrip("]")
        out(f"{code:6s} {groups[code]['title'][:50]:50s} {coef:+9.4f} {se:8.4f} {p:7.3f} {int(per_level.loc[code, 1]):8d} {int(per_level.loc[code, 0]):7d}")
    # %% Pooled model: headline statistics
    # The entry coefficient is the single number the pooled model delivers; the CI is from m.conf_int()
    # (95% by default). df_model excludes the intercept, hence the +1 in the parameter count.
    out("")
    out("Model statistics")
    out(f"  entry coefficient        {m.params['entry']:+.5f}   SE {m.bse['entry']:.5f}   t {m.tvalues['entry']:+.2f}   p {m.pvalues['entry']:.4f}")
    ci = m.conf_int().loc["entry"]
    out(f"  95% CI for entry         [{ci[0]:+.5f}, {ci[1]:+.5f}]")
    if ratio:
        # exponentiate the log-ratio coefficient and its CI bounds to get entry/experienced - 1
        out(f"  proportional difference  {np.exp(m.params['entry']) - 1:+.1%}   95% CI [{np.exp(ci[0]) - 1:+.1%}, {np.exp(ci[1]) - 1:+.1%}]   (entry / experienced - 1)")
        out(f"  pseudo R-squared         {m.pseudo_rsquared():.4f}")
    else:
        out(f"  R-squared                {m.rsquared:.4f}   adjusted {m.rsquared_adj:.4f}")
    out(f"  observations             {int(m.nobs):,}   occupations {d['osca_code'].nunique()}   parameters {int(m.df_model)+1}")
    out(f"  outcome mean             entry {d.loc[d['entry']==1, args.outcome].mean():.4f}   experienced {d.loc[d['entry']==0, args.outcome].mean():.4f}")

    # %% Write the pooled-model .txt and .csv
    # The CSV holds every coefficient (intercept, entry, covariates, occupation dummies) with SE, t, p, CI.
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

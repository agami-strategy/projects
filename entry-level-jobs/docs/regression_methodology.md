# Regression methodology: automation exposure of entry-level versus experienced ads

*Companion to the data collection and selection methodology. Prepared 20 September 2026.*

## 1. Question and unit of analysis

The regression asks whether, within an occupation, the tasks an employer describes in an entry-level ad are more or less exposed to automation than those described in an experienced-hire ad. The unit of analysis is the job advertisement. Each ad carries an OSCA unit group, a seniority label, and a judgement for every task in its occupation's standard task list: explicit, implied or not required.

## 2. Sample

The estimation sample is the task-matching sample drawn from the US snapshot of 17 September 2026: 18,753 ads across 80 OSCA unit groups, each group holding at least 20 entry-level and 20 experienced ads, with at most 200 ads per group and level. Ads labelled unclear are excluded. Internships are included in the entry-level group and removed in a robustness check. Ads with no explicitly matched task, 925 of them, drop out of outcomes defined over explicit tasks only.

## 3. Outcomes

Each task in the source data has a share of the occupation's working time split into automation, augmentation and unaffected time. Two families of outcome are built from the tasks an ad describes.

Count-share outcomes classify each task by its dominant category, where a task is largely automatable when automation accounts for at least half of its time, and likewise for augmentable and unaffected. The ad's count share for a category is the fraction of its matched tasks, explicit and implied, that fall in that category. The headline outcome is the automation count share; the net measure, augmentation count share minus automation count share, summarises the tilt of the described work.

Count shares are preferred to time-weighted measures for a specific reason. The task time shares are occupation-level averages and cannot vary by seniority, so weighting an ad's tasks by them presumes that entry-level and experienced hires spend their time across tasks in the same proportions, which is the allocation the study is testing. Count shares avoid that assumption by weighting each described task equally. They retain the occupation-level split within each task, so a task is taken to be automatable to the same degree whoever performs it; what they drop is only the assumption about how time is spread across tasks.

Time-weighted outcomes, automation time in matched tasks divided by total time in matched tasks, are reported alongside, together with the unweighted mean of the continuous automation fraction. The three measures separate two changes: dropping the time weights, and classifying tasks by dominant category. In this data the time-weighted share and the unweighted mean fraction show no entry-level effect while the automation count share does, so the difference between seniority levels lies in the mix of task types described, more tasks that are predominantly automatable and fewer that are predominantly augmentable, rather than in the average automation intensity of the described work.

Both families are also computed over explicit tasks alone. Two coverage outcomes, the number of explicit tasks and their combined time share, describe how much of the occupation each ad spells out.

## 4. Model

The main specification lets the entry-level effect differ by occupation. For ad i in occupation j,

y_ij = a_j + b_j · entry_ij + e_ij

where entry equals one for entry-level ads, a_j is a fixed effect giving occupation j its own intercept, and b_j is the occupation's own entry-level effect, the within-occupation difference between entry-level and experienced ads. The model is estimated by ordinary least squares with the occupation effects and the entry-by-occupation interactions entered as indicator variables. It is equivalent to estimating the model separately inside each occupation, but pooling the estimation allows joint tests across occupations. Occupations enter only if they hold at least 10 ads at each level after dropping ads with no matched task.

The output reports each b_j with its confidence interval and the counts behind it, the number of occupations with significantly positive and negative effects, the unweighted and ad-weighted means of the b_j, and two Wald tests: that all b_j are zero and that all b_j are equal. The second test is the formal check on whether a single pooled effect would be an adequate summary.

## 5. Inference

Standard errors are heteroskedasticity-robust, HC1. Ads within an occupation share a task list and are not independent draws, which would ordinarily argue for clustering standard errors on occupation. That is not available here: each b_j is identified by a single occupation, so a cluster-robust variance for it would be computed from one cluster and is degenerate. Robust errors are the appropriate choice for this specification, and are preferred to classical errors because the outcome is bounded and skewed within occupation.

## 6. Pooled summary

A restricted model with a single entry coefficient, y_ij = a_j + b · entry_ij + e_ij, is reported as a summary. Its b is an average of the b_j weighted by each occupation's contribution to within-occupation variance, and it is only a useful summary when the equality test in section 4 does not reject. In this data it rejects decisively, so the pooled coefficient is reported for orientation and the occupation-specific effects carry the findings. The pooled model can be estimated with occupation-clustered standard errors, and that version is retained in the output files as a conservative reference.

## 7. Robustness

Every outcome is re-estimated excluding internships and, for the pooled summary, restricting to occupations with at least 20 ads per level. As a check on dependence between ads, the interaction model is re-estimated with standard errors clustered on employer. Ads from the same employer are often templated and share language, and with several thousand employers in the sample the clustering is well defined and does not collide with the occupation-specific parameters, unlike clustering on occupation or on seniority. On the automation count share, employer clustering widens the intervals modestly: the number of occupations with a significant positive effect falls from 23 to 19 and with a significant negative effect from 16 to 12, while the joint tests are unchanged. The count shares are cross-checked against their time-weighted counterparts and against an alternative count share using the source data's binary automatable flag. Confidence in the occupation code is carried on every ad, so any result can be re-run with a stricter confidence floor.

## 8. Interpretation

Each b_j is the difference in the outcome between entry-level and experienced ads within occupation j, in units of the outcome. For the automation count share, a value of 0.08 for registered nurses means that among the tasks an ad describes, entry-level nursing ads have eight percentage points more in the largely-automatable category than experienced nursing ads. The findings are the distribution of the b_j: how many occupations show each sign, how large the effects are, and which occupations sit at the extremes. The comparison concerns the tasks employers choose to describe, not the full content of the job, and the ad text typically covers about a fifth of an occupation's working time. Results on the count shares and the time-weighted outcomes can differ in sign and significance, and both should be reported: the count share reflects which kinds of task are mentioned, the time-weighted result reflects how much of the described time is automatable.

## 9. Limitations

Occupation codes, seniority labels and task judgements are all model-generated and unreviewed at scale, so measurement error enters every variable. Task time shares for OSCA groups formed by splitting an ANZSCO group are identical across the resulting groups. The sample is capped at 200 ads per group and level, so large occupations are under-represented relative to their share of postings; the occupation fixed effects make this irrelevant for b but the ad-weighted means do not reflect the labour market's occupational mix. The data cover employers using three applicant-tracking systems and describe the US market on one day.

## 10. Reproduction

Two scripts produce the results. The first fits the interaction model for one outcome and prints the full statsmodels output, the per-occupation entry effects with unit group titles and confidence intervals, the joint tests and the model statistics, with options for extra covariates, excluding internships and the standard-error type; a flag switches it to the pooled summary model. The second fits the pooled model for every outcome across the three samples and writes one table of coefficients, standard errors, p-values and group means.

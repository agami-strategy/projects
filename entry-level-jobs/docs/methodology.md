# Entry-level and experienced job ads: data collection and selection methodology

*Status: data collection and sample selection complete; task matching and analysis in progress. Prepared 20 September 2026.*

## 1. Research question

Does the automation exposure of the tasks an employer asks for differ between entry-level and experienced hires within the same occupation? The unit of analysis is the job advertisement. Each ad is coded to an occupation, classified as entry-level or experienced, and matched against the standard task list for its occupation. Task-level automation exposure scores then give, for each ad, the exposure of the work it actually describes. Comparing entry-level and experienced ads within an occupation holds occupational composition fixed.

## 2. Data source

Ads come from the Open-Apply Jobs dataset published on Hugging Face (edwarddgao/open-apply-jobs). It is assembled daily from the public JSON APIs of three applicant-tracking systems: Greenhouse, Lever and Ashby. Each daily partition is a complete snapshot of the postings live that day, not a delta, so an ad that stays open for a month appears in about 30 consecutive snapshots. Every record carries a stable identifier, title, employer slug, location list, employment type, department, posting date, the full HTML description and, where the platform exposes it, salary fields.

Coverage is determined by the three platforms. They are the hiring tools of technology companies, start-ups and other knowledge-work employers, predominantly in the United States. Large Australian employers mostly hire through SEEK, Workday, PageUp and government job boards, none of which are in scope. The dataset therefore represents ads for roles at employers who use these platforms, not the labour market as a whole. This is stated as a scope condition on every result.

| Property | Value |
|---|---|
| Snapshots available | 127 daily partitions, 17 April to 18 September 2026 |
| Ads per snapshot | about 263,000 |
| Text field | full HTML description, median about 6,900 characters |
| Occupation coding in source | none |

Two known data quirks were handled explicitly. Several snapshot days contain duplicated parquet files, so all collection steps deduplicate by ad identifier. One partition stores the salary period column with a different type from the others, so all parquet reads unify schemas by column name.

## 3. Collection

### 3.1 Location filtering

Ads carry a free-text location list rather than a country code. Country membership was inferred from location text with per-country rules.

Australia: the string "australia" or one of the capital cities Sydney, Melbourne, Brisbane, Perth, Adelaide, Canberra, Hobart or Darwin, excluding strings that also name a United States or United Kingdom location or carry a US state abbreviation, which removes Melbourne in Florida, Brisbane in California, Perth in Scotland and Hobart in Indiana.

United States: the country name, a US state name, a US state abbreviation following a comma, or one of about 40 major US cities, excluding strings that also name another country or a well-known non-US city. Ads whose only location is "Remote" cannot be assigned a country and are excluded from both sets.

Ads listing more than three locations are excluded in both cases. These are multi-country remote roles whose labour market is undefined.

### 3.2 Australian ads: all snapshots

Because Australian ads are scarce, roughly 2,100 per snapshot, every unique Australian ad in the window was collected. A metadata scan read only the identifier, title and location columns from all 127 remote partitions to establish which ads were live on which dates. A greedy set-cover then chose the fewest snapshot days whose contents together included every unique ad, and only those days were downloaded in full. This captured all ads with 57 downloads rather than 127.

| Australian collection | Value |
|---|---|
| Unique ads, 17 April to 18 September 2026 | 7,258 |
| Median days an ad stayed live | 31 |
| New ads per weekday | about 38 |
| Ads live for at least 7 days | 6,512 |

### 3.3 United States ads: one snapshot

US ads are abundant, so a single snapshot, 17 September 2026, was used. It contains 152,386 US ads from 7,322 employers. Its occupational mix is markedly broader than the Australian set: alongside software, sales and management roles it includes large numbers of technician, nursing, allied health, driving, retail, cleaning and trades ads from employers such as home health providers, gym chains, car retailers and veterinary groups.

### 3.4 Text preparation

HTML descriptions were converted to plain text preserving paragraph and bullet structure. For model input, description text is truncated at 6,000 characters; about a quarter of ads exceed this, but duties and requirements sections almost always fall within it. Title, department, employment type, location and employer slug are supplied alongside the text.

## 4. Occupation coding

### 4.1 Classification

Ads are coded to the 4-digit unit group of OSCA 2024 version 1.0, the Occupation Standard Classification for Australia, which has 421 unit groups. OSCA rather than ANZSCO was chosen because it is the current Australian standard and its descriptions of newer technology and health occupations are better matched to the ad text. OSCA codes do not coincide with ANZSCO codes, and the two are never mixed.

### 4.2 Method

A large language model receives, as a fixed system prompt, the complete list of unit groups with code, title, lead statement, skill level, constituent occupation titles and alternative titles from the ABS category descriptions. The prompt instructs the model to code the work described rather than the title, to treat US title conventions with care, to use the ad's stated requirements to judge skill level, and to flag ads that are not a single real job. For each ad the model returns up to three candidate unit groups ranked by confidence with a one-sentence rationale, the skill level implied by the ad, and an issue flag for talent-community pages, multi-role postings, placeholders, non-English text or insufficient text. The fixed prompt is served from the provider's prefix cache, so the marginal cost per ad is the ad text alone.

The Australian and US sets were coded with DeepSeek flash. An earlier pilot on 100 Australian ads used Claude Opus 5, Sonnet 5 and Haiku 4.5 to compare model behaviour; Sonnet and Haiku agreed with Opus on most ads, with disagreements concentrated in adjacent unit groups.

### 4.3 Results on the US snapshot

| Outcome | Ads |
|---|---|
| Labelled | 152,386 |
| Valid unit group and no issue flag | 146,026 |
| Issue flagged | 5,695 |
| Code not in OSCA, returned by the model | 663 |
| Unit groups represented | 393 of 421 |
| Unit groups with 100 or more ads | 194 |

The model's confidence is bimodal. Ads are either coded with confidence 0.7 or above, about 38 percent, or the probability is split across candidates and the top code sits below 0.5. Low confidence is concentrated in occupations with fuzzy boundaries, such as analyst, architect and sales roles, rather than spread evenly.

### 4.4 Validation

Every labelling run writes a review workbook with the model's three candidates, confidence and rationale, and blank columns for a reviewer's gold code and notes. Downstream merges use the gold code where present and the model's top code otherwise. Gold review has been carried out on pilot samples only; the counts in this document rest on model labels.

## 5. Target occupations

The study targets 100 ANZSCO unit groups flagged as relevant to workers under 25 in the target workbook. These were mapped to OSCA through the ABS correspondence table from ANZSCO version 1.3 to OSCA 2024 at the 6-digit occupation level, taking every OSCA unit group linked to any occupation in a target ANZSCO unit rather than only the first match. The 100 ANZSCO units map to 189 OSCA unit groups, of which 171 have at least one US ad.

## 6. Selection of ads for seniority classification

Occupation coding is cheap and was applied to every ad. The later stages cost more per ad, so a confidence-based selection was applied per OSCA unit group: the highest of the confidence cutoffs 0.9, 0.8 and 0.7 that still leaves at least 1,000 ads in the group is used; if none does, a floor of 0.5 is used. Ads with an issue flag or an invalid code are excluded throughout.

| Selection outcome | Value |
|---|---|
| Groups using a cutoff of 0.7 or higher | 10 |
| Groups using the 0.5 floor | 161 |
| Ads selected | 56,600 of 90,307 eligible |
| Groups with 1,000 or more selected ads | 17 |
| Groups with 100 to 999 | 72 |
| Groups with 20 to 99 | 45 |

The 0.5 floor is deliberate. Raising it to 0.7 removes half the ads and eliminates whole occupations whose boundaries the model finds ambiguous, rather than removing random noise. Confidence is retained on every record so that any result can be re-run with a stricter floor as a robustness check.

## 7. Seniority classification

Each selected ad was classified as entry-level, experienced or unclear. Entry-level is defined relative to the occupation: the lowest rung a new entrant to that occupation could be hired into, so a graduate solicitor or a newly registered nurse is entry-level despite holding a qualification. The model prompt lists the signals to weigh, in order: stated minimum years of experience, whether prior experience in the work is required or merely preferred, seniority words in the title and text with a caution about the word "associate", explicit early-career framing, supervisory responsibility, and the gap between education and experience required. Stated years outrank the title when they conflict. Internships, placements, cadetships, apprenticeships and traineeships are always entry-level and are flagged separately so they can be excluded later. Each label carries a confidence, the phrase from the ad that most drove the decision, and a rationale.

| Seniority results, 56,600 selected US ads | Ads |
|---|---|
| Entry-level | 18,185 |
| Experienced | 36,927 |
| Unclear | 1,488 |
| Internships, within entry-level | 825 |
| Entry share overall | 33 percent |

The entry share tracks the occupation's skill level closely: near zero for manager groups, 5 to 15 percent for professional groups, above 85 percent for assistant and support groups, and 35 to 60 percent for nurses, sales representatives, civil engineers, storepersons and drivers. Groups with both levels present, which the within-occupation comparison requires, number 119 at a minimum of 5 per level and 80 at a minimum of 20.

## 8. Task data

Task-level time shares and AI exposure scores were supplied per ANZSCO version 1.3 unit group, with each task's share of the occupation's working time split into automation, augmentation and unaffected time. These were converted to OSCA unit groups through the same ABS correspondence. The correspondence carries no employment proportions, so each ANZSCO group's contribution to an OSCA group is weighted by its number of 6-digit occupation links into that group; a one-to-one mapping reproduces the ANZSCO row exactly, a split gives each OSCA child the full parent task list, and a merge blends task lists by link count. Time shares still sum to one within every OSCA group. Task-level attributes are unchanged because they describe the task rather than the occupation. Data exist for 416 of 421 OSCA groups; the five missing groups have no tasks in the source.

## 9. Sample for task matching

Task matching is the most expensive stage, because the model returns a judgement for every task in the occupation's list, about 70 on average. A capped sample was drawn from the seniority-labelled set: within each unit group and level cell, at most 200 ads are kept, randomly with a fixed seed, and cells with 200 or fewer are kept in full. Unclear-level ads are excluded. Each sampled ad carries the occupation's uncapped counts and entry share as metadata so that the raw split is preserved.

| Task-matching sample | Groups | Ads |
|---|---|---|
| Rule as stated | 171 | 23,110 |
| Requiring at least 5 ads per level | 119 | 21,557 |
| Requiring at least 20 ads per level | 80 | 18,753 |

The 5-per-level variant is the working choice: it drops only groups that cannot contribute to a within-occupation comparison.

## 10. Task matching and analysis, in progress

For each sampled ad the model receives the ad text and the OSCA group's task list and returns one of three judgements per task: explicit, when the ad describes the task or a clear equivalent, with the supporting phrase quoted; implied, when the role as described could not be done without it, used sparingly; or not required. From these, each ad's automation exposure is computed as automation time as a share of the time in matched tasks, alongside the occupation's overall share. The comparison of interest is the difference between entry-level and experienced ads in this measure within occupation, estimated as an excess over the occupation baseline and as an occupation fixed-effects regression.

A pilot on 94 Australian ads found no detectable difference on explicitly described tasks and a positive entry-level difference when implied tasks were included, on a sample too small to draw conclusions from. The full US sample is designed to resolve this.

## 11. Limitations

Coverage is limited to employers using three applicant-tracking systems and is skewed towards technology and knowledge work, with the US snapshot adding health, trades and service occupations but not public sector, education or large retail employers. Country is inferred from location text, and ads with only a "Remote" location are dropped. Occupation and seniority labels are model-generated and unreviewed at scale; confidence is bimodal and a floor of 0.5 was applied. Task profiles for OSCA groups formed by splitting an ANZSCO group are identical across the children, so their exposure scores are not independent. Ads describe a small fraction of an occupation's tasks explicitly, about 12 percent of working time in the pilot, so exposure measured on explicit tasks rests on a thin slice of each occupation and carries a tilt towards concrete, routine duties that affects both levels equally.

## 12. Reproducibility

All steps are scripted and resumable. In order: collect ads by country and date, code occupations, select by confidence, classify seniority, convert task data, draw the task-matching sample, merge labels, match tasks, analyse. Model calls go through a single provider layer with JSON validation and retry. Every labelling run writes a raw JSONL of model outputs and a review workbook, and every selection step writes a table of the counts behind it.

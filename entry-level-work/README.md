# Entry-level vs non-entry-level automation exposure by ANZSCO occupation

Lives in `entry-level-work/` inside the `projects` repo. Research pipeline using the [Open-Apply Jobs](https://huggingface.co/datasets/edwarddgao/open-apply-jobs)
dataset (daily snapshots of live postings from Greenhouse / Lever / Ashby career boards).

## Plan

1. Map job ads to ANZSCO codes (zero-shot classifier)
2. Segment ads into entry-level / non-entry-level
3. Map tasks (from the ad text) to jobs
4. Test whether the automatable share of tasks differs between entry-level and non-entry-level
   ads within the same ANZSCO occupation

## Layout

```
scripts/download_snapshot.py   download one daily snapshot (default: latest) into data/raw/
scripts/sample_jobs.py         random sample of ads -> data/samples/*.jsonl + *.md for review
scripts/label_osca.py          Claude labels ads with OSCA unit groups -> data/labels/*.jsonl + review *.xlsx
data/osca-descriptions.xlsx    ABS OSCA 2024 v1.0 category descriptions (Table 1 occupations, Table 2 structure)
scripts/label_seniority.py     Claude labels ads entry-level / experienced -> data/labels/seniority_*
scripts/merge_labels.py        joins ads + OSCA + seniority labels (gold overrides model) -> data/merged/
scripts/convert_tasks_to_osca.py  ANZSCO v1.3 task/time shares -> OSCA unit groups via ABS correspondence -> data/osca_task_share.*
data/anzsco_task_share.xlsx    task-level time shares and AI exposure per ANZSCO v1.3 unit group (input)
data/osca_correspondence_tables.xlsx  ABS OSCA 2024 correspondence tables v2 (Table 1 = ANZSCO v1.3 -> OSCA)
data/raw/                      gitignored parquet snapshots
data/samples/                  small committed samples for manual review
data/labels/                   model labels + review spreadsheets (gold_code column is filled by hand)
```

## Setup

```
pip install -r requirements.txt
python scripts/download_snapshot.py
python scripts/sample_jobs.py --n 100 --seed 42

# OSCA labelling needs an Anthropic API key
pip install anthropic openpyxl
cp .env.example .env            # then put your key in .env (gitignored)
python scripts/label_osca.py data/samples/sample_100.jsonl --dry-run   # prompt sizes + cost estimate, no calls
python scripts/label_osca.py data/samples/sample_100.jsonl --limit 5   # smoke test
python scripts/label_osca.py data/samples/sample_100.jsonl             # full run, resumable
```

## OSCA labelling method

- All 421 OSCA unit groups (code, title, lead statement, skill level, occupation and alternative titles) sit in a
  prompt-cached system prompt (~43k tokens). Each ad costs its own text plus a cache read. Default model is
  Sonnet 5 (`--model` switches to `opus-5` or `haiku-4-5`); roughly $2 per 100 ads on Sonnet, $5 on Opus, $0.65 on Haiku.
- Claude returns up to 3 candidate unit groups (best first) with confidence and a one-line rationale, an `issue` flag
  for ads that are not a single real job, and the skill level implied by the ad. The review sheet flags a mismatch
  between the ad's skill level and the chosen unit group's OSCA skill level.
- Fill `gold_code` in `data/labels/osca_review_*.xlsx` by hand; that becomes the evaluation set for any cheaper or
  faster labelling method later (retrieval shortlist, smaller model, fine-tuned classifier).

## Dataset notes

- Each `date=` partition is a full snapshot of active postings (~230k rows), not a delta.
  Never load `data/**` wholesale (34M rows / 37 GB) - pick one date.
- Coverage is skewed to tech / startup / knowledge-work employers, mostly US. Australian
  postings are a small subset; see `--country australia` in `sample_jobs.py`.
- `description_html` is the full ad; `sample_jobs.py` adds a `description_text` plain-text version.

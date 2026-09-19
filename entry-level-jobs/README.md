# Entry-level vs non-entry-level automation exposure by OSCA occupation (DeepSeek build)

Lives in `entry-level-jobs/` inside the `projects` repo. This is the DeepSeek build; `entry-level-work/` is the original Anthropic build with the same pipeline. Research pipeline using the [Open-Apply Jobs](https://huggingface.co/datasets/edwarddgao/open-apply-jobs)
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
scripts/label_osca.py          DeepSeek labels ads with OSCA unit groups -> data/labels/*.jsonl + review *.xlsx
data/osca-descriptions.xlsx    ABS OSCA 2024 v1.0 category descriptions (Table 1 occupations, Table 2 structure)
scripts/label_seniority.py     DeepSeek labels ads entry-level / experienced -> data/labels/seniority_*
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

# labelling needs a DeepSeek API key (https://platform.deepseek.com)
pip install -r requirements.txt
cp .env.example .env            # then put DEEPSEEK_API_KEY in .env (gitignored)
python scripts/label_osca.py data/samples/sample_100.jsonl --dry-run   # prompt sizes + cost estimate, no calls
python scripts/label_osca.py data/samples/sample_100.jsonl --limit 5   # smoke test
python scripts/label_osca.py data/samples/sample_100.jsonl             # full run, resumable
```

## OSCA labelling method

- All 421 OSCA unit groups (code, title, lead statement, skill level, occupation and alternative titles) sit in the
  system prompt (~43k tokens). DeepSeek caches repeated prompt prefixes automatically, so each ad pays the cached-input
  rate for the list plus the full rate for its own text.
- The model returns up to 3 candidate unit groups (best first) with confidence and a one-line rationale, an `issue` flag
  for ads that are not a single real job, and the skill level implied by the ad. The review sheet flags a mismatch
  between the ad's skill level and the chosen unit group's OSCA skill level.
- Fill `gold_code` in `data/labels/osca_review_*.xlsx` by hand; that becomes the evaluation set for any cheaper or
  faster labelling method later.

## DeepSeek specifics

- All model calls go through `scripts/llm.py`: OpenAI-compatible endpoint, JSON mode, Pydantic validation with retry.
  DeepSeek has no schema enforcement, so the JSON schema is placed in the prompt and replies are validated locally.
- Models: `--model flash` (deepseek-flash, default) or `--model pro` (deepseek-v4-pro). `--thinking` turns on reasoning
  mode (off by default for classification). Thinking mode ignores `temperature`.
- Prices printed by the scripts are PEAK rates (01:00-04:00 and 06:00-10:00 UTC weekdays); off-peak is half.
- Usage is reported as cache_hit / cache_miss / output tokens (DeepSeek's prompt_cache_hit_tokens / prompt_cache_miss_tokens).

## Pipeline

```
python scripts/collect_recent_days.py --days 10                     # or collate_australian_ads.py for the full window
python scripts/label_osca.py data/samples/sample_aus_recent10.jsonl
python scripts/label_seniority.py data/samples/sample_aus_recent10.jsonl
python scripts/merge_labels.py data/samples/sample_aus_recent10.jsonl
python scripts/match_tasks.py data/samples/sample_aus_recent10.jsonl --min-per-level 5
python scripts/analyse_automation.py --tag aus_recent10 --model flash
```

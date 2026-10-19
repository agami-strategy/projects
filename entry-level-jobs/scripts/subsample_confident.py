"""Step 2 of the pipeline: keep ads whose OSCA label is confident, spread across unit groups.

Reads the ads and their OSCA labels, drops ads with a flagged issue or a top-1 confidence
below the threshold, then caps the number of ads per OSCA unit group so a few huge groups
(software engineers, sales reps) do not dominate later spend. Writes a new sample JSONL plus a
matching filtered OSCA label file, so the rest of the pipeline runs on the subsample unchanged:

    python scripts/subsample_confident.py data/samples/sample_aus_all.jsonl --model haiku-4-5
    python scripts/label_seniority.py data/samples/sample_aus_confident.jsonl --model haiku-4-5
    python scripts/merge_labels.py    data/samples/sample_aus_confident.jsonl --model haiku-4-5
    python scripts/match_tasks.py     data/samples/sample_aus_confident.jsonl --model haiku-4-5

Options:
    --min-confidence 0.7   top-1 confidence threshold
    --max-per-group 150    cap per OSCA unit group (random, seeded)
    --min-per-group 1      drop unit groups with fewer confident ads than this
    --tag aus_confident    output name -> data/samples/sample_<tag>.jsonl
"""
# %% Status and imports
# NOTE: this is the earlier, simpler selection step (one fixed confidence floor + a per-group cap).
# It has been superseded by select_by_confidence.py, which picks the cutoff per OSCA group so that
# each group keeps >= 1,000 ads where possible. Kept for reference and for reproducing older runs.
# Standard library only; `label_osca` (sibling module) supplies the model registry and the OSCA list.
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the sibling module importable when run as `python scripts/subsample_confident.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_osca import DEFAULT_MODEL, DEFAULT_OSCA, MODELS, load_osca  # noqa: E402

# %% Directory layout
# ROOT is the project directory. Label files are read from / written to data/labels, samples to
# data/samples. Input files are located by the tag in the input file name; outputs use --tag.
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
SAMPLE_DIR = ROOT / "data" / "samples"


# %% Command line
# input             ads JSONL that label_osca.py was run on; its stem gives the input tag
# --model           which model's OSCA label file to read
# --min-confidence  keep an ad only if its top-1 OSCA confidence is >= this (default 0.7)
# --max-per-group   seeded random cap per OSCA unit group (default 150)
# --min-per-group   drop groups with fewer confident ads than this (default 1 = keep every group)
# --tag             output tag -> sample_<tag>.jsonl and osca_labels_<tag>_<model>.jsonl
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="JSONL of ads that label_osca.py was run on")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--max-per-group", type=int, default=150)
    ap.add_argument("--min-per-group", type=int, default=1)
    ap.add_argument("--tag", default="aus_confident")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # %% Load ads, OSCA labels and the OSCA code list
    # Labels are keyed by ad id. Records without a "label" key (failed API calls) are ignored here and
    # show up as "not labelled" in the exclusion tally below. `ads` keeps the input order.
    in_tag = Path(args.input).stem.replace("sample_", "")
    label_path = LABEL_DIR / f"osca_labels_{in_tag}_{args.model}.jsonl"
    if not label_path.exists():
        sys.exit(f"{label_path} not found; run label_osca.py on {args.input} first")
    labels = {}
    for line in open(label_path, encoding="utf-8"):
        r = json.loads(line)
        if "label" in r:
            labels[r["id"]] = r
    ads = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    groups = load_osca(Path(DEFAULT_OSCA))

    # %% Filter ads and bucket them by top-1 OSCA code
    # Each ad is either bucketed or excluded for exactly one reason (the first failing check wins);
    # `reasons` tallies exclusions for the console summary. Check order: labelled? -> no issue flag?
    # -> valid top-1 code (present in the OSCA list)? -> confidence >= --min-confidence?
    # by_group[code] = [(ad, label record), ...] so the label can be written alongside the ad later.
    # Unlike select_by_confidence.py the confidence floor is the same for every group.
    reasons = Counter()
    by_group: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for ad in ads:
        r = labels.get(ad["id"])
        if not r:
            reasons["not labelled"] += 1
            continue
        lab = r["label"]
        if lab["issue"] != "none":
            reasons[f"issue: {lab['issue']}"] += 1
            continue
        if not lab["candidates"] or lab["candidates"][0]["code"] not in groups:
            reasons["no valid code"] += 1
            continue
        top = lab["candidates"][0]
        if top["confidence"] < args.min_confidence:
            reasons["low confidence"] += 1
            continue
        by_group[top["code"]].append((ad, r))

    # %% Apply the per-group minimum and cap
    # Groups smaller than --min-per-group are dropped entirely. Groups larger than --max-per-group are
    # trimmed at random with one seeded RNG shared across groups, so the draw depends on --seed and on
    # the dict iteration order (insertion order = first appearance of each code in the input).
    # reasons["capped"] counts ads removed by the cap, not the number of groups capped.
    rng = random.Random(args.seed)
    kept, kept_labels = [], []
    dropped_small = 0
    for code, items in by_group.items():
        if len(items) < args.min_per_group:
            dropped_small += len(items)
            continue
        # sampling without replacement; `items` is rebound so by_group keeps the full list for the tally
        if len(items) > args.max_per_group:
            items = rng.sample(items, args.max_per_group)
            reasons["capped"] += len(by_group[code]) - args.max_per_group
        for ad, r in items:
            kept.append(ad)
            kept_labels.append(r)
    reasons["below min-per-group"] = dropped_small

    # %% Write the subsample and its matching OSCA label file
    # Both files are written in the same group-major order so the ids line up line for line.
    # `default=str` guards against non-JSON values (e.g. datetimes) in the ad records.
    out_sample = SAMPLE_DIR / f"sample_{args.tag}.jsonl"
    out_labels = LABEL_DIR / f"osca_labels_{args.tag}_{args.model}.jsonl"
    with open(out_sample, "w", encoding="utf-8") as f:
        for ad in kept:
            f.write(json.dumps(ad, ensure_ascii=False, default=str) + "\n")
    with open(out_labels, "w", encoding="utf-8") as f:
        for r in kept_labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # %% Console summary
    # Group sizes are recomputed from the kept labels. The >= 100 / 50 / 20 counts are a quick check of
    # how many groups are large enough for an entry-vs-experienced comparison downstream.
    sizes = Counter(r["label"]["candidates"][0]["code"] for r in kept_labels)
    print(f"ads in: {len(ads):,} | kept: {len(kept):,} across {len(sizes)} OSCA unit groups | excluded: {dict(reasons)}")
    print(f"unit groups with >= 100 ads: {sum(1 for v in sizes.values() if v >= 100)}, >= 50: {sum(1 for v in sizes.values() if v >= 50)}, >= 20: {sum(1 for v in sizes.values() if v >= 20)}")
    print("largest groups:")
    for code, n in sizes.most_common(15):
        print(f"  {code} {groups[code]['title'][:45]:45s} {n}")
    print(f"wrote {out_sample}\nwrote {out_labels}")


if __name__ == "__main__":
    main()

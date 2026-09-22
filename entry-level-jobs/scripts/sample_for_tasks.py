"""Draw the ads to send to task matching: per (OSCA unit group, seniority level) cell, keep at
most --per-cell ads (random, seeded), otherwise keep them all. Unclear-level ads are dropped.

Each kept ad gets occupation metadata from the FULL labelled set (before sampling):
    occ_n_entry, occ_n_experienced, occ_entry_share
so the raw entry share is preserved even where a cell was capped.

Usage:
    python scripts/sample_for_tasks.py data/samples/sample_us_2026-09-17_selected.jsonl            # counts only
    python scripts/sample_for_tasks.py data/samples/sample_us_2026-09-17_selected.jsonl --write    # write files
    python scripts/sample_for_tasks.py ... --per-cell 200 --min-per-level 5

With --write, produces (tag = input tag + "_tasks"):
    data/samples/sample_<tag>.jsonl                        the sampled ads (+ occ_* metadata fields)
    data/labels/osca_labels_<tag>_<model>.jsonl            their OSCA labels
    data/labels/seniority_labels_<tag>_<model>.jsonl       their seniority labels
    data/occupation_meta_<tag>.csv                         one row per unit group: counts, entry share, sampled counts
so merge_labels.py and match_tasks.py run on the new sample file unchanged.
"""
# %% Imports
# Standard library only. Sibling modules: `llm` for the model registry (used to build label file names)
# and `label_osca.load_osca` for the OSCA code -> title map.
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make the sibling modules importable when run as `python scripts/sample_for_tasks.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL, MODELS  # noqa: E402
from label_osca import DEFAULT_OSCA, load_osca  # noqa: E402

# %% Directory layout and the two analysis levels
# Inputs are found by tag in data/labels (osca_labels_*, seniority_labels_*) and outputs go to
# data/samples, data/labels and data/ (occupation_meta CSV) under tag + "_tasks".
ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
SAMPLE_DIR = ROOT / "data" / "samples"
# label_seniority.py emits entry_level / experienced / unclear. Only the first two enter the analysis;
# "unclear" ads are dropped here and count toward neither the cells nor the occupation totals.
LEVELS = ("entry_level", "experienced")


# %% Command line
# input            a sample JSONL that has BOTH osca_labels_<tag>_<model>.jsonl and
#                  seniority_labels_<tag>_<model>.jsonl (normally the "_selected" sample)
# --per-cell       cap per (OSCA unit group, level) cell; seeded random draw (default 200)
# --min-per-level  drop unit groups with fewer than this many ads at EITHER level (default 0 = keep all)
# --write          write the sampled files; without it only the occupation_meta CSV and counts appear
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--per-cell", type=int, default=200, help="cap per (unit group, level) cell")
    ap.add_argument("--min-per-level", type=int, default=0, help="drop unit groups with fewer than this at either level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    # %% Tags and the OSCA code list
    # out_tag appends "_tasks": "us_2026-09-17_selected" -> "us_2026-09-17_selected_tasks".
    tag = Path(args.input).stem.replace("sample_", "")
    out_tag = tag + "_tasks"
    groups = load_osca(Path(DEFAULT_OSCA))

    # %% Load OSCA and seniority labels keyed by ad id
    # OSCA: labelled, issue-free records with at least one candidate (same rule as the selection step,
    # so on a "_selected" sample nothing should be lost here). Seniority: every labelled record; the
    # level filter is applied when the cells are built.
    osca, sen = {}, {}
    for line in open(LABEL_DIR / f"osca_labels_{tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r and r["label"]["issue"] == "none" and r["label"]["candidates"]:
            osca[r["id"]] = r
    for line in open(LABEL_DIR / f"seniority_labels_{tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r:
            sen[r["id"]] = r

    # %% Build (unit group, level) cells from the UNCAPPED labelled set
    # cells[(code, level)] = [ad ids]. An ad enters a cell only if its top-1 code is a known OSCA group,
    # it has a seniority label, and that level is in LEVELS. The gold overrides from the review
    # workbooks are NOT applied here (merge_labels.py applies them later); this stage uses model labels.
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for id_, r in osca.items():
        code = r["label"]["candidates"][0]["code"]
        s = sen.get(id_)
        if code in groups and s and s["label"]["level"] in LEVELS:
            cells[(code, s["label"]["level"])].append(id_)

    # %% Occupation metadata from the full counts (before any capping)
    # For each group with at least one cell: n_entry, n_experienced, entry_share = e / (e + x).
    # These are carried onto every sampled ad as occ_n_entry / occ_n_experienced / occ_entry_share, so
    # the raw entry share of the occupation survives even where a cell is capped at --per-cell.
    # A group with ads at only one level gets 0 for the empty cell (defaultdict) and a share of 0 or 1.
    # eligible: groups meeting --min-per-level at BOTH levels; others are excluded from sampling but
    # still appear in the occupation_meta CSV with eligible = False.
    meta = {}
    for code in {c for c, _ in cells}:
        e, x = len(cells[(code, "entry_level")]), len(cells[(code, "experienced")])
        meta[code] = {"n_entry": e, "n_experienced": x, "entry_share": round(e / (e + x), 4) if e + x else None}
    eligible = {c for c, m in meta.items() if m["n_entry"] >= args.min_per_level and m["n_experienced"] >= args.min_per_level}

    # %% Seeded per-cell sampling
    # Cells are visited in sorted (code, level) order with ONE shared RNG, so the draw is reproducible
    # for a given --seed and input but changes if any earlier cell's membership changes. A cell larger
    # than --per-cell is trimmed to exactly --per-cell ads; smaller cells are kept whole. Capping both
    # levels at the same size means the sampled entry share is ~50% for big groups; the true share is
    # in meta. chosen[ad id] = code marks the ads to send on; sampled[code][level] = kept counts.
    rng = random.Random(args.seed)
    chosen: dict[str, str] = {}
    sampled = defaultdict(lambda: {"entry_level": 0, "experienced": 0})
    for (code, level), ids in sorted(cells.items()):
        if code not in eligible:
            continue
        # random.sample = without replacement; `ids` is in OSCA label-file order
        keep = rng.sample(ids, args.per_cell) if len(ids) > args.per_cell else ids
        sampled[code][level] = len(keep)
        for i in keep:
            chosen[i] = code

    # %% Console counts
    # n_capped counts cells (not groups) that exceeded --per-cell and were trimmed.
    n_groups = len(sampled)
    n_capped = sum(1 for (c, l), ids in cells.items() if c in eligible and len(ids) > args.per_cell)
    print(f"unit groups: {len(meta)} labelled, {n_groups} eligible (min {args.min_per_level} per level) | "
          f"cells capped at {args.per_cell}: {n_capped}")
    print(f"ads to send to task matching: {len(chosen):,} "
          f"(entry {sum(v['entry_level'] for v in sampled.values()):,}, experienced {sum(v['experienced'] for v in sampled.values()):,})")

    # %% Write the occupation metadata CSV (always, even without --write)
    # One row per labelled unit group, largest first: uncapped counts, entry share, sampled counts and
    # whether it passed --min-per-level. This is the per-occupation table for the regression stage.
    meta_path = ROOT / "data" / f"occupation_meta_{out_tag}.csv"
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["osca_code", "osca_title", "n_entry", "n_experienced", "entry_share", "sampled_entry", "sampled_experienced", "eligible"])
        for code in sorted(meta, key=lambda c: -(meta[c]["n_entry"] + meta[c]["n_experienced"])):
            m = meta[code]
            w.writerow([code, groups[code]["title"], m["n_entry"], m["n_experienced"], m["entry_share"],
                        sampled[code]["entry_level"], sampled[code]["experienced"], code in eligible])
    print(f"wrote {meta_path}")

    # %% Write the sampled ads and their label files (--write only)
    # Re-reads the input so ads are emitted in input order. Each kept ad gains four fields
    # (occ_osca_code, occ_n_entry, occ_n_experienced, occ_entry_share) before being written. The OSCA
    # and seniority label files are written in the same order with the same ids under out_tag, so
    # merge_labels.py and match_tasks.py run on the new sample with no extra arguments.
    if not args.write:
        return
    out_sample = SAMPLE_DIR / f"sample_{out_tag}.jsonl"
    with open(out_sample, "w", encoding="utf-8") as fs, \
         open(LABEL_DIR / f"osca_labels_{out_tag}_{args.model}.jsonl", "w", encoding="utf-8") as fo, \
         open(LABEL_DIR / f"seniority_labels_{out_tag}_{args.model}.jsonl", "w", encoding="utf-8") as fsen:
        n = 0
        for line in open(args.input, encoding="utf-8"):
            ad = json.loads(line)
            code = chosen.get(ad["id"])
            if not code:
                continue
            m = meta[code]
            ad.update(occ_osca_code=code, occ_n_entry=m["n_entry"], occ_n_experienced=m["n_experienced"], occ_entry_share=m["entry_share"])
            fs.write(json.dumps(ad, ensure_ascii=False, default=str) + "\n")
            fo.write(json.dumps(osca[ad["id"]], ensure_ascii=False) + "\n")
            fsen.write(json.dumps(sen[ad["id"]], ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n:,} ads -> {out_sample} (+ matching OSCA and seniority label files)")
    print(f"next: python scripts/merge_labels.py {out_sample.relative_to(ROOT).as_posix()}\n"
          f"      python scripts/match_tasks.py {out_sample.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()

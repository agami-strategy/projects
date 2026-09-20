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
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import DEFAULT_MODEL, MODELS  # noqa: E402
from label_osca import DEFAULT_OSCA, load_osca  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "labels"
SAMPLE_DIR = ROOT / "data" / "samples"
LEVELS = ("entry_level", "experienced")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--per-cell", type=int, default=200, help="cap per (unit group, level) cell")
    ap.add_argument("--min-per-level", type=int, default=0, help="drop unit groups with fewer than this at either level")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    tag = Path(args.input).stem.replace("sample_", "")
    out_tag = tag + "_tasks"
    groups = load_osca(Path(DEFAULT_OSCA))

    osca, sen = {}, {}
    for line in open(LABEL_DIR / f"osca_labels_{tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r and r["label"]["issue"] == "none" and r["label"]["candidates"]:
            osca[r["id"]] = r
    for line in open(LABEL_DIR / f"seniority_labels_{tag}_{args.model}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if "label" in r:
            sen[r["id"]] = r

    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for id_, r in osca.items():
        code = r["label"]["candidates"][0]["code"]
        s = sen.get(id_)
        if code in groups and s and s["label"]["level"] in LEVELS:
            cells[(code, s["label"]["level"])].append(id_)

    meta = {}
    for code in {c for c, _ in cells}:
        e, x = len(cells[(code, "entry_level")]), len(cells[(code, "experienced")])
        meta[code] = {"n_entry": e, "n_experienced": x, "entry_share": round(e / (e + x), 4) if e + x else None}
    eligible = {c for c, m in meta.items() if m["n_entry"] >= args.min_per_level and m["n_experienced"] >= args.min_per_level}

    rng = random.Random(args.seed)
    chosen: dict[str, str] = {}
    sampled = defaultdict(lambda: {"entry_level": 0, "experienced": 0})
    for (code, level), ids in sorted(cells.items()):
        if code not in eligible:
            continue
        keep = rng.sample(ids, args.per_cell) if len(ids) > args.per_cell else ids
        sampled[code][level] = len(keep)
        for i in keep:
            chosen[i] = code

    n_groups = len(sampled)
    n_capped = sum(1 for (c, l), ids in cells.items() if c in eligible and len(ids) > args.per_cell)
    print(f"unit groups: {len(meta)} labelled, {n_groups} eligible (min {args.min_per_level} per level) | "
          f"cells capped at {args.per_cell}: {n_capped}")
    print(f"ads to send to task matching: {len(chosen):,} "
          f"(entry {sum(v['entry_level'] for v in sampled.values()):,}, experienced {sum(v['experienced'] for v in sampled.values()):,})")

    meta_path = ROOT / "data" / f"occupation_meta_{out_tag}.csv"
    with open(meta_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["osca_code", "osca_title", "n_entry", "n_experienced", "entry_share", "sampled_entry", "sampled_experienced", "eligible"])
        for code in sorted(meta, key=lambda c: -(meta[c]["n_entry"] + meta[c]["n_experienced"])):
            m = meta[code]
            w.writerow([code, groups[code]["title"], m["n_entry"], m["n_experienced"], m["entry_share"],
                        sampled[code]["entry_level"], sampled[code]["experienced"], code in eligible])
    print(f"wrote {meta_path}")

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

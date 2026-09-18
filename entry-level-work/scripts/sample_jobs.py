"""Draw a random sample of job ads from a downloaded snapshot for manual review.

Outputs (in data/samples/):
    sample_<n>.jsonl  - one job per line, all original fields plus `description_text`
    sample_<n>.md     - human-readable version for skimming

Usage:
    python scripts/sample_jobs.py --n 100 --seed 42
    python scripts/sample_jobs.py --n 100 --seed 42 --date 2026-09-17
    python scripts/sample_jobs.py --n 100 --country australia   # preset: 'australia' + capital cities in location text
    python scripts/sample_jobs.py --n 3000 --country australia --max-locations 3   # all Australian ads, no multi-country remote
    python scripts/sample_jobs.py --n 100 --country "new zealand,auckland,wellington"
"""
import argparse
import html
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "data"
OUT_DIR = ROOT / "data" / "samples"

# Location strings often name only a city. "victoria"/"nsw" etc. are left out because they
# collide with Victoria TX/BC and similar; state abbreviations are too short to be safe.
COUNTRY_PRESETS = {
    "australia": ["australia", "sydney", "melbourne", "brisbane", "perth", "adelaide", "canberra", "hobart", "darwin"],
}
# Same-named cities elsewhere (Melbourne FL, Brisbane CA, Perth Scotland): drop when these appear.
COUNTRY_EXCLUDE_REGEX = {
    "australia": r"united states|\busa\b|florida|california|scotland|united kingdom|\buk\b|,\s*(fl|ca|tx|ny|in|nj|pa|il|oh|ga|va|ma|nc|wi|mi|on|bc)\b",
}

_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|blockquote)[^>]*>", re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(raw: str | None) -> str:
    """Cheap HTML -> plain text that keeps paragraph / bullet structure."""
    if not raw:
        return ""
    s = html.unescape(raw)  # ATS feeds are often double-escaped
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = _BLOCK_TAGS.sub("\n", s)
    s = _TAGS.sub("", s)
    s = html.unescape(s)
    s = _WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = re.sub(r"^- *\n+", "- ", s, flags=re.M)  # <li><p>..</p></li> -> keep bullet text on same line
    s = _NL.sub("\n\n", s)
    return s.strip()


def available_dates() -> list[str]:
    return sorted(p.name.split("=")[1] for p in RAW_DIR.glob("date=*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=float, default=42, help="duckdb setseed() value; any number is fine")
    ap.add_argument("--date", default=None, help="snapshot date; default = latest downloaded")
    ap.add_argument("--country", default=None,
                    help="preset name (australia) or comma-separated substrings to match in location text, case-insensitive")
    ap.add_argument("--max-locations", type=int, default=None,
                    help="drop ads listing more than this many locations (multi-country remote ads)")
    ap.add_argument("--tag", default=None, help="suffix for output file names")
    args = ap.parse_args()

    dates = available_dates()
    if not dates:
        raise SystemExit("No snapshot found in data/raw. Run scripts/download_snapshot.py first.")
    date = args.date or dates[-1]
    glob = str(RAW_DIR / f"date={date}" / "*" / "*.parquet").replace("\\", "/")

    con = duckdb.connect()
    con.execute("SET threads = 1")  # ORDER BY random() is only reproducible single-threaded
    # setseed wants a value in [-1, 1]; map any seed into that range deterministically
    con.execute(f"SELECT setseed({(args.seed % 2) - 1})")
    conds = []
    if args.country:
        terms = COUNTRY_PRESETS.get(args.country.lower()) or [t.strip().lower() for t in args.country.split(",") if t.strip()]
        loc = "lower(array_to_string(locations, ' | '))"
        conds.append("(" + " OR ".join(f"{loc} LIKE '%{t}%'" for t in terms) + ")")
        if args.country.lower() in COUNTRY_EXCLUDE_REGEX:
            conds.append(f"NOT regexp_matches({loc}, '{COUNTRY_EXCLUDE_REGEX[args.country.lower()]}')")
    if args.max_locations is not None:
        conds.append(f"len(locations) <= {args.max_locations}")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = con.execute(
        f"""
        SELECT id, source, source_slug, title, apply_url, description_html, employment_type,
               department, locations, remote, posted_at, salary_min, salary_max,
               salary_currency, salary_period
        FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=true)
        {where}
        ORDER BY random()
        LIMIT {args.n}
        """
    ).fetchall()
    cols = [d[0] for d in con.description]

    country_slug = re.sub(r"[^a-z0-9]+", "-", args.country.lower()).strip("-") if args.country else ""
    tag = args.tag or (f"{args.n}_{country_slug}" if args.country else str(args.n))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"sample_{tag}.jsonl"
    md_path = OUT_DIR / f"sample_{tag}.md"

    with jsonl_path.open("w", encoding="utf-8") as fj, md_path.open("w", encoding="utf-8") as fm:
        fm.write(f"# Sample of {len(rows)} job ads (snapshot {date}, seed {args.seed}"
                 f"{', country=' + args.country if args.country else ''})\n\n")
        for i, r in enumerate(rows, 1):
            rec = dict(zip(cols, r))
            rec["description_text"] = html_to_text(rec["description_html"])
            fj.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

            locs = ", ".join(rec["locations"] or []) or "(none)"
            sal = ""
            if rec["salary_min"] or rec["salary_max"]:
                sal = f" | salary: {rec['salary_min']}-{rec['salary_max']} {rec['salary_currency'] or ''}"
            fm.write(f"## {i}. {rec['title']}\n\n")
            fm.write(f"**Company:** {rec['source_slug']} ({rec['source']}) | **Dept:** {rec['department'] or '-'} | "
                     f"**Type:** {rec['employment_type'] or '-'} | **Remote:** {rec['remote']}{sal}\n\n")
            fm.write(f"**Location:** {locs}\n\n**URL:** {rec['apply_url']}\n\n")
            fm.write(rec["description_text"] + "\n\n---\n\n")

    print(f"wrote {len(rows)} jobs -> {jsonl_path} and {md_path}")


if __name__ == "__main__":
    main()

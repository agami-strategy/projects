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
# %% Imports and paths
# argparse for the CLI; html and re for the HTML-to-text converter; json for the JSONL writer;
# duckdb reads the parquet snapshot directly (no pandas). RAW_DIR is data/raw/data: the extra "data"
# level is the repo layout mirrored by download_snapshot.py. OUT_DIR receives the sample files.
# This module is also imported (not run) by the three collector scripts for country_condition and
# html_to_text, so nothing below the constants should have import-time side effects.
import argparse
import html
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "data"
OUT_DIR = ROOT / "data" / "samples"

# %% US location vocabulary
# Three alternation strings spliced into DuckDB regexes by the "us" preset. _US_STATES and _US_CITIES
# are matched against the lower-cased location text; _US_ABBR is matched case-sensitively (see
# COUNTRY_PRESET_REGEX_CS) because lower-cased codes such as "in", "or", "me", "hi" are ordinary words.
# "washington" is in the state list, so "Washington, DC" also matches via the state alternation.
# Reviewer: this is a hand-built list, not a gazetteer. A US ad that names only a smaller city with
# no state ("Boulder") is missed, and city names are plain substrings ("austin" matches "Austintown").
# Location strings often name only a city. "victoria"/"nsw" etc. are left out because they
# collide with Victoria TX/BC and similar; state abbreviations are too short to be safe.
_US_STATES = ("alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|"
              "indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|"
              "nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
              "pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|"
              "wisconsin|wyoming")
# Major tech-hiring cities plus "bay area". "\\b" in a non-raw Python string becomes "\b" in the SQL,
# which DuckDB reads as a word boundary, so "nyc" and "sf" must stand alone. "washington, d\\.?c"
# covers both "Washington, DC" and "Washington, D.C.".
_US_CITIES = ("san francisco|los angeles|chicago|boston|seattle|austin|denver|atlanta|san jose|san diego|dallas|houston|miami|"
              "palo alto|mountain view|menlo park|sunnyvale|redwood city|santa clara|oakland|brooklyn|philadelphia|phoenix|"
              "portland|minneapolis|nashville|charlotte|raleigh|salt lake city|pittsburgh|detroit|las vegas|san mateo|irvine|"
              "bay area|\\bnyc\\b|\\bsf\\b|washington, d\\.?c")
# Two-letter state codes (plus DC) for the ", XX" pattern. Only used in the case-sensitive regex.
_US_ABBR = ("AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|"
            "RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC")

# %% Country presets: inclusion terms
# COUNTRY_PRESETS maps a preset name to a list of terms OR-ed together in country_condition().
# Australia: the country name or one of the eight capital cities. No states or abbreviations, for the
# reason in the comment below. US: country name, "usa", "u.s." plus the state and city alternations.
# A term containing "|" or "\" is treated as a regex; anything else becomes a LIKE '%term%'.
# These presets define the study population, so any change here changes every downstream sample.
# Location strings often name only a city. "victoria"/"nsw" etc. are left out because they
# collide with Victoria TX/BC and similar; state abbreviations are too short to be safe.
COUNTRY_PRESETS = {
    "australia": ["australia", "sydney", "melbourne", "brisbane", "perth", "adelaide", "canberra", "hobart", "darwin"],
    "us": ["united states", "\\busa\\b", "\\bu\\.s\\.", _US_STATES, _US_CITIES],
}
# %% Country presets: case-sensitive additions
# Applied to the raw (not lower-cased) location text and OR-ed with the inclusion terms. The only
# entry is the US ", TX" style: a comma, optional spaces, a two-letter code, then a word boundary.
# Case-sensitive regex applied to the raw location text in addition to the lower-cased term match.
COUNTRY_PRESET_REGEX_CS = {
    "us": r",\s*(" + _US_ABBR + r")\b",  # "Austin, TX" style
}
# %% Country presets: exclusions
# AND NOT-ed after the inclusions, so a single foreign marker anywhere in the location list removes
# the ad even when it also names an Australian city. This is how "Melbourne, FL", "Brisbane, CA" and
# "Perth, Scotland" are dropped, and also how a genuinely multi-country ad ("Sydney | London") is
# dropped. Australia's list ends with a lower-case ", xx" state/province pattern (US states plus
# Ontario and BC). The US list names other countries, Canadian provinces and big non-US cities.
# Reviewer: the exclusions are matched on lower-cased text, so ", in" would also fire on
# "Sydney, in the CBD" - rare in ATS location fields, but possible.
# Same-named cities elsewhere (Melbourne FL, Brisbane CA, Perth Scotland): drop when these appear.
COUNTRY_EXCLUDE_REGEX = {
    "australia": r"united states|\busa\b|florida|california|scotland|united kingdom|\buk\b|,\s*(fl|ca|tx|ny|in|nj|pa|il|oh|ga|va|ma|nc|wi|mi|on|bc)\b",
    "us": (r"canada|united kingdom|\buk\b|australia|germany|india|mexico|brazil|ireland|france|spain|netherlands|singapore|"
           r"japan|ontario|british columbia|quebec|london|toronto|vancouver|sydney|melbourne|bangalore|bengaluru|dublin|berlin|paris|amsterdam"),
}


# %% Build the SQL WHERE fragment for a country
# Input: a preset name ("australia", "us") or a comma-separated list of substrings. Output: a string
# like "(lower(...) LIKE '%australia%' OR regexp_matches(...)) AND NOT regexp_matches(...)" that the
# caller splices into a DuckDB query. `locations` is a list column, so it is joined with ' | ' first
# and a hit anywhere in any listed location counts. An ad whose only location is "Remote" matches
# nothing and is therefore excluded from every country sample.
# Shared by scan_australian_ads.py, collect_recent_days.py and collate_australian_ads.py.
# Known limits: terms are not SQL-escaped (a user term containing a single quote breaks the query);
# plain terms match inside longer words ("perth" also matches "Perthshire").
def country_condition(country: str) -> str:
    """SQL WHERE fragment selecting ads whose location text matches a preset (or comma-separated terms)."""
    # Two views of the same text: lower-cased for the term list, raw for the case-sensitive regex.
    loc = "lower(array_to_string(locations, ' | '))"
    loc_cs = "array_to_string(locations, ' | ')"
    key = country.lower()
    # Preset lookup first; if the name is not a preset, split the argument on commas into terms.
    terms = COUNTRY_PRESETS.get(key) or [t.strip().lower() for t in country.split(",") if t.strip()]
    # Regex terms (any "|" or "\" inside) go through regexp_matches; everything else is a LIKE.
    parts = [f"regexp_matches({loc}, '{t}')" if any(ch in t for ch in "|\\") else f"{loc} LIKE '%{t}%'" for t in terms]
    if key in COUNTRY_PRESET_REGEX_CS:
        parts.append(f"regexp_matches({loc_cs}, '{COUNTRY_PRESET_REGEX_CS[key]}')")
    cond = "(" + " OR ".join(parts) + ")"
    # Exclusions apply only to presets; user-supplied term lists get no exclusion.
    if key in COUNTRY_EXCLUDE_REGEX:
        cond += f" AND NOT regexp_matches({loc}, '{COUNTRY_EXCLUDE_REGEX[key]}')"
    return cond

# %% HTML -> text regexes
# Compiled once at import. _BLOCK_TAGS: opening or closing block-level tags (p, div, br, li, ul, ol,
# headings, table parts...), replaced by a newline so paragraphs and bullets stay on separate lines.
# _TAGS: any remaining tag, deleted. _WS: runs of horizontal whitespace (not newlines), squeezed to
# one space. _NL: three or more consecutive newlines, collapsed to one blank line.
_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article|blockquote)[^>]*>", re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


# %% HTML -> text conversion
# Produces `description_text` from `description_html` for every sample and collector script. It keeps
# bullet structure (a "- " prefix per <li>) because the downstream task-extraction step reads
# responsibilities from bullet lists. Input: raw HTML or None. Output: stripped plain text.
# This is deliberately cheap and not a real HTML parser: entities are unescaped, tags are removed,
# links lose their href, table cells on one row run together. Good enough for LLM labelling.
def html_to_text(raw: str | None) -> str:
    """Cheap HTML -> plain text that keeps paragraph / bullet structure."""
    if not raw:
        return ""
    # First unescape: some feeds ship "&lt;p&gt;" instead of "<p>", so the tags only become visible
    # (and removable) after one round of unescaping.
    s = html.unescape(raw)  # ATS feeds are often double-escaped
    # Each opening <li> becomes newline + "- " so every bullet starts its own line.
    s = re.sub(r"<li[^>]*>", "\n- ", s, flags=re.I)
    s = _BLOCK_TAGS.sub("\n", s)
    s = _TAGS.sub("", s)
    # Second unescape: turns "&amp;" left over from the double escaping into "&", etc.
    s = html.unescape(s)
    s = _WS.sub(" ", s)
    # Strip indentation and trailing spaces on every line; HTML source formatting is noise here.
    s = "\n".join(line.strip() for line in s.split("\n"))
    # A bullet whose text was wrapped in <p> now reads "- \n text"; pull the text back onto the line.
    s = re.sub(r"^- *\n+", "- ", s, flags=re.M)  # <li><p>..</p></li> -> keep bullet text on same line
    s = _NL.sub("\n\n", s)
    return s.strip()


# %% List locally downloaded snapshot dates
# Looks for data/raw/data/date=* folders and returns their dates sorted ascending; main() uses the
# last one as the default snapshot. A partially downloaded day would also appear here.
def available_dates() -> list[str]:
    return sorted(p.name.split("=")[1] for p in RAW_DIR.glob("date=*"))


# %% Main: CLI, DuckDB sample, JSONL + Markdown output
def main() -> None:
    # --- CLI ---
    # --n sample size; --seed reproducibility; --date which local snapshot; --country location filter;
    # --max-locations cap on listed locations (multi-country remote ads); --tag output-name suffix.
    # The defaults reproduce the original 100-ad manual review sample.
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

    # --- Resolve the snapshot to read ---
    # Default is the newest date on disk. The glob spans all three source=<ats> folders and every
    # parquet file in them. Forward slashes because DuckDB mis-parses Windows backslashes in globs.
    dates = available_dates()
    if not dates:
        raise SystemExit("No snapshot found in data/raw. Run scripts/download_snapshot.py first.")
    date = args.date or dates[-1]
    glob = str(RAW_DIR / f"date={date}" / "*" / "*.parquet").replace("\\", "/")

    # --- Random sample inside DuckDB ---
    # Sampling is ORDER BY random() LIMIT n. Reproducibility needs both threads=1 (a parallel scan
    # changes row order and hence the draw) and setseed(). The seed is mapped into setseed's [-1, 1]
    # range with (seed % 2) - 1: 42 -> -1, 42.5 -> -0.5. Reviewer: every even integer maps to -1 and
    # every odd integer to 0, so integer seeds give only two distinct draws; use fractional seeds if
    # you need more than two independent samples from one snapshot.
    con = duckdb.connect()
    con.execute("SET threads = 1")  # ORDER BY random() is only reproducible single-threaded
    # setseed wants a value in [-1, 1]; map any seed into that range deterministically
    con.execute(f"SELECT setseed({(args.seed % 2) - 1})")
    # WHERE clause is assembled from the optional country filter and the optional location cap.
    conds = []
    if args.country:
        conds.append(country_condition(args.country))
    if args.max_locations is not None:
        conds.append(f"len(locations) <= {args.max_locations}")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    # hive_partitioning=1 exposes the date= and source= folder names as columns. union_by_name=true is
    # required because one day (2026-04-17) stores salary_period as VARCHAR while the others store a
    # NULL-typed column; without it read_parquet fails on the schema mismatch. Known quirk: some days
    # carry duplicated parquet files, and this query does not dedupe by id, so the same ad can appear
    # twice in a sample drawn from such a day.
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
    # Column names in SELECT order, used to turn each tuple into a dict.
    cols = [d[0] for d in con.description]

    # --- Output file names ---
    # Tag is "<n>" or "<n>_<country-slug>" unless --tag overrides. The slug replaces any run of
    # non-alphanumerics with "-", so "new zealand,auckland" -> "new-zealand-auckland".
    country_slug = re.sub(r"[^a-z0-9]+", "-", args.country.lower()).strip("-") if args.country else ""
    tag = args.tag or (f"{args.n}_{country_slug}" if args.country else str(args.n))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / f"sample_{tag}.jsonl"
    md_path = OUT_DIR / f"sample_{tag}.md"

    # --- Write JSONL and Markdown ---
    # JSONL: one dict per ad with every selected column plus description_text. description_html is
    # kept here (the multi-day collectors drop it to save space). default=str serialises datetimes
    # and Decimals. Markdown: a "## i. title" section per ad with a one-line header and the text,
    # meant for skimming in an editor. `locations` may be None, hence the `or []`.
    with jsonl_path.open("w", encoding="utf-8") as fj, md_path.open("w", encoding="utf-8") as fm:
        fm.write(f"# Sample of {len(rows)} job ads (snapshot {date}, seed {args.seed}"
                 f"{', country=' + args.country if args.country else ''})\n\n")
        for i, r in enumerate(rows, 1):
            rec = dict(zip(cols, r))
            rec["description_text"] = html_to_text(rec["description_html"])
            fj.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

            # Salary line only when at least one bound is present; currency may be missing.
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

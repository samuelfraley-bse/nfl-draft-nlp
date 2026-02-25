"""
Scraper for nfldraftbuzz.com individual player scouting report pages.

Reads player_url from scripts/data/nfldraftbuzz_rankings.csv and scrapes
each player page, extracting metadata and scouting report sections.

Two page formats exist on the site:
  - "structured"   : named playerDesc* divs (Bio, Strengths, Weaknesses,
                     Summary, Honors).  Most players from ~2022+.
  - "unstructured" : single post__content div with h5/h6/<p> blocks.
                     Common in 2021 and some 2022 entries.

Output:    scripts/data/nfldraftbuzz_scouting.csv
Checkpoint: scripts/data/.player_scrape_checkpoint.json  (set of done URLs)
"""

import csv
import json
import time
import os
from bs4 import BeautifulSoup
from curl_cffi import requests
import pandas as pd

BASE_URL = "https://www.nfldraftbuzz.com"
IMPERSONATE = "chrome124"

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RANKINGS_CSV   = os.path.join(_DATA_DIR, "nfldraftbuzz_rankings.csv")
OUTPUT_CSV     = os.path.join(_DATA_DIR, "nfldraftbuzz_scouting.csv")
CHECKPOINT_PATH = os.path.join(_DATA_DIR, ".player_scrape_checkpoint.json")

DELAY = 1.2  # seconds between requests

CSV_COLUMNS = [
    # --- join keys ---
    "player_url",
    "year",
    "rank",
    # --- from rankings csv ---
    "first_name",
    "last_name",
    # --- from player page metadata ---
    "position",
    "college",
    "height",
    "weight",
    "player_class",
    "hometown",
    # --- report ---
    "report_format",   # "structured" | "unstructured"
    "bio",
    "strengths",
    "weaknesses",
    "summary",
    "honors",
    "raw_text",        # full concatenated text (all formats)
]

# Map normalised section header keywords → CSV column
SECTION_MAP = {
    "bio":        "bio",
    "strength":   "strengths",
    "weakness":   "weaknesses",
    "summary":    "summary",
    "honor":      "honors",
    "award":      "honors",
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return " ".join(text.split())


def parse_metadata(soup: BeautifulSoup) -> dict:
    """Extract Height/Weight/College/Position/Class/Hometown from sidebar."""
    meta = {}
    for item in soup.find_all("div", class_="player-info-details__item"):
        parts = [p.strip() for p in item.get_text(separator="|").split("|") if p.strip()]
        if len(parts) == 2:
            label, value = parts
            meta[label.lower()] = value
    return {
        "position":     meta.get("position", ""),
        "college":      meta.get("college", ""),
        "height":       meta.get("height", ""),
        "weight":       meta.get("weight", ""),
        "player_class": meta.get("class", ""),
        "hometown":     meta.get("home town", meta.get("hometown", "")),
    }


def _map_header(header_text: str) -> str:
    """Map a section header string to a CSV column name, or '' if unknown."""
    lower = header_text.lower()
    for keyword, col in SECTION_MAP.items():
        if keyword in lower:
            return col
    return ""


def parse_structured(soup: BeautifulSoup) -> dict:
    """Parse pages that use playerDescIntro / playerDescPro / playerDescNeg divs."""
    sections = {col: "" for col in ["bio", "strengths", "weaknesses", "summary", "honors"]}
    all_text_parts = []

    for div in soup.find_all("div", class_=lambda c: c and "playerDesc" in str(c)):
        h_tag = div.find(["h4", "h5", "h6"])
        header = h_tag.get_text(strip=True) if h_tag else ""
        if h_tag:
            h_tag.extract()          # remove header from text extraction
        text = _clean(div.get_text(separator=" "))

        col = _map_header(header)
        if col:
            sections[col] = text
        all_text_parts.append(text)

    return {**sections, "raw_text": " ".join(all_text_parts), "report_format": "structured"}


# Text after these markers is noise (navigation, footnotes, share widgets)
_NOISE_MARKERS = [
    "How other scouting services rate",
    "Share scouting report",
    "PercentileRanking",
    "Previous:",
    "Next:",
]


def parse_unstructured(soup: BeautifulSoup) -> dict:
    """Parse pages where all content lives in a single post__content div.

    Handles three sub-formats found in the wild:
      1. <h5>/<h6> + <p> paragraphs  (e.g. Trevor Lawrence)
      2. <p> only, no headings        (e.g. Asante Samuel Jr)
      3. <h5> + <ul>/<li> bullets     (e.g. Michael Carter) — no <p> tags at all
    """
    sections = {col: "" for col in ["bio", "strengths", "weaknesses", "summary", "honors"]}

    pc = soup.find("div", class_="post__content")
    if not pc:
        return {**sections, "raw_text": "", "report_format": "unstructured"}

    # raw_text: use full get_text() then trim everything after the first noise marker
    full_text = _clean(pc.get_text(separator=" "))
    for marker in _NOISE_MARKERS:
        idx = full_text.find(marker)
        if idx > 50:  # guard: only trim if marker is past meaningful content
            full_text = full_text[:idx].strip()
            break
    raw_text = full_text

    # Section mapping: walk headings + ALL content-bearing tags (p AND li)
    current_col = ""
    section_texts: dict[str, list] = {}

    for tag in pc.find_all(["h4", "h5", "h6", "p", "li"], recursive=True):
        if tag.name in ("h4", "h5", "h6"):
            current_col = _map_header(tag.get_text(strip=True))
        else:  # p or li
            text = _clean(tag.get_text())
            if text and current_col:
                section_texts.setdefault(current_col, []).append(text)

    for col, parts in section_texts.items():
        sections[col] = " ".join(parts)

    return {**sections, "raw_text": raw_text, "report_format": "unstructured"}


def scrape_player(session, url: str) -> dict:
    """Fetch and parse a single player page. Returns a dict of report fields."""
    resp = session.get(url, impersonate=IMPERSONATE, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")

    meta = parse_metadata(soup)

    # Determine format
    has_structured = bool(soup.find("div", class_=lambda c: c and "playerDesc" in str(c)))
    if has_structured:
        report = parse_structured(soup)
    else:
        report = parse_unstructured(soup)

    return {**meta, **report}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_DATA_DIR, exist_ok=True)

    # Load rankings as the source of player URLs + join metadata
    df = pd.read_csv(RANKINGS_CSV)
    # Drop duplicates: same player can appear multiple times per year if rankings
    # data had duplicates; keep one row per (player_url, year)
    df = df.drop_duplicates(subset=["player_url", "year"])
    total = len(df)
    print(f"Loaded {total} player-year rows from rankings CSV.")

    checkpoint = load_checkpoint()
    already_done = len(checkpoint)
    if already_done:
        print(f"Checkpoint: {already_done} URLs already scraped. Resuming...")

    # Open CSV for appending
    csv_is_new = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if csv_is_new:
        writer.writeheader()
        csv_file.flush()

    session = requests.Session(impersonate=IMPERSONATE)

    scraped_this_run = 0
    errors = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        url = row.player_url

        if url in checkpoint:
            continue  # already done

        print(
            f"  [{i}/{total}] {row.year} #{row.rank:>4}  "
            f"{row.first_name} {row.last_name} ({row.position}) ...",
            end=" ",
            flush=True,
        )

        try:
            page_data = scrape_player(session, url)
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            # Don't checkpoint on error — will retry next run
            time.sleep(DELAY)
            continue

        record = {
            "player_url":  url,
            "year":        row.year,
            "rank":        row.rank,
            "first_name":  row.first_name,
            "last_name":   row.last_name,
            **page_data,
        }

        writer.writerow(record)
        csv_file.flush()

        checkpoint.add(url)
        # Save checkpoint every 10 players to avoid excessive I/O
        if scraped_this_run % 10 == 0:
            save_checkpoint(checkpoint)

        fmt = page_data.get("report_format", "?")
        raw_len = len(page_data.get("raw_text", ""))
        print(f"ok  [{fmt}, {raw_len} chars]")

        scraped_this_run += 1
        time.sleep(DELAY)

    save_checkpoint(checkpoint)
    csv_file.close()

    print(f"\nDone. Scraped {scraped_this_run} players this run. Errors: {errors}.")
    print(f"CSV: {os.path.abspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()

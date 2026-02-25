"""
Scraper for nfldraftbuzz.com prospect rankings.

Fetches all player rows from /positions/ALL/{page}/{year}/RATING/DESC
for years 2021-2027, writing results to CSV incrementally.

Resume support: a checkpoint file tracks the last completed page per year.
If interrupted, re-running will continue from where it left off.

Output: scripts/data/nfldraftbuzz_rankings.csv
Checkpoint: scripts/data/.scrape_checkpoint.json
"""

import csv
import json
import time
import os
from bs4 import BeautifulSoup
from curl_cffi import requests

BASE_URL = "https://www.nfldraftbuzz.com"
YEARS = range(2021, 2028)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(_DATA_DIR, "nfldraftbuzz_rankings.csv")
CHECKPOINT_PATH = os.path.join(_DATA_DIR, ".scrape_checkpoint.json")

IMPERSONATE = "chrome124"  # browser fingerprint for Cloudflare bypass

CSV_COLUMNS = [
    "year",
    "rank",
    "first_name",
    "last_name",
    "position",
    "nfl_team",
    "weight",
    "height",
    "forty_yard",
    "avg_pos_rank",
    "avg_ovr_rank",
    "rating",
    "summary",
    "player_url",
]

DELAY_BETWEEN_PAGES = 1.5  # seconds


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_last_page(soup: BeautifulSoup) -> int:
    """Return the last page number from the pagination nav (varies per year)."""
    nav = soup.find("nav", {"aria-label": "Blog navigation"})
    if not nav:
        return 1
    max_page = 1
    for link in nav.find_all("a", class_="page-link"):
        text = link.get_text(strip=True)
        if text.isdigit():
            max_page = max(max_page, int(text))
    return max_page


def parse_player_row(tr, year: int) -> dict:
    """Extract all fields from a single player <tr> element."""
    player_url = tr.get("data-href", "")
    tds = tr.find_all("td")

    # Rank
    rank_tag = tds[0].find("i")
    rank = rank_tag.get_text(strip=True) if rank_tag else ""

    # Name
    fn_tag = tds[1].find("span", class_="firstName")
    ln_tag = tds[1].find("span", class_="lastName")
    first_name = fn_tag.get_text(strip=True) if fn_tag else ""
    last_name = ln_tag.get_text(strip=True) if ln_tag else ""

    # Position (desktop column, index 3)
    position = tds[3].get_text(strip=True) if len(tds) > 3 else ""

    # NFL Team — from logo img alt, e.g. "JAX   Mascot" -> "JAX"
    nfl_team = ""
    if len(tds) > 4:
        img = tds[4].find("img")
        if img:
            alt = img.get("alt", "").split()
            nfl_team = alt[0] if alt else ""

    # Weight (5), Height (6), 40YD (7)
    weight    = tds[5].get_text(strip=True) if len(tds) > 5 else ""
    height    = tds[6].get_text(strip=True) if len(tds) > 6 else ""
    forty_yard = tds[7].get_text(strip=True) if len(tds) > 7 else ""

    # Avg Pos Rank (8), Avg Ovr Rank (9)
    avg_pos_rank = tds[8].get_text(strip=True) if len(tds) > 8 else ""
    avg_ovr_rank = tds[9].get_text(strip=True) if len(tds) > 9 else ""

    # Rating — first <div> inside whiteBold td (index 10)
    rating = ""
    if len(tds) > 10:
        div = tds[10].find("div")
        if div:
            rating = div.get_text(strip=True)

    # Summary (index 11)
    summary = tds[11].get_text(strip=True) if len(tds) > 11 else ""

    return {
        "year": year,
        "rank": rank,
        "first_name": first_name,
        "last_name": last_name,
        "position": position,
        "nfl_team": nfl_team,
        "weight": weight,
        "height": height,
        "forty_yard": forty_yard,
        "avg_pos_rank": avg_pos_rank,
        "avg_ovr_rank": avg_ovr_rank,
        "rating": rating,
        "summary": summary,
        "player_url": BASE_URL + player_url,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_DATA_DIR, exist_ok=True)

    checkpoint = load_checkpoint()

    # Open CSV in append mode; write header only if file is new/empty
    csv_is_new = not os.path.exists(OUTPUT_PATH) or os.path.getsize(OUTPUT_PATH) == 0
    csv_file = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if csv_is_new:
        writer.writeheader()
        csv_file.flush()

    session = requests.Session(impersonate=IMPERSONATE)

    total_written = 0

    for year in YEARS:
        key = str(year)
        last_done = checkpoint.get(key, {}).get("last_page", 0)
        known_total = checkpoint.get(key, {}).get("total_pages")

        # Skip fully completed years
        if known_total and last_done >= known_total:
            print(f"\nYear {year}: already complete ({known_total} pages). Skipping.")
            continue

        print(f"\n{'='*52}")
        if last_done > 0:
            print(f"Resuming {year} from page {last_done + 1} ...")
        else:
            print(f"Scraping {year} ...")

        page = last_done + 1
        year_written = 0
        total_pages = known_total  # may be None on first run

        while True:
            url = f"{BASE_URL}/positions/ALL/{page}/{year}/RATING/DESC"

            try:
                resp = session.get(url, impersonate=IMPERSONATE, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                print(f"  Page {page}: ERROR - {e}")
                break

            soup = BeautifulSoup(resp.content, "lxml")

            # Detect total pages (do this every page in case it wasn't known yet)
            if total_pages is None:
                total_pages = get_last_page(soup)

            table = soup.find("table", id="positionRankTable")
            if not table:
                print(f"  Page {page}: no table found, stopping year {year}.")
                break

            rows = table.find_all("tr", {"data-href": True})
            if not rows:
                print(f"  Page {page}: no player rows, stopping year {year}.")
                break

            page_players = []
            for tr in rows:
                try:
                    page_players.append(parse_player_row(tr, year))
                except Exception as e:
                    print(f"    Warning: failed to parse a row: {e}")

            # Write this page's rows immediately to disk
            writer.writerows(page_players)
            csv_file.flush()
            year_written += len(page_players)
            total_written += len(page_players)

            # Save checkpoint after every page
            checkpoint[key] = {"last_page": page, "total_pages": total_pages}
            save_checkpoint(checkpoint)

            print(
                f"  [{year}] Page {page}/{total_pages}"
                f"  -> {len(page_players)} players"
                f"  (year total: {year_written}, overall: {total_written})"
            )

            if page >= total_pages:
                print(f"  Year {year} complete.")
                break

            page += 1
            time.sleep(DELAY_BETWEEN_PAGES)

    csv_file.close()
    print(f"\nDone. {total_written} rows written this session.")
    print(f"CSV: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

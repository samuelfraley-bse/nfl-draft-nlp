"""
Join nfl_contracts.csv onto nfl_draft_enriched_all.csv by player name.

Strategy:
  1. Normalize names: lowercase, strip suffixes (Jr./Sr./II/III/IV).
  2. Split into first name and last name.
  3. Exact match on last name to get candidates (eliminates cross-last-name
     false positives entirely).
  4. If exactly one candidate shares last name + draft year + draft team,
     accept it even with a nickname mismatch (Bucky/Temuchin, Sauce/Ahmad).
     Two unrelated players sharing all three identifiers is essentially impossible.
  5. Otherwise, fuzzy match on first name among same-last-name candidates.
     This handles Josh/Joshua, Pat/Patrick, etc. without any lookup table.
  6. Flag borderline first-name matches for manual review.
  If no candidate shares the exact last name -> no_match immediately.

Output: data/processed/draft_enriched_with_contracts.csv
  - contract columns are suffixed with _contract to avoid collisions
  - match_score: 0-100 fuzzy score on first name (100 = exact)
  - match_flag: "exact", "fuzzy_high", "team_match", "no_match"
"""

import re
import logging
import pandas as pd
from rapidfuzz import fuzz, process
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

THRESHOLD_HIGH = 90   # minimum first-name score for fuzzy path (no fuzzy_low — use year+team instead)

SUFFIX_RE = re.compile(
    r"\b(jr\.?|sr\.?|ii|iii|iv|v|esq\.?)\s*$", re.IGNORECASE
)

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(",", " ")          # "Jones, II" -> "Jones  II"
    name = name.replace(".", "")           # "J.C." -> "JC", "Jr." -> "Jr"
    name = " ".join(name.split())          # collapse extra spaces
    name = SUFFIX_RE.sub("", name).strip()
    return name

def team_nick(name) -> str:
    """Normalize team name to its nickname for comparison.
    'Los Angeles Chargers' -> 'chargers', 'Chargers' -> 'chargers'
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().lower().split()[-1]

def split_name(name: str) -> tuple[str, str]:
    parts = name.split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return name, name

def first_name_score(a: str, b: str, **kwargs) -> float:
    """Boost score when one name is a prefix of the other (Will/William, Pat/Patrick, etc.)."""
    if a.startswith(b) or b.startswith(a):
        return 95.0
    return fuzz.ratio(a, b)

ROOT = Path(__file__).parents[1]
DRAFT_PATH    = ROOT / "data" / "raw" / "nfl_com" / "nfl_draft_enriched_all.csv"
CONTRACT_PATH = ROOT / "data" / "raw" / "contract" / "nfl_contracts.csv"
OUT_PATH      = ROOT / "data" / "processed" / "draft_enriched_with_contracts.csv"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.info(f"Loading draft data from {DRAFT_PATH}")
draft     = pd.read_csv(DRAFT_PATH)
logging.info(f"Loading contract data from {CONTRACT_PATH}")
contracts = pd.read_csv(CONTRACT_PATH)
logging.info(f"Draft rows: {len(draft)} | Contract rows: {len(contracts)}")

draft["_norm"]          = draft["player_name"].apply(normalize)
contracts["_norm"]      = contracts["player"].apply(normalize)
draft[["_first","_last"]]     = pd.DataFrame(draft["_norm"].apply(split_name).tolist(), index=draft.index)
contracts[["_first","_last"]] = pd.DataFrame(contracts["_norm"].apply(split_name).tolist(), index=contracts.index)

# Index contracts by last name for fast lookup
last_name_groups = contracts.groupby("_last")

contract_cols = [c for c in contracts.columns if c not in ("_norm", "_first", "_last")]

results = []
total_rows = len(draft)

for i, (_, row) in enumerate(draft.iterrows(), 1):
    if i % 250 == 0 or i == total_rows:
        logging.info(f"Progress: {i}/{total_rows} rows processed")

    last  = row["_last"]
    first = row["_first"]
    draft_year       = row.get("year", None)
    draft_team_nick  = team_nick(row.get("team", ""))

    # Step 1: exact last-name filter
    if last not in last_name_groups.groups:
        flag, score = "no_match", 0
        match_row = pd.Series({c: pd.NA for c in contract_cols})
        logging.debug(f"NO MATCH (last name) | {row['player_name']!r}")
    else:
        candidates = last_name_groups.get_group(last)

        # Step 2: narrow to candidates matching BOTH draft year AND draft team.
        # Requiring two independent identifiers makes false positives essentially
        # impossible (different people with same last name rarely share year+team).
        year_team_cands = candidates[
            (candidates["draft_year"] == draft_year) &
            (candidates["draft_team"].apply(team_nick) == draft_team_nick)
        ] if (draft_year and draft_team_nick) else pd.DataFrame()

        if len(year_team_cands) == 1:
            # Unique last-name + year + team match: accept even with nickname mismatch
            idx = year_team_cands.index[0]
            match_row = contracts.loc[idx, contract_cols]
            team_first = contracts.loc[idx, "_first"]
            score = first_name_score(first, team_first)
            flag = "exact" if score == 100 else ("fuzzy_high" if score >= THRESHOLD_HIGH else "team_match")
            if flag == "team_match":
                logging.warning(
                    f"TEAM MATCH (nickname?) | {row['player_name']!r} -> {contracts.loc[idx, 'player']!r} "
                    f"(first: {first!r} -> {team_first!r}, score={score:.0f})"
                )
        else:
            # Multiple year+team candidates or no match: fuzzy on first name
            search_pool = year_team_cands if len(year_team_cands) > 1 else candidates
            first_names = search_pool["_first"].tolist()

            # Step 3: fuzzy on first name only (prefix match boosted to 95)
            best = process.extractOne(first, first_names, scorer=first_name_score)

            if best is None or best[1] < THRESHOLD_HIGH:
                flag, score = "no_match", best[1] if best else 0
                match_row = pd.Series({c: pd.NA for c in contract_cols})
                logging.debug(f"NO MATCH (first name) | {row['player_name']!r} best={best}")
            else:
                score = best[1]
                flag  = "exact" if score == 100 else "fuzzy_high"
                idx   = search_pool.index[first_names.index(best[0])]
                match_row = contracts.loc[idx, contract_cols]

    out = row.drop(labels=["_norm", "_first", "_last"]).to_dict()
    out["match_score"] = score
    out["match_flag"]  = flag
    for c in contract_cols:
        out[f"{c}_contract"] = match_row.get(c, pd.NA)
    results.append(out)

logging.info("Building output dataframe and writing CSV...")
out_df = pd.DataFrame(results)

# ── Position group mapping ────────────────────────────────────────────────────
POS_GROUP = {
    "C": "OL", "G": "OL", "OG": "OL", "OL": "OL", "OT": "OL", "T": "OL",
    "CB": "DB", "DB": "DB", "FS": "DB", "S": "DB", "SAF": "DB", "SS": "DB",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DT": "DT",
    "ILB": "LB", "LB": "LB",
    "K": "SPECIAL", "P": "SPECIAL", "LS": "SPECIAL", "KR": "SPECIAL",
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "TE": "TE",
    "WR": "WR", "WR/CB": "WR",
    "DL": "DT", "NT": "DT",
}
GROUP = {
    "OL":      "BIG O",
    "DB":      "SKILL D",
    "EDGE":    "COMBO D",
    "DT":      "BIG D",
    "LB":      "COMBO D",
    "SPECIAL": "SPECIAL",
    "QB":      "QB",
    "RB":      "COMBO O",
    "TE":      "COMBO O",
    "WR":      "SKILL O",
}
out_df["Pos_Group"] = out_df["position"].map(POS_GROUP)
out_df["Group"]     = out_df["Pos_Group"].map(GROUP)

unmapped = out_df[out_df["Pos_Group"].isna()]["position"].unique()
if len(unmapped):
    logging.warning(f"Unmapped positions (no Pos_Group): {unmapped}")

out_df.to_csv(OUT_PATH, index=False)

# Summary
total      = len(out_df)
exact      = (out_df["match_flag"] == "exact").sum()
high       = (out_df["match_flag"] == "fuzzy_high").sum()
team_match = (out_df["match_flag"] == "team_match").sum()
no_match   = (out_df["match_flag"] == "no_match").sum()

logging.info(f"Done -> {OUT_PATH}")
logging.info(f"  {total} draft rows")
logging.info(f"  exact matches        : {exact}")
logging.info(f"  fuzzy high (>=90)    : {high}")
logging.info(f"  team match (nickname): {team_match}  <- review these")
logging.info(f"  no match (<90)       : {no_match}")
print()
print("Team-match rows to review manually (possible nicknames):")
print(out_df[out_df["match_flag"] == "team_match"][
    ["year", "player_name", "player_contract", "match_score"]
].to_string(index=False))

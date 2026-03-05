"""
Join nfl_contracts.csv onto nfl_draft_enriched_all.csv by player name.

Strategy:
  1. Normalize names: lowercase, strip suffixes (Jr./Sr./II/III/IV).
  2. Split into first name and last name.
  3. Exact match on last name to get candidates (eliminates cross-last-name
     false positives entirely).
  4. Among same-last-name candidates, fuzzy match on first name only.
     This handles Josh/Joshua, Pat/Patrick, etc. without any lookup table.
  5. Flag borderline first-name matches for manual review.
  If no candidate shares the exact last name -> no_match immediately.

Output: data/processed/draft_enriched_with_contracts.csv
  - contract columns are suffixed with _contract to avoid collisions
  - match_score: 0-100 fuzzy score on first name (100 = exact)
  - match_flag: "exact", "fuzzy_high", "fuzzy_low", "no_match"
"""

import re
import logging
import pandas as pd
from rapidfuzz import fuzz, process
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

THRESHOLD_HIGH = 90   # confident first-name match
THRESHOLD_LOW  = 84   # flag for manual review (still joined)

SUFFIX_RE = re.compile(
    r"\b(jr\.?|sr\.?|ii|iii|iv|v|esq\.?)\s*$", re.IGNORECASE
)

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = name.replace(",", " ")          # "Jones, II" -> "Jones  II"
    name = " ".join(name.split())          # collapse extra spaces
    name = SUFFIX_RE.sub("", name).strip().rstrip(".,")
    return name

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

contract_cols = [c for c in contracts.columns if c not in ("player", "_norm", "_first", "_last")]

results = []
total_rows = len(draft)

for i, (_, row) in enumerate(draft.iterrows(), 1):
    if i % 250 == 0 or i == total_rows:
        logging.info(f"Progress: {i}/{total_rows} rows processed")

    last  = row["_last"]
    first = row["_first"]

    # Step 1: exact last-name filter
    if last not in last_name_groups.groups:
        flag, score = "no_match", 0
        match_row = pd.Series({c: pd.NA for c in contract_cols})
        logging.debug(f"NO MATCH (last name) | {row['player_name']!r}")
    else:
        candidates   = last_name_groups.get_group(last)
        first_names  = candidates["_first"].tolist()

        # Step 2: fuzzy on first name only (prefix match boosted to 95)
        best = process.extractOne(first, first_names, scorer=first_name_score)

        if best is None or best[1] < THRESHOLD_LOW:
            flag, score = "no_match", best[1] if best else 0
            match_row = pd.Series({c: pd.NA for c in contract_cols})
            logging.debug(f"NO MATCH (first name) | {row['player_name']!r} best={best}")
        else:
            score = best[1]
            flag  = "exact" if score == 100 else ("fuzzy_high" if score >= THRESHOLD_HIGH else "fuzzy_low")
            idx   = candidates.index[first_names.index(best[0])]
            match_row = contracts.loc[idx, contract_cols]
            if flag == "fuzzy_low":
                logging.warning(
                    f"FUZZY LOW | {row['player_name']!r} -> {contracts.loc[idx, 'player']!r} "
                    f"(first: {first!r} -> {best[0]!r}, score={score})"
                )

    out = row.drop(labels=["_norm", "_first", "_last"]).to_dict()
    out["match_score"] = score
    out["match_flag"]  = flag
    for c in contract_cols:
        out[f"{c}_contract"] = match_row.get(c, pd.NA)
    results.append(out)

logging.info("Building output dataframe and writing CSV...")
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_PATH, index=False)

# Summary
total   = len(out_df)
exact   = (out_df["match_flag"] == "exact").sum()
high    = (out_df["match_flag"] == "fuzzy_high").sum()
low     = (out_df["match_flag"] == "fuzzy_low").sum()
no_match= (out_df["match_flag"] == "no_match").sum()

logging.info(f"Done -> {OUT_PATH}")
logging.info(f"  {total} draft rows")
logging.info(f"  exact matches    : {exact}")
logging.info(f"  fuzzy high (>=90): {high}")
logging.info(f"  fuzzy low  (75-89): {low}  <- review these")
logging.info(f"  no match (<75)   : {no_match}")
print()
print("Fuzzy-low rows to review manually:")
print(out_df[out_df["match_flag"] == "fuzzy_low"][
    ["year", "player_name", "player_contract", "match_score"]
].to_string(index=False))

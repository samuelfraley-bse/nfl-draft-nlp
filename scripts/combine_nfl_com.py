import pandas as pd
from pathlib import Path

input_dir = Path(__file__).parents[1] / "data" / "raw" / "nfl_com"
output_path = Path(__file__).parents[1] / "data" / "raw" / "nfl_com" / "nfl_draft_enriched_all.csv"

csvs = sorted(input_dir.glob("nfl_draft_enriched_*.csv"))
# Exclude the combined file if it already exists
csvs = [f for f in csvs if f.name != "nfl_draft_enriched_all.csv"]

dfs = [pd.read_csv(f) for f in csvs]
combined = pd.concat(dfs, ignore_index=True)

combined.to_csv(output_path, index=False)
print(f"Combined {len(csvs)} files -> {output_path} ({len(combined)} rows)")

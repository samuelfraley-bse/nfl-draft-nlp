from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_player_contract_summary(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"id", "player", "draft_year", "year_signed"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    # Ensure contract years are numeric for ordering and filtering.
    df = df.copy()
    df["year_signed"] = pd.to_numeric(df["year_signed"], errors="coerce")
    df["draft_year"] = pd.to_numeric(df["draft_year"], errors="coerce")
    df = df.dropna(subset=["id", "year_signed"])

    summary = (
        df.groupby(["id", "player"], as_index=False)
        .agg(
            draft_year=("draft_year", "min"),
            first_contract_year=("year_signed", "min"),
            total_contracts=("year_signed", "count"),
        )
        .sort_values(["draft_year", "player"])
    )

    summary["got_second_contract"] = (summary["total_contracts"] >= 2).astype(int)

    # Keep year fields as nullable ints instead of floats.
    summary["draft_year"] = summary["draft_year"].round().astype("Int64")
    summary["first_contract_year"] = summary["first_contract_year"].round().astype("Int64")

    return summary[
        [
            "id",
            "player",
            "draft_year",
            "first_contract_year",
            "total_contracts",
            "got_second_contract",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a player-level contract summary with draft year, first contract year, "
            "total contracts, and a second-contract flag."
        )
    )
    parser.add_argument(
        "--input",
        default="data/raw/contract/combined_data_2000-2023.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/player_contract_summary_2000-2023.csv",
        help="Path to output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    summary = build_player_contract_summary(df)
    summary.to_csv(output_path, index=False)

    print(f"Wrote {len(summary):,} player rows to: {output_path}")


if __name__ == "__main__":
    main()

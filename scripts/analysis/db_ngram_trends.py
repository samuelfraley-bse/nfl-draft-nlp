import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-IDF n-grams for top-tier players by Pos_Group/year."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/draft_enriched_with_contracts.csv"),
        help="Processed csv with grading + text.",
    )
    parser.add_argument(
        "--pos-groups",
        nargs="+",
        default=["DB"],
        help="Position groups to include (default: DB).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Earliest year to analyze.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Latest year to analyze.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.25,
        help="Top fraction (0-1) used to select the best graded players per year/Group.",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=3,
        help="Minimum players required to build a doc for a year/Group.",
    )
    parser.add_argument(
        "--ngram-range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=(1, 2),
        help="N-gram range passed to TfidfVectorizer.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="Limit on vocabulary size for TfidfVectorizer.",
    )
    parser.add_argument(
        "--top-terms",
        type=int,
        default=12,
        help="Number of top n-grams to display per year/Group.",
    )
    return parser.parse_args()


def build_documents(
    df: pd.DataFrame,
    pos_groups,
    start_year: int,
    end_year: int,
    top_percent: float,
    min_players: int,
):
    df = df[
        df["year"].notna()
        & df["grade"].notna()
        & df["Pos_Group"].notna()
        & df["year"].between(start_year, end_year)
    ].copy()
    df["year"] = df["year"].astype(int)

    documents = []
    missing_count = defaultdict(int)
    for year in range(start_year, end_year + 1):
        subset = df[df["year"] == year]
        if subset.empty:
            continue
        for group in pos_groups:
            group_df = subset[subset["Pos_Group"] == group]
            if len(group_df) < min_players:
                missing_count[group] += 1
                continue
            cutoff = group_df["grade"].quantile(1 - top_percent, interpolation="higher")
            top_players = group_df[group_df["grade"] >= cutoff]
            if len(top_players) < min_players:
                top_players = group_df.nlargest(min_players, "grade")
            if top_players.empty:
                continue
            text = " ".join(top_players["all_text"])
            if not text.strip():
                continue
            documents.append(
                {
                    "year": year,
                    "group": group,
                    "text": text,
                    "n_players": len(top_players),
                }
            )
    return documents, missing_count


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)
    text_cols = ["overview", "strengths", "weaknesses"]
    df[text_cols] = df[text_cols].fillna("")
    df["all_text"] = (
        df[text_cols].agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    documents, missing = build_documents(
        df,
        args.pos_groups,
        args.start_year,
        args.end_year,
        args.top_percent,
        args.min_players,
    )

    if not documents:
        raise SystemExit("No documents generated. Check filters and data availability.")

    df_meta = pd.DataFrame(documents)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    dtm = vectorizer.fit_transform(df_meta["text"])
    features = vectorizer.get_feature_names_out()

    print("TF-IDF n-grams per Pos_Group/year (top terms):")
    for idx, row in df_meta.iterrows():
        vec = dtm[idx].toarray().ravel()
        top_indices = np.argsort(vec)[-args.top_terms:][::-1]
        term_weights = [
            (features[i], float(vec[i])) for i in top_indices if vec[i] > 0
        ]
        if not term_weights:
            print(f"{row['year']} {row['group']}: (no nonzero terms)")
            continue
        terms_display = ", ".join(
            f"{term} ({weight:.3f})" for term, weight in term_weights
        )
        print(
            f"{row['year']} {row['group']} (n={row['n_players']}): {terms_display}"
        )

    if missing:
        print("\nSkipped year/Group combinations with too few players:")
        for group, count in missing.items():
            print(f"- {group}: {count} year(s)")


if __name__ == "__main__":
    main()

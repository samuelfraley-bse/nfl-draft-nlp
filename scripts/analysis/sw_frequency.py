"""
Analysis: Top N unigrams & bigrams + TF-IDF in strengths vs. weaknesses by position group.

Outputs:
  data/interim/sw_freq_<position>.png   -- bar charts, top N terms per position group
  data/interim/tfidf_strengths.csv      -- TF-IDF top terms per position group (strengths)
  data/interim/tfidf_weaknesses.csv     -- TF-IDF top terms per position group (weaknesses)

Run:
  .venv/Scripts/python.exe scripts/analysis/sw_frequency.py
"""

from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[2] / "data" / "raw"
OUT_DIR  = Path(__file__).parents[2] / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20  # top terms to display per group / field

POSITION_MAP = {
    # Offensive line
    "OT": "OL", "T": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    # Linebackers
    "LB": "LB", "ILB": "LB",
    # Edge rushers (includes OLB in 3-4 schemes)
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    # Interior defensive line
    "DT": "DL", "NT": "DL", "DL": "DL",
    # Defensive backs (all lumped)
    "SAF": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "CB": "DB",
    # Skill positions
    "RB": "RB", "FB": "RB",
    "QB": "QB",
    "WR": "WR", "WR/CB": "WR",
    "TE": "TE",
}

# Positions to drop entirely from text analysis
EXCLUDE_POSITIONS = {"K", "P", "LS", "KR", "-"}

# Words that appear everywhere in scouting reports but carry no analytical signal
FOOTBALL_STOP_WORDS = {
    "nfl", "player", "football", "game", "ball", "play", "field", "team",
    "also", "ability", "well", "even", "still", "one", "two", "need",
    "get", "run", "make", "use", "level", "draft", "college", "year",
    "snap", "side", "first", "good", "great", "will", "can",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("nfl_draft_enriched_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Drop rows with no usable position or excluded special-teams positions
    df = df[~df["position"].isin(EXCLUDE_POSITIONS)]
    df = df[df["position"].notna()]

    # Map to position groups; drop anything not in the map (e.g. stray values)
    df["pos_group"] = df["position"].map(POSITION_MAP)
    df = df[df["pos_group"].notna()].copy()

    return df


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()
STOP_WORDS = stopwords.words("english") + list(FOOTBALL_STOP_WORDS)
STOP_WORDS_SET = set(STOP_WORDS)


def preprocess(text: str) -> list[str]:
    """Lowercase, strip punctuation, tokenize, remove stop words, lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS_SET and len(t) > 2]
    return tokens


def get_bigrams(tokens: list[str]) -> list[str]:
    """Return list of 'word1 word2' bigram strings."""
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def top_n_terms(token_lists: list[list[str]], n: int = TOP_N, include_bigrams: bool = True):
    """Count unigrams (and optionally bigrams) across a list of token lists."""
    unigram_counts: Counter = Counter()
    bigram_counts:  Counter = Counter()
    for tokens in token_lists:
        unigram_counts.update(tokens)
        if include_bigrams:
            bigram_counts.update(get_bigrams(tokens))
    top_uni = unigram_counts.most_common(n)
    top_bi  = bigram_counts.most_common(n)
    return top_uni, top_bi


# ---------------------------------------------------------------------------
# Frequency plots
# ---------------------------------------------------------------------------

def plot_top_terms(pos_group: str, strengths_tokens, weaknesses_tokens) -> None:
    """Save a figure with top-N unigrams and bigrams for one position group."""
    s_uni, s_bi = top_n_terms(strengths_tokens)
    w_uni, w_bi = top_n_terms(weaknesses_tokens)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{pos_group} — Strengths vs. Weaknesses: Top {TOP_N} Terms", fontsize=14, fontweight="bold")

    datasets = [
        (axes[0, 0], s_uni, "Strengths — Unigrams",  "#2196F3"),
        (axes[0, 1], w_uni, "Weaknesses — Unigrams", "#F44336"),
        (axes[1, 0], s_bi,  "Strengths — Bigrams",   "#2196F3"),
        (axes[1, 1], w_bi,  "Weaknesses — Bigrams",  "#F44336"),
    ]

    for ax, counts, title, color in datasets:
        if not counts:
            ax.set_visible(False)
            continue
        terms, freqs = zip(*counts)
        y = range(len(terms))
        ax.barh(y, freqs, color=color, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(terms, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = OUT_DIR / f"sw_freq_{pos_group.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def tfidf_top_terms(group_texts: dict[str, list[str]], n: int = TOP_N) -> pd.DataFrame:
    """
    For each position group, treat the concatenated text of all players as one
    document, then find the top-N TF-IDF terms per group.

    Returns a DataFrame with columns: pos_group, term, tfidf_score
    """
    pos_groups = list(group_texts.keys())
    # One document per position group (concatenate all player texts)
    docs = [" ".join(group_texts[pg]) for pg in pos_groups]

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for i, pg in enumerate(pos_groups):
        scores = tfidf_matrix[i].toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:n]
        for idx in top_idx:
            if scores[idx] > 0:
                rows.append({"pos_group": pg, "term": feature_names[idx], "tfidf_score": round(scores[idx], 4)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_data()

    print(f"Total players (with position group): {len(df)}")
    print("\nPosition group counts:")
    print(df["pos_group"].value_counts().to_string())
    print()

    # Preprocess text for each player
    df["s_tokens"] = df["strengths"].apply(preprocess)
    df["w_tokens"] = df["weaknesses"].apply(preprocess)

    pos_groups = sorted(df["pos_group"].unique())

    # --- Frequency plots ---
    print("Generating frequency bar charts...")
    for pg in pos_groups:
        sub = df[df["pos_group"] == pg]
        s_tokens = sub["s_tokens"].tolist()
        w_tokens = sub["w_tokens"].tolist()
        plot_top_terms(pg, s_tokens, w_tokens)

    # --- TF-IDF ---
    print("\nRunning TF-IDF (strengths)...")
    s_group_texts = {
        pg: [t for tokens in df[df["pos_group"] == pg]["s_tokens"] for t in tokens]
        for pg in pos_groups
    }
    tfidf_s = tfidf_top_terms(s_group_texts)
    out_s = OUT_DIR / "tfidf_strengths.csv"
    tfidf_s.to_csv(out_s, index=False)
    print(f"  Saved: {out_s.name}")

    print("Running TF-IDF (weaknesses)...")
    w_group_texts = {
        pg: [t for tokens in df[df["pos_group"] == pg]["w_tokens"] for t in tokens]
        for pg in pos_groups
    }
    tfidf_w = tfidf_top_terms(w_group_texts)
    out_w = OUT_DIR / "tfidf_weaknesses.csv"
    tfidf_w.to_csv(out_w, index=False)
    print(f"  Saved: {out_w.name}")

    # Preview TF-IDF in console
    print("\n--- TF-IDF top 5 terms per group (STRENGTHS) ---")
    print(tfidf_s.groupby("pos_group").head(5).to_string(index=False))

    print("\n--- TF-IDF top 5 terms per group (WEAKNESSES) ---")
    print(tfidf_w.groupby("pos_group").head(5).to_string(index=False))

    print("\nDone. All outputs written to data/interim/")


if __name__ == "__main__":
    main()

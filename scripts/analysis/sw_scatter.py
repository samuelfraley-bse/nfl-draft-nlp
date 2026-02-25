"""
Viz: Strengths vs. Weaknesses term scatter plot by position group.

For each position group, plots each term's relative frequency in the strengths
corpus (x-axis) against its relative frequency in the weaknesses corpus (y-axis).

  - Terms far RIGHT  → characteristic of strengths
  - Terms far UP     → characteristic of weaknesses
  - Terms near the diagonal → appear in both equally

Outputs:
  data/interim/sw_scatter_<position>.png  -- one scatter per position group
  data/interim/sw_scatter_all.png         -- all groups in one grid

Run:
  .venv/Scripts/python.exe scripts/analysis/sw_scatter.py
"""

from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Config  (same as sw_frequency.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[2] / "data" / "raw"
OUT_DIR  = Path(__file__).parents[2] / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_ANNOTATE = 15   # how many terms to label per plot
MIN_FREQ = 5          # minimum raw count to include a term (filters noise)
NGRAM = True          # include bigrams alongside unigrams

POSITION_MAP = {
    "OT": "OL", "T": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "LB": "LB", "ILB": "LB",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL",
    "SAF": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "CB": "DB",
    "RB": "RB", "FB": "RB",
    "QB": "QB",
    "WR": "WR", "WR/CB": "WR",
    "TE": "TE",
}

EXCLUDE_POSITIONS = {"K", "P", "LS", "KR", "-"}

FOOTBALL_STOP_WORDS = {
    "nfl", "player", "football", "game", "ball", "play", "field", "team",
    "also", "ability", "well", "even", "still", "one", "two", "need",
    "get", "run", "make", "use", "level", "draft", "college", "year",
    "snap", "side", "first", "good", "great", "will", "can",
}

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

lemmatizer = WordNetLemmatizer()
STOP_WORDS_SET = set(stopwords.words("english")) | FOOTBALL_STOP_WORDS


def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("nfl_draft_enriched_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[~df["position"].isin(EXCLUDE_POSITIONS)]
    df = df[df["position"].notna()]
    df["pos_group"] = df["position"].map(POSITION_MAP)
    df = df[df["pos_group"].notna()].copy()
    return df


def preprocess(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS_SET and len(t) > 2]
    return tokens


def get_bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def term_freq_proportions(token_lists: list[list[str]]) -> dict[str, float]:
    """Return term -> proportion (count / total tokens) across all token lists."""
    all_tokens = []
    for tokens in token_lists:
        all_tokens.extend(tokens)
        if NGRAM:
            all_tokens.extend(get_bigrams(tokens))
    total = len(all_tokens)
    if total == 0:
        return {}
    counts = Counter(all_tokens)
    return {term: count / total for term, count in counts.items()}, counts


# ---------------------------------------------------------------------------
# Scatter plot for one position group
# ---------------------------------------------------------------------------

def make_scatter(ax, pos_group: str, s_tokens, w_tokens, annotate: bool = True) -> None:
    (s_freq, s_counts), (w_freq, w_counts) = (
        term_freq_proportions(s_tokens),
        term_freq_proportions(w_tokens),
    )

    # Union of terms that meet min frequency in either corpus
    terms = {t for t, c in s_counts.items() if c >= MIN_FREQ} | \
            {t for t, c in w_counts.items() if c >= MIN_FREQ}

    if not terms:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(pos_group)
        return

    s_vals = np.array([s_freq.get(t, 0.0) for t in terms])
    w_vals = np.array([w_freq.get(t, 0.0) for t in terms])
    term_list = list(terms)

    # Color by which axis dominates
    colors = []
    for s, w in zip(s_vals, w_vals):
        if s > w * 1.5:
            colors.append("#2196F3")   # blue  = strength-heavy
        elif w > s * 1.5:
            colors.append("#F44336")   # red   = weakness-heavy
        else:
            colors.append("#9E9E9E")   # grey  = shared

    ax.scatter(s_vals, w_vals, c=colors, alpha=0.6, s=30, linewidths=0)

    # Diagonal reference line (equal frequency in both)
    lim = max(s_vals.max(), w_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4, zorder=0)

    if annotate:
        # Label the terms furthest from diagonal (most distinctive)
        diff = np.abs(s_vals - w_vals)
        top_idx = np.argsort(diff)[::-1][:TOP_N_ANNOTATE]
        for i in top_idx:
            ax.annotate(
                term_list[i],
                (s_vals[i], w_vals[i]),
                fontsize=6.5,
                xytext=(3, 3),
                textcoords="offset points",
                color=colors[i],
            )

    ax.set_xlabel("Proportion in Strengths", fontsize=8)
    ax.set_ylabel("Proportion in Weaknesses", fontsize=8)
    ax.set_title(pos_group, fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    df = load_data()
    df["s_tokens"] = df["strengths"].apply(preprocess)
    df["w_tokens"] = df["weaknesses"].apply(preprocess)

    pos_groups = sorted(df["pos_group"].unique())

    # --- Individual plots ---
    print("Generating individual scatter plots...")
    for pg in pos_groups:
        sub = df[df["pos_group"] == pg]
        fig, ax = plt.subplots(figsize=(7, 6))
        make_scatter(ax, pg, sub["s_tokens"].tolist(), sub["w_tokens"].tolist())

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", label="Strength-heavy", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336", label="Weakness-heavy", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E", label="Shared",          markersize=8),
            Line2D([0], [0], linestyle="--", color="black", alpha=0.4, label="Equal frequency"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="upper left")
        fig.suptitle(
            f"{pg} — Term Frequency: Strengths vs. Weaknesses\n"
            f"(top {TOP_N_ANNOTATE} most distinctive terms labeled)",
            fontsize=10,
        )
        plt.tight_layout()
        out = OUT_DIR / f"sw_scatter_{pg.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}")

    # --- Combined grid (all positions) ---
    print("Generating combined grid...")
    n_cols = 3
    n_rows = int(np.ceil(len(pos_groups) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5.5))
    axes_flat = axes.flatten()

    for i, pg in enumerate(pos_groups):
        sub = df[df["pos_group"] == pg]
        make_scatter(axes_flat[i], pg, sub["s_tokens"].tolist(), sub["w_tokens"].tolist(), annotate=True)

    # Hide unused subplots
    for j in range(len(pos_groups), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Strengths vs. Weaknesses — Term Frequency by Position Group\n"
        "Blue = strength-heavy  |  Red = weakness-heavy  |  Grey = shared  |  -- = equal frequency",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / "sw_scatter_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

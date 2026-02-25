"""
Analysis: Which terms associate with QB rank (within-year, within-position)?

For each term appearing in overview / strengths / weaknesses, computes:
  - mean within-position rank of players whose text contains that term
  - Low mean rank  = term appears for HIGHER-ranked players
  - High mean rank = term appears for LOWER-ranked players

Separating by field lets you ask:
  - Strengths  + low rank  → valued elite quality
  - Weaknesses + low rank  → forgivable flaw
  - Weaknesses + high rank → disqualifying flaw

Outputs:
  data/interim/rank_terms_<pos_group>.png      -- bar chart: top/bottom associated terms
  data/interim/rank_terms_<pos_group>.csv      -- full table of term rank associations

Run:
  .venv/Scripts/python.exe scripts/analysis/rank_terms.py

Config:
  Change POS_GROUP to any of: QB WR RB TE OL DL EDGE LB DB
"""

from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR  = Path(__file__).parents[2] / "data" / "raw"
OUT_DIR   = Path(__file__).parents[2] / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POS_GROUP = "QB"    # <-- change this to run for a different position group

TOP_N     = 20      # terms to show at each end of the rank spectrum
MIN_COUNT = 8       # minimum number of players a term must appear in

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
# Data loading
# ---------------------------------------------------------------------------

def load_data(pos_group: str) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("nfl_draft_enriched_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[~df["position"].isin(EXCLUDE_POSITIONS)]
    df["pos_group"] = df["position"].map(POSITION_MAP)
    df = df[df["pos_group"] == pos_group].copy()

    # Within-year, within-position rank: sort by overall rank each year,
    # assign 1 = highest-ranked player at this position that year
    df = df.sort_values(["year", "rank"])
    df["pos_rank"] = df.groupby("year").cumcount() + 1

    return df


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

lemmatizer    = WordNetLemmatizer()
STOP_WORDS_SET = set(stopwords.words("english")) | FOOTBALL_STOP_WORDS


def preprocess(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS_SET and len(t) > 2]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


# ---------------------------------------------------------------------------
# Core: term → mean rank
# ---------------------------------------------------------------------------

def term_rank_table(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    For each unique term in `field`, compute:
      - player_count: how many distinct players mention this term
      - mean_pos_rank: average within-position rank of those players
      - median_pos_rank
    """
    rank_lists: dict[str, list[int]] = defaultdict(list)

    for _, row in df.iterrows():
        tokens = preprocess(row[field])
        seen = set(tokens)          # one mention per player per term
        for term in seen:
            rank_lists[term].append(row["pos_rank"])

    rows = []
    for term, ranks in rank_lists.items():
        if len(ranks) >= MIN_COUNT:
            rows.append({
                "term":           term,
                "player_count":   len(ranks),
                "mean_pos_rank":  round(np.mean(ranks), 2),
                "median_pos_rank": round(np.median(ranks), 2),
            })

    return pd.DataFrame(rows).sort_values("mean_pos_rank")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FIELD_COLORS = {
    "overview":   "#7B1FA2",   # purple
    "strengths":  "#1565C0",   # blue
    "weaknesses": "#C62828",   # red
}

FIELD_LABELS = {
    "overview":   "Overview",
    "strengths":  "Strengths",
    "weaknesses": "Weaknesses",
}


def plot_summary(pos_group: str, tables: dict[str, pd.DataFrame], top_n: int = 10) -> None:
    """
    Compact single-figure summary: one row per field, each row shows the
    top_n high-rank terms (left, colored) and top_n low-rank terms (right, grey).
    Terms are shown as dot-and-line lollipops sorted by mean rank.
    """
    fields   = [f for f, tbl in tables.items() if not tbl.empty]
    n_fields = len(fields)

    fig, axes = plt.subplots(n_fields, 2, figsize=(13, n_fields * 3.8))
    if n_fields == 1:
        axes = axes[np.newaxis, :]

    for row_i, field in enumerate(fields):
        tbl   = tables[field]
        color = FIELD_COLORS[field]
        label = FIELD_LABELS[field]

        hi = tbl.head(top_n).copy()              # lowest mean rank  → elite
        lo = tbl.tail(top_n).iloc[::-1].copy()   # highest mean rank → fringe

        for col_i, (subset, title, dot_color) in enumerate([
            (hi, f"↑ Higher-ranked {pos_group}s", color),
            (lo, f"↓ Lower-ranked {pos_group}s",  "#9E9E9E"),
        ]):
            ax = axes[row_i, col_i]
            terms = subset["term"].tolist()
            ranks = subset["mean_pos_rank"].tolist()
            y     = range(len(terms))

            # lollipop: horizontal line + dot
            ax.hlines(y, 0, ranks, color=dot_color, linewidth=1.2, alpha=0.6)
            ax.scatter(ranks, y, color=dot_color, s=55, zorder=3)

            ax.set_yticks(y)
            ax.set_yticklabels(terms, fontsize=8.5)
            ax.set_xlabel("Mean within-position rank", fontsize=7.5)
            ax.set_xlim(left=0)
            ax.set_title(f"{label}  —  {title}", fontsize=9, fontweight="bold", pad=4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.5)
            ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(
        f"{pos_group} — Terms Most Associated with Rank  (top {top_n} per field)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = OUT_DIR / f"rank_terms_{pos_group.lower()}_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_rank_terms(pos_group: str, tables: dict[str, pd.DataFrame]) -> None:
    """
    For each field, plot the TOP_N lowest-mean-rank terms (elite-associated)
    and TOP_N highest-mean-rank terms (fringe-associated) as horizontal bars.
    """
    n_fields = len(tables)
    fig, axes = plt.subplots(n_fields, 2, figsize=(16, n_fields * 5))
    if n_fields == 1:
        axes = axes[np.newaxis, :]

    total_players = None

    for row_i, (field, tbl) in enumerate(tables.items()):
        color = FIELD_COLORS[field]

        if tbl.empty:
            for col in range(2):
                axes[row_i, col].set_visible(False)
            continue

        if total_players is None:
            # approximate from the most-common term's count (generous upper bound)
            total_players = tbl["player_count"].max()

        top_elite  = tbl.head(TOP_N)
        top_fringe = tbl.tail(TOP_N).sort_values("mean_pos_rank", ascending=False)

        for col_i, (subset, subtitle) in enumerate([
            (top_elite,  f"Associated with HIGHER rank\n(lower = better)"),
            (top_fringe, f"Associated with LOWER rank"),
        ]):
            ax = axes[row_i, col_i]
            terms  = subset["term"].tolist()
            ranks  = subset["mean_pos_rank"].tolist()
            counts = subset["player_count"].tolist()

            y = range(len(terms))
            bars = ax.barh(y, ranks, color=color, alpha=0.75)
            ax.set_yticks(y)
            ax.set_yticklabels(terms, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Mean within-position rank (lower = higher-ranked player)", fontsize=8)
            ax.set_title(f"{FIELD_LABELS[field]}  —  {subtitle}", fontsize=9, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)

            # Annotate bars with player count
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={count}",
                    va="center", fontsize=7, color="#555555",
                )

    fig.suptitle(
        f"{pos_group} — Terms by Associated Rank  (n≥{MIN_COUNT} players per term)\n"
        f"Min appearances to include a term: {MIN_COUNT}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / f"rank_terms_{pos_group.lower()}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading data for position group: {POS_GROUP}...")
    df = load_data(POS_GROUP)
    print(f"  Players found: {len(df)}  |  Years: {df['year'].min()}–{df['year'].max()}")
    print(f"  Within-position rank range: {df['pos_rank'].min()}–{df['pos_rank'].max()}")
    print()

    tables = {}
    for field in ["overview", "strengths", "weaknesses"]:
        has_text = df[field].notna() & (df[field].str.strip() != "")
        print(f"  {field}: {has_text.sum()} players with text")
        tbl = term_rank_table(df[has_text], field)
        tables[field] = tbl

        # Save CSV
        out_csv = OUT_DIR / f"rank_terms_{POS_GROUP.lower()}_{field}.csv"
        tbl.to_csv(out_csv, index=False)
        print(f"    Saved: {out_csv.name}  ({len(tbl)} terms)")

    print()
    plot_summary(POS_GROUP, tables)
    plot_rank_terms(POS_GROUP, tables)

    # Console preview: top 10 / bottom 10 per field
    for field, tbl in tables.items():
        if tbl.empty:
            continue
        print(f"\n--- {field.upper()} — top 8 terms (high-ranked {POS_GROUP}s) ---")
        print(tbl.head(8)[["term","mean_pos_rank","player_count"]].to_string(index=False))
        print(f"\n--- {field.upper()} — top 8 terms (low-ranked {POS_GROUP}s) ---")
        print(tbl.tail(8).sort_values("mean_pos_rank", ascending=False)[["term","mean_pos_rank","player_count"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

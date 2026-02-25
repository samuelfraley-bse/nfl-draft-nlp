"""
Analysis: What QB evaluation topics appear as strengths vs. weaknesses?

For each topic bucket (arm strength, accuracy, pocket presence, etc.), counts
what % of QBs have that topic mentioned in their strengths vs. weaknesses text.

The same topic (e.g. "tight window") can appear in either field — the field
label carries the valence (strength = can do it, weakness = can't).

Outputs:
  data/interim/qb_topics.png       -- grouped bar chart (% strength vs % weakness)
  data/interim/qb_topics.csv       -- full table

Run:
  .venv/Scripts/python.exe scripts/analysis/qb_topics.py
"""

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parents[2] / "data" / "raw"
OUT_DIR  = Path(__file__).parents[2] / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {
    "OT": "OL", "T": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "LB": "LB", "ILB": "LB",
    "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE",
    "DT": "DL", "NT": "DL", "DL": "DL",
    "SAF": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "CB": "DB",
    "RB": "RB", "FB": "RB",
    "QB": "QB", "WR": "WR", "WR/CB": "WR", "TE": "TE",
}

# ---------------------------------------------------------------------------
# Topic buckets — seed words matched against raw lowercased text
# Each seed is matched as a whole word (word-boundary regex), so "arm"
# won't match "harm" and "throw" catches "throwing", "throws", "thrown".
# ---------------------------------------------------------------------------

QB_TOPICS = {
    "Arm Strength":     ["velocity", "arm strength", "zip", "cannon", "spiral",
                         "arm talent", "strong arm", "powerful"],
    "Accuracy":         ["accuracy", "accurate", "placement", "tight window",
                         "touch", "precise", "precision"],
    "Pocket Presence":  ["pocket", "pressure", "navigate", "feel", "extend",
                         "escape", "pocket awareness", "pocket presence"],
    "Release":          ["release", "delivery", "footwork", "mechanics",
                         "throwing motion", "drop back"],
    "Reading / Vision": ["read", "progression", "recognition", "coverage",
                         "diagnose", "pre.snap", "eyes", "vision", "scan"],
    "Decision Making":  ["decision", "turnover", "interception", "protect",
                         "hold", "force", "careless", "check down"],
    "Anticipation":     ["anticipat", "timing"],
    "Mobility":         ["mobile", "mobility", "scramble", "athletic", "rush",
                         "runner", "foot speed", "speed"],
    "Deep Ball":        ["deep ball", "downfield", "vertical", "go route",
                         "deep throw", "deep pass"],
    "Touch / Trajectory": ["touch", "trajectory", "loft", "arc",
                            "ball placement", "float"],
    "Consistency":      ["inconsistent", "consistent", "tendency", "struggle",
                         "average", "adequate"],
    "Leadership":       ["leader", "leadership", "poise", "composure",
                         "confident", "prepare", "compete", "intangible"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def contains_any(text: str, seeds: list[str]) -> bool:
    """Return True if any seed word/phrase appears as a whole word in text."""
    if not isinstance(text, str):
        return False
    text = text.lower()
    for seed in seeds:
        # build a word-boundary pattern; handle phrases (spaces are fine)
        pattern = r"\b" + re.escape(seed) + r"\b"
        if re.search(pattern, text):
            return True
    return False


def load_qbs() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("nfl_draft_enriched_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["pos_group"] = df["position"].map(POSITION_MAP)
    qbs = df[df["pos_group"] == "QB"].copy()
    # Only players who have at least one text field
    has_text = qbs["strengths"].notna() | qbs["weaknesses"].notna()
    return qbs[has_text].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_topic_rates(qbs: pd.DataFrame) -> pd.DataFrame:
    n = len(qbs)
    rows = []
    for topic, seeds in QB_TOPICS.items():
        s_count = qbs["strengths"].apply(lambda t: contains_any(t, seeds)).sum()
        w_count = qbs["weaknesses"].apply(lambda t: contains_any(t, seeds)).sum()
        both    = (
            qbs["strengths"].apply(lambda t: contains_any(t, seeds)) &
            qbs["weaknesses"].apply(lambda t: contains_any(t, seeds))
        ).sum()
        rows.append({
            "topic":       topic,
            "n_strength":  int(s_count),
            "n_weakness":  int(w_count),
            "n_both":      int(both),
            "pct_strength": round(s_count / n * 100, 1),
            "pct_weakness": round(w_count / n * 100, 1),
            "n_total":     n,
        })
    return pd.DataFrame(rows).sort_values("pct_strength", ascending=False)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_topics(df: pd.DataFrame) -> None:
    # Sort by strength % for readability
    df = df.sort_values("pct_strength", ascending=True)
    topics = df["topic"].tolist()
    s_pct  = df["pct_strength"].tolist()
    w_pct  = df["pct_weakness"].tolist()
    n      = df["n_total"].iloc[0]

    y = np.arange(len(topics))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    bars_s = ax.barh(y + height/2, s_pct, height, label="Strength",  color="#1565C0", alpha=0.82)
    bars_w = ax.barh(y - height/2, w_pct, height, label="Weakness",  color="#C62828", alpha=0.82)

    # Annotate with raw count
    for bar, row in zip(bars_s, df.sort_values("pct_strength", ascending=True).itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"n={row.n_strength}", va="center", fontsize=7, color="#1565C0")
    for bar, row in zip(bars_w, df.sort_values("pct_strength", ascending=True).itertuples()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"n={row.n_weakness}", va="center", fontsize=7, color="#C62828")

    ax.set_yticks(y)
    ax.set_yticklabels(topics, fontsize=9.5)
    ax.set_xlabel("% of QBs where topic is mentioned", fontsize=9)
    ax.set_title(
        f"QB Evaluation Topics: Strengths vs. Weaknesses\n"
        f"(n={n} QBs with scouting text, 2010–2026)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(0, max(s_pct + w_pct) * 1.18)

    plt.tight_layout()
    out = OUT_DIR / "qb_topics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading QB data...")
    qbs = load_qbs()
    print(f"  {len(qbs)} QBs with scouting text ({qbs['year'].min()}–{qbs['year'].max()})")

    print("\nComputing topic rates...")
    results = compute_topic_rates(qbs)

    out_csv = OUT_DIR / "qb_topics.csv"
    results.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    print()
    print(results[["topic","pct_strength","n_strength","pct_weakness","n_weakness","n_both"]].to_string(index=False))

    print("\nGenerating plot...")
    plot_topics(results)

    print("\nDone.")


if __name__ == "__main__":
    main()

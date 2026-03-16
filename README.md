# NFL Draft NLP

Predict whether an NFL draft pick earns a real second contract using scouting report text and pre-draft scores.

---

## Project Goal

NFL scouting reports are written before players are drafted. The question: **does the language scouts use predict long-term success, beyond athletic/production scores alone?**

**Target variable:** `made_it_contract` — 1 if the player earned a genuine second NFL contract (not just a practice-squad or minimum deal), 0 otherwise.

---

## Data

| Source | Description | Location |
|---|---|---|
| NFL.com scouting reports | Overview, strengths, weaknesses text + pre-draft scores | `data/raw/nfl_com/` |
| NFL contracts | Contract data joined by fuzzy name match | `data/raw/contract/` |
| Processed | Draft rows with contract outcome joined | `data/processed/draft_enriched_with_contracts.csv` |

**Coverage:** 2010–2025 draft classes (~7,400 players after joining).
**Positive rate:** ~29–32% depending on filter (rows with text and numeric scores).

### Position Mapping

Raw position strings are mapped to two aggregation levels, both stored as columns in the processed CSV:

| Column | Example values | Use |
|---|---|---|
| `position` | CB, QB, OT, … | Fine-grained position model |
| `Pos_Group` | DB, OL, LB, EDGE, … | Mid-level grouping |
| `Group` | SKILL D, BIG O, COMBO D, QB, … | Broad side-of-ball grouping |

Mapping is defined in [`scripts/join_contracts.py`](scripts/join_contracts.py).

---

## Pipeline

```
data/raw/nfl_com/           → scripts/combine_nfl_com.py   → nfl_draft_enriched_all.csv
data/raw/contract/          → scripts/join_contracts.py    → draft_enriched_with_contracts.csv
                                                              (adds Pos_Group, Group columns)
```

---

## Models

All models predict `made_it_contract`. Metric: **PR-AUC (Average Precision)** — appropriate for the class imbalance (~70/30). Random baseline ≈ positive rate.

| Notebook | Features | 5-fold CV PR-AUC | Notes |
|---|---|---|---|
| [`rf_scores_made_it`](notebooks/rf_scores_made_it.ipynb) | `production_score`, `athleticism_score` | **0.408 ± 0.010** | Best so far; numeric only |
| [`rf_combined_made_it`](notebooks/rf_combined_made_it.ipynb) | TF-IDF (all words) + numeric scores | ~0.377 | FeatureUnion pipeline |
| [`rf_text_made_it`](notebooks/rf_text_made_it.ipynb) | TF-IDF unigrams+bigrams (all words) | ~0.321 | Text-only baseline |
| [`rf_text_adjverb_made_it`](notebooks/rf_text_adjverb_made_it.ipynb) | POS-filtered TF-IDF (adj+verb) + word_count + adj_density | TBD | Filters noun noise |

All text models use CountVectorizer → TfidfTransformer → RandomForestClassifier with grid search over hyperparameters (3-fold CV, `class_weight="balanced"`).

---

## EDA

[`notebooks/eda_adjectives_by_position.ipynb`](notebooks/eda_adjectives_by_position.ipynb)

Tests which words are statistically associated with success vs bust **per position**, using:

- **Log-Likelihood Ratio G²** (Dunning 1993) — robust to rare words; standard in corpus linguistics
- **Frequency Ratio** with Laplace smoothing — easy to interpret

Two configurable feature types (set `FEATURE_TYPE` at the top of the notebook):

| Value | What it extracts |
|---|---|
| `"adjectives"` | Single adjective lemmas (`"explosive"`, `"stiff"`, …) — fast |
| `"noun_chunks"` | spaCy noun phrases (`"quick feet"`, `"elite burst"`, …) — requires parser |
| `"both"` | Runs both analyses back-to-back |

Position aggregation level is also configurable via `POS_LEVEL` (`"position"`, `"Pos_Group"`, or `"Group"`).

Outputs domain dictionaries to:
- `data/processed/domain_adjectives_dict.csv`
- `data/processed/domain_noun_chunks_dict.csv`

---

## Key Findings So Far

- **Numeric scores beat text** — production + athleticism scores (PR-AUC 0.408) outperform TF-IDF on raw text (~0.321). Scores encode pre-draft consensus efficiently.
- **Combining text + scores doesn't yet help** — the combined model (0.377) underperforms scores alone, suggesting the text features are adding noise rather than signal.
- **POS filtering is promising** — restricting to adjectives/verbs removes cohort noise (player names, school names) that varies by year. The adj+verb notebook is still being evaluated.
- **Position context matters** — the same adjective can flip sign across positions (e.g. `"heavy"` is positive for DT, negative for CB). The EDA is designed to surface this.

---

## Next Steps

- [ ] Evaluate `rf_text_adjverb_made_it` — compare POS-filtered text vs raw text PR-AUC
- [ ] Add noun chunk features to the RF models (following EDA findings)
- [ ] Position-stratified models — train separate models per `Pos_Group` or `Group`
- [ ] Use domain dictionary as feature engineering input (signal-adjective count scores)

---

## Setup

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### Install

```bash
# Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtualenv and install dependencies
uv sync

# Activate (PowerShell)
.venv\Scripts\Activate.ps1
```

### Data

Download from the shared Drive folder and place contents into `data/raw/`, preserving the folder structure. Then run:

```bash
python scripts/combine_nfl_com.py   # merge per-year CSVs → nfl_draft_enriched_all.csv
python scripts/join_contracts.py    # join contracts + add position mapping
```

`data/` is git-ignored — keep large files local or on Drive.

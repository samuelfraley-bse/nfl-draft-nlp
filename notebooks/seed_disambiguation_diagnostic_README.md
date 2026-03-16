# Seed Disambiguation Diagnostic

**Notebook:** `seed_disambiguation_diagnostic.ipynb`
**Purpose:** Validate whether each seed term in `ANCHOR_SEEDS` and `CENTROID_EXTRAS` actually belongs to its assigned bin — using the corpus itself as the judge, not manual intuition.

---

## Why This Exists

The v2 scoring notebook (`nfl_pillar_scoring_NP_v2.ipynb`) classifies noun phrases from scouting reports by comparing them to three bin centroids: **Physical**, **Technique**, and **Character**. Each centroid is the average Word2Vec vector of its hand-picked seed terms.

The problem: if a seed is ambiguous — i.e. it appears in multiple bin-relevant contexts in real scouting language — it pulls the centroid toward a rival bin, silently biasing every classification that follows. This notebook detects those seeds empirically before they do damage.

---

## Method

For each seed in `ANCHOR_SEEDS` and `CENTROID_EXTRAS`:

1. Remove the seed from its bin temporarily (leave-one-out)
2. Rebuild all three centroids from the remaining seeds
3. Compute cosine similarity between the seed's W2V vector and each centroid
4. Calculate the **disambiguation margin**:

```
margin = sim(seed, own_centroid) − max(sim(seed, rival_centroids))
```

| Margin | Verdict | Meaning |
|---|---|---|
| ≥ 0.05 | **ok** | Seed unambiguously points to its own bin |
| 0 – 0.05 | **ambiguous** | Seed is equidistant — weak signal |
| < 0 | **misassigned** | Seed is closer to a rival bin — contaminates the centroid |

Seeds are also tagged `A` (anchor) or `E` (extra) to show which pool they come from.

---

## How to Run

Run all cells top to bottom. No external dependencies beyond the main project environment.
The notebook retrains W2V from scratch using identical config to v2 (`W2V_DIM=100`, `W2V_WINDOW=6`, `W2V_EPOCHS=30`, `seed=42`).

---

## Outputs

| Section | Output |
|---|---|
| 6 | Filtered table — flagged seeds only (ambiguous + misassigned), colour-coded |
| 7 | Full table — all seeds colour-coded by verdict |
| 8 | Per-bin summary (clean rate) + worst 5 seeds per bin printed to console |
| 9 | Horizontal bar chart of all margins, one panel per bin |

---

## Results (First Run)

**Overall:** 110 in-vocab seeds evaluated across 3 bins

| Bin | ok | ambiguous | misassigned | % clean |
|---|---|---|---|---|
| physical | 29 | 2 | 5 | 80.6% |
| character | 30 | 6 | 4 | 75.0% |
| technique | 24 | 4 | 6 | **70.6%** |

Technique has the most contaminated centroid.

**Worst misassigned seeds:**

| Seed | Bin | Margin | Why |
|---|---|---|---|
| `productive` | technique | −0.187 | W2V places it in outcome/character language, not technique execution |
| `velocity` | physical | −0.163 | "Release velocity", "ball velocity" → technique territory |
| `run_support` | technique | −0.127 | Appears next to physicality terms → physical |
| `focus` | character | −0.064 | "Eye discipline and focus" → technique context |
| `processing` | character | −0.042 | "Processes the defense" → technique in scouting language |
| `recognition` | character | −0.023 | "Route/blitz/play recognition" → technique |

**OOV seeds** (below `min_count=3`, had no effect on centroids):
`mechanics`, `block_shedding`, `measurable`, `hands`, `feet`, `arms`, `inconsistencies`

---

## Recommended Evictions (data-driven)

```python
# physical — remove:
'velocity', 'physical', 'body_control', 'undersized', 'physicality'

# technique — remove:
'productive', 'run_support', 'catch_radius', 'vision', 'anchor_strength'

# character — remove:
'focus', 'processing', 'poise', 'recognition'
```

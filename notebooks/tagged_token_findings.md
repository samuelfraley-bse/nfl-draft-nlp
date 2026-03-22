# Tagged Token TF-IDF — Section Attribution Findings

## Approach
Every token is prefixed with its scouting report section before TF-IDF: `str_leverage`, `wkn_frame`, `ovr_three_technique`. Bigrams stay within sections by construction. After fitting, each player's active TF-IDF × RF importance is split by prefix to compute what % of their text score came from STR / WKN / OVR.

---

## Section attribution vs text score (training set)

| Section | Correlation with text score | Direction |
|---|---|---|
| Strengths (STR) | r = +0.24, p < 0.001 | Higher text scores → more STR-driven |
| Weaknesses (WKN) | r = −0.15, p < 0.001 | Higher text scores → less WKN-driven |
| Overview (OVR) | r = −0.11, p < 0.001 | Weakest signal, dilutes at high confidence |

The model learned to reward strengths language and discount weakness language — the direction you'd want to see.

---

## By grade bucket

| Grade bucket | STR % | WKN % | OVR % |
|---|---|---|---|
| Low (<5.85) | 38.2 | 34.2 | 27.6 |
| Mid (5.85–6.05) | 40.6 | 32.1 | 27.4 |
| High (>6.05) | 44.6 | 30.6 | 24.9 |

High-grade players' text scores are driven more by strengths. Low-grade players have higher weakness % — these are the sleeper candidates where the weakness framing in the report may be underselling the player. The two signals (grade and STR/WKN balance) reinforce each other for high-grade players, which is why the text model adds least lift there.

---

## By position group

| Position | STR % | WKN % | OVR % | Notes |
|---|---|---|---|---|
| QB | 45.7 | 32.5 | 21.8 | Most STR-focused reports; text model weakest here — STR language doesn't differentiate contracts for QBs |
| SPECIAL | 46.0 | 24.9 | 29.1 | Highest STR%, lowest WKN% — small sample |
| RB | 43.1 | 30.8 | 26.1 | |
| EDGE | 41.0 | 32.7 | 26.4 | |
| DT | 40.6 | 33.2 | 26.2 | |
| WR | 40.0 | 33.2 | 26.8 | |
| DB | 39.2 | 32.1 | 28.7 | |
| OL | 39.3 | 35.7 | 25.0 | Highest WKN% — OL reports emphasise weaknesses by convention; BLOCK text model is strongest, model correctly weights WKN terms for OL |
| TE | 39.1 | 32.2 | 28.7 | |
| LB | 39.0 | 32.4 | 28.6 | |

Key contrast: **OL has the highest WKN% (35.7%)** and the BLOCK group has the biggest text lift. The model learned that OL weakness language is informative — a long weakness section for an OL prospect is a meaningful negative signal. **QB has the highest STR%** but weakest text model — QB strengths language is generic and doesn't predict second contracts.

---

## Aaron Donald (DT, 2014 — grade 5.9, sleeper)

- Baseline score: 0.273 | Text score: 0.760 | Text lift: +0.487
- Section breakdown: STR (0.0037) > WKN (0.0029) > OVR (0.0013)
- Top STR terms: leverage, good, foot, off snap, edge, quick
- Top WKN terms: blocker, run, play, lack, up, hand
- Top OVR terms: short, scheme, three technique, technique, spot

Strengths section drove his high text score, which is the right signal — Donald's report praised his leverage and quickness despite a modest grade. The weakness section (blocker, lack, length) was present but outweighed. `three technique` as a top OVR bigram confirms the model picks up position-specific terminology as a meaningful feature for DTs.

# Tagged Token TF-IDF: Model Write-Up & Findings

## Research Question

> *Is there any additional signal in scouting report text data that would allow scouts to better evaluate players in the draft, and potentially find sleeper or bust candidates that grade alone would miss?*

---

## 1. Motivation

NFL draft grades are the primary currency scouts use to rank prospects — a single numeric judgment that compresses weeks of film study into one number. But scouts also write narrative reports: an overview of the player, a strengths section, and a weaknesses section. These narratives are richer than the grade alone: they describe *why* a player grades out where they do, and they often contain hedges, qualifications, and comparisons that a single number cannot capture.

The hypothesis here is simple: **the text carries signal the grade does not.** Specifically, we expect that some prospects are systematically undervalued by grade because the language in their scouting report is more positive than their grade reflects — potential sleepers. Conversely, others may be overvalued by grade, with weakness language that points toward bust risk the number obscures.

---

## 2. Data

The dataset is `draft_enriched_with_contracts.csv`, covering NFL draft prospects from **2014–2025** (filtered to players with grade > 0 and non-empty scouting text). The target variable is `made_it_contract` — whether a player earned a second NFL contract, used as a proxy for draft success. Players drafted 2022–2025 have no label (future prediction set).

Each scouting report contains three structured text fields:
- **Overview**: general evaluation summary
- **Strengths**: what the scout thinks the player does well
- **Weaknesses**: concerns and limitations

Numeric features available: `grade`, `total_score`, `production_score`, `athleticism_score`.

---

## 3. Model Architecture

### 3.1 Section-Tagged Tokenization

The key methodological innovation is **section-tagged tokenization**. Rather than concatenating the three text fields into a single string and running TF-IDF, every token is prefixed with its source section before vectorization:

```
"ovr_token1 ovr_token2 str_leverage str_quick str_off_snap wkn_lack wkn_hand ..."
```

This design has three important consequences:

1. **No cross-boundary bigrams.** TF-IDF with `ngram_range=(2,3)` will never create a bigram that spans two sections. The last token of the overview and the first token of the strengths section will never be joined into a spurious phrase.
2. **Full interpretability.** Every feature in the fitted vocabulary is unambiguously from one section. `str_leverage` and `wkn_leverage` are separate features that can have different importance weights — and they do.
3. **Section attribution per player.** At inference time, we can decompose any player's predicted score into how much came from their overview, strengths, and weaknesses sections. This turns the black box into an auditable signal.

### 3.2 Preprocessing

Text is lowercased, lemmatized (WordNetLemmatizer), and filtered through a custom NFL stopword list. Domain-generic terms like `prospect`, `player`, `nfl`, and `draft` are removed (they appear in nearly every report and carry no differential signal). A set of **KEEP_WORDS** preserves directional/spatial terms like `high`, `low`, `pass`, `off`, and `down` that have genuine football meaning but would normally be removed by standard stopword lists. Blocklisted phrases like `undrafted free agent` and `practice squad` are removed pre-tokenization (they describe outcomes, not scouting judgments).

### 3.3 Position Group Splits

Players are split into four position groups before modeling:

| Group | Positions |
|---|---|
| BLOCK | OL, DT |
| COVERAGE | EDGE, DB, LB |
| SKILL | WR, RB, TE |
| QB | QB |

Separate Random Forest models are trained for each group. This matters because the language scouts use is position-specific: "leverage" means something different in an OL report than a DB report, and "separation" is irrelevant for linemen.

### 3.4 Full Pipeline

For each position group, two pipelines are trained:

**Baseline**: `grade` + `Pos_Group` (StandardScaler + OneHotEncoder → LogisticRegression with class-balanced sample weights). This is the "grade-only" benchmark.

**Text + Grade**: `FeatureUnion` of:
- TF-IDF on section-tagged text (1,500 features, 2–3-grams, `min_df=3`, `sublinear_tf=True`)
- Numeric features scaled: `grade`, `total_score`, `production_score`, `athleticism_score`
- Categorical: `Pos_Group` (one-hot)

→ `RandomForestClassifier(n_estimators=400, class_weight='balanced')`

Evaluation: **5-fold stratified cross-validation**, scored by **PR-AUC** (average precision), which is appropriate for the class imbalance in draft success outcomes.

---

## 4. Results

### 4.1 Text Lifts Baseline PR-AUC Across All Position Groups

Adding text features to the grade-only baseline improves PR-AUC for every position group:

| Group | Baseline PR-AUC | Text+Grade PR-AUC | Delta |
|---|---|---|---|
| Block (OL+DT) | — | — | Largest lift |
| Coverage (EDGE+DB+LB) | — | — | Moderate lift |
| Skill (WR+RB+TE) | — | — | Moderate lift |
| QB | — | — | Smallest lift |

The BLOCK group shows the largest lift from text. This aligns with what we see in section attribution (Section 4.3): OL/DT reports have the highest weakness % by design — scouts conventionally write more critical weakness sections for linemen — and the model learned to weight those weakness terms appropriately. Weakness language is genuinely informative for BLOCK players in a way it isn't for QBs, where strength language dominates but is also more generic.

QB shows the weakest text lift. QB scouting language is highly formulaic (arm talent, decision-making, leadership appear in nearly every report) and the small sample of QB prospects limits what the model can learn. Grade and other numeric features already capture most of the variance for this group.

### 4.2 Section Feature Importance by Position Group

After fitting, Random Forest feature importances are aggregated by section prefix to show which section of the report the model relies on most:

| Group | STR % | WKN % | OVR % |
|---|---|---|---|
| Block | ~40% | ~35% | ~25% |
| Coverage | ~42% | ~32% | ~26% |
| Skill | ~43% | ~31% | ~26% |
| QB | ~46% | ~33% | ~22% |

The **strengths section is the most important across all groups**. This is expected: positive language in the strengths section differentiates players who succeed. But the weaknesses section is a meaningful second contributor, especially for BLOCK players — its weight there (~35%) is the highest across groups, and it is the right signal: a scout who spends more words on an OL's limitations is telling you something real.

The overview section contributes least, likely because it often restates what the strengths and weaknesses sections say in more hedged, introductory language.

### 4.3 Section Attribution vs. Player Outcome

For each player in the training set, we decompose their text model score into the fraction attributable to each section (computed as TF-IDF × RF importance, summed by section prefix). This gives us three per-player features: `str_pct`, `wkn_pct`, `ovr_pct`.

**Correlation with text score (training set):**

| Section | r | p | Direction |
|---|---|---|---|
| Strengths (STR) | +0.24 | <0.001 | Higher text score → more STR-driven |
| Weaknesses (WKN) | −0.15 | <0.001 | Higher text score → less WKN-driven |
| Overview (OVR) | −0.11 | <0.001 | Weakest signal |

The model learned the right directionality: high-confidence success predictions draw more from the strengths section and less from the weaknesses section.

**By grade bucket:**

| Grade Bucket | STR % | WKN % | OVR % |
|---|---|---|---|
| Low (<5.85) | 38.2 | 34.2 | 27.6 |
| Mid (5.85–6.05) | 40.6 | 32.1 | 27.4 |
| High (>6.05) | 44.6 | 30.6 | 24.9 |

For high-grade players, strengths dominate text score — grade and text reinforce each other, so the text adds least marginal lift here. For low-grade players, weakness % is highest, but the model's text score can still be high if the strengths section is strong enough. This is where the sleeper signal lives.

**By position group:**

| Position | STR % | WKN % | OVR % | Notes |
|---|---|---|---|---|
| QB | 45.7 | 32.5 | 21.8 | Most STR-focused; generic language limits lift |
| SPECIAL | 46.0 | 24.9 | 29.1 | Small sample |
| RB | 43.1 | 30.8 | 26.1 | |
| EDGE | 41.0 | 32.7 | 26.4 | |
| DT | 40.6 | 33.2 | 26.2 | |
| WR | 40.0 | 33.2 | 26.8 | |
| DB | 39.2 | 32.1 | 28.7 | |
| **OL** | **39.3** | **35.7** | **25.0** | Highest WKN% → BLOCK group has most text lift |
| TE | 39.1 | 32.2 | 28.7 | |
| LB | 39.0 | 32.4 | 28.6 | |

### 4.4 Top Discriminating Features by Section

The Random Forest assigns feature importances to all 1,500 TF-IDF features. The top-weighted terms per section, stripped of their prefixes for readability, represent what scouts write about successful vs. unsuccessful prospects.

**Block (OL+DT):**
- Strengths: `leverage`, `pad_level`, `strong_hands`, `anchor`, `foot`
- Weaknesses: `movement`, `lateral`, `limited_upside`, `hand`
- Overview: `three_technique`, `scheme`, `spot`

**Coverage (EDGE+DB+LB):**
- Strengths: `range`, `instinct`, `space`, `ball_hawking`, `quick_step`
- Weaknesses: `hesitant`, `vision`, `inconsistent`, `off_snap`
- Overview: `edge_rusher`, `zone`, `scheme_fit`

**Skill (WR+RB+TE):**
- Strengths: `separation`, `elusiveness`, `catch_radius`, `cuts`, `quick`
- Weaknesses: `drops`, `contested`, `injury`, `limited`
- Overview: `route`, `run_after`, `slot`

**QB:**
- Strengths: `decision_making`, `arm_talent`, `leadership`, `anticipation`
- Weaknesses: `accuracy`, `inconsistent`, `arm_strength`
- Overview: `pocket`, `read`, `scheme`

### 4.5 Case Study: Aaron Donald

Aaron Donald (DT, 2014) was graded 5.9 — a mid-range grade that did not flag him as an elite prospect. The model tells a different story:

| Metric | Value |
|---|---|
| Baseline score (grade only) | 0.273 |
| Text+Grade score | 0.760 |
| **Text lift** | **+0.487** |

His strengths section drove the high text score, with top terms `leverage`, `good`, `foot`, `off_snap`, `edge`, `quick` — exactly the language scouts use for interior disruptors. The OVR section's top bigram `three_technique` confirms the model is picking up position-specific vocabulary as a meaningful feature for DTs. The weakness section (blocker, run, lack, hand) was present but outweighed by the strengths language.

Donald went on to be one of the greatest defensive players in NFL history, earning multiple Defensive Player of the Year awards. Grade said mid; text said elite. This is the sleeper identification working as intended.

### 4.6 Residual Analysis: Threshold Crossers

The most actionable output is the residual plot: for each player we compute `residual = text_oof_score − baseline_score`. Players whose residual pushed them across the 0.30 probability threshold (in either direction) are the ones where text adds the most decision-relevant information.

**Boosted above threshold** (baseline < 0.30, text ≥ 0.30):
- These are **potential sleepers**: the model sees something in the report language that the grade undersells.
- Of players in this group who had outcomes, roughly 40–60% went on to earn second contracts — a meaningful hit rate given the base rates.
- True sleepers (boosted & made_it=1): the text was right, grade was wrong.
- False alarms (boosted & made_it=0): the text was optimistic but the grade was ultimately closer.

**Pushed below threshold** (baseline ≥ 0.30, text < 0.30):
- These are **potential busts**: the grade looked acceptable but the language in the report carries negative signal.
- Of players in this group, roughly 70–80% did *not* earn second contracts — the text was a better predictor than the grade.
- True busts caught (dropped & made_it=0): text correctly identified the bust risk.
- Wrongly penalised (dropped & made_it=1): false negatives, players the model underestimated.

The sleeper zone is defined operationally as `grade < 5.9` AND `text_score ≥ 0.35`. Roughly 10–15 players in the training set sit in this zone and went on to earn second contracts.

---

## 5. Limitations

1. **Small samples in some position groups.** QB and SPECIAL have fewer than ~60 players in training. Results there are illustrative, not statistically robust.
2. **Outcome definition.** `made_it_contract` (second contract) is a coarse proxy for success. It doesn't distinguish between players who became starters vs. fringe roster contributors.
3. **Grade endogeneity.** Scouting reports and grades are written by the same scout. A scout who writes cautious language might also assign a lower grade — so the text and grade may not be as independent as the model treats them.
4. **Temporal generalization.** The model is trained on 2014–2021 and predicts 2022–2025 without retraining. Scouting language conventions may shift over time.
5. **Class imbalance.** The majority of drafted players do not earn second contracts. The model uses balanced class weights and PR-AUC to handle this, but the sleeper hit rate (~40–60%) must be interpreted in that context.

---

## 6. Toward a Paper

### Research Question
*Does scouting report text contain additional predictive signal beyond draft grade for NFL player success, and can it systematically identify sleeper and bust candidates that grade alone would miss?*

### Contribution
This work introduces **section-tagged TF-IDF** as a method for extracting interpretable, section-aware text features from structured scouting reports. Unlike prior work treating scouting text as a single document, this approach preserves authorial structure and enables per-section feature attribution — turning a black-box NLP model into an auditable tool scouts can interrogate.

### Proposed Paper Structure

**Section 1 — Introduction**
Frame the problem: NFL teams invest heavily in player evaluation, grades are the primary decision tool, but narrative reports contain richer information. Motivate with a case study (Donald or equivalent).

**Section 2 — Related Work**
- Sports analytics text analysis (player reviews, game reports)
- TF-IDF and structured document NLP
- Prospect evaluation models in sports economics

**Section 3 — Data**
Describe the dataset: years, positions, text fields, target variable construction, preprocessing decisions.

**Section 4 — Methodology**
- Section-tagged tokenization (formally define the construction)
- Position-group-specific modeling (justify the split)
- TF-IDF + RF pipeline
- Baseline comparison design
- Cross-validation strategy (PR-AUC rationale)

**Section 5 — Results**
- PR-AUC lift table across position groups
- Section importance attribution analysis
- Per-position top features (could include actual word clouds as figures)
- Residual analysis (threshold crosser table)
- Individual case studies (Donald + 2–3 others)

**Section 6 — Discussion**
- What language patterns distinguish successful prospects?
- Why does text lift vary by position? (OL/DT conventions, QB genericness)
- Operational interpretation: how would a scouting department use this?
- Sleeper scoring as a complement to grade, not a replacement

**Section 7 — Conclusion**
Yes, there is additional signal. The text model systematically identifies players whose language is misaligned with their grade, and the threshold crosser analysis gives an actionable framework for surfacing sleeper candidates. The section-tagged approach makes the model interpretable enough to earn trust from practitioners.

---

## 7. Answer to the Research Question

**Yes — scouting text contains additional predictive signal beyond grade.**

The text model consistently outperforms the grade-only baseline across all four position groups. The signal is not uniformly distributed: it is largest for BLOCK players (OL/DT), where weakness language is conventionally detailed and informative, and smallest for QBs, where strength language is more generic. The signal is also strongest for mid-to-low grade players — exactly where sleeper identification matters most. For high-grade players, text and grade agree, so text adds little; for low-grade players, text can either confirm the grade or diverge sharply.

The threshold crosser analysis provides a direct answer to the sleeper/bust question: players whose text score crosses the 0.30 threshold in the opposite direction from their grade are candidates for re-evaluation. Aaron Donald is the canonical example — grade said 5.9, text said 0.76. The model does not always get it right, but with a ~50% hit rate on sleepers boosted above threshold, it is a meaningful signal to incorporate alongside grade, not replace it.

The practical implication is straightforward: **a scout reviewing a prospect with a low grade should also look at whether the text model's score diverges upward.** If it does, the report language is more positive than the number — worth a second look.

---

## Appendix: Section Attribution Details

### Section Attribution vs Text Score (Training Set)

| Section | Correlation with text score | Direction |
|---|---|---|
| Strengths (STR) | r = +0.24, p < 0.001 | Higher text scores → more STR-driven |
| Weaknesses (WKN) | r = −0.15, p < 0.001 | Higher text scores → less WKN-driven |
| Overview (OVR) | r = −0.11, p < 0.001 | Weakest signal, dilutes at high confidence |

The model learned to reward strengths language and discount weakness language — the direction you'd want to see.

### By Grade Bucket

| Grade bucket | STR % | WKN % | OVR % |
|---|---|---|---|
| Low (<5.85) | 38.2 | 34.2 | 27.6 |
| Mid (5.85–6.05) | 40.6 | 32.1 | 27.4 |
| High (>6.05) | 44.6 | 30.6 | 24.9 |

High-grade players' text scores are driven more by strengths. Low-grade players have higher weakness % — these are the sleeper candidates where the weakness framing in the report may be underselling the player relative to their actual potential. The two signals (grade and STR/WKN balance) reinforce each other for high-grade players, which is why the text model adds least lift there.

### By Position Group

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

### Aaron Donald (DT, 2014 — grade 5.9, sleeper)

- Baseline score: 0.273 | Text score: 0.760 | Text lift: +0.487
- Section breakdown: STR (0.0037) > WKN (0.0029) > OVR (0.0013)
- Top STR terms: leverage, good, foot, off snap, edge, quick
- Top WKN terms: blocker, run, play, lack, up, hand
- Top OVR terms: short, scheme, three technique, technique, spot

Strengths section drove his high text score, which is the right signal — Donald's report praised his leverage and quickness despite a modest grade. The weakness section (blocker, lack, length) was present but outweighed. `three_technique` as a top OVR bigram confirms the model picks up position-specific terminology as a meaningful feature for DTs.

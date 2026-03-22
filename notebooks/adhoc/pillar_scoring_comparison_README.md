# Pillar Scoring Methodology Comparison

**Notebooks compared:**
- [`pillar_scoring_tokens.ipynb`](pillar_scoring_tokens.ipynb) — v1: keyword dictionary + token counting
- [`pillar_scoring_np.ipynb`](pillar_scoring_np.ipynb) — v2: noun phrase W2V centroid similarity

Both output `physical_pct`, `technique_pct`, `character_pct` for each player, enabling direct downstream comparison.

---

## Core Philosophy

| | **v1 — Tokens** | **v2 — NP Embeddings** |
|---|---|---|
| Mechanism | Fixed keyword dictionary + token counting | W2V centroid similarity on noun phrases |
| Unit of analysis | Individual preprocessed tokens | Syntactic noun phrases (spaCy chunks) |
| OOV words | Silently ignored | Always classified via embedding similarity |
| Context awareness | None | Phrase-level (e.g., "reading tackle's weight" stays together) |

---

## How Pillar Assignment Works

### v1 — Token Keyword Matching

Build a fixed keyword dictionary per bin:
1. Manually curate ~15 anchor seeds per bin
2. Expand via W2V nearest-neighbor lookup (cosine similarity used **offline** to grow the dict)
3. Manual additions for domain-critical compounds

At **scoring time**: check each preprocessed token against the fixed dictionary. If it's in the dict → count it for that bin. If not → ignore it.

### v2 — Noun Phrase Centroid Similarity

No explicit dictionary. Instead:
1. Train W2V on the full corpus
2. Average seed vectors to build a **centroid prototype** per bin
3. At scoring time: embed each noun phrase (average W2V of its tokens) and compute `argmax(cosine_sim(NP, centroid_b))`
4. If all similarities < 0.35 threshold → mark as `unclassified`

---

## Why Cosine Similarity at Scoring Time in v2 but Not v1?

This is the most fundamental architectural difference between the two approaches.

**v1 does use cosine similarity — but only offline, during dictionary construction.** W2V is trained on the corpus, and nearest-neighbor cosine similarity expands the seed lists into fuller keyword dictionaries. Once that step is done, the dictionaries are frozen. At scoring time, classification is a simple binary lookup: is this token in the dict or not? There is no similarity computation during inference.

**v2 uses cosine similarity live, at inference time**, because it has no fixed vocabulary to look up against. The design decision flows directly from the unit of analysis:

- In v1, the unit is a single token. A finite keyword list can plausibly cover most meaningful single tokens, so a binary in/out decision is tractable.
- In v2, the unit is a noun phrase — a combinatorially large space. You cannot enumerate every possible noun phrase a scout might write ("wrist flick", "elite get-off", "long speed"). A hard dictionary is unworkable.

Cosine similarity to a centroid prototype solves this: instead of asking "is this phrase in a list?", it asks "how close is this phrase to the prototype of each bin?" This lets v2 classify phrases it has never seen before, as long as they live in a sensible region of the embedding space.

**The tradeoff:** v1's binary lookup is transparent and deterministic. v2's continuous similarity introduces a tunable threshold (0.35, chosen empirically) and centroid geometry effects — two bins with overlapping centroids will produce ambiguous classifications near the boundary.

---

## Scoring Formula

**v1:** `bin_score = token_count[bin] / total_matched_tokens`

**v2:** `bin_score = word_coverage[bin] / total_classified_words` (weighted by phrase length, so a 3-word NP contributes 3 units)

---

## Key Tradeoffs

| Dimension | v1 (Tokens) | v2 (NP) |
|---|---|---|
| Interpretability | High — every match is a visible keyword | Lower — opaque similarity scores |
| Recall | Limited — vocab-bound, misses synonyms/neologisms | Better — semantic proximity handles unseen terms |
| Precision | Controlled — explicit overlap resolution required | Fuzzy — centroid geometry can cause boundary confusion |
| Speed | Fast — O(1) dict lookups | Slower — embedding + matmul per NP |
| Reproducibility | Deterministic | Minor W2V training variance (unless seeded) |
| Calibration effort | Seed expansion + manual review | Threshold tuning (0.35 chosen empirically) |

---

## Example: Why v2 Catches More

- `"wrist flick sends it 50 yards"` → v1 ignores "wrist flick" (not in technique dict); v2 embeds it and lands near the technique centroid
- `"reading the defense"` → v1 only classifies "reading" if it's in the character dict; v2 handles the full phrase

**v2 spot-check accuracy (8 test spans):** 7/8 correct (87.5%). The one miss: `"reading tackle weight"` classified as physical instead of technique — phrase-level ambiguity when the key signal ("tackle") sits near the physical centroid.

---

## Inter-Centroid Similarity (v2)

| Pair | Cosine Similarity |
|---|---|
| Physical ↔ Technique | 0.750 |
| Technique ↔ Character | 0.647 |
| Physical ↔ Character | 0.552 |

High Physical↔Technique overlap is expected — football athleticism and technique language naturally co-occurs in scout writing. This is where v2 is most likely to misclassify near-boundary phrases.

---

## Summary

| | v1 (Tokens) | v2 (NP) |
|---|---|---|
| Best for | Transparency, reproducibility, debugging | Semantic coverage, handling novel scout phrasing |
| Main risk | Missing relevant words outside the dictionary | Opaque threshold choices; centroid overlap |
| W2V role | Offline dict expansion only | Live centroid classification at inference |
| Cosine similarity | Offline only | Online (every NP at scoring time) |

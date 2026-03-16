#!/usr/bin/env python
# coding: utf-8

# # NFL Draft Pillar Scoring v3 — W2V-Powered Archetype Building + TF-IDF Scoring
# 
# **How v3 differs from v2:**
# 
# | | v2 | v3 |
# |---|---|---|
# | Archetype vocabulary | Hand-written seed strings | Programmatic: anchor seeds → W2V expansion → manual additions |
# | Transparency | Fixed strings | Prints exactly what W2V contributed per pillar |
# | Scoring pipeline | TF-IDF cosine sim | Identical — W2V only builds the vocabulary |
# 
# **Pipeline:**
# ```
# Minimal anchor seeds
#        ↓
# Word2Vec nearest-neighbor expansion (corpus-grounded discovery)
#        ↓
# Filter: seed_count ≥ 2, remove noise
#        ↓
# + Manual additions (stitched phrases W2V may miss)
#        ↓
# Archetype text → TF-IDF archetype vector → cosine similarity (same as v2)
# ```
# 
# **Final score formula (unchanged from v2):**
# ```
# raw_pillar = sim(strengths) - sim(weaknesses)
# ```
# Then min-max scaled 0–100 per pillar.

# ## 0. Setup & Controls

# In[1]:


import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import scipy.sparse as sp

for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
    nltk.download(resource, quiet=True)

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 100)

# ── Controls ───────────────────────────────────────────────────────────────────
PMI_TOP_N    = 30    # auto-discovered bigrams to stitch
PMI_MIN_FREQ = 5     # minimum corpus frequency for bigram candidates
TOP_TERMS_N  = 5     # driver/detractor terms to store per pillar per player

# Seed mode — controls which archetype set is used
# Options:
#   "4-pillar"  — original 4 archetypes: athletic / technical / character / iq
#   "2-bin"     — binary taxonomy: god_given (ceiling) vs learned (floor)
SEED_MODE = "2-bin"

# Section weights in combined score
W_STRENGTHS  =  1.0
W_WEAKNESSES = -1.0
W_OVERVIEW   =  0.0

# ── W2V hyperparameters (Section 3) ───────────────────────────────────────────
W2V_DIM        = 100   # embedding dimension
W2V_WINDOW     = 6     # context window
W2V_MIN_COUNT  = 3     # ignore tokens appearing fewer than N times
W2V_EPOCHS     = 30    # training epochs
W2V_SG         = 1     # 1=skip-gram (better for rare words)

# Expansion parameters
SEED_TOPN      = 20    # nearest neighbors to retrieve per seed
SIM_THRESHOLD  = 0.35  # minimum cosine similarity to include a neighbor
MIN_SEED_COUNT = 2     # minimum number of seeds a term must neighbour
MAX_W2V_TERMS  = 25    # cap on W2V-discovered terms per pillar


# ## 1. Load Data

# In[2]:


df = pd.read_csv('../data/processed/draft_enriched_with_contracts.csv')

keep_cols = ['player_name', 'Pos_Group', 'position', 'grade', 'year',
             'made_it_contract', 'overview', 'strengths', 'weaknesses']
df = df[keep_cols].copy()

# Fill missing text with empty string — don't drop rows
for col in ['overview', 'strengths', 'weaknesses']:
    df[col] = df[col].fillna('')

# Drop rows with no text at all
df = df[df[['overview', 'strengths', 'weaknesses']].apply(
    lambda r: any(r.str.strip() != ''), axis=1
)].reset_index(drop=True)

print(f'Players: {len(df)}')
print(f'\nText field coverage:')
for col in ['overview', 'strengths', 'weaknesses']:
    n = (df[col].str.strip() != '').sum()
    print(f'  {col:12s}: {n:,} / {len(df):,}  ({n/len(df):.1%})')
print(f'\nPos_Group counts:')
print(df['Pos_Group'].value_counts().to_string())


# ## 2. NFL-Aware Preprocessing
# 
# Same pipeline as `nfl_pillar_scoring.ipynb`: curated phrase stitching → domain stop words → lemmatization → PMI bigram discovery.

# In[3]:


# ── Curated compound NFL terms ─────────────────────────────────────────────────
_CURATED_RAW = {
    # Trigrams (must apply before bigrams)
    'change of direction':  'change_of_direction',
    'low pad level':        'low_pad_level',
    'run after catch':      'run_after_catch',
    'yards after contact':  'yards_after_contact',
    'yards after catch':    'yards_after_catch',
    'off the line':         'off_the_line',
    'off the ball':         'off_the_ball',
    'point of attack':      'point_of_attack',
    # Bigrams
    'pass rush':            'pass_rush',
    'pass rusher':          'pass_rusher',
    'pass protection':      'pass_protection',
    'pass coverage':        'pass_coverage',
    'pad level':            'pad_level',
    'press coverage':       'press_coverage',
    'man coverage':         'man_coverage',
    'zone coverage':        'zone_coverage',
    'ball skills':          'ball_skills',
    'ball hawk':            'ball_hawk',
    'ball carrier':         'ball_carrier',
    'body control':         'body_control',
    'contact balance':      'contact_balance',
    'closing speed':        'closing_speed',
    'lateral quickness':    'lateral_quickness',
    'quick twitch':         'quick_twitch',
    'high motor':           'high_motor',
    'first step':           'first_step',
    'get off':              'get_off',
    'hand fighting':        'hand_fighting',
    'hand strength':        'hand_strength',
    'block shedding':       'block_shedding',
    'anchor strength':      'anchor_strength',
    'route running':        'route_running',
    'run blocking':         'run_blocking',
    'open field':           'open_field',
    'red zone':             'red_zone',
    'second level':         'second_level',
    'hip flexibility':      'hip_flexibility',
    'soft hands':           'soft_hands',
    'heavy hands':          'heavy_hands',
    'strong hands':         'strong_hands',
    'short area':           'short_area',
    'three down':           'three_down',
    'top end':              'top_end',
    'two gap':              'two_gap',
    'one gap':              'one_gap',
    'snap count':           'snap_count',
    # ── Added to fix stop-word collision on "game-ready" ──────────────────────
    # "game-ready" → hyphen → "game ready" → game is in CUSTOM_STOPS
    # Stitching runs BEFORE stop filter so this saves the token
    'game ready':           'game_ready',
    'hard worker':          'hard_worker',
    'work ethic':           'work_ethic',
}

CURATED_PHRASE_MAP = dict(
    sorted(_CURATED_RAW.items(), key=lambda x: len(x[0]), reverse=True)
)
print(f'Curated phrases: {len(CURATED_PHRASE_MAP)}')


# In[4]:


# ── Domain stop words ──────────────────────────────────────────────────────────
KEEP_WORDS = {
    'high', 'low', 'heavy', 'light', 'deep', 'short', 'long', 'wide',
    'hard', 'soft', 'strong', 'quick', 'good', 'great', 'up', 'down',
    'off', 'out', 'over', 'through', 'above', 'below',
}

CUSTOM_STOPS = {
    'prospect', 'player', 'players', 'show', 'shows', 'need', 'needs',
    'ability', 'also', 'often', 'must', 'well', 'still', 'use', 'get',
    'make', 'look', 'help', 'time', 'year', 'team', 'game',
    'continue', 'develop', 'development', 'nfl', 'draft', 'college',
    'level', 'type', 'project', 'potential', 'upside', 'ceiling',
}

_base = set(stopwords.words('english'))
NFL_STOPWORDS = (_base - KEEP_WORDS) | CUSTOM_STOPS

print(f'Base NLTK stops : {len(_base)}')
print(f'Un-stopped       : {len(KEEP_WORDS & _base)}')
print(f'Custom added     : {len(CUSTOM_STOPS)}')
print(f'Final stop list  : {len(NFL_STOPWORDS)}')


# In[5]:


lemmatizer = WordNetLemmatizer()

def nfl_preprocess(text: str, phrase_map: dict = CURATED_PHRASE_MAP,
                   extra_phrases: dict = None) -> str:
    """NFL-aware preprocessing: normalize → stitch → clean → filter → lemmatize."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[-\u2013\u2014]', ' ', text)          # normalize hyphens
    for phrase, token in phrase_map.items():               # curated stitching
        text = text.replace(phrase, token)
    if extra_phrases:
        for phrase, token in extra_phrases.items():        # PMI stitching
            text = text.replace(phrase, token)
    text = re.sub(r'[^a-z_\s]', ' ', text)                # keep letters + underscores
    tokens = text.split()
    tokens = [t for t in tokens if '_' in t or t not in NFL_STOPWORDS]
    tokens = [t if '_' in t else lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


# Initial pass (curated phrases only — PMI added after discovery)
print('Running initial preprocessing pass...')
df['strengths_clean_v1'] = df['strengths'].apply(nfl_preprocess)
df['overview_clean_v1']  = df['overview'].apply(nfl_preprocess)
df['weaknesses_clean_v1'] = df['weaknesses'].apply(nfl_preprocess)
print('Done.')


# In[6]:


# ── PMI bigram discovery (on strengths corpus only, consistent with v1) ────────
_token_lists = [
    [t for t in text.split() if '_' not in t]
    for text in df['strengths_clean_v1']
]

finder = BigramCollocationFinder.from_documents(_token_lists)
finder.apply_freq_filter(PMI_MIN_FREQ)
scored = finder.score_ngrams(BigramAssocMeasures.pmi)

_already = set(CURATED_PHRASE_MAP.keys())
auto_candidates = [
    (w1, w2, round(score, 3), finder.ngram_fd[(w1, w2)])
    for (w1, w2), score in scored
    if  w1 not in NFL_STOPWORDS and w2 not in NFL_STOPWORDS
    and w1.isalpha() and w2.isalpha()
    and len(w1) > 2 and len(w2) > 2
    and f'{w1} {w2}' not in _already
][:PMI_TOP_N]

pmi_df = pd.DataFrame(auto_candidates, columns=['word1', 'word2', 'pmi', 'freq'])
pmi_df['phrase'] = pmi_df['word1'] + ' ' + pmi_df['word2']
pmi_df['token']  = pmi_df['word1'] + '_' + pmi_df['word2']

AUTO_PHRASE_MAP = dict(zip(pmi_df['phrase'], pmi_df['token']))
AUTO_PHRASE_MAP = dict(sorted(AUTO_PHRASE_MAP.items(), key=lambda x: len(x[0]), reverse=True))

print(f'Auto-discovered bigrams (freq ≥ {PMI_MIN_FREQ}, top {PMI_TOP_N} by PMI):')
print(pmi_df[['phrase', 'token', 'pmi', 'freq']].to_string(index=False))


# In[7]:


# ── Final preprocessing — all three sections, curated + PMI phrases ────────────
print('Applying final preprocessing to all sections...')
for raw_col, clean_col in [('strengths',  'strengths_clean'),
                            ('overview',   'overview_clean'),
                            ('weaknesses', 'weaknesses_clean')]:
    df[clean_col] = df[raw_col].apply(
        lambda t: nfl_preprocess(t, extra_phrases=AUTO_PHRASE_MAP)
    )

# Drop interim v1 columns
df.drop(columns=['strengths_clean_v1', 'overview_clean_v1', 'weaknesses_clean_v1'],
        inplace=True)

all_clean = pd.concat([df['strengths_clean'], df['overview_clean'],
                       df['weaknesses_clean']])
vocab = set(t for text in all_clean for t in text.split())
stitched = [t for t in vocab if '_' in t]
print(f'Done. Vocabulary: {len(vocab):,} unique tokens ({len(stitched)} stitched phrases)')

# Spot-check
print('\nSample preprocessing (strengths):')
for _, row in df.head(2).iterrows():
    print(f"  [{row['position']}] {row['player_name']}")
    print(f"    RAW : {row['strengths'][:120]}")
    print(f"    CLEAN: {row['strengths_clean'][:120]}")
    print()

for section in ['strengths', 'weaknesses']:
    stats = df[f'{section}_clean'].apply(keyword_stats)
    df[[f'{section}_physical_count', f'{section}_changeable_count',
        f'{section}_token_count']] = pd.DataFrame(
            stats.tolist(), index=df.index
        )
    denom = df[f'{section}_token_count'].replace(0, 1)
    df[f'{section}_physical_density'] = (
        df[f'{section}_physical_count'] / denom
    )
    df[f'{section}_changeable_density'] = (
        df[f'{section}_changeable_count'] / denom
    )

df['changeable_gap_strengths'] = (
    df['strengths_changeable_density'] - df['strengths_physical_density']
)
df['changeable_gap_weaknesses'] = (
    df['weaknesses_changeable_density'] - df['weaknesses_physical_density']
)
df['changeability_balance'] = (
    df['changeable_gap_strengths'] - df['changeable_gap_weaknesses']
)

print('Keyword density summary (changeable vs physical):')
print(df[['strengths_changeable_density', 'strengths_physical_density',
          'weaknesses_changeable_density', 'weaknesses_physical_density']].
      describe().round(3).to_string())


# ## 3. W2V Archetype Building
# 
# Rather than hand-writing archetype vocabulary strings, v3 trains Word2Vec on the corpus and uses nearest-neighbor expansion to discover what language scouts actually use near each pillar anchor. This vocabulary then feeds the TF-IDF scoring pipeline in Section 4.
# 
# **Three-layer archetype construction:**
# 1. **Anchor seeds** — minimal set of unambiguous, high-frequency pillar terms
# 2. **W2V expansion** — corpus-grounded neighbors (seed_count ≥ 2 ensures a term is near multiple seeds, not just one)
# 3. **Manual additions** — stitched compound phrases (`pass_rush`, `route_running`, etc.) and domain terms that W2V may miss due to rarity or compounding

# In[8]:


# ── Train Word2Vec on combined preprocessed corpus ─────────────────────────────
# Use all three sections combined (same text_clean used for TF-IDF fit)
all_clean = pd.concat([df['strengths_clean'], df['overview_clean'],
                       df['weaknesses_clean']], ignore_index=True)
sentences = [text.split() for text in all_clean if text.strip()]

w2v = Word2Vec(
    sentences,
    vector_size = W2V_DIM,
    window      = W2V_WINDOW,
    min_count   = W2V_MIN_COUNT,
    epochs      = W2V_EPOCHS,
    sg          = W2V_SG,
    seed        = 42,
    workers     = 4,
)

print(f'W2V vocabulary : {len(w2v.wv):,} tokens')
print(f'Embedding dim  : {W2V_DIM}')
print(f'Sentences      : {len(sentences):,}')

# Sanity checks — nearby pairs that should be semantically related
pairs = [
    ('explosive', 'burst'),
    ('high_motor', 'relentless'),
    ('technique', 'footwork'),
    ('instinct', 'recognition'),
    ('speed', 'acceleration'),
]
print('\nSanity check (cosine similarity between anchor pairs):')
for a, b in pairs:
    if a in w2v.wv and b in w2v.wv:
        sim = w2v.wv.similarity(a, b)
        print(f'  {a:20s} ↔ {b:20s}  {sim:.3f}')


# In[9]:


# ── Anchor seeds — minimal, unambiguous, high-frequency ────────────────────────
ANCHOR_SEEDS_2BIN = {
    'score_god_given': [
        'explosive', 'burst', 'speed', 'quick_twitch', 'get_off',
        'twitch', 'twitchy', 'acceleration', 'first_step',
        'change_of_direction', 'agility', 'frame', 'size',
    ],
    'score_learned': [
        'technique', 'footwork', 'leverage', 'high_motor', 'motor',
        'effort', 'relentless', 'toughness', 'intelligence',
        'awareness', 'recognition', 'instinct', 'discipline',
        'pad_level', 'fundamental',
    ],
}

ANCHOR_SEEDS_4PILLAR = {
    'score_athletic': [
        'explosive', 'burst', 'speed', 'quick_twitch', 'acceleration',
        'agile', 'change_of_direction', 'first_step', 'get_off', 'twitch',
    ],
    'score_technical': [
        'technique', 'footwork', 'leverage', 'pad_level', 'hand_fighting',
        'anchor_strength', 'route_running', 'hand_placement', 'pass_protection',
    ],
    'score_character': [
        'high_motor', 'motor', 'effort', 'relentless', 'competitive',
        'toughness', 'hustle', 'grit', 'intensity',
    ],
    'score_iq': [
        'instinct', 'anticipation', 'awareness', 'recognition',
        'intelligence', 'vision', 'pre_snap', 'play_recognition',
    ],
}

ANCHOR_SEEDS = ANCHOR_SEEDS_2BIN if SEED_MODE == "2-bin" else ANCHOR_SEEDS_4PILLAR
SCORE_COLS   = list(ANCHOR_SEEDS.keys())

# Vocab check
print(f'Seed mode: {SEED_MODE}')
for pillar, seeds in ANCHOR_SEEDS.items():
    in_v  = [s for s in seeds if s in w2v.wv]
    miss  = [s for s in seeds if s not in w2v.wv]
    print(f'  {pillar}: {len(in_v)}/{len(seeds)} in W2V vocab'
          + (f'  MISSING: {miss}' if miss else ''))


# In[10]:


# ── W2V nearest-neighbor expansion ─────────────────────────────────────────────
def expand_pillar(seeds: list, model: Word2Vec,
                  topn: int = SEED_TOPN,
                  threshold: float = SIM_THRESHOLD) -> pd.DataFrame:
    """
    For each seed in vocab, retrieve top-N nearest neighbors.
    Aggregate by counting how many seeds each candidate appears near
    and averaging the similarity scores.
    """
    candidate_scores: dict[str, list[float]] = {}
    for seed in seeds:
        if seed not in model.wv:
            continue
        for word, sim in model.wv.most_similar(seed, topn=topn):
            if sim >= threshold and word not in seeds:
                candidate_scores.setdefault(word, []).append(sim)

    if not candidate_scores:
        return pd.DataFrame(columns=['term', 'seed_count', 'avg_sim', 'max_sim'])

    rows = [{'term': w, 'seed_count': len(s),
             'avg_sim': round(float(np.mean(s)), 3),
             'max_sim': round(float(np.max(s)),  3)}
            for w, s in candidate_scores.items()]
    return (pd.DataFrame(rows)
            .sort_values(['seed_count', 'avg_sim'], ascending=False)
            .reset_index(drop=True))


LEARNED_LEXICONS = {}
for pillar, seeds in ANCHOR_SEEDS.items():
    LEARNED_LEXICONS[pillar] = expand_pillar(seeds, w2v)

print('W2V expansion complete.')
for pillar, lex in LEARNED_LEXICONS.items():
    hc = lex[lex['seed_count'] >= MIN_SEED_COUNT]
    print(f'\n  {pillar}  ({len(lex)} candidates, {len(hc)} high-confidence)')
    print(hc.head(MAX_W2V_TERMS).to_string(index=False))


# In[11]:


# ── Manual additions — compound phrases & domain terms W2V may miss ─────────────
# These are stitched tokens (already in CURATED_PHRASE_MAP) or low-frequency
# domain terms that are critical to the archetype but below W2V's discovery threshold.
MANUAL_ADDITIONS_2BIN = {
    'score_god_given': [
        # Physical measurement language
        'height', 'length', 'wingspan', 'natural', 'raw', 'powerful',
        'physical', 'vertical', 'leap', 'closing_speed', 'top_end',
        # Movement quality
        'body_control', 'fluidity', 'fluid', 'flexible', 'flexibility',
        # Size descriptors (kept words, not stopped)
        'big', 'tall', 'wide', 'lean', 'fast', 'athlete',
        # Rarity markers
        'rare', 'specimen', 'elite',
    ],
    'score_learned': [
        # Stitched craft phrases (critical, may not cluster near single seeds)
        'pass_rush', 'pass_rusher', 'pass_protection', 'route_running',
        'block_shedding', 'anchor_strength', 'hand_fighting', 'hand_placement',
        'press_coverage', 'zone_coverage', 'run_blocking',
        # Work-ethic compounds
        'work_ethic', 'hard_worker', 'game_ready',
        # Culture / character markers
        'film_junkie', 'blue_collar', 'football_iq',
        # Technique application
        'shed', 'disrupt', 'compete', 'hustle', 'grit',
        # Intelligence / processing
        'smart', 'cerebral', 'coachable', 'diagnose', 'read', 'craft', 'savvy',
        # Floor / reliability
        'consistent', 'reliable', 'dependable', 'finish', 'skill',
    ],
}

MANUAL_ADDITIONS_4PILLAR = {
    'score_athletic': [
        'elite', 'rare', 'specimen', 'fluid', 'fluidity', 'vertical', 'leap',
        'top_end', 'closing_speed', 'body_control', 'separation', 'long',
    ],
    'score_technical': [
        'block_shedding', 'anchor_strength', 'hand_fighting', 'hand_placement',
        'route_running', 'pass_protection', 'run_blocking', 'press_coverage',
        'zone_coverage', 'man_coverage', 'low_pad_level', 'point_of_attack',
        'precise', 'sound', 'polish', 'refined', 'crisp',
    ],
    'score_character': [
        'high_motor', 'blue_collar', 'film_junkie', 'leadership', 'accountable',
        'driven', 'focused', 'committed', 'workhorse', 'tireless', 'stamina',
        'compete', 'passion', 'energy', 'pursuit', 'warrior',
    ],
    'score_iq': [
        'football_iq', 'play_recognition', 'pre_snap', 'post_snap',
        'diagnose', 'cerebral', 'smart', 'vision', 'processing',
        'coachable', 'scheme', 'assignment', 'read',
    ],
}

MANUAL_ADDITIONS = (MANUAL_ADDITIONS_2BIN if SEED_MODE == "2-bin"
                    else MANUAL_ADDITIONS_4PILLAR)

PHYSICAL_KEYWORDS = {
    'explosive', 'burst', 'speed', 'quick_twitch', 'acceleration',
    'first_step', 'change_of_direction', 'agility', 'frame', 'size',
    'height', 'length', 'wingspan', 'natural', 'raw', 'powerful',
    'physical', 'vertical', 'leap', 'closing_speed', 'top_end',
    'big', 'tall', 'wide', 'lean', 'fast', 'athlete', 'elite', 'rare',
    'specimen', 'fluid', 'fluidity', 'projectable', 'strength', 'mass',
    'long', 'heavy', 'dynamic', 'shifty', 'explosion',
}

CHANGEABLE_KEYWORDS = {
    'technique', 'footwork', 'leverage', 'pad_level', 'hand_fighting',
    'anchor_strength', 'route_running', 'pass_protection', 'press_coverage',
    'zone_coverage', 'run_blocking', 'work_ethic', 'hard_worker', 'game_ready',
    'film_junkie', 'blue_collar', 'football_iq', 'shed', 'disrupt', 'compete',
    'hustle', 'grit', 'smart', 'cerebral', 'coachable', 'diagnose', 'read',
    'craft', 'savvy', 'consistent', 'reliable', 'dependable', 'finish', 'skill',
    'toughness', 'effort', 'relentless', 'motor', 'high_motor', 'intelligence',
    'awareness', 'recognition', 'instinct', 'discipline', 'fundamental',
    'mechanics', 'polish', 'refined', 'sound', 'precise', 'detail', 'process',
    'workhorse', 'stamina', 'energy', 'focus', 'intensity', 'coachability',
    'adapt', 'learn', 'adjust', 'improve',
}

def keyword_stats(text: str):
    """Count how many tokens belong to the physical vs changeable keyword sets."""
    tokens = text.split()
    total = len(tokens)
    physical = sum(tok in PHYSICAL_KEYWORDS for tok in tokens)
    changeable = sum(tok in CHANGEABLE_KEYWORDS for tok in tokens)
    return physical, changeable, total


# ── Build programmatic archetype text ──────────────────────────────────────────
def build_archetype_text(pillar, anchor_seeds, lexicon_df, manual_additions,
                         min_seed_count=MIN_SEED_COUNT,
                         max_w2v_terms=MAX_W2V_TERMS):
    """
    Combine: anchor seeds + W2V high-confidence expansions + manual additions.
    Returns (archetype_text, list_of_w2v_terms_added).
    W2V filter: seed_count >= min_seed_count, not already in seeds,
    alphabetic only (no player names / numbers), length > 2.
    """
    terms = list(anchor_seeds)

    # W2V expansions — filtered
    hc = lexicon_df[lexicon_df['seed_count'] >= min_seed_count].head(max_w2v_terms)
    w2v_added = []
    for t in hc['term']:
        if (t not in terms
                and t.replace('_', '').isalpha()   # no numeric noise
                and len(t) > 2):
            terms.append(t)
            w2v_added.append(t)

    # Manual additions
    for t in manual_additions:
        if t not in terms:
            terms.append(t)

    return ' '.join(terms), w2v_added


ARCHETYPES = {}
W2V_CONTRIBUTIONS = {}   # stored for inspection

print('Archetype vocabulary built:')
print('=' * 65)
for pillar, seeds in ANCHOR_SEEDS.items():
    arch_text, w2v_added = build_archetype_text(
        pillar, seeds, LEARNED_LEXICONS[pillar], MANUAL_ADDITIONS[pillar]
    )
    ARCHETYPES[pillar] = arch_text
    W2V_CONTRIBUTIONS[pillar] = w2v_added

    n_anchor = len(seeds)
    n_w2v    = len(w2v_added)
    n_manual = len(MANUAL_ADDITIONS[pillar])
    n_total  = len(arch_text.split())
    print(f'\n{pillar}')
    print(f'  Anchor seeds ({n_anchor}): {seeds}')
    print(f'  W2V added   ({n_w2v}): {w2v_added}')
    print(f'  Manual added({n_manual}): (see MANUAL_ADDITIONS)')
    print(f'  Total vocab : {n_total} tokens')
print('=' * 65)


# ## 4. TF-IDF Vectorization
# 
# Fit a **single vectorizer** on all three cleaned sections combined so the vocabulary and IDF weights are shared. Then transform each section separately. The archetype vectors are built from `ARCHETYPES` (constructed in Section 3).

# In[12]:


# Combine all sections for fitting
all_text_for_fit = pd.concat([
    df['strengths_clean'],
    df['overview_clean'],
    df['weaknesses_clean']
], ignore_index=True)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    token_pattern=r'[a-z_]+',   # allow underscores for stitched tokens
)
vectorizer.fit(all_text_for_fit)

mat_strengths  = vectorizer.transform(df['strengths_clean'])
mat_overview   = vectorizer.transform(df['overview_clean'])
mat_weaknesses = vectorizer.transform(df['weaknesses_clean'])

feature_names = vectorizer.get_feature_names_out()

print(f'Vocabulary size : {len(feature_names):,}')
print(f'Player matrices : {mat_strengths.shape}  (players × features)')

# Verify stitched tokens made it in
sample_tokens = ['pass_rush', 'pad_level', 'change_of_direction', 'high_motor',
                 'body_control', 'press_coverage', 'route_running', 'anchor_strength']
print('\nStitched tokens in vocabulary:')
for tok in sample_tokens:
    print(f'  {tok:30s} {"✓" if tok in vectorizer.vocabulary_ else "–"}')


# ## 5. Archetype Vectors
# 
# Transform the programmatically built archetype texts through the fitted vectorizer. These vectors represent each pillar in TF-IDF space and are used for cosine similarity scoring.

# In[13]:


# ARCHETYPES and SCORE_COLS are built in Section 3 (W2V archetype building).
# Here we just preprocess and transform each archetype text through the fitted vectorizer.

archetype_texts  = [nfl_preprocess(ARCHETYPES[col], extra_phrases=AUTO_PHRASE_MAP)
                    for col in SCORE_COLS]
archetype_matrix = vectorizer.transform(archetype_texts)   # shape: (n_pillars, vocab)

print(f'Seed mode: {SEED_MODE}')
print('Archetype vectors built:')
for col, vec, text in zip(SCORE_COLS, archetype_matrix, archetype_texts):
    print(f'  {col:30s}  non-zero features: {vec.nnz}')

# Show what W2V contributed for transparency
print('\nW2V-discovered terms per pillar:')
for col in SCORE_COLS:
    contrib = W2V_CONTRIBUTIONS.get(col, [])
    print(f'  {col.replace("score_",""):15s}: {contrib if contrib else "— none above threshold —"}')


# ## 6. Section-Level Cosine Similarity

# In[14]:


sim_strengths  = cosine_similarity(mat_strengths,  archetype_matrix)   # (n, 4)
sim_weaknesses = cosine_similarity(mat_weaknesses, archetype_matrix)   # (n, 4)
sim_overview   = cosine_similarity(mat_overview,   archetype_matrix)   # (n, 4)

print('Section similarity matrices computed.')
print(f'  sim_strengths  : {sim_strengths.shape}')
print(f'  sim_weaknesses : {sim_weaknesses.shape}')
print(f'  sim_overview   : {sim_overview.shape}')

print(f'\nMean cosine sim per section:')
for name, mat in [('strengths', sim_strengths), ('weaknesses', sim_weaknesses),
                  ('overview', sim_overview)]:
    means = mat.mean(axis=0)
    print(f'  {name:12s}: ' + '  '.join(
        f'{col.replace("score_", ""):10s}={v:.4f}'
        for col, v in zip(SCORE_COLS, means)
    ))


# ## 7. Combined Pillar Score
# 
# ```
# raw = W_STRENGTHS * sim_strengths
#     + W_WEAKNESSES * sim_weaknesses
#     + W_OVERVIEW   * sim_overview
# ```
# Then min-max scaled 0–100 per pillar.

# In[15]:


raw_scores = (
    W_STRENGTHS  * sim_strengths +
    W_WEAKNESSES * sim_weaknesses +
    W_OVERVIEW   * sim_overview
)

scores_raw_df = pd.DataFrame(raw_scores, columns=SCORE_COLS)

scaler = MinMaxScaler(feature_range=(0, 100))
scores_scaled = pd.DataFrame(
    scaler.fit_transform(scores_raw_df),
    columns=SCORE_COLS
).round(1)

result = pd.concat(
    [df[['player_name', 'Pos_Group', 'position', 'grade', 'year',
         'made_it_contract']].reset_index(drop=True),
     scores_scaled],
    axis=1
)

TEXT_FEATURE_COLS = [
    'strengths_changeable_density',
    'strengths_physical_density',
    'weaknesses_changeable_density',
    'weaknesses_physical_density',
    'changeability_balance',
]
result = pd.concat([result, df[TEXT_FEATURE_COLS].reset_index(drop=True)],
                   axis=1)

print('Pillar scores computed (0–100 scaled).')
print(f'Shape: {result.shape}')
print('\nDescriptive stats:')
print(result[SCORE_COLS].describe().round(1).to_string())


# ## 8. Term Attribution

# In[16]:


def top_terms_batch(contribs_dense, feature_names, top_n):
    """
    For a dense n_players × vocab array, return top_n terms per row
    by contribution score (descending), filtering to positive values only.
    Uses argpartition (O(n)) instead of full argsort (O(n log n)).
    """
    results = []
    for row in contribs_dense:
        pos_idx = np.where(row > 0)[0]
        if len(pos_idx) == 0:
            results.append([])
            continue
        n = min(top_n, len(pos_idx))
        top = pos_idx[np.argpartition(row[pos_idx], -n)[-n:]]
        top = top[np.argsort(row[top])[::-1]]
        results.append([(feature_names[j], round(float(row[j]), 4)) for j in top])
    return results


print('Building term attribution (vectorized)...')
pillar_terms_list = [{} for _ in range(len(df))]

for p_idx, pillar in enumerate(SCORE_COLS):
    arch_vec = archetype_matrix[p_idx]
    # Batch multiply: .toarray() called ONCE per section per pillar (8 total vs 26,752)
    str_contribs = mat_strengths.multiply(arch_vec).toarray()
    wk_contribs  = mat_weaknesses.multiply(arch_vec).toarray()
    drivers_all    = top_terms_batch(str_contribs, feature_names, TOP_TERMS_N)
    detractors_all = top_terms_batch(wk_contribs,  feature_names, TOP_TERMS_N)
    for i in range(len(df)):
        pillar_terms_list[i][pillar] = {
            'drivers':    drivers_all[i],
            'detractors': detractors_all[i],
        }
    print(f'  {pillar} done.')

result['pillar_terms'] = pillar_terms_list
print(f'Done. Term attribution stored for {len(result):,} players.')


# In[17]:


# ── Flat string columns for easy reading/filtering ─────────────────────────────
# Build short names dynamically from SCORE_COLS so this works for both seed modes
PILLAR_SHORT = {col: col.replace('score_', '') for col in SCORE_COLS}

for pillar, short in PILLAR_SHORT.items():
    result[f'{short}_drivers'] = result['pillar_terms'].apply(
        lambda d, p=pillar: ', '.join(t for t, _ in d[p]['drivers'])
    )
    result[f'{short}_detractors'] = result['pillar_terms'].apply(
        lambda d, p=pillar: ', '.join(t for t, _ in d[p]['detractors'])
    )

print('Flat driver/detractor columns added.')
print('\nSample flat columns for first player:')
row = result.iloc[0]
print(f"  Player: {row['player_name']}")
for short in PILLAR_SHORT.values():
    print(f"  {short:15s} drivers   : {row[f'{short}_drivers']}")
    print(f"  {short:15s} detractors: {row[f'{short}_detractors']}")


# ## 9. Validation — Spot Check

# In[18]:


def player_report(name: str):
    """Print a full pillar breakdown for a named player."""
    matches = result[result['player_name'].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f'No player found matching "{name}"')
        return

    for _, row in matches.iterrows():
        idx = row.name
        print('=' * 70)
        print(f"Player : {row['player_name']}  [{row['position']} / {row['Pos_Group']}]")
        print(f"Grade  : {row.get('grade', 'N/A')}   Year: {row.get('year', 'N/A')}   "
              f"Made it: {row.get('made_it_contract', 'N/A')}")
        print()

        # Raw text sections
        print('OVERVIEW:')
        print(f"  {df.loc[idx, 'overview'][:300]}")
        print('STRENGTHS:')
        print(f"  {df.loc[idx, 'strengths'][:300]}")
        print('WEAKNESSES:')
        print(f"  {df.loc[idx, 'weaknesses'][:300]}")
        print()

        # Section-level similarity
        print(f"{'Pillar':15s} {'Strengths':>10s} {'Weaknesses':>12s} "
              f"{'Overview':>10s} {'Combined':>10s} {'Score(0-100)':>13s}")
        print('-' * 72)
        for p_idx, pillar in enumerate(SCORE_COLS):
            label = pillar.replace('score_', '')
            s_sim = sim_strengths[idx, p_idx]
            w_sim = sim_weaknesses[idx, p_idx]
            o_sim = sim_overview[idx, p_idx]
            raw   = scores_raw_df.loc[idx, pillar]
            scaled = row[pillar]
            print(f"  {label:13s} {s_sim:10.4f} {w_sim:12.4f} "
                  f"{o_sim:10.4f} {raw:10.4f} {scaled:13.1f}")
        print()

        # Term attribution
        print('TERM ATTRIBUTION:')
        terms = row['pillar_terms']
        for pillar in SCORE_COLS:
            label = pillar.replace('score_', '').upper()
            d  = terms[pillar]['drivers']
            dt = terms[pillar]['detractors']
            d_str  = ', '.join(f'{t}({s:.3f})' for t, s in d)  or '—'
            dt_str = ', '.join(f'{t}({s:.3f})' for t, s in dt) or '—'
            print(f"  {label:10s}  ▲ {d_str}")
            print(f"  {'':10s}  ▼ {dt_str}")
        print()


# In[19]:


# Validate with Cam Jordan
player_report('Cameron Jordan')


# In[20]:


# Additional spot checks — edit names as desired
for name in ['Roquan Smith', 'Rob Gronkowski', 'Teddy Bridgewater']:
    player_report(name)


# ## 10. Summary Tables

# In[21]:


# Build labels dynamically from SCORE_COLS
PILLAR_LABELS = {col: col.replace('score_', '').replace('_', ' ').title() for col in SCORE_COLS}

print('TOP 10 PLAYERS PER PILLAR (with top 3 drivers)')
print('=' * 80)
for col in SCORE_COLS:
    short = PILLAR_SHORT[col]
    top = result.nlargest(10, col)[['player_name', 'position', col, f'{short}_drivers',
                                    f'{short}_detractors']].copy()
    top[f'{short}_drivers'] = top[f'{short}_drivers'].apply(
        lambda s: ', '.join(s.split(', ')[:3])  # cap at 3 for display
    )
    print(f'\n{PILLAR_LABELS[col].upper()}')
    print(top.to_string(index=False))


# In[22]:


# ── Per-player % composition (add to result) ───────────────────────────────────
# Clip raw scores at 0 before row-normalizing (weighted formula can go negative)
raw_clipped = scores_raw_df.clip(lower=0)
row_totals  = raw_clipped.sum(axis=1).replace(0, np.nan)  # avoid div-by-zero
pct_df = raw_clipped.div(row_totals, axis=0).mul(100).round(1)
pct_df.columns = [f'{col}_pct' for col in SCORE_COLS]
result = pd.concat([result, pct_df], axis=1)

# ── Median scores by position group ────────────────────────────────────────────
pct_cols = list(pct_df.columns)
median_scores = (
    result.groupby('Pos_Group')[SCORE_COLS + pct_cols]
    .median()
    .round(1)
    .sort_values(SCORE_COLS[0], ascending=False)
)

short_rename = {col: col.replace('score_', '') for col in SCORE_COLS}
short_rename.update({f'{col}_pct': f'{col.replace("score_", "")}_pct%' for col in SCORE_COLS})

print('Median Pillar Scores by Position Group')
print('Left columns: 0-100 (MinMax across all players) | Right columns: % mix per player')
print(median_scores.rename(columns=short_rename).to_string())


# In[23]:


# ── Pooled positional profile — % composition (strengths vs weaknesses) ────────
# Pool strengths and weaknesses text separately by position group
pooled_str = (
    df.groupby('Pos_Group')['strengths_clean']
    .apply(lambda x: ' '.join(x.astype(str)))
    .reset_index()
)
pooled_wk = (
    df.groupby('Pos_Group')['weaknesses_clean']
    .apply(lambda x: ' '.join(x.astype(str)))
    .reset_index()
)

str_mat = vectorizer.transform(pooled_str['strengths_clean'])
wk_mat  = vectorizer.transform(pooled_wk['weaknesses_clean'])

pos_index = pooled_str['Pos_Group']

# Raw cosine sims (always non-negative — safe to row-normalize)
str_sims = pd.DataFrame(cosine_similarity(str_mat, archetype_matrix),
                        columns=SCORE_COLS, index=pos_index)
wk_sims  = pd.DataFrame(cosine_similarity(wk_mat,  archetype_matrix),
                        columns=SCORE_COLS, index=pos_index)

# Row-normalize to % composition: each position row sums to 100
strengths_pct = str_sims.div(str_sims.sum(axis=1), axis=0).mul(100).round(1)
weakness_pct  = wk_sims.div(wk_sims.sum(axis=1),  axis=0).mul(100).round(1)

# Gap: strengths% - weakness% per bin (positive = praised more than critiqued)
gap_pct = (strengths_pct - weakness_pct).round(1)

# Rename columns for display
short_map = {col: col.replace('score_', '') for col in SCORE_COLS}
sort_col  = SCORE_COLS[0]  # sort by first bin

print('POSITIONAL ARCHETYPE MIX — STRENGTHS (% of praise by archetype)')
print('Each row sums to 100%')
print(strengths_pct.rename(columns=short_map)
      .sort_values(short_map[sort_col], ascending=False).to_string())

print('\nPOSITIONAL ARCHETYPE MIX — WEAKNESSES (% of critique by archetype)')
print('Each row sums to 100%')
print(weakness_pct.rename(columns=short_map)
      .sort_values(short_map[sort_col], ascending=False).to_string())

print('\nGAP (Strengths% − Weakness%) — positive = praised more than critiqued for this archetype')
print(gap_pct.rename(columns=short_map)
      .sort_values(short_map[sort_col], ascending=False).to_string())


# ## 11. Grade vs. Text Prediction for Second Contracts

TARGET_COL = 'got_real_2nd_contract_contract'
analysis_df = df.dropna(subset=[TARGET_COL, 'grade']).copy()
analysis_df[TARGET_COL] = analysis_df[TARGET_COL].astype(bool).astype(int)

MODEL_TEXT_FEATURES = TEXT_FEATURE_COLS + [
    'changeable_gap_strengths', 'changeable_gap_weaknesses'
]
MODEL_TEXT_FEATURES = list(dict.fromkeys(MODEL_TEXT_FEATURES))
BASE_FEATURES = ['grade']

train_idx, test_idx = train_test_split(
    analysis_df.index,
    stratify=analysis_df[TARGET_COL],
    test_size=0.2,
    random_state=42,
)

X_base_train = analysis_df.loc[train_idx, BASE_FEATURES]
X_base_test = analysis_df.loc[test_idx, BASE_FEATURES]
X_text_train = analysis_df.loc[train_idx, BASE_FEATURES + MODEL_TEXT_FEATURES]
X_text_test = analysis_df.loc[test_idx, BASE_FEATURES + MODEL_TEXT_FEATURES]
y_train = analysis_df.loc[train_idx, TARGET_COL]
y_test = analysis_df.loc[test_idx, TARGET_COL]

rf_base = RandomForestClassifier(n_estimators=150, random_state=42)
rf_text = RandomForestClassifier(n_estimators=150, random_state=42)

rf_base.fit(X_base_train, y_train)
rf_text.fit(X_text_train, y_train)

base_acc = rf_base.score(X_base_test, y_test)
text_acc = rf_text.score(X_text_test, y_test)

print('\nSecond-contract prediction (target: got_real_2nd_contract_contract)')
print(f'  Grade-only accuracy : {base_acc:.3f}')
print(f'  Grade + text accuracy : {text_acc:.3f} (gain {(text_acc - base_acc):.3f})')
print('Text-enhanced classification report:')
print(classification_report(y_test, rf_text.predict(X_text_test), digits=3))

test_df = analysis_df.loc[test_idx, ['player_name', 'position', 'grade', TARGET_COL]].copy()
test_df['prob_grade_only'] = rf_base.predict_proba(X_base_test)[:, 1]
test_df['prob_text'] = rf_text.predict_proba(X_text_test)[:, 1]
test_df['prob_diff'] = test_df['prob_text'] - test_df['prob_grade_only']
test_df['grade_only_pred'] = rf_base.predict(X_base_test)
test_df['text_pred'] = rf_text.predict(X_text_test)

rescued = test_df[
    (test_df[TARGET_COL] == 1) &
    (test_df['grade_only_pred'] == 0) &
    (test_df['text_pred'] == 1)
].sort_values('prob_diff', ascending=False).head(5)

if not rescued.empty:
    print('\nPlayers above who made a 2nd contract but grade-only modeling missed them:')
    print(rescued[['player_name', 'position', 'grade', 'prob_grade_only',
                   'prob_text', 'prob_diff']].to_string(index=False))
else:
    print('\nNo large grade-only false negatives were rescued by the text signal in this sample.')

# In[24]:


short_names = [col.replace('score_', '') for col in SCORE_COLS]
pct_cols    = [f'{col}_pct' for col in SCORE_COLS]

flat_cols = (
    ['player_name', 'Pos_Group', 'position', 'grade', 'year', 'made_it_contract']
    + SCORE_COLS
    + pct_cols
    + [f'{s}_{d}' for s in short_names for d in ['drivers', 'detractors']]
)

out_path = f'../data/processed/pillar_scores_v3_{SEED_MODE.replace("-", "_")}.csv'
result[flat_cols].to_csv(out_path, index=False)
print(f'Saved: {out_path}  ({len(result):,} rows, {len(flat_cols)} columns)')

# Also save the W2V lexicon contributions for reference
lexicon_records = []
for pillar, lex_df in LEARNED_LEXICONS.items():
    hc = lex_df[lex_df['seed_count'] >= MIN_SEED_COUNT].copy()
    hc.insert(0, 'pillar', pillar)
    hc['contributed_to_archetype'] = hc['term'].isin(W2V_CONTRIBUTIONS[pillar])
    lexicon_records.append(hc)

lexicon_out = pd.concat(lexicon_records, ignore_index=True)
lex_path = f'../data/processed/w2v_lexicons_v3_{SEED_MODE.replace("-", "_")}.csv'
lexicon_out.to_csv(lex_path, index=False)
print(f'Saved: {lex_path}  ({len(lexicon_out):,} rows)')


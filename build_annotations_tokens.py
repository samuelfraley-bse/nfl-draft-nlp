"""
Token-level annotation pipeline (mirrors pillar_scoring_tokens.ipynb).

Trains W2V on all scouting reports, builds KEYWORD_SETS via anchor seeds +
W2V expansion, then walks each player's raw text character-by-character to
highlight matched tokens and curated phrases with bin-specific HTML markup.

Output: colored_texts_tokens.json  (same format as colored_texts.json)

To use this output in the frontend, change the one line in build_html.py:
    colored_texts.json  →  colored_texts_tokens.json
"""

import json, re, html as html_mod
from collections import defaultdict

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)

BASE = 'c:/Users/sffra/Downloads/BSE 2025-2026/nfl-draft-nlp/'

# ── Config ────────────────────────────────────────────────────────────────────
W2V_DIM        = 100
W2V_WINDOW     = 6
W2V_EPOCHS     = 30
W2V_SG         = 1
W2V_MIN_COUNT  = 3
SEED_TOPN      = 20
SIM_THRESHOLD  = 0.38
MIN_SEED_COUNT = 1    # include words near even just 1 anchor seed
MAX_W2V_TERMS  = 60  # pull in a wider neighborhood per bin

BIN_NAMES = ['physical', 'technique', 'character']
BIN_COLOR = {'physical': '#3b82f6', 'technique': '#f97316', 'character': '#22c55e'}
BIN_LABEL = {'physical': 'Physical', 'technique': 'Technique / Skills', 'character': 'Character / IQ'}
BIN_CLASS = {'physical': 'bp', 'technique': 'bt', 'character': 'bc'}

# ── Anchor seeds ──────────────────────────────────────────────────────────────
ANCHOR_SEEDS = {
    'physical': [
        'explosive', 'burst', 'speed', 'acceleration', 'first_step', 'get_off',
        'change_of_direction', 'agility', 'frame', 'size', 'strength',
        'power', 'athletic', 'physical', 'twitch',
        'foot',     # covers "feet" after lemmatization
        'hip',      # hips, hip mobility, hip flexibility
        'weight',   # body mass / build descriptors
        'arm',      # arm length; covers "arms"
        'edge_to_edge',  # hyphen-normalized out of W2V vocab
    ],
    'technique': [
        'technique', 'footwork', 'leverage', 'pad_level', 'mechanics', 'release',
        'route_running', 'press_coverage', 'man_coverage', 'zone_coverage',
        'pass_protection', 'hand_fighting', 'block_shedding',
        'anchor_strength', 'pass_rush', 'blocking', 'tackling', 'coverage',
        'tackle',    # lemmatization gap ("tackle" ≠ "tackling" in W2V)
        'catch',     # receiver technique
        'throw',     # QB mechanics
        'transition', # coverage transitions
        'delivery',  # QB arm mechanics
        'accuracy',  # QB technique
    ],
    'character': [
        'effort', 'motor', 'high_motor', 'relentless', 'competitive', 'ego',
        'toughness', 'instinct', 'awareness', 'intelligence', 'football_iq',
        'coachable', 'discipline', 'leadership', 'work_ethic', 'recognition',
    ],
}

# ── Curated phrase map (longest-first after sort) ─────────────────────────────
_CURATED_RAW = {
    # Trigrams
    'change of direction':   'change_of_direction',
    'stack and shed':        'stack_and_shed',
    'run after catch':       'run_after_catch',
    'yards after contact':   'yards_after_contact',
    'point of attack':       'point_of_attack',
    'block shedding':        'block_shedding',
    'anchor strength':       'anchor_strength',
    # Seed-critical bigrams
    'pass protection':       'pass_protection',
    'pass coverage':         'pass_coverage',
    'pass rusher':           'pass_rusher',
    'pass rush':             'pass_rush',
    'press coverage':        'press_coverage',
    'man coverage':          'man_coverage',
    'man to man':            'man_coverage',
    'zone coverage':         'zone_coverage',
    'zone coverages':        'zone_coverages',
    'hand fighting':         'hand_fighting',
    'route running':         'route_running',
    'pad level':             'pad_level',
    'high motor':            'high_motor',
    'football iq':           'football_iq',
    'work ethic':            'work_ethic',
    'first step':            'first_step',
    'get off':               'get_off',
    # Physical extras
    'body control':          'body_control',
    'lateral quickness':     'lateral_quickness',
    'quick twitch':          'quick_twitch',
    'arm strength':          'arm_strength',
    'closing speed':         'closing_speed',
    'change of pace':        'change_of_pace',
    'short area':            'short_area',
    'top end':               'top_end',
    'lower half':            'lower_half',
    'hip flexibility':       'hip_flexibility',
    'catch radius':          'catch_radius',
    'edge-to-edge':          'edge_to_edge',      # hyphen form for raw-text colorize matching
    # Technique extras
    'run blocking':          'run_blocking',
    'ball skills':           'ball_skills',
    'contact balance':       'contact_balance',
    'hand placement':        'hand_placement',
    'hand strength':         'hand_strength',
    'hand usage':            'hand_usage',
    'soft hands':            'soft_hands',
    'heavy hands':           'heavy_hands',
    'strong hands':          'strong_hands',
    'active hands':          'active_hands',
    'high point':            'high_point',
    'bull rush':             'bull_rush',
    'bull rusher':           'bull_rusher',
    'spin move':             'spin_move',
    'swim move':             'swim_move',
    'speed rush':            'speed_rush',
    'run support':           'run_support',
    'press technique':       'press_technique',
    'contested catch':       'contested_catch',
    'run defense':           'run_defense',
    'pursuit angle':         'pursuit_angle',
    'punch-and-pull':        'punch_and_pull',    # hyphen form; "punch" seeds individual word color
    # Character / IQ extras
    'football intelligence': 'football_intelligence',
    'play recognition':      'play_recognition',
    'decision making':       'decision_making',
    'field vision':          'field_vision',
    'film study':            'film_study',
    'film room':             'film_room',
    'locker room':           'locker_room',
    'mental toughness':      'mental_toughness',
    'blitz awareness':       'blitz_awareness',
    'pocket awareness':      'pocket_awareness',
    'high effort':           'high_effort',
    'hard worker':           'hard_worker',
}

# Sort longest-first so greedy matching hits multi-word phrases before sub-phrases
CURATED_PHRASE_MAP = dict(sorted(_CURATED_RAW.items(), key=lambda x: len(x[0]), reverse=True))

# ── Stopwords ─────────────────────────────────────────────────────────────────
KEEP_WORDS = {
    'high', 'low', 'heavy', 'light', 'deep', 'short', 'long', 'wide',
    'hard', 'soft', 'strong', 'quick', 'great',
    'up', 'down', 'off', 'out', 'over', 'through', 'above', 'below',
    'hand', 'hands', 'back', 'field',
    # NOTE: 'good', 'ball', 'man' removed — too generic; compound phrases
    # (ball_skills, man_coverage) are handled by CURATED_PHRASE_MAP before filtering.
}
CUSTOM_STOPS = {
    'prospect', 'player', 'players', 'show', 'shows', 'need', 'needs',
    'also', 'often', 'must', 'well', 'still', 'use', 'get',
    'make', 'look', 'help', 'time', 'year', 'team', 'game',
    'continue', 'develop', 'development', 'nfl', 'draft', 'college',
    'type', 'project', 'potential', 'upside',
    'starter', 'backup', 'senior', 'junior', 'cb', 'rb', 'wr', 'qb',
    # Generic evaluation language — high-frequency but carry no pillar signal:
    'play', 'run', 'average', 'lack', 'ability', 'position',
    'good', 'ball', 'man', 'line',
}
_base_stops    = set(stopwords.words('english'))
NFL_STOPWORDS  = (_base_stops - KEEP_WORDS) | CUSTOM_STOPS

lemmatizer = WordNetLemmatizer()


def nfl_preprocess(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    for phrase, token in CURATED_PHRASE_MAP.items():
        text = text.replace(phrase, token)
    text   = re.sub(r'[^a-z_\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if '_' in t or t not in NFL_STOPWORDS]
    tokens = [t if '_' in t else lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
scout = pd.read_csv(BASE + 'data/raw/nfldraftbuzz/nfldraftbuzz_scouting.csv')
scout['player_name'] = scout['first_name'] + ' ' + scout['last_name']


def get_text(row, col):
    v = row.get(col, '')
    return str(v) if isinstance(v, str) and v.strip() and v != 'nan' else ''


def all_text(row):
    if row.get('report_format') == 'structured':
        return ' '.join(filter(None, [get_text(row, 'bio'),
                                      get_text(row, 'strengths'),
                                      get_text(row, 'weaknesses')]))
    return get_text(row, 'raw_text')


scout['_all_text']   = scout.apply(all_text, axis=1)
scout['_all_tokens'] = scout['_all_text'].apply(nfl_preprocess).apply(str.split)

# ── Train W2V ─────────────────────────────────────────────────────────────────
print('Training Word2Vec...')
sentences = [t for t in scout['_all_tokens'].tolist() if t]
w2v = Word2Vec(
    sentences,
    vector_size=W2V_DIM,
    window=W2V_WINDOW,
    epochs=W2V_EPOCHS,
    sg=W2V_SG,
    min_count=W2V_MIN_COUNT,
    workers=4,
    seed=42,
)
print(f'  vocab size: {len(w2v.wv):,}')

# ── Seed coverage ─────────────────────────────────────────────────────────────
for bin_name, seeds in ANCHOR_SEEDS.items():
    in_vocab = [s for s in seeds if s in w2v.wv]
    print(f'  {bin_name}: {len(in_vocab)}/{len(seeds)} seeds in vocab')

# ── W2V expansion → KEYWORD_SETS ─────────────────────────────────────────────
print('Building keyword sets via W2V expansion...')

# Block player names — they appear near athletic words in the corpus and pollute expansion
_player_tokens = set()
for _name in scout['player_name'].dropna():
    for _part in str(_name).lower().split():
        _player_tokens.add(lemmatizer.lemmatize(_part))
print(f'  Player name blocklist: {len(_player_tokens)} tokens')


def expand_bin(seeds, model, topn=SEED_TOPN, threshold=SIM_THRESHOLD,
               blocked=_player_tokens):
    candidate_scores = {}
    for seed in seeds:
        if seed not in model.wv:
            continue
        for word, sim in model.wv.most_similar(seed, topn=topn):
            if sim >= threshold and word not in seeds and word not in blocked:
                candidate_scores.setdefault(word, []).append(sim)
    if not candidate_scores:
        return pd.DataFrame(columns=['term', 'seed_count', 'avg_sim'])
    rows = [{'term': w, 'seed_count': len(s), 'avg_sim': float(np.mean(s))}
            for w, s in candidate_scores.items()]
    return (pd.DataFrame(rows)
              .sort_values(['seed_count', 'avg_sim'], ascending=False)
              .reset_index(drop=True))


KEYWORD_SETS = {}
for bin_name in BIN_NAMES:
    seeds    = ANCHOR_SEEDS[bin_name]
    lex      = expand_bin(seeds, w2v)
    w2v_best = set(
        lex[lex['seed_count'] >= MIN_SEED_COUNT].head(MAX_W2V_TERMS)['term']
    ) - set(seeds)
    KEYWORD_SETS[bin_name] = set(seeds) | w2v_best
    print(f'  {bin_name}: {len(seeds)} seeds + {len(w2v_best)} W2V = {len(KEYWORD_SETS[bin_name])} total')

# Warn on cross-bin overlaps
for b1, b2 in [('physical', 'technique'), ('physical', 'character'), ('technique', 'character')]:
    overlap = KEYWORD_SETS[b1] & KEYWORD_SETS[b2]
    if overlap:
        print(f'  ⚠  overlap {b1} ∩ {b2}: {sorted(overlap)}')

# ── Build phrase → bin lookup ─────────────────────────────────────────────────
_PHRASE_LIST = list(CURATED_PHRASE_MAP.keys())   # already sorted longest-first

_PHRASE_BIN = {}
for phrase, token in CURATED_PHRASE_MAP.items():
    assigned = 'none'
    for bn in BIN_NAMES:
        if token in KEYWORD_SETS[bn]:
            assigned = bn
            break
    _PHRASE_BIN[phrase] = assigned

_re_word = re.compile(r"[\w']+")


def _bin_of_word(word: str) -> str:
    token = lemmatizer.lemmatize(word.lower())
    for bn in BIN_NAMES:
        if token in KEYWORD_SETS[bn]:
            return bn
    return 'none'


# ── Colorize raw text at token level ─────────────────────────────────────────
def colorize_tokens(raw_text: str) -> str:
    """
    Walk raw_text greedily:
      1. Try longest-first phrase match (from CURATED_PHRASE_MAP)
      2. Fall through to single-word match via _bin_of_word()
      3. Pass through non-alpha characters unchanged
    Emits <b class="bp/bt/bc">…</b> for matched spans.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return html_mod.escape(raw_text or '')

    lo = raw_text.lower()
    parts = []
    i = 0
    n = len(raw_text)

    while i < n:
        # 1. Try curated phrases (longest-first)
        matched = False
        for phrase in _PHRASE_LIST:
            pl = len(phrase)
            if lo[i:i + pl] == phrase:
                bn = _PHRASE_BIN[phrase]
                seg = html_mod.escape(raw_text[i:i + pl])
                if bn != 'none':
                    parts.append(f'<b class="{BIN_CLASS[bn]}">{seg}</b>')
                else:
                    parts.append(seg)
                i += pl
                matched = True
                break
        if matched:
            continue

        # 2. Non-alpha character → pass through
        if not raw_text[i].isalpha() and raw_text[i] != "'":
            parts.append(html_mod.escape(raw_text[i]))
            i += 1
            continue

        # 3. Single word
        m = _re_word.match(raw_text, i)
        if m:
            word = m.group(0)
            bn   = _bin_of_word(word)
            seg  = html_mod.escape(word)
            if bn != 'none':
                parts.append(f'<b class="{BIN_CLASS[bn]}">{seg}</b>')
            else:
                parts.append(seg)
            i += len(word)
        else:
            parts.append(html_mod.escape(raw_text[i]))
            i += 1

    return ''.join(parts)


# ── Annotate all players ───────────────────────────────────────────────────────
print('Annotating players...')
annotated = {}

rows = scout.to_dict('records')
for idx, row in enumerate(rows):
    key = row['player_name'] + '||' + str(int(row['year']))
    fmt = row.get('report_format', '')
    sections = {}
    if fmt == 'structured':
        for col in ['bio', 'strengths', 'weaknesses']:
            t = get_text(row, col)
            if t:
                sections[col] = colorize_tokens(t)
    else:
        t = get_text(row, 'raw_text')
        if t:
            sections['raw_text'] = colorize_tokens(t)
    if sections:
        annotated[key] = sections
    if (idx + 1) % 1000 == 0:
        print(f'  {idx + 1}/{len(rows)} done...')

print(f'  Annotation complete: {len(annotated)} players')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = BASE + 'colored_texts_tokens.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(annotated, f)

print(f'Saved → {out_path}  ({len(annotated)} players)')
print()
print('To use in the frontend, update build_html.py:')
print("  colored_texts.json  →  colored_texts_tokens.json")

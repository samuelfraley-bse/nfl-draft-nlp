"""
Runs the NP annotation pipeline (from nfl_pillar_scoring_NP_v2.ipynb) on all
scouting reports and saves colored HTML snippets to colored_texts.json.
"""

import json, re, html as html_mod
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

BASE = 'c:/Users/sffra/Downloads/BSE 2025-2026/nfl-draft-nlp/'

# ── Config ────────────────────────────────────────────────────────────────────
W2V_DIM       = 100
W2V_WINDOW    = 6
W2V_EPOCHS    = 30
W2V_SG        = 1
W2V_MIN_COUNT = 3
SIM_THRESHOLD = 0.40
BATCH_SIZE    = 128

BIN_NAMES  = ['physical', 'technique', 'character']
BIN_COLOR  = {'physical': '#3b82f6', 'technique': '#f97316', 'character': '#22c55e', 'none': '#888'}
BIN_LABEL  = {'physical': 'Physical', 'technique': 'Technique / Skills', 'character': 'Character / IQ'}

# ── Seed terms ────────────────────────────────────────────────────────────────
ANCHOR_SEEDS = {
    'physical': [
        'explosive', 'burst', 'speed', 'acceleration', 'first_step', 'get_off',
        'change_of_direction', 'agility', 'frame', 'size', 'strength',
        'power', 'athletic', 'physical', 'twitch', 'measurable',
    ],
    'technique': [
        'technique', 'footwork', 'leverage', 'pad_level', 'mechanics', 'productive',
        'route_running', 'pass_protection', 'hand_fighting', 'block_shedding',
        'anchor_strength', 'pass_rush', 'blocking', 'tackling', 'coverage',
    ],
    'character': [
        'effort', 'motor', 'high_motor', 'relentless', 'competitive',
        'toughness', 'instinct', 'awareness', 'intelligence', 'football_iq',
        'coachable', 'discipline', 'leadership', 'work_ethic', 'recognition',
    ],
}

CENTROID_EXTRAS = {
    'physical': [
        'height', 'length', 'wingspan', 'arm', 'velocity', 'cannon',
        'speed', 'quickness', 'burst', 'lean', 'thick', 'frame',
        'undersized', 'size', 'weight', 'hands', 'feet',
        'arms', 'lower_half', 'body_control',
        'explosion', 'vertical', 'hip_flexibility',
        'measurable', 'physicality',
    ],
    'technique': [
        'throw', 'release', 'delivery', 'pocket', 'accuracy', 'timing',
        'separation', 'route_running', 'pass_rush', 'bull_rush',
        'vision', 'tackling', 'coverage', 'blocking', 'sack', 'ball_skills',
        'catch_radius', 'catches', 'pressure', 'spin_move', 'swim_move', 'run_support',
        'production',
    ],
    'character': [
        'hustle', 'grit', 'relentless', 'compete', 'smart', 'cerebral',
        'coachable', 'leadership', 'patient', 'composure', 'instinct',
        'mature', 'immature', 'processing', 'dedicated', 'personality',
        'fearless', 'inconsistencies', 'aggressive', 'durable',
        'focus', 'confidence', 'poise', 'decisive', 'decisiveness', 'competitive',
    ],
}

# ── Curated phrase map (multi-word → single token) ────────────────────────────
CURATED_PHRASE_MAP = {
    'change of direction': 'change_of_direction',
    'first step': 'first_step',
    'get off': 'get_off',
    'pass rush': 'pass_rush',
    'bull rush': 'bull_rush',
    'route running': 'route_running',
    'pass protection': 'pass_protection',
    'hand fighting': 'hand_fighting',
    'block shedding': 'block_shedding',
    'anchor strength': 'anchor_strength',
    'pad level': 'pad_level',
    'high motor': 'high_motor',
    'football iq': 'football_iq',
    'work ethic': 'work_ethic',
    'lower half': 'lower_half',
    'body control': 'body_control',
    'ball skills': 'ball_skills',
    'catch radius': 'catch_radius',
    'swim move': 'swim_move',
    'spin move': 'spin_move',
    'run support': 'run_support',
    'hip flexibility': 'hip_flexibility',
}

NFL_STOPWORDS = set(stopwords.words('english')) - {
    'no', 'not', 'very', 'more', 'most', 'above', 'below', 'up', 'down',
    'over', 'under', 'high', 'low', 'long', 'short', 'strong', 'light',
    'hard', 'quick', 'fast', 'good', 'well', 'better', 'best',
}
NFL_STOPWORDS |= {
    'nfl', 'draft', 'player', 'prospect', 'team', 'season', 'game',
    'year', 'round', 'pick', 'scout',
}

_COORD_CONJ  = re.compile(r'\s+(?:and|or)\s+', re.IGNORECASE)
_SKIP_ENT_TYPES = {'ORG', 'GPE', 'NORP', 'FAC', 'LOC'}
lemmatizer   = WordNetLemmatizer()

def _apply_phrases(text):
    for phrase, token in CURATED_PHRASE_MAP.items():
        text = text.replace(phrase, token)
    return text

def nfl_preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ''
    text = text.lower()
    text = re.sub(r'[-\u2013\u2014]', ' ', text)
    text = _apply_phrases(text)
    text = re.sub(r'[^a-z_\s]', ' ', text)
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
        return ' '.join(filter(None, [get_text(row, 'bio'), get_text(row, 'strengths'), get_text(row, 'weaknesses')]))
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

# ── Build centroids ───────────────────────────────────────────────────────────
print('Building centroids...')
BIN_CENTROIDS = {}
for bin_name in BIN_NAMES:
    terms = list(set(ANCHOR_SEEDS[bin_name]) | set(CENTROID_EXTRAS[bin_name]))
    vecs  = [w2v.wv[t] for t in terms if t in w2v.wv]
    BIN_CENTROIDS[bin_name] = np.mean(vecs, axis=0) if vecs else None
    print(f'  {bin_name}: {sum(1 for t in terms if t in w2v.wv)}/{len(terms)} terms in vocab')

_centroid_matrix = np.stack([BIN_CENTROIDS[b] for b in BIN_NAMES])
_norms = np.linalg.norm(_centroid_matrix, axis=1, keepdims=True)
_centroid_normed = _centroid_matrix / _norms

# ── Embedding + classification ────────────────────────────────────────────────
_embed_cache = {}

def embed_span(text):
    if text in _embed_cache:
        return _embed_cache[text]
    t = text.lower()
    t = re.sub(r'[-\u2013\u2014]', ' ', t)
    t = _apply_phrases(t)
    t = re.sub(r'[^a-z_\s]', ' ', t)
    tokens = t.split()
    tokens = [tok if '_' in tok else lemmatizer.lemmatize(tok)
              for tok in tokens if '_' in tok or tok not in NFL_STOPWORDS]
    tokens = [tok for tok in tokens if len(tok) > 1]
    vecs = [w2v.wv[tok] for tok in tokens if tok in w2v.wv]
    result = np.mean(vecs, axis=0).astype(np.float32) if vecs else None
    _embed_cache[text] = result
    return result

def classify_span(text):
    vec = embed_span(text)
    if vec is None:
        return {'bin': 'none', 'sims': {}}
    norm = np.linalg.norm(vec)
    if norm == 0:
        return {'bin': 'none', 'sims': {}}
    sims_arr = _centroid_normed @ (vec / norm)
    sims     = {b: float(sims_arr[i]) for i, b in enumerate(BIN_NAMES)}
    best_idx = int(np.argmax(sims_arr))
    assigned = BIN_NAMES[best_idx] if sims_arr[best_idx] >= SIM_THRESHOLD else 'none'
    return {'bin': assigned, 'sims': sims}

def annotate_doc(doc):
    raw_text = doc.text
    if not raw_text.strip():
        return []
    spans = []
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == 'PROPN':
            continue
        ent_char_set = set()
        for ent in doc.ents:
            if ent.label_ in _SKIP_ENT_TYPES:
                for ch in range(ent.start_char, ent.end_char):
                    ent_char_set.add(ch)
        if any(ch in ent_char_set for ch in range(chunk.start_char, chunk.end_char)):
            continue
        # split on coordinating conjunctions
        sub_spans = []
        text_c = chunk.text
        pos    = 0
        for m in _COORD_CONJ.finditer(text_c):
            part = text_c[pos:m.start()]
            if part.strip():
                sub_spans.append((part, chunk.start_char + pos, chunk.start_char + m.start()))
            pos = m.end()
        tail = text_c[pos:]
        if tail.strip():
            sub_spans.append((tail, chunk.start_char + pos, chunk.start_char + len(text_c)))
        if len(sub_spans) <= 1:
            sub_spans = [(text_c, chunk.start_char, chunk.end_char)]

        for sub_text, sub_start, sub_end in sub_spans:
            result  = classify_span(sub_text)
            n_words = len(sub_text.split())
            spans.append({
                'text':    sub_text,
                'start':   sub_start,
                'end':     sub_end,
                'bin':     result['bin'],
                'sims':    result['sims'],
                'n_words': n_words,
            })
    return sorted(spans, key=lambda s: s['start'])

# Short class names: bp=physical, bt=technique, bc=character
BIN_CLASS = {'physical': 'bp', 'technique': 'bt', 'character': 'bc'}

def colorize(raw_text, annotations):
    if not annotations or not raw_text:
        return html_mod.escape(raw_text or '')
    parts    = []
    prev_end = 0
    for span in sorted(annotations, key=lambda s: s['start']):
        if span['start'] > prev_end:
            parts.append(html_mod.escape(raw_text[prev_end:span['start']]))
        seg = html_mod.escape(raw_text[span['start']:span['end']])
        if span['bin'] != 'none':
            cls = BIN_CLASS[span['bin']]
            parts.append(f'<b class="{cls}">{seg}</b>')
        else:
            parts.append(seg)
        prev_end = span['end']
    if prev_end < len(raw_text):
        parts.append(html_mod.escape(raw_text[prev_end:]))
    return ''.join(parts)

# ── Run annotation pipeline ───────────────────────────────────────────────────
print('Loading spaCy model...')
nlp = spacy.load('en_core_web_sm')

print('Annotating texts...')
results = {}   # key: "player_name||year"

rows = scout.to_dict('records')

def process_row(row):
    key = row['player_name'] + '||' + str(int(row['year']))
    fmt = row.get('report_format', '')
    sections = {}
    if fmt == 'structured':
        for col in ['bio', 'strengths', 'weaknesses']:
            t = get_text(row, col)
            if t:
                doc = nlp(t)
                ann = annotate_doc(doc)
                sections[col] = colorize(t, ann)
    else:
        t = get_text(row, 'raw_text')
        if t:
            doc = nlp(t)
            ann = annotate_doc(doc)
            sections['raw_text'] = colorize(t, ann)
    return key, sections

# Batch process using spaCy pipe for speed
# Build list of (key, field, text)
tasks = []
for row in rows:
    fmt = row.get('report_format', '')
    key = row['player_name'] + '||' + str(int(row['year']))
    if fmt == 'structured':
        for col in ['bio', 'strengths', 'weaknesses']:
            t = get_text(row, col)
            if t:
                tasks.append((key, col, t))
    else:
        t = get_text(row, 'raw_text')
        if t:
            tasks.append((key, 'raw_text', t))

print(f'  {len(tasks)} text segments to annotate...')

all_texts = [t for _, _, t in tasks]
annotated = {}

for i, doc in enumerate(nlp.pipe(all_texts, batch_size=BATCH_SIZE)):
    key, col, raw = tasks[i]
    ann = annotate_doc(doc)
    colored = colorize(raw, ann)
    if key not in annotated:
        annotated[key] = {}
    annotated[key][col] = colored
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(tasks)} done...')

print(f'  Annotation complete: {len(annotated)} players')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = BASE + 'colored_texts.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(annotated, f)

print(f'Saved to {out_path} ({len(annotated)} players)')

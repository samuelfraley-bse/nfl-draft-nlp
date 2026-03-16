import json, math

BASE = 'c:/Users/sffra/Downloads/BSE 2025-2026/nfl-draft-nlp/'

# ── Merge player_data with colored HTML from colored_texts.json ───────────────
with open(BASE + 'player_data.json', encoding='utf-8') as f:
    data = json.load(f)

with open(BASE + 'colored_texts.json', encoding='utf-8') as f:
    colored = json.load(f)

for p in data:
    key = p['name'] + '||' + str(p['year'])
    c = colored.get(key, {})
    p['bio_html']       = c.get('bio', '')
    p['strengths_html'] = c.get('strengths', '')
    p['weaknesses_html']= c.get('weaknesses', '')
    p['raw_html']       = c.get('raw_text', '')

# ── Write draft_data.js (external, so browser parses it separately) ───────────
with open(BASE + 'draft_data.js', 'w', encoding='utf-8') as f:
    f.write('window.PLAYERS=')
    json.dump(data, f)
    f.write(';')

print('draft_data.js written')

# ── Write draft_cards.html ────────────────────────────────────────────────────
lines = []
A = lines.append

A('<!DOCTYPE html>')
A('<html lang="en">')
A('<head>')
A('<meta charset="UTF-8">')
A('<title>NFL Draft Scouting Cards</title>')
A('<style>')
A('* { box-sizing: border-box; margin: 0; padding: 0; }')
A('body { background: #1a1a2e; color: #e0e0e0; font-family: Segoe UI,sans-serif; font-size: 14px; }')
A('#controls { position: sticky; top: 0; z-index: 100; background: #16213e; padding: 12px 16px;')
A('  border-bottom: 1px solid #0f3460; display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-end; }')
A('#controls label { font-size: 12px; color: #aaa; display: block; margin-bottom: 3px; }')
A('#controls input, #controls select { background: #0f3460; color: #e0e0e0; border: 1px solid #1a4a7a;')
A('  padding: 5px 8px; border-radius: 4px; font-size: 13px; }')
A('#controls input { width: 200px; }')
A('#count { color: #aaa; font-size: 12px; margin-left: auto; align-self: center; }')
A('#grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; padding: 14px; }')
A('.card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; overflow: hidden; }')
A('.card-header { background: #0f3460; padding: 10px 14px; display: flex; justify-content: space-between; align-items: flex-start; }')
A('.card-name { font-size: 17px; font-weight: 700; color: #fff; }')
A('.card-meta { font-size: 12px; color: #8899bb; margin-top: 2px; }')
A('.card-right { text-align: right; flex-shrink: 0; margin-left: 10px; }')
A('.grade { font-size: 22px; font-weight: 700; color: #f0c040; }')
A('.rank-lbl { font-size: 12px; color: #8899bb; }')
A('.pillar-bar { height: 8px; display: flex; }')
A('.bar-p { background: #3b82f6; } .bar-t { background: #f97316; } .bar-c { background: #22c55e; } .bar-r { background: #2a3555; flex: 1; }')
A('.plbls { display: flex; flex-wrap: wrap; gap: 8px; padding: 6px 14px; font-size: 11px; align-items: center; }')
A('.lbl { display: flex; align-items: center; gap: 4px; }')
A('.dot { width: 8px; height: 8px; border-radius: 50%; }')
A('.dp { background: #3b82f6; } .dt { background: #f97316; } .dc { background: #22c55e; }')
A('.wl { color: #445; font-size: 11px; margin-left: auto; }')
A('.card-body { padding: 10px 14px; }')
A('.stitle { font-size: 10px; font-weight: 700; color: #556; text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 4px; }')
A('.stitle:first-child { margin-top: 0; }')
A('.ctext { font-size: 12px; line-height: 1.5; color: #ccc; }')
A('.ctext.col { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }')
A('.tbtn { background: none; border: none; color: #3b82f6; cursor: pointer; font-size: 11px; padding: 2px 0; margin-top: 2px; }')
A('.tbtn:hover { text-decoration: underline; }')
A('.none { padding: 40px; text-align: center; color: #445; grid-column: 1/-1; font-size: 16px; }')
A('b.bp { color: #3b82f6; font-weight: bold; }')
A('b.bt { color: #f97316; font-weight: bold; }')
A('b.bc { color: #22c55e; font-weight: bold; }')
A('</style>')
A('</head>')
A('<body>')
A('<div id="controls">')
A('  <div><label>Search</label><input type="text" id="search" placeholder="Name or college..."></div>')
A('  <div><label>Position</label><select id="pos">')
A('    <option value="">All</option><option>QB</option><option>RB</option><option>FB</option>')
A('    <option>WR</option><option>TE</option><option>OT</option><option>C</option>')
A('    <option>DE</option><option>DT</option><option>LB</option><option>CB</option>')
A('  </select></div>')
A('  <div><label>Year</label><select id="yr">')
A('    <option value="">All</option><option value="2026">2026</option><option value="2025">2025</option>')
A('    <option value="2024">2024</option><option value="2023">2023</option>')
A('    <option value="2022">2022</option><option value="2021">2021</option>')
A('  </select></div>')
A('  <div><label>Sort</label><select id="srt">')
A('    <option value="rank">Rank</option><option value="grade">Grade</option>')
A('    <option value="name">Name</option><option value="pp">Physical%</option>')
A('    <option value="tp">Technique%</option><option value="cp">Character%</option>')
A('  </select></div>')
A('  <div><label>Order</label><select id="ord"><option value="asc">Asc</option><option value="desc">Desc</option></select></div>')
A('  <span id="count"></span>')
A('</div>')
A('<div id="grid"><p style="padding:40px;color:#8899bb;text-align:center">Loading data...</p></div>')
A('<script src="draft_data.js"></script>')
A('<script>')

js = r"""
var ci = 0;
function p1(v) { return Math.round(v * 1000) / 10; }
function es(s) {
  if (!s) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function tgl(id, btn) {
  var el = document.getElementById(id);
  if (!el) return;
  if (el.classList.contains('col')) {
    el.classList.remove('col'); btn.textContent = 'Show less';
  } else {
    el.classList.add('col'); btn.textContent = 'Show more';
  }
}
function mkcard(p) {
  var ph = p1(p.physical_pct), tc = p1(p.technique_pct), ch = p1(p.character_pct);
  var gr = p.grade ? p.grade.toFixed(2) : '-';
  var rk = (p.rank && p.rank < 9999) ? '#' + p.rank : '-';
  var id = 'c' + (ci++);
  // Use colored HTML if available, else fall back to plain text (escaped)
  var secs = [];
  if (p.report_format === 'structured') {
    if (p.bio_html || p.bio)       secs.push(['OVERVIEW',   p.bio_html   || es(p.bio)]);
    if (p.strengths_html || p.strengths) secs.push(['STRENGTHS',  p.strengths_html || es(p.strengths)]);
    if (p.weaknesses_html || p.weaknesses) secs.push(['WEAKNESSES', p.weaknesses_html || es(p.weaknesses)]);
  } else {
    var t = p.raw_html || es(p.raw_text) || es(p.summary);
    if (t) secs.push(['OVERVIEW', t]);
  }
  var sh = secs.map(function(s, i) {
    return '<div class="stitle">' + s[0] + '</div>' +
      '<div class="ctext col" id="' + id + 'x' + i + '">' + s[1] + '</div>' +
      '<button class="tbtn" onclick="tgl(\'' + id + 'x' + i + '\',this)">Show more</button>';
  }).join('');
  return '<div class="card">' +
    '<div class="card-header">' +
      '<div>' +
        '<div class="card-name">' + es(p.name) + '</div>' +
        '<div class="card-meta">' + es(p.position) + ' &middot; ' + p.year + ' &middot; ' + es(p.college) + '</div>' +
        '<div class="card-meta">' + es(p.height) + ' / ' + es(p.weight) + '</div>' +
      '</div>' +
      '<div class="card-right"><div class="grade">' + gr + '</div><div class="rank-lbl">' + rk + '</div></div>' +
    '</div>' +
    '<div class="pillar-bar">' +
      '<div class="bar-p" style="width:' + ph + '%"></div>' +
      '<div class="bar-t" style="width:' + tc + '%"></div>' +
      '<div class="bar-c" style="width:' + ch + '%"></div>' +
      '<div class="bar-r"></div>' +
    '</div>' +
    '<div class="plbls">' +
      '<span class="lbl"><span class="dot dp"></span>Physical ' + ph + '%</span>' +
      '<span class="lbl"><span class="dot dt"></span>Technique ' + tc + '%</span>' +
      '<span class="lbl"><span class="dot dc"></span>Character ' + ch + '%</span>' +
      (p.total_words ? '<span class="wl">(' + p.total_words + ' words)</span>' : '') +
    '</div>' +
    '<div class="card-body">' + sh + '</div>' +
  '</div>';
}
function render() {
  if (!window.PLAYERS) {
    document.getElementById('grid').innerHTML = '<div class="none">Error: draft_data.js not loaded. Both files must be in the same folder.</div>';
    return;
  }
  var q   = document.getElementById('search').value.toLowerCase();
  var pos = document.getElementById('pos').value;
  var yr  = document.getElementById('yr').value;
  var srt = document.getElementById('srt').value;
  var ord = document.getElementById('ord').value;
  var list = window.PLAYERS.filter(function(p) {
    if (pos && p.position !== pos) return false;
    if (yr  && p.year !== parseInt(yr, 10)) return false;
    if (q) { var h = (p.name + ' ' + p.college).toLowerCase(); if (h.indexOf(q) === -1) return false; }
    return true;
  });
  list.sort(function(a, b) {
    var va, vb;
    if      (srt === 'rank')  { va = a.rank  || 9999; vb = b.rank  || 9999; }
    else if (srt === 'grade') { va = a.grade || 0;    vb = b.grade || 0; }
    else if (srt === 'name')  { va = a.name;           vb = b.name; }
    else if (srt === 'pp')    { va = a.physical_pct;   vb = b.physical_pct; }
    else if (srt === 'tp')    { va = a.technique_pct;  vb = b.technique_pct; }
    else if (srt === 'cp')    { va = a.character_pct;  vb = b.character_pct; }
    if (typeof va === 'string') return ord === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    return ord === 'asc' ? va - vb : vb - va;
  });
  document.getElementById('count').textContent = list.length + ' players';
  ci = 0;
  var g = document.getElementById('grid');
  g.innerHTML = list.length ? list.map(mkcard).join('') : '<div class="none">No players found</div>';
}
document.getElementById('srt').addEventListener('change', function() {
  var s = this.value;
  document.getElementById('ord').value = (s === 'grade' || s === 'pp' || s === 'tp' || s === 'cp') ? 'desc' : 'asc';
  render();
});
['search','pos','yr','ord'].forEach(function(id) {
  document.getElementById(id).addEventListener('input',  render);
  document.getElementById(id).addEventListener('change', render);
});
document.getElementById('yr').value = '2026';
render();
"""

A(js)
A('</script>')
A('</body>')
A('</html>')

html = '\n'.join(lines)

with open(BASE + 'draft_cards.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('draft_cards.html written,', len(html) // 1024, 'KB')
print('Open draft_cards.html in a browser (both files must be in the same folder)')

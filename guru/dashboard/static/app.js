// guru usage dashboard. Read-only; every value from the API is inserted as
// text (textContent / SVG text nodes), never as HTML.
'use strict';

const SVG = 'http://www.w3.org/2000/svg';
const SLOTS = 7;                      // categorical slots; the rest is "Other"
const state = { range: '7d', source: 'all', offset: 0, limit: 50 };

// --- formatting --------------------------------------------------------------
const money = (v) => {
  const n = Number(v) || 0;
  if (n === 0) return '$0';
  return n < 0.01 ? '$' + n.toFixed(4) : '$' + n.toFixed(2);
};
const count = (v) => {
  const n = Number(v) || 0;
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e4) return Math.round(n / 1e3) + 'k';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
};
const when = (ts) => {
  const d = new Date(ts);
  return isNaN(d) ? String(ts || '') : d.toLocaleString();
};

// --- small DOM helpers -------------------------------------------------------
function el(tag, attrs, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

// --- colour follows the entity, never its rank --------------------------------
// A model keeps the slot it first got (persisted), so a filter never repaints.
function slotFor(model) {
  let map = {};
  try { map = JSON.parse(localStorage.getItem('guru-model-slots') || '{}'); }
  catch (e) { map = {}; }
  if (!(model in map)) {
    const used = new Set(Object.values(map));
    let free = 0;
    for (let i = 1; i <= SLOTS; i++) if (!used.has(i)) { free = i; break; }
    map[model] = free;                                  // 0 = Other
    try { localStorage.setItem('guru-model-slots', JSON.stringify(map)); }
    catch (e) { /* private mode: colours just are not remembered */ }
  }
  return map[model];
}
const colour = (slot) => slot ? `var(--series-${slot})` : 'var(--series-other)';

// --- tooltip -----------------------------------------------------------------
const tip = document.getElementById('tip');
function showTip(evt, lines) {
  clear(tip);
  for (const [k, v] of lines) {
    const row = el('div');
    if (k) row.append(el('span', { class: 'k' }, k + ' '));
    row.append(document.createTextNode(v));
    tip.append(row);
  }
  tip.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
}
function hideTip() { tip.hidden = true; }
function hover(node, lines) {
  node.setAttribute('tabindex', '0');
  node.addEventListener('mousemove', (e) => showTip(e, lines));
  node.addEventListener('mouseleave', hideTip);
  node.addEventListener('focus', () => {
    const r = node.getBoundingClientRect();
    showTip({ clientX: r.right, clientY: r.top }, lines);
  });
  node.addEventListener('blur', hideTip);
}

// A bar whose data end (right or top) is rounded 4px, square at the baseline.
function barPath(x, y, w, h, end) {
  const r = Math.min(4, end === 'right' ? w / 2 : h / 2, end === 'right' ? h / 2 : w / 2);
  if (w <= 0 || h <= 0) return '';
  if (end === 'right') {
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}` +
           `V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  }
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}` +
         `H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

// --- tiles -------------------------------------------------------------------
function renderTiles(t) {
  const box = document.getElementById('tiles');
  clear(box);
  const tiles = [
    ['Cost', money(t.cost), t.unpriced ? `${t.unpriced} calls not priced` : ''],
    ['Calls', count(t.calls), ''],
    ['Tokens', count((t.tokens_in || 0) + (t.tokens_out || 0)),
     `${count(t.tokens_in)} in · ${count(t.tokens_out)} out`],
    ['Topics', count(t.topics), ''],
    ['Projects', count(t.projects), ''],
    ['Models', count(t.models), ''],
  ];
  for (const [label, value, note] of tiles) {
    const tile = el('div', { class: 'tile' });
    tile.append(el('div', { class: 'label' }, label),
                el('div', { class: 'value' }, value));
    if (note) tile.append(el('div', { class: 'note' }, note));
    box.append(tile);
  }
}

// --- cost per day, stacked by model ------------------------------------------
function renderDaily(rows) {
  const box = document.getElementById('daily');
  const legend = document.getElementById('legend');
  clear(box); clear(legend);
  if (!rows.length) { box.append(el('div', { class: 'empty' }, 'No calls in this range.')); return; }
  // Every day from the first to the last: a day without calls is a gap
  // on the axis, not a missing column.
  const seen = [...new Set(rows.map((r) => r.day))].sort();
  const days = [];
  for (let d = new Date(seen[0] + 'T00:00:00Z');
       d <= new Date(seen[seen.length - 1] + 'T00:00:00Z');
       d.setUTCDate(d.getUTCDate() + 1)) {
    days.push(d.toISOString().slice(0, 10));
  }
  // Series: the models in this range; models without a slot fold into Other.
  const series = new Map();
  for (const r of rows) {
    const slot = slotFor(r.model || '?');
    const key = slot ? r.model : 'Other';
    if (!series.has(key)) series.set(key, slot);
  }
  const order = [...series.entries()].sort((a, b) => (a[1] || 99) - (b[1] || 99));
  const stack = new Map(days.map((d) => [d, []]));
  for (const r of rows) {
    const slot = slotFor(r.model || '?');
    const key = slot ? r.model : 'Other';
    const seg = stack.get(r.day).find((s) => s.key === key);
    if (seg) { seg.cost += r.cost; seg.calls += r.calls; }
    else stack.get(r.day).push({ key, slot, cost: r.cost, calls: r.calls });
  }
  const top = Math.max(...days.map((d) => stack.get(d).reduce((a, s) => a + s.cost, 0)));
  const max = niceMax(top);
  // Drawn at the container's real width: text stays at its CSS size.
  const W = Math.max(320, box.clientWidth || 900), H = 240, L = 56, B = 26, T = 8;
  const plotW = W - L - 8, plotH = H - B - T;
  const band = plotW / days.length;
  const barW = Math.max(2, Math.min(24, band * 0.6));
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
                         'aria-label': 'Cost per day, stacked by model' });
  for (let i = 0; i <= 4; i++) {
    const v = (max * i) / 4, y = T + plotH - (plotH * i) / 4;
    g.append(svg('line', { x1: L, x2: W - 8, y1: y, y2: y,
                           class: i ? 'gridline' : 'baseline' }));
    g.append(svg('text', { x: L - 8, y: y + 4, 'text-anchor': 'end' }, money(v)));
  }
  const every = Math.ceil(days.length / 12);
  days.forEach((day, i) => {
    const cx = L + band * i + band / 2;
    if (i % every === 0) {
      g.append(svg('text', { x: cx, y: H - 8, 'text-anchor': 'middle' }, day.slice(5)));
    }
    let base = T + plotH;
    const segs = stack.get(day).sort((a, b) => (a.slot || 99) - (b.slot || 99));
    segs.forEach((s, j) => {
      const h = (s.cost / max) * plotH;
      const gap = j ? 2 : 0;                           // 2px surface gap
      const hh = Math.max(0, h - gap);
      if (hh <= 0) return;
      const last = j === segs.length - 1;
      const y = base - gap - hh;
      const mark = last
        ? svg('path', { d: barPath(cx - barW / 2, y, barW, hh, 'top'), class: 'mark',
                        fill: colour(s.slot) })
        : svg('rect', { x: cx - barW / 2, y, width: barW, height: hh, class: 'mark',
                        fill: colour(s.slot) });
      hover(mark, [['', day], ['model', s.key], ['cost', money(s.cost)],
                   ['calls', count(s.calls)]]);
      g.append(mark);
      base = y;
    });
  });
  box.append(g);
  for (const [key, slot] of order) {
    const item = el('span');
    const sw = el('i');
    sw.style.background = colour(slot);
    item.append(sw, document.createTextNode(key));
    legend.append(item);
  }
  const tableRows = [];
  for (const d of days) for (const s of stack.get(d)) {
    tableRows.push({ day: d, model: s.key, cost: s.cost, calls: s.calls });
  }
  renderTable(document.getElementById('daily-table'), tableRows, [
    ['day', 'Day'], ['model', 'Model'], ['cost', 'Cost', money, true],
    ['calls', 'Calls', count, true]], 'day');
}

// --- horizontal bars per group (one series: slot 1) ---------------------------
function renderBars(by, rows) {
  const box = document.getElementById('bars-' + by);
  clear(box);
  if (!rows.length) { box.append(el('div', { class: 'empty' }, 'No calls in this range.')); return; }
  const shown = rows.slice(0, 10);
  const max = Math.max(...shown.map((r) => r.cost)) || 1;
  const W = Math.max(320, box.clientWidth || 640), row = 26, R = 70;
  const L = Math.min(240, Math.round(W * 0.38));
  const chars = Math.max(12, Math.floor((L - 14) / 6.5));
  const H = shown.length * row + 4;
  const g = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img',
                         'aria-label': `Cost by ${by}` });
  shown.forEach((r, i) => {
    const y = i * row + 4;
    const name = String(r.key || '(none)');
    const label = name.length > chars ? name.slice(0, chars - 1) + '…' : name;
    g.append(svg('text', { x: L - 10, y: y + 13, 'text-anchor': 'end', class: 'label' }, label));
    const w = Math.max(r.cost > 0 ? 2 : 0, ((W - L - R) * r.cost) / max);
    const mark = svg('path', { d: barPath(L, y + 3, w, 14, 'right'), class: 'mark',
                               fill: by === 'model' ? colour(slotFor(name)) : 'var(--series-1)' });
    const hit = svg('rect', { x: 0, y, width: W, height: row - 2, class: 'hit' });
    const lines = [['', name], ['cost', money(r.cost)], ['calls', count(r.calls)],
                   ['tokens', `${count(r.tokens_in)} in · ${count(r.tokens_out)} out`]];
    if (r.unpriced) lines.push(['not priced', `${r.unpriced} calls`]);
    hover(hit, lines);
    g.append(mark);
    g.append(svg('text', { x: L + w + 6, y: y + 13, class: 'value' }, money(r.cost)));
    g.append(hit);
  });
  g.prepend(svg('line', { x1: L, x2: L, y1: 0, y2: H, class: 'baseline' }));
  box.append(g);
  renderTable(document.getElementById('table-' + by), rows, [
    ['key', by[0].toUpperCase() + by.slice(1)], ['cost', 'Cost', money, true],
    ['calls', 'Calls', count, true], ['tokens_in', 'Tokens in', count, true],
    ['tokens_out', 'Tokens out', count, true], ['last', 'Last', when]], 'cost');
}

// --- sortable table ----------------------------------------------------------
function renderTable(box, rows, columns, sortKey) {
  clear(box);
  if (!rows.length) return;
  let key = sortKey, dir = -1;
  const table = el('table');
  const head = el('tr');
  const body = el('tbody');
  function draw() {
    const sorted = [...rows].sort((a, b) => {
      const x = a[key], y = b[key];
      if (typeof x === 'number' || typeof y === 'number') return dir * ((x || 0) - (y || 0));
      return dir * String(x || '').localeCompare(String(y || ''));
    });
    clear(body);
    for (const r of sorted) {
      const tr = el('tr');
      for (const [k, , fmt, num] of columns) {
        tr.append(el('td', { class: num ? 'num' : 'wrap' },
                     fmt ? fmt(r[k]) : String(r[k] ?? '')));
      }
      body.append(tr);
    }
    for (const th of head.children) {
      th.setAttribute('aria-sort', th.dataset.key === key
        ? (dir > 0 ? 'ascending' : 'descending') : 'none');
    }
  }
  for (const [k, title, , num] of columns) {
    const th = el('th', { class: num ? 'num' : '', scope: 'col' }, title);
    th.dataset.key = k;
    th.addEventListener('click', () => {
      dir = key === k ? -dir : (num ? -1 : 1);
      key = k;
      draw();
    });
    head.append(th);
  }
  const thead = el('thead');
  thead.append(head);
  table.append(thead, body);
  box.append(table);
  draw();
}

// --- recent calls ------------------------------------------------------------
function renderCalls(data) {
  const box = document.getElementById('calls');
  clear(box);
  const rows = data.calls || [];
  if (!rows.length && !state.offset) {
    box.append(el('div', { class: 'empty' }, 'No calls in this range.'));
  } else {
    renderTable(box, rows.map((r) => ({ ...r, tokens: (r.tokens_in || 0) + (r.tokens_out || 0) })), [
      ['ts', 'Time', when], ['source', 'Source'], ['project', 'Project'],
      ['topic', 'Topic'], ['task', 'Task'], ['model', 'Model'],
      ['tokens', 'Tokens', count, true], ['cost_usd', 'Cost',
        (v) => (v === null || v === undefined ? 'n/a' : money(v)), true]], 'ts');
  }
  document.getElementById('prev').disabled = state.offset === 0;
  document.getElementById('next').disabled = rows.length < state.limit;
  document.getElementById('page').textContent =
    rows.length ? `${state.offset + 1}–${state.offset + rows.length}` : '';
}

// --- data --------------------------------------------------------------------
async function api(path) {
  const q = new URLSearchParams({ range: state.range, source: state.source });
  const res = await fetch(`${path}?${q}${path === '/api/calls'
    ? `&limit=${state.limit}&offset=${state.offset}` : ''}`);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function loadCalls() {
  renderCalls(await api('/api/calls'));
}

let last = null;                       // the last summary, for re-layout
function draw(s) {
  renderTiles(s.totals);
  renderDaily(s.daily);
  for (const by of ['model', 'project', 'topic']) renderBars(by, s.groups[by]);
}

async function load() {
  const err = document.getElementById('error');
  try {
    const s = await api('/api/summary');
    last = s;
    draw(s);
    await loadCalls();
    err.hidden = true;
  } catch (e) {
    err.textContent = 'Could not load usage: ' + e.message;
    err.hidden = false;
  }
}

document.getElementById('range').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  for (const x of document.querySelectorAll('#range button')) x.removeAttribute('aria-checked');
  b.setAttribute('aria-checked', 'true');
  state.range = b.dataset.range;
  state.offset = 0;
  load();
});
document.getElementById('source').addEventListener('change', (e) => {
  state.source = e.target.value;
  state.offset = 0;
  load();
});
document.getElementById('prev').addEventListener('click', () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadCalls();
});
document.getElementById('next').addEventListener('click', () => {
  state.offset += state.limit;
  loadCalls();
});
let resizing = 0;
window.addEventListener('resize', () => {
  clearTimeout(resizing);
  resizing = setTimeout(() => { if (last) draw(last); }, 150);
});
load();
setInterval(load, 30000);              // live-ish: refresh every 30 s

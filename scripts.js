/* scripts.js — Shared utilities */

/* ── Colour helpers ─────────────────────────────────────────────── */
function riskColor(score) {
  if (score < 0.30) return '#00e676';
  if (score < 0.60) return '#ffab00';
  return '#ff1744';
}

function riskClass(state) {
  if (!state) return 'safe';
  return state.toLowerCase();
}

function stateEmoji(state) {
  return { SAFE: '✅', WARNING: '⚠️', DANGER: '🔴' }[state] || '—';
}

/* ── Format ─────────────────────────────────────────────────────── */
function fmt2(n) { return Number(n).toFixed(2); }
function fmt3(n) { return Number(n).toFixed(3); }
function fmtPct(n) { return (n * 100).toFixed(0) + '%'; }

function timeStr() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

/* ── Gaussian noise (Box-Muller) ───────────────────────────────── */
function gaussNoise(std = 1) {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ── Clamp ──────────────────────────────────────────────────────── */
function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }

/* ── Update DOM risk display ────────────────────────────────────── */
function applyRiskUI(scoreEl, stateEl, barEl, score, state) {
  const col = riskColor(score);
  if (scoreEl) { scoreEl.textContent = fmt3(score); scoreEl.style.color = col; }
  if (stateEl) { stateEl.textContent = state;       stateEl.style.color = col; }
  if (barEl)   barEl.style.width = (score * 100).toFixed(1) + '%';
}

/* ── Debounce ───────────────────────────────────────────────────── */
function debounce(fn, ms = 300) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* ── Vent chip state ────────────────────────────────────────────── */
function setVentChip(el, on) {
  if (!el) return;
  el.className = 'vent-chip ' + (on ? 'on' : 'off');
  el.innerHTML = `<span>${on ? '⇑' : '○'}</span> VENT ${on ? 'ON' : 'OFF'}`;
}

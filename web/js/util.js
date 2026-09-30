// Small shared helpers: DOM, time formatting, toast, confirm dialog.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "dataset") Object.assign(node.dataset, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}

export function fmtTime(s, digits = 1) {
  if (!isFinite(s)) s = 0;
  const sign = s < 0 ? "-" : "";
  s = Math.abs(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  const sec = (s % 60).toFixed(digits).padStart(digits ? 3 + digits : 2, "0");
  return sign + (h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`);
}

export function fmtDur(s) {
  if (!isFinite(s) || s == null) return "–";
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}


let toastTimer;
export function toast(message, { action, onAction, ms = 4000 } = {}) {
  const t = $("#toast");
  t.replaceChildren(el("span", { text: message }));
  if (action) t.append(el("button", { text: action, onclick: () => { t.hidden = true; onAction?.(); } }));
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

// In-page confirm (never window.confirm). Resolves true/false.
export function confirmDialog(title, body, ok = "OK") {
  return new Promise((resolve) => {
    const m = $("#modal");
    $("#modal-title").textContent = title;
    $("#modal-body").textContent = body;
    $("#modal-ok").textContent = ok;
    m.hidden = false;
    const done = (v) => {
      m.hidden = true;
      $("#modal-ok").onclick = $("#modal-cancel").onclick = m.onclick = null;
      document.removeEventListener("keydown", onKey, true);
      resolve(v);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); done(false); }
      if (e.key === "Enter") { e.stopPropagation(); e.preventDefault(); done(true); }
    };
    document.addEventListener("keydown", onKey, true);
    $("#modal-ok").onclick = () => done(true);
    $("#modal-cancel").onclick = () => done(false);
    m.onclick = (e) => { if (e.target === m) done(false); };
    $("#modal-ok").focus();
  });
}

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const pct = (x, digits = 1) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);

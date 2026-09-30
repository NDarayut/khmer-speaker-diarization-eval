// Comparison view: DiariZen's, Nemotron's and your system's speakers on one timeline, the regions
// kept for the pseudo-labels, the error track, the scores (against DiariZen on the whole file, or
// against the labels where the two references agree), and the largest disagreements to listen to.
import { api } from "./api.js";
import { STAGES } from "./library.js";
import { swatchStyle } from "./palette.js";
import { Timeline, ERROR_COLORS, ERROR_NAMES } from "./timeline.js";
import { $, $$, el, fmtTime, pct, toast } from "./util.js";

const PAD = 0.4;  // seconds of context around a played disagreement
const KEPT = "#6f6468";  // the "kept for labels" lane

let V = null;         // current view state
let bound = false;
let pollTimer = null;

export async function showCompare(id) {
  if (!bound) bind();
  $("#compare").hidden = false;
  if (V?.id !== id) await load(id);
  else render();
}

export function hideCompare() {
  $("#compare").hidden = true;
  clearTimeout(pollTimer);
  V?.audio.pause();
}

async function load(id) {
  clearTimeout(pollTimer);
  V?.audio.pause();
  const data = await api.get(id);
  const audio = V?.audio || new Audio();
  audio.src = api.audioUrl(id);
  V = {
    id, data, audio, variant: V?.variant || "system", collar: V?.collar ?? "0", skipOverlap: V?.skipOverlap || false,
    target: V?.target || "labels",
    errFilter: "all", time: 0, peaks: null, playEnd: null, follow: true,
    duration: data.meta.duration || 0, groups: [], errors: [],
    onHover: (t) => ($("#hover-time").textContent = t == null ? "" : fmtTime(t, 2)),
    onViewChange: () => ($("#zoom-label").textContent = `${Math.round(V.tl.viewDur)} s`),
    seek: (t) => seek(t),
    playSpan: (s, e) => playSpan(s, e),
  };
  V.tl = new Timeline(V, $("#timeline"), $("#overview"));
  V.tl.setZoomSeconds(Math.min(120, V.duration || 120), 0);
  api.peaks(id).then((p) => { V.peaks = p; V.tl.invalidate(); }).catch(() => {});
  crumbs(data.meta);
  render();
  watchJob();
}

// ------------------------------------------------------------------ rendering
function scoresKey() {
  return `collar=${V.collar}` + (V.skipOverlap ? ",no-overlap" : "");
}

function crumbs(meta) {
  const name = el("input", { class: "rec-name", value: meta.name, spellcheck: "false", title: "Rename" });
  name.addEventListener("change", async () => {
    try { await api.rename(V.id, name.value); } catch (err) { toast(err.message); }
  });
  name.addEventListener("keydown", (e) => { if (e.key === "Enter") name.blur(); e.stopPropagation(); });
  $("#crumbs").replaceChildren(el("a", { href: "#/", text: "All recordings" }), el("span", { class: "sep", text: "/" }), name);
  document.title = `${meta.name} · Diarization Bench`;
}

// the scores shown: against the labels (kept regions) when there are any, else against DiariZen
function target() {
  const m = V.data.metrics;
  return V.target === "labels" && m?.vs_labels ? "labels" : "diarizen";
}

function evaluation() {
  const m = V.data.metrics;
  return target() === "labels" ? m?.vs_labels?.[V.variant] : m?.[V.variant];
}

function render() {
  const { data } = V;
  const m = data.metrics, ref = data.turns.reference || [], hyp = data.turns[V.variant] || [];
  const sys = evaluation();
  const refSpk = speakersByTalk(ref);
  const color = new Map(refSpk.map((s, i) => [s, i]));
  const mapping = sys?.mapping || {};
  const hypSpk = speakersByTalk(hyp);
  let extra = refSpk.length;
  const hypColor = new Map(hypSpk.map((s) => [s, mapping[s] !== undefined ? color.get(mapping[s]) : extra++]));

  V.groups = [
    { title: `Reference · DiariZen · ${refSpk.length} speakers`,
      lanes: refSpk.map((s) => ({ label: s, color: color.get(s), turns: ref.filter((t) => t[2] === s) })) },
  ];
  const second = data.turns.second;
  if (second) {
    const map2 = m?.second?.mapping || {};
    const spk2 = speakersByTalk(second);
    let extra2 = refSpk.length;
    V.groups.push({ title: `Second opinion · Nemotron · ${spk2.length} speakers`,
      lanes: spk2.map((s) => ({
        label: map2[s] ? `${s} → ${map2[s]}` : `${s} (unmatched)`,
        color: map2[s] !== undefined ? color.get(map2[s]) : extra2++, turns: second.filter((t) => t[2] === s) })) });
  }
  if (data.labels) {
    V.groups.push({ title: `Labels · ${data.labels.status}${data.labels.reason ? ` (${data.labels.reason})` : ""}`,
      lanes: [{ label: "Kept", fill: KEPT, turns: data.labels.uem }] });
  }
  V.groups.push({ title: `Yours · ${V.variant === "system" ? "own count" : "reference count"} · ${hypSpk.length} speakers`,
    lanes: hypSpk.map((s) => ({ label: mapping[s] ? `${s} → ${mapping[s]}` : `${s} (unmatched)`,
      color: hypColor.get(s), turns: hyp.filter((t) => t[2] === s) })) });
  V.errors = sys?.errors || [];
  V.errorsLabel = target() === "labels" ? "vs. labels" : "vs. DiariZen";
  V.shade = target() === "labels" && data.labels ? complement(data.labels.uem, V.duration) : [];
  V.tl.invalidate();

  $$("#variant button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.v === V.variant)));
  $$("#target button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.t === target()));
    b.disabled = b.dataset.t === "labels" && !m?.vs_labels;
  });
  $$("#collar button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.c === V.collar)));
  $("#skip-overlap").checked = V.skipOverlap;
  $("#clock-total").textContent = fmtTime(V.duration, 0);
  $("#legend").replaceChildren(...Object.entries(ERROR_NAMES).map(([k, n]) =>
    el("span", { class: "lg" }, el("span", { class: "sw", style: `background:${ERROR_COLORS[k]}` }), n)));

  renderScores(m, sys);
  renderSpeakers(sys, color, mapping);
  renderErrors();
  renderLabels();
  renderSystems();
}

function complement(uem, duration) {
  const out = [];
  let t = 0;
  for (const [a, b] of uem) { if (a > t) out.push([t, a]); t = b; }
  if (t < duration) out.push([t, duration]);
  return out;
}

function speakersByTalk(turns) {
  const talk = new Map();
  for (const [s, e, spk] of turns) talk.set(spk, (talk.get(spk) || 0) + e - s);
  return [...talk.entries()].sort((a, b) => b[1] - a[1]).map(([s]) => s);
}

function kv(pairs) {
  return pairs.flatMap(([k, v, title]) => [el("dt", { text: k, title }), el("dd", { text: v })]);
}

function renderScores(m, sys) {
  const sc = sys?.scores?.[scoresKey()];
  $("#der-big").textContent = sc ? pct(sc.der) : "–";
  const share = (x) => (sc?.speech ? `${pct(x / sc.speech)}  (${x.toFixed(1)} s)` : "–");
  $("#score-kv").replaceChildren(...kv(sc ? [
    ["Missed", share(sc.missed)],
    ["False alarm", share(sc.false_alarm)],
    ["Confusion", share(sc.confusion)],
    ["JER", pct(sc.jer)],
    ["Scored speech", `${(sc.speech / 60).toFixed(1)} min`],
  ] : [["", "not scored yet"]]));
  const t = target() === "labels" ? m?.vs_labels : m;
  $("#count-kv").replaceChildren(...kv(m ? [
    ["DiariZen", String(m.reference.speakers)],
    ...(m.second ? [["Nemotron", String(m.second.speakers)]] : []),
    ["Yours, own count", String(m.system?.speakers ?? "–")],
    ["DER, own count", pct(t?.system?.scores?.[scoresKey()]?.der)],
    ["DER, reference count", pct(t?.system_oracle?.scores?.[scoresKey()]?.der)],
  ] : []));
}

function renderSpeakers(sys, color, mapping) {
  const t = $("#spk-table");
  if (!sys) { t.replaceChildren(); return; }
  const inv = Object.fromEntries(Object.entries(mapping).map(([h, r]) => [r, h]));
  t.replaceChildren(
    el("thead", {}, el("tr", {}, ...["Reference", "Yours", "Speech", "Found", "Confused", "Missed"].map((h) => el("th", { text: h })))),
    el("tbody", {}, ...sys.per_speaker.map((p) => el("tr", {},
      el("td", {}, el("span", { class: "sw", style: swatchStyle(color.get(p.speaker)) }), p.speaker),
      el("td", { text: inv[p.speaker] || "–" }),
      el("td", { text: fmtTime(p.speech, 0) }),
      el("td", { text: pct(p.speech ? p.found / p.speech : null, 0) }),
      el("td", { text: pct(p.speech ? p.confused / p.speech : null, 0) }),
      el("td", { text: pct(p.speech ? p.missed / p.speech : null, 0) })))));
}

function filteredErrors() {
  return V.errors.filter((e) => V.errFilter === "all" || e[2] === V.errFilter);
}

function renderErrors() {
  $$("#err-filters button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.f === V.errFilter)));
  const list = filteredErrors().slice().sort((a, b) => (b[1] - b[0]) - (a[1] - a[0])).slice(0, 150);
  $("#err-list").replaceChildren(...(list.length ? list.map(([s, e, kind]) =>
    el("button", { class: "err-row", onclick: () => playSpan(s, e) },
      el("span", { class: "sw", style: `background:${ERROR_COLORS[kind]}` }),
      el("span", { class: "err-kind", text: ERROR_NAMES[kind] }),
      el("span", { class: "err-time", text: `${fmtTime(s, 1)} – ${fmtTime(e, 1)}` }),
      el("span", { class: "err-dur", text: `${(e - s).toFixed(1)} s` }),
      el("span", { class: "err-who", text: whoAt((s + e) / 2) })))
    : [el("p", { class: "meta muted2 note", text: V.data.metrics ? "No disagreement of this kind." : "Not scored yet." })]));
}

// "R1 / S2" style: who each system says is talking at time t
function whoAt(t) {
  const at = (turns) => [...new Set((turns || []).filter(([s, e]) => s <= t && t < e).map((x) => x[2]))].join("+") || "—";
  const second = V.data.turns.second ? ` · nemotron ${at(V.data.turns.second)}` : "";
  return `diarizen ${at(V.data.turns.reference)}${second} · yours ${at(V.data.turns[V.variant])}`;
}

function renderLabels() {
  const lab = V.data.metrics?.labels;
  const box = $("#labels-kv");
  const sel = $("#split-select");
  sel.value = V.data.meta.split || "";
  sel.options[0].textContent = `Auto (${V.data.split})`;
  if (!lab) { box.replaceChildren(...kv([["", "no labels yet"]])); return; }
  const d = lab.dropped || {};
  const sec = (x) => (x == null ? "–" : x >= 90 ? `${(x / 60).toFixed(1)} min` : `${x.toFixed(1)} s`);
  box.replaceChildren(...kv([
    ["Status", lab.status + (lab.reason ? ` — ${lab.reason}` : "")],
    ["Reference speech kept", pct(lab.speech_kept_share, 0)],
    ["Kept, with silence", sec(lab.kept)],
    ["Frame agreement", pct(lab.agreement, 0), "DiariZen and Nemotron, after speaker mapping"],
    ...(lab.coverage ? [
      ["Overlap kept", lab.coverage.overlap_seconds ? `${pct(lab.coverage.overlap, 0)} of ${sec(lab.coverage.overlap_seconds)}` : "no overlap",
        "Overlapped speech rarely survives: both models must agree on every speaker"],
      ["Turn changes kept", pct(lab.coverage.near_turn_change, 0), "speech within 0.5 s of a turn change"]] : []),
    ["Dropped: disagreement", sec((d.disagreement || 0) + (d["near disagreement"] || 0))],
    ["Dropped: music", sec(d.music)],
    ["Dropped: short pieces", sec(d.short)],
    ...(V.data.metrics.second ? [["DiariZen vs Nemotron DER", pct(V.data.metrics.second.scores?.["collar=0"]?.der)]] : []),
  ]));
}

function renderSystems() {
  const { meta, metrics: m, job } = V.data;
  const t = meta.timings || {};
  $("#sys-kv").replaceChildren(...kv([
    ["Reference", m?.reference?.model?.split("/").pop() || "DiariZen"],
    ["Yours", "Khmer segmentation + Community-1"],
    ["Diarize time", t.diarize != null ? `${t.diarize.toFixed(0)} s` : "–", "both systems, in parallel"],
  ]));
  const running = job && !["done", "error"].includes(job.stage);
  $$("[data-rerun]").forEach((b) => (b.disabled = running));
  $("#job-info").textContent = running
    ? `${STAGES[job.stage] || job.stage}… ${Math.round((job.progress || 0) * 100)}%`
    : job?.stage === "error" || meta.status === "error" ? `Failed: ${job?.error || meta.error || ""}`
    : (meta.warnings || []).join(" ");
  const names = { reference: "DiariZen", second: "Nemotron", system: "Yours", system_oracle: "Yours (ref. count)" };
  $("#rttm-links").replaceChildren(
    ...Object.keys(names).filter((n) => V.data.turns[n])
      .map((n) => el("a", { class: "pill small ghost", href: api.rttmUrl(V.id, n), download: "", text: names[n] })),
    ...(V.data.labels ? [el("a", { class: "pill small ghost", href: api.rttmUrl(V.id, "labels.uem"), download: "", text: "Labels UEM" })] : []));
}

// ------------------------------------------------------------------ jobs
async function watchJob() {
  clearTimeout(pollTimer);
  const job = V.data.job;
  if (!job || ["done", "error"].includes(job.stage)) return;
  const id = V.id;
  pollTimer = setTimeout(async () => {
    if (V?.id !== id || $("#compare").hidden) return;
    try {
      const j = await api.job(job.id);
      V.data.job = j;
      if (j.stage === "done") { const t = V.time; await reload(); seek(t); return; }
      if (j.stage === "error") { V.data = await api.get(id); render(); return; }
      renderSystems();
    } catch (err) { toast(err.message); }
    watchJob();
  }, 1500);
}

async function reload() {
  V.data = await api.get(V.id);
  render();
  watchJob();
}

async function rerun(system) {
  try {
    const r = await api.rerun(V.id, system);
    V.data.job = r.job;
    renderSystems();
    watchJob();
  } catch (err) { toast(err.message); }
}

// ------------------------------------------------------------------ playback
function seek(t) {
  V.time = Math.max(0, Math.min(V.duration, t));
  V.audio.currentTime = V.time;
  V.playEnd = null;
  tick();
}

function playSpan(s, e) {
  const a = Math.max(0, s - PAD);
  V.tl.reveal(a, e + PAD);
  V.time = a;
  V.audio.currentTime = a;
  V.playEnd = Math.min(V.duration, e + PAD);
  V.audio.play();
}

function togglePlay() {
  if (V.audio.paused) { V.playEnd = null; V.audio.play(); } else V.audio.pause();
}

function tick() {
  if (!V) return;
  V.time = V.audio.currentTime;
  if (V.playEnd != null && V.time >= V.playEnd) { V.audio.pause(); V.playEnd = null; }
  $("#clock-now").textContent = fmtTime(V.time, 1);
  $("#play-btn").classList.toggle("playing", !V.audio.paused);
  if (V.follow && !V.audio.paused) V.tl.reveal(V.time);
  V.tl.requestDraw();
}

function loop() {
  if (V && !V.audio.paused) tick();
  requestAnimationFrame(loop);
}

function nextError(dir) {
  const errs = filteredErrors();
  const t = V.time + (dir > 0 ? PAD + 0.05 : -PAD - 0.05);
  const e = dir > 0 ? errs.find(([s]) => s > t) : [...errs].reverse().find(([s]) => s < t - 0.3);
  if (e) playSpan(e[0], e[1]);
}

// ------------------------------------------------------------------ events
function bind() {
  bound = true;
  requestAnimationFrame(loop);
  $("#play-btn").addEventListener("click", () => togglePlay());
  $("#zoom-in").addEventListener("click", () => V.tl.zoomAt(1.5));
  $("#zoom-out").addEventListener("click", () => V.tl.zoomAt(1 / 1.5));
  $("#follow-btn").addEventListener("click", (e) => {
    V.follow = !V.follow;
    e.currentTarget.setAttribute("aria-pressed", String(V.follow));
  });
  $$("#variant button").forEach((b) => b.addEventListener("click", () => { V.variant = b.dataset.v; render(); }));
  $$("#target button").forEach((b) => b.addEventListener("click", () => { V.target = b.dataset.t; render(); }));
  $("#split-select").addEventListener("change", async (e) => {
    try {
      V.data.meta = await api.setSplit(V.id, e.target.value || null);
      if (e.target.value) V.data.split = e.target.value;
      renderLabels();
    } catch (err) { toast(err.message); }
  });
  $$("#collar button").forEach((b) => b.addEventListener("click", () => { V.collar = b.dataset.c; render(); }));
  $("#skip-overlap").addEventListener("change", (e) => { V.skipOverlap = e.target.checked; render(); });
  $$("#err-filters button").forEach((b) => b.addEventListener("click", () => {
    V.errFilter = b.dataset.f; renderErrors();
  }));
  $$("[data-rerun]").forEach((b) => b.addEventListener("click", () => rerun(b.dataset.rerun)));
  document.addEventListener("keydown", (e) => {
    if ($("#compare").hidden || !V || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key;
    if (k === " ") { e.preventDefault(); togglePlay(); }
    else if (k === "ArrowRight" || k === "ArrowLeft") {
      e.preventDefault();
      seek(V.audio.currentTime + (k === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 5 : 1));
      V.tl.reveal(V.time);
    }
    else if (k === "n" || k === "N") nextError(e.shiftKey ? -1 : 1);
    else if (k === "-") V.tl.zoomAt(1 / 1.5);
    else if (k === "=" || k === "+") V.tl.zoomAt(1.5);
    else if (k === "f" || k === "F") $("#follow-btn").click();
  });
}

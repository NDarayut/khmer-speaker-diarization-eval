// Spot check: random labelled windows, one at a time. Listen, then mark correct / wrong / can't
// tell; the accuracy estimate (with a 95% interval) goes into the dataset card.
import { api } from "./api.js";
import { Timeline } from "./timeline.js";
import { $, $$, el, fmtTime, pct, toast } from "./util.js";

const CONTEXT = 2.0;  // seconds shown around the window

let R = null;         // { doc, index, recs: Map(rec_id -> {data, peaks}), audio, tl, view }
let bound = false;

export async function showReview() {
  if (!bound) bind();
  $("#review").hidden = false;
  $("#crumbs").replaceChildren();
  document.title = "Spot check · Diarization Bench";
  R = R || { index: 0, recs: new Map(), audio: new Audio() };
  await refresh(true);
}

export function hideReview() {
  $("#review").hidden = true;
  R?.audio.pause();
}

async function refresh(jumpToPending = false) {
  R.doc = await api.review();
  renderStats();
  const items = R.doc.items;
  $("#review-item").hidden = !items.length;
  $("#review-empty").hidden = !!items.length;
  $("#review-empty").textContent = "No windows yet. Draw some above (only recordings with accepted labels are sampled).";
  if (!items.length) return;
  if (jumpToPending) {
    const i = items.findIndex((x) => !x.verdict);
    R.index = i >= 0 ? i : items.length - 1;
  }
  await showItem(false);
}

function renderStats() {
  const s = R.doc.stats;
  const stat = (label, value, sub) => el("div", { class: "stat" }, el("span", { class: "eyebrow muted2", text: label }),
    el("span", { class: "stat-value", text: value }), sub ? el("span", { class: "meta muted2", text: sub }) : null);
  $("#review-stats").replaceChildren(
    stat("Labels correct", s.judged ? pct(s.accuracy, 0) : "–",
      s.judged ? `95% interval ${pct(s.low, 0)}–${pct(s.high, 0)}` : "nothing judged yet"),
    stat("Judged", `${s.judged}`, `${s.correct} correct · ${s.wrong} wrong`),
    stat("Still to listen to", `${s.pending}`, `of ${s.items} windows`),
  );
}

async function recording(id) {
  if (!R.recs.has(id)) {
    const [data, peaks] = await Promise.all([api.get(id), api.peaks(id).catch(() => null)]);
    R.recs.set(id, { data, peaks });
  }
  return R.recs.get(id);
}

function speakersByTalk(turns) {
  const talk = new Map();
  for (const [s, e, spk] of turns) talk.set(spk, (talk.get(spk) || 0) + e - s);
  return [...talk.entries()].sort((a, b) => b[1] - a[1]).map(([s]) => s);
}

async function showItem(autoplay = true) {
  const items = R.doc.items;
  R.index = Math.max(0, Math.min(items.length - 1, R.index));
  const item = items[R.index];
  let rec;
  try { rec = await recording(item.rec_id); } catch (err) { toast(err.message); return; }
  const ref = rec.data.turns.reference || [];
  const spk = speakersByTalk(ref);
  const a = Math.max(0, item.start - CONTEXT), b = Math.min(rec.data.meta.duration, item.end + CONTEXT);

  R.view = {
    duration: rec.data.meta.duration, peaks: rec.peaks, time: item.start,
    groups: [{ title: "Labels (DiariZen)", lanes: spk.map((s, i) => ({ label: s, color: i, turns: ref.filter((t) => t[2] === s) })) }],
    errors: null, shade: [[0, item.start], [item.end, rec.data.meta.duration]],
    seek: (t) => { R.audio.currentTime = t; R.view.time = t; R.tl.requestDraw(); },
    playSpan: (s, e) => play(s, e),
  };
  if (!R.tl) R.tl = new Timeline(R.view, $("#review-timeline"), $("#review-overview"));
  else { R.tl.v = R.view; }
  R.tl.resize();
  R.tl.setZoomSeconds(b - a, (a + b) / 2);

  if (R.audio.dataset.rec !== item.rec_id) {
    R.audio.src = api.audioUrl(item.rec_id);
    R.audio.dataset.rec = item.rec_id;
  }
  $("#review-title").textContent = `Window ${R.index + 1} of ${items.length}` +
    (item.verdict ? ` · judged ${item.verdict}` : "");
  $("#review-where").textContent = `${rec.data.meta.name} · ${fmtTime(item.start, 1)} – ${fmtTime(item.end, 1)}`;
  $("#review-note").value = item.note || "";
  $$("[data-verdict]").forEach((btn) => btn.classList.toggle("chosen", btn.dataset.verdict === item.verdict));
  if (autoplay) play(item.start, item.end);
}

function play(s, e) {
  R.audio.currentTime = s;
  R.playEnd = e;
  R.audio.play().catch(() => {});
}

function tick() {
  if (R && !R.audio.paused) {
    if (R.playEnd != null && R.audio.currentTime >= R.playEnd) { R.audio.pause(); R.playEnd = null; }
    R.view.time = R.audio.currentTime;
    R.tl?.requestDraw();
  }
  requestAnimationFrame(tick);
}

async function judge(verdict) {
  const item = R.doc.items[R.index];
  try {
    await api.judge(item.id, verdict, $("#review-note").value);
  } catch (err) { toast(err.message); return; }
  R.audio.pause();
  const pending = R.doc.items.findIndex((x, i) => i > R.index && !x.verdict);
  await refresh(false);
  R.index = pending >= 0 ? pending : Math.min(R.index + 1, R.doc.items.length - 1);
  await showItem(true);
}

function bind() {
  bound = true;
  requestAnimationFrame(tick);
  $("#sample-btn").addEventListener("click", async () => {
    try {
      R.doc = await api.sampleReview(Number($("#sample-n").value) || 20);
      await refresh(true);
    } catch (err) { toast(err.message); }
  });
  $("#review-play").addEventListener("click", () => {
    const item = R.doc.items[R.index];
    if (R.audio.paused) play(item.start, item.end); else R.audio.pause();
  });
  $$("[data-verdict]").forEach((b) => b.addEventListener("click", () => judge(b.dataset.verdict)));
  $("#review-prev").addEventListener("click", () => { R.index--; showItem(true); });
  $("#review-next").addEventListener("click", () => { R.index++; showItem(true); });
  document.addEventListener("keydown", (e) => {
    if ($("#review").hidden || !R?.doc?.items.length || e.ctrlKey || e.metaKey || e.altKey) return;
    if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    const k = e.key.toLowerCase();
    if (k === " ") { e.preventDefault(); $("#review-play").click(); }
    else if (k === "y") judge("correct");
    else if (k === "n") judge("wrong");
    else if (k === "s") judge("skip");
    else if (k === "arrowleft") $("#review-prev").click();
    else if (k === "arrowright") $("#review-next").click();
  });
}

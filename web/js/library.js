// Library view: import form, pooled scores, recordings table.
import { api } from "./api.js";
import { $, $$, el, fmtDur, pct, toast, confirmDialog } from "./util.js";

export const STAGES = { queued: "Queued", download: "Downloading", transcode: "Converting audio",
  waveform: "Drawing waveform", diarize: "Diarizing", oracle: "Re-clustering", label: "Labelling",
  score: "Scoring", done: "Ready", error: "Failed" };

// a playlist or channel link (the server decides for sure)
const isCollection = (u) => /[?&]list=|youtube\.com\/(@|channel\/|c\/|user\/)/.test(u);

let pollTimer = null;
let status = null;
let bound = false;

export async function showLibrary() {
  $("#library").hidden = false;
  $("#crumbs").replaceChildren();
  if (!bound) bind();
  status = await api.status().catch(() => null);
  renderNote();
  loadInbox();
  await refresh();
}

export function hideLibrary() {
  $("#library").hidden = true;
  clearTimeout(pollTimer);
}

function bind() {
  bound = true;
  $$("#import-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$("#import-tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    $$("#import-form [data-for]").forEach((f) => (f.hidden = f.dataset.for !== b.dataset.tab));
    $(`#import-form [data-for=${b.dataset.tab}] input`)?.focus();
  }));
  $("#spk-mode").addEventListener("change", (e) => {
    const m = e.target.value;
    $("#spk-n1").hidden = m === "auto";
    $("#spk-n2").hidden = $("#spk-and").hidden = m !== "range";
    if (m !== "auto") $("#spk-n1").focus();
  });
  $("#import-form").addEventListener("submit", onImport);
  $("#import-url").addEventListener("input", (e) => {
    const many = isCollection(e.target.value);
    $("#batch-opts").hidden = !many;
    $("#import-btn").textContent = many ? "Queue videos" : "Run";
  });
}

function renderNote() {
  const note = $("#import-note");
  if (!status) { note.textContent = ""; return; }
  const { system: s, reference: r, second: n } = status;
  const bits = [`Yours: ${s.segmentation} + ${s.rest} on ${s.device}.`,
    `Labels: ${r.model} on ${r.device}, checked by ${n.model} on ${n.device}.`];
  if (!r.available) bits.push("DiariZen is not set up yet: run scripts/setup_diarizen.sh.");
  if (!n.available) bits.push("Nemotron is not set up yet (no labels until it is): run scripts/setup_nemotron.sh.");
  note.textContent = bits.join(" ");
}

// Files in the inbox folder are offered as paths.
async function loadInbox() {
  const r = await api.inbox().catch(() => null);
  const files = r?.files || [];
  $("#inbox-files").replaceChildren(...files.map((f) => el("option", { value: f.path, text: f.name })));
  if (files.length) $("#import-path").placeholder = `${files[0].path}  (${files.length} in the inbox; click to choose)`;
}

async function onImport(e) {
  e.preventDefault();
  const tab = $("#import-tabs [aria-selected=true]").dataset.tab;
  const fd = new FormData();
  if (tab === "path") {
    const p = $("#import-path").value.trim();
    if (!p) { toast("Enter the path of an audio file on this computer."); $("#import-path").focus(); return; }
    fd.append("path", p);
  } else if (tab === "youtube") {
    const u = $("#import-url").value.trim();
    if (!u) { toast("Paste the link of a YouTube video, playlist or channel."); $("#import-url").focus(); return; }
    fd.append("url", u);
    if (isCollection(u)) {
      fd.append("limit", $("#batch-limit").value);
      fd.append("min_minutes", $("#batch-min").value);
      fd.append("max_minutes", $("#batch-max").value);
    }
  } else {
    const f = $("#import-file").files[0];
    if (!f) { toast("Choose a file to upload."); return; }
    fd.append("file", f);
  }
  const mode = $("#spk-mode").value;
  const n1 = $("#spk-n1").value, n2 = $("#spk-n2").value;
  if (mode === "exact") fd.append("num_speakers", n1);
  if (mode === "range") { fd.append("min_speakers", n1); fd.append("max_speakers", n2); }
  const btn = $("#import-btn");
  btn.disabled = true;
  btn.textContent = tab === "upload" ? "Uploading…" : "Starting…";
  try {
    const r = await api.create(fd);
    if (r.collection) {
      const why = {};
      for (const s of r.skipped) why[s.reason] = (why[s.reason] || 0) + 1;
      const skipped = Object.entries(why).map(([k, n]) => `${n} ${k}`).join(", ");
      toast(`${r.created.length} of ${r.found} videos queued${skipped ? `; skipped: ${skipped}` : ""}.`, { ms: 8000 });
    }
    $("#batch-opts").hidden = true;
    $("#import-path").value = "";
    $("#import-file").value = "";
    $("#import-url").value = "";
    await refresh();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run";
  }
}

async function refresh() {
  clearTimeout(pollTimer);
  let recs = [];
  try { recs = await api.recordings(); } catch (err) { toast(err.message); }
  const body = $("#rec-table tbody");
  body.replaceChildren(...recs.map(row));
  $("#rec-empty").hidden = recs.length > 0;
  $("#rec-table").hidden = recs.length === 0;
  renderSummary(await api.summary().catch(() => null));
  const active = recs.some((r) => r.job && !["done", "error"].includes(r.job.stage));
  if (!$("#library").hidden) pollTimer = setTimeout(refresh, active ? 1500 : 8000);
}

function stat(label, value, sub, small = false) {
  return el("div", { class: "stat" }, el("span", { class: "eyebrow muted2", text: label }),
    el("span", { class: small ? "stat-value sm" : "stat-value", text: value }), sub ? el("span", { class: "meta muted2", text: sub }) : null);
}

function renderSummary(s) {
  const box = $("#summary");
  box.hidden = !s || !s.recordings;
  if (box.hidden) return;
  const L = s.labels, lab = L.system["collar=0"], labOrc = L.system_oracle["collar=0"];
  const own = s.system["collar=0"];
  $("#summary-sub").textContent =
    `${s.recordings} recording${s.recordings === 1 ? "" : "s"}; ${L.accepted} with accepted labels ` +
    `(${L.speech_hours.toFixed(2)} h of labelled speech); no collar`;
  const split = (x) => (x?.speech ? [x.missed, x.false_alarm, x.confusion].map((v) => pct(v / x.speech)).join(" · ") : "–");
  $("#summary-stats").replaceChildren(
    stat("DER vs labels", pct(lab?.der), `reference count given: ${pct(labOrc?.der)}`),
    stat("Missed · FA · Confusion", split(lab), "vs labels", true),
    stat("DER vs DiariZen", pct(own.der), "whole files, all recordings"),
    stat("Labelled speech", `${L.speech_hours.toFixed(1)} h`, `${L.labelled_hours.toFixed(1)} h incl. silence`),
    stat("Speaker count right", `${s.speaker_count_exact} / ${s.recordings}`),
  );
}

function row(r) {
  const job = r.job, sc = r.scores, sys = sc?.system;
  let statusCell;
  const running = job && !["done", "error"].includes(job.stage);
  if (running) {
    const pctTxt = ["diarize", "download", "oracle"].includes(job.stage) ? ` ${Math.round(job.progress * 100)}%` : "";
    statusCell = el("span", { class: "status" }, el("span", { class: "status-dot soon" }),
      el("span", { class: "meta", text: (STAGES[job.stage] || job.stage) + pctTxt }));
  } else if (r.status === "error" || job?.stage === "error") {
    statusCell = el("span", { class: "status", title: r.error || job?.error || "" },
      el("span", { class: "status-dot" }), el("span", { class: "eyebrow muted2", text: "Failed" }));
  } else if (r.status === "ready") {
    statusCell = el("span", { class: "status" }, el("span", { class: "status-dot live" }),
      el("span", { class: "eyebrow", style: "color: var(--primary)", text: "Scored" }));
  } else {
    statusCell = el("span", { class: "status" }, el("span", { class: "status-dot soon" }),
      el("span", { class: "meta", text: "Interrupted" }));
  }
  const openable = !!sc;
  const open = () => { if (openable) location.hash = `#/r/${r.id}`; };
  const lab = sc?.labels;
  const labCell = !lab ? el("td", { text: "–" })
    : lab.status === "accepted" ? el("td", { class: "meta", text: pct(lab.speech_kept_share, 0), title: `frame agreement ${pct(lab.agreement, 0)}` })
    : el("td", { class: "meta muted2", text: "rejected", title: lab.reason || "" });
  return el("tr", { onclick: open, class: openable ? "" : "inert" },
    el("td", {}, el("div", { class: "rec-title", text: r.name }),
      el("div", { class: "rec-sub", text: r.source?.startsWith("upload:") ? r.source.slice(7) : r.source })),
    el("td", { text: fmtDur(r.duration) }),
    el("td", { text: sc ? `${sc.ref_speakers} / ${sys?.speakers ?? "–"}` : "–" }),
    labCell,
    el("td", { class: "der", text: lab?.system ? pct(lab.system.der) : "–" }),
    el("td", { class: "meta", text: sys ? pct(sys.der) : "–" }),
    el("td", {}, statusCell),
    el("td", {},
      openable ? el("button", { class: "pill small", text: "Open", onclick: (e) => { e.stopPropagation(); open(); } }) : null,
      " ",
      el("button", { class: "pill small ghost", text: "Delete", onclick: async (e) => {
        e.stopPropagation();
        if (await confirmDialog("Delete this recording?", `“${r.name}”, its audio, results and labels are deleted from this machine. Datasets already built keep it.`, "Delete")) {
          try { await api.remove(r.id); refresh(); } catch (err) { toast(err.message); }
        }
      } })));
}

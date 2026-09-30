// Dataset page: what an export holds, build it, push it to a private Hugging Face repo, and the
// command that evaluates a checkpoint on it.
import { api } from "./api.js";
import { $, el, confirmDialog, toast } from "./util.js";

let bound = false;
let pollTimer = null;
let info = null;

export async function showDataset() {
  if (!bound) bind();
  $("#dataset").hidden = false;
  $("#crumbs").replaceChildren();
  document.title = "Dataset · Diarization Bench";
  await refresh();
}

export function hideDataset() {
  $("#dataset").hidden = true;
  clearTimeout(pollTimer);
}

async function refresh() {
  clearTimeout(pollTimer);
  try { info = await api.dataset(); } catch (err) { toast(err.message); return; }
  render();
  if (info.task?.state === "running" && !$("#dataset").hidden) pollTimer = setTimeout(refresh, 1500);
}

function stat(label, value, sub) {
  return el("div", { class: "stat" }, el("span", { class: "eyebrow muted2", text: label }),
    el("span", { class: "stat-value", text: value }), sub ? el("span", { class: "meta muted2", text: sub }) : null);
}

function render() {
  const sp = info.splits, rv = info.review;
  const total = Object.values(sp).reduce((a, s) => a + s.recordings, 0);
  $("#ds-sub").textContent = `${total} accepted recording${total === 1 ? "" : "s"}`;
  $("#ds-stats").replaceChildren(
    ...Object.entries(sp).map(([name, s]) =>
      stat(name, `${s.speech_hours.toFixed(2)} h`, `${s.recordings} recordings · ${s.labelled_hours.toFixed(2)} h incl. silence`)),
    stat("Spot check", rv.judged ? `${Math.round(rv.accuracy * 100)}%` : "–",
      rv.judged ? `${rv.correct}/${rv.judged} correct · 95% ${Math.round(rv.low * 100)}–${Math.round(rv.high * 100)}%` : "not done yet"),
  );
  const rej = info.rejected;
  $("#ds-rejected-box").hidden = !rej.length;
  $("#ds-rejected-sum").textContent = `${rej.length} recording${rej.length === 1 ? "" : "s"} left out`;
  $("#ds-rejected").replaceChildren(...rej.map((r) =>
    el("li", {}, el("a", { href: `#/r/${r.id}`, text: r.name }), ` — ${r.reason}`)));
  if (!sp.test.recordings && total) {
    $("#ds-sub").textContent += " · no recording falls in the test split yet: pin one on its page";
  }

  // build
  const t = info.task;
  const note = $("#ds-export-note");
  if (t?.kind === "export") {
    note.textContent = t.state === "running" ? "Building…"
      : t.state === "error" ? `Failed: ${t.error}`
      : `Built ${t.result.recordings} recordings in ${t.result.path}`;
  } else {
    note.textContent = `Datasets are built in ${info.export_dir}.`;
  }

  // push
  const sel = $("#ds-push-name");
  const current = sel.value;
  sel.replaceChildren(...info.exports.map((x) => el("option", { value: x.name,
    text: `${x.name} (${x.recordings} recordings${x.audio ? "" : ", no audio"})` })));
  if (info.exports.some((x) => x.name === current)) sel.value = current;
  if (!$("#ds-repo").value && info.repo) $("#ds-repo").value = info.repo;
  $("#ds-hf-user").textContent = info.hf_user ? `logged in as ${info.hf_user}`
    : "not logged in: run  hf auth login  (a write token), then restart the server";
  $("#ds-push-btn").disabled = !info.hf_user || !info.exports.length || t?.state === "running";
  const pn = $("#ds-push-note");
  if (t?.kind === "push") {
    pn.replaceChildren(t.state === "running" ? "Uploading…"
      : t.state === "error" ? `Failed: ${t.error}`
      : el("span", {}, "Pushed: ", el("a", { href: t.result.url, target: "_blank", rel: "noopener", text: t.result.url })));
  }

  // evaluate
  const name = sel.value || $("#ds-name").value || "khmer-diarization-eval";
  const repo = $("#ds-repo").value || "your-user/khmer-diarization-eval";
  $("#ds-eval").textContent = [
    "# your current checkpoint (per-recording, per-channel and pooled DER with 95% intervals)",
    `python -m bench.evaluate ${info.export_dir}/${name} --out eval-current.json`,
    "# another checkpoint, and whether it is really better than the current one",
    `python -m bench.evaluate ${info.export_dir}/${name} --segmentation weights/new.ckpt --compare eval-current.json`,
    "# any other system's RTTMs (<uri>.rttm), or the copy on the Hugging Face Hub",
    `python -m bench.evaluate ${info.export_dir}/${name} --hyp-dir other_system/`,
    `python -m bench.evaluate ${repo}`,
  ].join("\n");
}

function bind() {
  bound = true;
  $("#ds-export-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api.exportDataset($("#ds-name").value.trim(), $("#ds-audio").checked, $("#ds-eval-only").checked);
      await refresh();
    } catch (err) { toast(err.message); }
  });
  $("#ds-push-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $("#ds-push-name").value, repo = $("#ds-repo").value.trim();
    const ok = await confirmDialog("Upload to Hugging Face?",
      `“${name}” is uploaded to the private dataset repo ${repo} (created if needed). ` +
      "Audio in it belongs to its YouTube owners: keep the repo private.", "Upload");
    if (!ok) return;
    try {
      await api.pushDataset(name, repo);
      await refresh();
    } catch (err) { toast(err.message); }
  });
  $("#ds-second-missing").addEventListener("click", async () => {
    try {
      const r = await api.rerunAll("second", true);
      toast(r.queued.length ? `Nemotron queued for ${r.queued.length} recordings.` : "Every recording has a second opinion.");
    } catch (err) { toast(err.message); }
  });
  $("#ds-relabel").addEventListener("click", async () => {
    try {
      const r = await api.rerunAll("labels", false);
      toast(`Labels rebuilt for ${r.queued.length} recordings (see Recordings).`);
    } catch (err) { toast(err.message); }
  });
}

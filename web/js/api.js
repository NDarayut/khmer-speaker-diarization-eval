// Thin fetch wrappers around the server API.

async function request(method, url, body, isForm = false) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  }
  const r = await fetch(url, opts);
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) {
    const err = new Error((data && data.detail) || `HTTP ${r.status}`);
    err.status = r.status;
    throw err;
  }
  return data;
}

export const api = {
  status: () => request("GET", "/api/status"),
  summary: () => request("GET", "/api/summary"),
  recordings: () => request("GET", "/api/recordings"),
  create: (form) => request("POST", "/api/recordings", form, true),
  get: (id) => request("GET", `/api/recordings/${id}`),
  rename: (id, name) => request("PATCH", `/api/recordings/${id}`, { name }),
  setSplit: (id, split) => request("PATCH", `/api/recordings/${id}`, { split }),
  remove: (id) => request("DELETE", `/api/recordings/${id}`),
  rerun: (id, system) => request("POST", `/api/recordings/${id}/rerun`, { system }),
  rerunAll: (system, only_missing) => request("POST", "/api/rerun", { system, only_missing }),
  dataset: () => request("GET", "/api/dataset"),
  exportDataset: (name, include_audio, eval_only) => request("POST", "/api/dataset/export", { name, include_audio, eval_only }),
  pushDataset: (name, repo_id) => request("POST", "/api/dataset/push", { name, repo_id }),
  review: () => request("GET", "/api/review"),
  sampleReview: (n) => request("POST", "/api/review/sample", { n }),
  judge: (id, verdict, note = "") => request("PUT", `/api/review/${id}`, { verdict, note }),
  inbox: () => request("GET", "/api/inbox"),
  job: (id) => request("GET", `/api/jobs/${id}`),
  peaks: async (id) => {
    const r = await fetch(`/api/recordings/${id}/peaks`);
    if (!r.ok) throw new Error(`peaks: HTTP ${r.status}`);
    return new Int8Array(await r.arrayBuffer());
  },
  audioUrl: (id) => `/api/recordings/${id}/audio`,
  rttmUrl: (id, name) => `/api/recordings/${id}/rttm/${name}`,
};

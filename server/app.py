"""
HTTP API + the benchmark UI. Single user, local: bind to localhost.

    GET    /api/status                          both models, devices, whether DiariZen is set up
    GET    /api/recordings                      library (with headline scores)
    POST   /api/recordings                      import: a YouTube video, playlist or channel link,
                                                a server path, or an upload
    GET    /api/recordings/{id}                 meta + turns of each system + labels + metrics
    PATCH  /api/recordings/{id}                 {name} rename, {split} pin the dataset split
    DELETE /api/recordings/{id}
    POST   /api/recordings/{id}/rerun           {system: reference | second | system | labels}
    POST   /api/rerun                           {system, only_missing}: the same for every recording
    GET    /api/recordings/{id}/audio           16 kHz WAV (Range requests)
    GET    /api/recordings/{id}/peaks           int8 (min, max) per 10 ms
    GET    /api/recordings/{id}/rttm/{name}     reference | second | system | system_oracle | labels.uem
    GET    /api/summary                         scores pooled over all scored recordings
    GET    /api/dataset                         what an export holds, spot check, exports, HF login
    POST   /api/dataset/export                  {name, include_audio, eval_only (default: true)}
    POST   /api/dataset/push                    {name, repo_id}: to a PRIVATE Hugging Face dataset
    GET    /api/review                          spot-check windows + accuracy
    POST   /api/review/sample                   {n}: draw n more random labelled windows
    PUT    /api/review/{item_id}                {verdict: correct | wrong | skip | null, note}
    GET    /api/inbox                           audio files in the inbox folder
    GET    /api/jobs/{job_id}
"""

import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bench import dataset, metrics, review, rttm, youtube
from bench.diarize import Diarizer
from bench.jobs import RERUN, Jobs, load_metrics
from bench.nonspeech import MusicDetector
from bench.reference import Reference, SecondOpinion
from bench.vad import Vad
from bench.store import Store, read_json
from config import ROOT, Settings

log = logging.getLogger("bench")
WEB = ROOT / "web"
RTTMS = ("reference", "second", "system", "system_oracle")
MEDIA_EXT = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".amr",
             ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}


class App:
    """Everything the routes share, created once at startup."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.diarizer = Diarizer(settings.segmentation, settings.device, settings.hf_token)
        self.reference = Reference(settings.diarizen_python, settings.diarizen_model,
                                   settings.diarizen_device)
        self.second = SecondOpinion(settings.nemotron_python, settings.nemotron_model,
                                    settings.nemotron_device)
        self.music = MusicDetector(str(self.diarizer.device))
        self.jobs = Jobs(self.store, self.diarizer, self.reference, self.second, self.music, Vad())
        self.task: dict | None = None  # the running / last dataset export or push
        self._hf_user: tuple[float, str | None] | None = None

    def hf_user(self) -> str | None:
        """Cached for a minute: whoami is a network call."""
        if self._hf_user is None or time.time() - self._hf_user[0] > 60:
            self._hf_user = (time.time(), dataset.hf_user(self.settings.hf_token))
        return self._hf_user[1]

    def run_task(self, kind: str, fn):
        if self.task and self.task["state"] == "running":
            raise HTTPException(409, f"a dataset {self.task['kind']} is still running")
        task = self.task = {"kind": kind, "state": "running", "started": time.time(),
                            "result": None, "error": None}

        def go():
            try:
                task["result"] = fn()
                task["state"] = "done"
            except Exception as e:
                log.error("dataset %s failed:\n%s", kind, traceback.format_exc())
                task["state"], task["error"] = "error", f"{type(e).__name__}: {e}"
            task["ended"] = time.time()
        threading.Thread(target=go, name=f"dataset-{kind}", daemon=True).start()
        return task


state: App | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    settings = Settings.from_env()
    state = App(settings)
    log.info("data: %s", settings.data_dir)
    log.info("system: %s + Community-1 on %s", Path(settings.segmentation).name,
             state.diarizer.device)
    log.info("reference: %s on %s%s", settings.diarizen_model, state.reference.device,
             "" if state.reference.available else " (NOT SET UP: run scripts/setup_diarizen.sh)")
    log.info("second opinion: %s on %s%s", settings.nemotron_model, state.second.device,
             "" if state.second.available else " (NOT SET UP: run scripts/setup_nemotron.sh)")
    resumed = resume_imports()
    if resumed:
        log.info("resumed %d interrupted imports", len(resumed))
    yield


def resume_imports() -> list[str]:
    """The job queue lives in memory: imports interrupted by a restart (still "importing"), and
    downloads that failed (an error before any audio, e.g. YouTube's occasional 403), are queued
    again. The rest of a recording's data is on disk and needs nothing."""
    out = []
    for meta in state.store.list():
        rid = meta["id"]
        d = state.store.dir(rid)
        failed_download = meta.get("status") == "error" and not (d / "audio.wav").is_file()
        if meta.get("status") != "importing" and not failed_download:
            continue
        opts = dict(meta.get("import_options") or {})
        vid = youtube.video_id(meta.get("source") or "")
        if vid:
            opts.update(youtube_id=vid, delete_source=True,
                        title_as_name=meta["name"] == f"youtube-{vid}")
        elif Path(meta.get("source") or "").is_file():
            opts.update(source_path=meta["source"], delete_source=False)
        else:  # an upload whose file is gone: nothing to resume from
            continue
        state.store.update_meta(rid, status="importing", error=None)
        state.jobs.submit("import", rid, **opts)
        out.append(rid)
    return out


app = FastAPI(title="Khmer Diarization Bench", lifespan=lifespan)


@app.middleware("http")
async def revalidate_ui(request, call_next):
    """UI files are revalidated on every load, so an updated app.js is never stale."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _rec(rec_id: str) -> dict:
    if not state.store.exists(rec_id):
        raise HTTPException(404, "no such recording")
    return state.store.meta(rec_id)


def _job_view(job: dict | None) -> dict | None:
    if not job:
        return None
    return {k: job[k] for k in ("id", "kind", "stage", "progress", "parts", "error", "warnings",
                                "timings")}


def _int(v) -> int | None:
    try:
        v = int(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


@app.get("/api/status")
def status():
    s = state.settings
    return {
        "system": {"segmentation": Path(s.segmentation).name, "rest": "pyannote Community-1 (VBx)",
                   "device": str(state.diarizer.device)},
        "reference": {"model": s.diarizen_model, "device": state.reference.device,
                      "available": state.reference.available},
        "second": {"model": s.nemotron_model, "device": state.second.device,
                   "available": state.second.available},
        "queued": state.jobs.pending(),
    }


@app.get("/api/recordings")
def recordings():
    return [{**meta, "job": _job_view(state.jobs.latest_for(meta["id"]))}
            for meta in state.store.list()]


@app.post("/api/recordings")
async def create_recording(
    file: UploadFile | None = File(None), path: str = Form(""), url: str = Form(""),
    num_speakers: str = Form(""), min_speakers: str = Form(""), max_speakers: str = Form(""),
    name: str = Form(""), limit: str = Form("50"), min_minutes: str = Form("1"),
    max_minutes: str = Form("180"),
):
    opts = {"num_speakers": _int(num_speakers), "min_speakers": _int(min_speakers),
            "max_speakers": _int(max_speakers)}
    if url.strip() and youtube.collection_url(url):
        return _import_collection(url, opts, limit=_int(limit) or 50,
                                  min_s=float(min_minutes or 0) * 60, max_s=float(max_minutes or 1e9) * 60)
    if url.strip():
        vid = youtube.video_id(url)
        if not vid:
            raise HTTPException(400, "not a YouTube video, playlist or channel link")
        if youtube.canonical_url(vid) in _sources():
            raise HTTPException(409, "this video is already in the library")
        rec_id = state.store.create(name or f"youtube-{vid}", youtube.canonical_url(vid))
        # downloaded by the import job; the recording takes the video's title unless named
        opts.update(youtube_id=vid, delete_source=True, title_as_name=not name)
    elif file is not None and file.filename:
        rec_id = state.store.create(name or file.filename, f"upload:{file.filename}")
        src = state.store.dir(rec_id) / ("source" + Path(file.filename).suffix.lower())
        with src.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
        opts.update(source_path=str(src), delete_source=True)
    elif path.strip():
        p = Path(path.strip()).expanduser()
        if not p.is_file():
            raise HTTPException(400, f"file not found on this machine: {p}")
        rec_id = state.store.create(name or p.name, str(p))
        opts.update(source_path=str(p), delete_source=False)
    else:
        raise HTTPException(400, "paste a YouTube link, choose a file or enter a path")
    state.store.update_meta(rec_id, import_options={k: v for k, v in opts.items()
                                                    if k.endswith("_speakers")})
    job = state.jobs.submit("import", rec_id, **opts)
    return {"id": rec_id, "job": _job_view(job)}


def _sources() -> set[str]:
    return {m.get("source") for m in state.store.list()}


def _import_collection(url: str, opts: dict, limit: int, min_s: float, max_s: float) -> dict:
    """A playlist or channel: one recording per video, skipping known, too short or too long
    videos (the length is known before downloading)."""
    try:
        videos = youtube.expand(url, limit=limit)
    except Exception as e:
        raise HTTPException(400, f"could not list that playlist/channel: {e}")
    known, created, skipped = _sources(), [], []
    for v in videos:
        src = youtube.canonical_url(v["id"])
        why = ("already in the library" if src in known
               else "length unknown (live or upcoming?)" if not v["duration"]
               else "too short" if v["duration"] < min_s
               else "too long" if v["duration"] > max_s else None)
        if why:
            skipped.append({"id": v["id"], "title": v["title"], "reason": why})
            continue
        rec_id = state.store.create(v["title"], src)
        state.store.update_meta(rec_id, import_options={k: o for k, o in opts.items() if o})
        state.jobs.submit("import", rec_id, **opts, youtube_id=v["id"], delete_source=True,
                          title_as_name=True)
        created.append(rec_id)
        known.add(src)
    return {"collection": youtube.collection_url(url), "found": len(videos), "created": created,
            "skipped": skipped}


@app.get("/api/recordings/{rec_id}")
def get_recording(rec_id: str):
    meta = _rec(rec_id)
    d = state.store.dir(rec_id)
    turns = {name: rttm.read(d / f"{name}.rttm") if (d / f"{name}.rttm").is_file() else None
             for name in RTTMS}
    c = read_json(d / "consensus.json")
    music = (read_json(d / "nonspeech.json") or {}).get("regions")
    return {"meta": meta, "turns": turns, "metrics": load_metrics(state.store, rec_id),
            "labels": {"uem": c["uem"], "excluded": c["excluded"], "status": c["status"],
                       "reason": c.get("reason")} if c else None,
            "music": music, "split": dataset.split_of(dataset.uri_of(meta), meta),
            "job": _job_view(state.jobs.latest_for(rec_id))}


@app.patch("/api/recordings/{rec_id}")
def update_recording(rec_id: str, body: dict = Body(...)):
    _rec(rec_id)
    changes = {}
    if "name" in body:
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is empty")
        changes["name"] = name[:120]
    if "split" in body:
        if body["split"] not in (*dataset.SPLITS, None):
            raise HTTPException(400, f"split must be one of {dataset.SPLITS} or null")
        changes["split"] = body["split"]
    return state.store.update_meta(rec_id, **changes)


@app.delete("/api/recordings/{rec_id}")
def delete_recording(rec_id: str):
    _rec(rec_id)
    if state.jobs.busy(rec_id):
        raise HTTPException(409, "still running")
    state.store.delete(rec_id)
    review.forget_recording(state.store, rec_id)
    return {"deleted": rec_id}


@app.post("/api/recordings/{rec_id}/rerun")
def rerun(rec_id: str, body: dict = Body(...)):
    meta = _rec(rec_id)
    which = body.get("system")
    if which not in RERUN:
        raise HTTPException(400, f"system must be one of {tuple(RERUN)}")
    if state.jobs.busy(rec_id):
        raise HTTPException(409, "a job is already running for this recording")
    if not meta.get("duration"):
        raise HTTPException(409, "the recording was not imported")
    return {"job": _job_view(state.jobs.submit("rerun", rec_id, system=which))}


@app.post("/api/rerun")
def rerun_all(body: dict = Body(...)):
    """Queue a rerun for every imported recording; with only_missing, only where that part's
    result is missing (e.g. the second opinion, after setting Nemotron up)."""
    which = body.get("system")
    if which not in RERUN:
        raise HTTPException(400, f"system must be one of {tuple(RERUN)}")
    file = {"reference": "reference.rttm", "second": "second.rttm", "system": "system.rttm",
            "labels": "consensus.json"}[which]
    queued = []
    for meta in state.store.list():
        rid = meta["id"]
        if not meta.get("duration") or state.jobs.busy(rid):
            continue
        if body.get("only_missing") and (state.store.dir(rid) / file).is_file():
            continue
        state.jobs.submit("rerun", rid, system=which)
        queued.append(rid)
    return {"queued": queued}


@app.get("/api/recordings/{rec_id}/audio")
def get_audio(rec_id: str):
    _rec(rec_id)
    p = state.store.audio_path(rec_id)
    if not p.is_file():
        raise HTTPException(404, "audio not ready")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/recordings/{rec_id}/peaks")
def get_peaks(rec_id: str):
    _rec(rec_id)
    p = state.store.dir(rec_id) / "peaks.bin"
    if not p.is_file():
        raise HTTPException(404, "waveform not ready")
    return FileResponse(p, media_type="application/octet-stream")


@app.get("/api/recordings/{rec_id}/rttm/{name}")
def get_rttm(rec_id: str, name: str):
    meta = _rec(rec_id)
    fname = "labels.uem" if name == "labels.uem" else f"{name}.rttm"
    p = state.store.dir(rec_id) / fname
    if (name not in RTTMS and name != "labels.uem") or not p.is_file():
        raise HTTPException(404, "no such result")
    return FileResponse(p, media_type="text/plain", filename=f"{meta['id']}.{fname}")


@app.get("/api/summary")
def summary():
    """Scores pooled over every scored recording (components summed, then divided): against
    DiariZen on whole files, and against the pseudo-labels on the accepted recordings' kept
    regions."""
    per = [m for m in (load_metrics(state.store, meta["id"]) for meta in state.store.list()) if m]
    acc = [m for m in per if (m.get("labels") or {}).get("status") == "accepted" and m.get("vs_labels")]
    return {
        "recordings": len(per),
        "speech_hours": round(sum(m["reference"]["speech"] for m in per) / 3600, 3),
        **{name: metrics.aggregate([m[name]["scores"] for m in per])
           for name in ("system", "system_oracle")},
        "speaker_count_exact": sum(m["system"]["speakers"] == m["reference"]["speakers"] for m in per),
        "labels": {"accepted": len(acc),
                   "labelled_hours": round(sum(m["labels"]["kept"] for m in acc) / 3600, 3),
                   "speech_hours": round(sum(m["labels"]["speech_kept"] for m in acc) / 3600, 3),
                   **{name: metrics.aggregate([m["vs_labels"][name]["scores"] for m in acc])
                      for name in ("system", "system_oracle")}},
    }


# ------------------------------------------------------------------ dataset
def _review_doc() -> dict:
    doc = review.load(state.store)
    return {**doc, "stats": review.stats(doc)}


def _default_repo() -> str | None:
    if state.settings.hf_dataset_repo:
        return state.settings.hf_dataset_repo
    user = state.hf_user()
    return f"{user}/khmer-diarization-eval" if user else None


@app.get("/api/dataset")
def dataset_info():
    root = Path(state.settings.export_dir)
    exports = []
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (d / "metadata.jsonl").is_file():
                exports.append({"name": d.name, "path": str(d), "built": d.stat().st_mtime,
                                "audio": (d / "audio").is_dir(),
                                "recordings": sum(1 for _ in (d / "metadata.jsonl").open())})
    return {**dataset.summary(state.store), "review": review.stats(review.load(state.store)),
            "exports": exports, "export_dir": str(root), "task": state.task,
            "hf_user": state.hf_user(), "repo": _default_repo()}


@app.post("/api/dataset/export")
def dataset_export(body: dict = Body(...)):
    name = str(body.get("name") or "khmer-diarization-eval").strip()
    include_audio = bool(body.get("include_audio", True))
    eval_only = bool(body.get("eval_only", True))
    models = {"reference": state.settings.diarizen_model, "second": state.settings.nemotron_model}
    return state.run_task("export", lambda: dataset.export(
        state.store, state.settings.export_dir, name, include_audio, _review_doc(), models, eval_only))


@app.post("/api/dataset/push")
def dataset_push(body: dict = Body(...)):
    name = str(body.get("name") or "").strip()
    folder = Path(state.settings.export_dir) / name
    if not name or "/" in name or not (folder / "metadata.jsonl").is_file():
        raise HTTPException(400, "build the dataset first")
    repo = str(body.get("repo_id") or _default_repo() or "").strip()
    if repo.count("/") != 1:
        raise HTTPException(400, "repo_id must look like user/name")
    if not state.hf_user():
        raise HTTPException(400, "not logged in to Hugging Face: run  hf auth login  with a write "
                                 "token (or set HF_TOKEN in .env), then restart")
    return state.run_task("push", lambda: {"url": dataset.push(folder, repo, state.settings.hf_token),
                                           "repo": repo, "name": name})


# ------------------------------------------------------------------ spot check
@app.get("/api/review")
def review_get():
    return _review_doc()


@app.post("/api/review/sample")
def review_sample(body: dict = Body(default={})):
    n = max(1, min(200, _int(body.get("n")) or 20))
    try:
        review.sample(state.store, dataset.eligible(state.store), n)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _review_doc()


@app.put("/api/review/{item_id}")
def review_judge(item_id: str, body: dict = Body(...)):
    try:
        return review.judge(state.store, item_id, body.get("verdict"), str(body.get("note") or ""))
    except KeyError:
        raise HTTPException(404, "no such window")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/inbox")
def inbox():
    """Audio/video files in the inbox folder, offered as import paths."""
    root = Path(state.settings.inbox_dir)
    if not root.is_dir():
        return {"dir": str(root), "files": []}
    files = [{"path": str(p), "name": str(p.relative_to(root)), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in MEDIA_EXT and not p.name.startswith(".")]
    return {"dir": str(root), "files": files}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = state.jobs.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return _job_view(job)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=WEB), name="web")

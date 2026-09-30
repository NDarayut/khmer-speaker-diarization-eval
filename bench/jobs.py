"""
Import and re-run jobs. One background worker runs them in order:

    import:  [download from YouTube] -> transcode -> waveform
             -> diarize (DiariZen ‖ Nemotron ‖ your system, then music detection)
             -> oracle -> label -> score
    rerun:   one step again, and everything after it that depends on it

"diarize" starts the external diarizers (subprocesses, each on its GPU) and runs the system under
test and the music detector in this process meanwhile. "oracle" re-clusters the system with the
reference's speaker count (seconds, from its cached embeddings). "label" builds the pseudo-labels:
DiariZen's turns where Nemotron agrees and there is no music (consensus.py). "score" scores the
system against DiariZen (whole file) and against the labels (agreed regions only).
"""

import logging
import os
import queue
import threading
import time
import traceback

import numpy as np

from . import audio, consensus, metrics, rttm, vad as vadlib, youtube
from .diarize import DiarizationCache, Diarizer
from .store import Store, read_json, write_json

log = logging.getLogger(__name__)

# what a rerun of each part redoes
RERUN = {"reference": (("reference",), True, True),   # (diarize which, oracle?, label?)
         "second": (("second",), False, True),
         "system": (("system",), True, False),
         "labels": ((), False, True)}
HEADLINE = metrics.condition_key(0.0, False)  # the score shown in the library


def _turns(result) -> list:
    return [[t.start, t.end, t.speaker] for t in result.turns]


def headline(m: dict) -> dict:
    """The library row's numbers from a metrics.json."""
    out = {"ref_speakers": m["reference"]["speakers"]}
    for name in ("system", "system_oracle"):
        if m.get(name):
            out[name] = {"speakers": m[name]["speakers"], **m[name]["scores"][HEADLINE]}
    lab = m.get("labels")
    if lab:
        out["labels"] = {k: lab.get(k) for k in ("status", "reason", "kept", "speech_kept",
                                                 "speech_kept_share", "agreement")}
        vs = (m.get("vs_labels") or {}).get("system")
        if vs:
            out["labels"]["system"] = vs["scores"][HEADLINE]
    return out


class Jobs:
    def __init__(self, store: Store, diarizer: Diarizer, reference, second=None, music=None,
                 vad=None):
        self.store = store
        self.vad = vad
        self.diarizer = diarizer
        self.reference = reference
        self.second = second
        self.music = music
        self.jobs: dict[str, dict] = {}
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="jobs", daemon=True).start()

    # -- public -------------------------------------------------------------------------
    def submit(self, kind: str, rec_id: str, **options) -> dict:
        job = {"id": f"{kind}-{rec_id}-{time.time_ns() // 1000}", "kind": kind, "rec_id": rec_id,
               "stage": "queued", "progress": 0.0, "parts": {}, "error": None, "warnings": [],
               "options": options, "created": time.time(), "timings": {}}
        self.jobs[job["id"]] = job
        self._q.put(job)
        return job

    def latest_for(self, rec_id: str) -> dict | None:
        mine = [j for j in self.jobs.values() if j["rec_id"] == rec_id]
        return max(mine, key=lambda j: j["created"]) if mine else None

    def busy(self, rec_id: str) -> bool:
        j = self.latest_for(rec_id)
        return bool(j and j["stage"] not in ("done", "error"))

    def pending(self) -> int:
        return sum(j["stage"] not in ("done", "error") for j in self.jobs.values())

    # -- worker -------------------------------------------------------------------------
    def _run(self):
        while True:
            job = self._q.get()
            try:
                getattr(self, f"_{job['kind']}")(job)
                job["stage"], job["progress"] = "done", 1.0
            except Exception as e:
                log.exception("job %s failed", job["id"])
                error = f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc(limit=5)
                try:  # the recording first: whoever sees the job fail must see this too
                    self.store.update_meta(job["rec_id"], status="error", error=error)
                except Exception:
                    pass
                job["stage"], job["error"] = "error", error

    def _stage(self, job, name):
        job["stage"], job["progress"] = name, 0.0
        job["_t"] = time.time()

    def _end_stage(self, job, name):
        job["timings"][name] = round(time.time() - job.pop("_t"), 2)

    def _import(self, job):
        rec_id, opts = job["rec_id"], job["options"]
        d = self.store.dir(rec_id)
        if opts.get("youtube_id"):
            self._stage(job, "download")
            video: dict = {}
            path, title = youtube.download(opts["youtube_id"], d,
                                           progress=lambda p: job.__setitem__("progress", p),
                                           info=video)
            opts["source_path"] = str(path)
            self.store.update_meta(rec_id, youtube={k: v for k, v in video.items() if v})
            if opts.get("title_as_name"):
                self.store.update_meta(rec_id, name=title)
            self._end_stage(job, "download")
        src = opts["source_path"]

        self._stage(job, "transcode")
        duration = audio.transcode(src, d / "audio.wav")
        if opts.get("delete_source"):
            os.remove(src)
        self.store.update_meta(rec_id, duration=round(duration, 3))
        self._end_stage(job, "transcode")

        self._stage(job, "waveform")
        wav = audio.load(d / "audio.wav")
        audio.peaks(wav).tofile(d / "peaks.bin")
        self._end_stage(job, "waveform")

        self._diarize(job, wav, ("reference", "second", "system"), music=True)
        self._oracle(job, wav)
        self._label(job)
        self._score(job)

    def _rerun(self, job):
        rec_id, which = job["rec_id"], job["options"]["system"]
        meta = self.store.update_meta(rec_id, status="running", error=None)
        # the speaker-count options given at import apply again
        job["options"] = {**(meta.get("import_options") or {}), **job["options"]}
        d = self.store.dir(rec_id)
        if which != "reference" and not (d / "reference.rttm").is_file():
            raise RuntimeError("there is no reference yet: re-run the reference first")
        parts, oracle, label = RERUN[which]
        wav = audio.load(self.store.audio_path(rec_id))
        if parts:
            self._diarize(job, wav, parts, music=not (d / "nonspeech.json").is_file())
        elif not (d / "nonspeech.json").is_file():
            self._diarize(job, wav, (), music=True)
        if oracle:
            self._oracle(job, wav)
        if label:
            self._label(job)
        self._score(job)

    # -- steps --------------------------------------------------------------------------
    def _diarize(self, job, wav, which, music=False):
        rec_id, opts = job["rec_id"], job["options"]
        d = self.store.dir(rec_id)
        count = {k: opts.get(k) for k in ("num_speakers", "min_speakers", "max_speakers")}
        self._stage(job, "diarize")
        parts = job["parts"] = {w: 0.0 for w in (*which, *(("music",) if music else ()))}

        def progress(name):
            def set_(p):
                parts[name] = p
                job["progress"] = sum(parts.values()) / len(parts)
            return set_

        runs = {}
        if "reference" in which:
            runs["reference"] = self.reference.start(d / "audio.wav", d / "reference.rttm",
                                                     progress=progress("reference"), **count)
        if "second" in which:
            if self.second is not None and self.second.available:
                runs["second"] = self.second.start(d / "audio.wav", d / "second.rttm",
                                                   progress=progress("second"))
            else:
                (d / "second.rttm").unlink(missing_ok=True)
                job["warnings"].append("second opinion (Nemotron) is not set up: "
                                       "run scripts/setup_nemotron.sh")
                parts["second"] = 1.0
        try:
            if "system" in which:
                result = self.diarizer(wav, **count, progress=progress("system"))
                (d / "diar_cache.npz").unlink(missing_ok=True)
                if result.cache is not None:
                    result.cache.save(d / "diar_cache.npz")
                rttm.write(rttm.relabel(_turns(result), "S"), d / "system.rttm")
            if music and self.music is not None:
                write_json(d / "nonspeech.json", self.music(wav, progress=progress("music")))
            if "second" in runs:
                try:
                    rttm.write(rttm.relabel(runs["second"].wait(), "N"), d / "second.rttm")
                except RuntimeError as e:  # no labels for this recording, but the rest stands
                    (d / "second.rttm").unlink(missing_ok=True)
                    job["warnings"].append(str(e))
                    log.warning("%s: %s", rec_id, e)
                parts["second"] = 1.0
            if "reference" in runs:
                rttm.write(rttm.relabel(runs["reference"].wait(), "R"), d / "reference.rttm")
                parts["reference"] = 1.0
        finally:
            for run in runs.values():
                run.kill()
        self._end_stage(job, "diarize")

    def _oracle(self, job, wav):
        """The system again, told the reference's speaker count (re-clustering only)."""
        d = self.store.dir(job["rec_id"])
        self._stage(job, "oracle")
        n = len(rttm.speakers(rttm.read(d / "reference.rttm")))
        cache_path = d / "diar_cache.npz"
        cache = DiarizationCache.load(cache_path) if cache_path.is_file() else None
        result = self.diarizer(wav, num_speakers=max(1, n), cache=cache,
                               progress=lambda p: job.__setitem__("progress", p))
        if cache is None and result.cache is not None:
            result.cache.save(cache_path)
        rttm.write(rttm.relabel(_turns(result), "S"), d / "system_oracle.rttm")
        self._end_stage(job, "oracle")

    def _label(self, job):
        """Pseudo-labels: DiariZen where Nemotron agrees and there is no music."""
        rec_id = job["rec_id"]
        d = self.store.dir(rec_id)
        self._stage(job, "label")
        duration = self.store.meta(rec_id)["duration"]
        second = rttm.read(d / "second.rttm") if (d / "second.rttm").is_file() else None
        music = (read_json(d / "nonspeech.json") or {}).get("regions", [])
        vad = None
        if self.vad is not None and second is not None:
            if not (d / "vad.npy").is_file():  # CPU, seconds; also for recordings made before it
                np.save(d / "vad.npy", self.vad(audio.load(d / "audio.wav")))
            vad = vadlib.frames(np.load(d / "vad.npy"), duration)
        c = consensus.build(rttm.read(d / "reference.rttm"), second, duration, music, vad=vad)
        write_json(d / "consensus.json", c)
        consensus.write_uem(c["uem"], d / "labels.uem")
        self._end_stage(job, "label")

    def _score(self, job):
        rec_id = job["rec_id"]
        d = self.store.dir(rec_id)
        self._stage(job, "score")
        duration = self.store.meta(rec_id)["duration"]
        ref = rttm.read(d / "reference.rttm")
        m = {"reference": {"speakers": len(rttm.speakers(ref)), "model": self.reference.model,
                           "speech": round(metrics.speech(ref), 3)}}
        hyps = {name: rttm.read(d / f"{name}.rttm") for name in ("system", "system_oracle")}
        for name, hyp in hyps.items():
            m[name] = metrics.evaluate(ref, hyp, duration)
        if (d / "second.rttm").is_file():  # how far the two references are apart
            m["second"] = {"model": getattr(self.second, "model", None),
                           **metrics.evaluate(ref, rttm.read(d / "second.rttm"), duration)}
        c = read_json(d / "consensus.json")
        if c:
            m["labels"] = {k: v for k, v in c.items() if k not in ("uem", "excluded")}
            if c["uem"]:
                m["vs_labels"] = {name: metrics.evaluate(ref, hyp, duration, c["uem"])
                                  for name, hyp in hyps.items()}
        m["scored"] = time.time()
        write_json(d / "metrics.json", m)
        self._end_stage(job, "score")
        self.store.update_meta(rec_id, status="ready", error=None, scores=headline(m),
                               warnings=job["warnings"] or None,
                               timings={**(self.store.meta(rec_id).get("timings") or {}),
                                        **job["timings"]})


def load_metrics(store: Store, rec_id: str) -> dict | None:
    return read_json(store.dir(rec_id) / "metrics.json")

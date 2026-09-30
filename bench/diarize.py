"""
The system under test: offline (whole-file) diarization with the fine-tuned Khmer segmentation
model, and everything else from pyannote Community-1 (embedding, PLDA, VBx clustering, its
default parameters). Community-1 is gated: it needs HF_TOKEN once, then it stays in the HF cache.
"""

import itertools
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.core import SlidingWindow, SlidingWindowFeature

from .audio import SAMPLE_RATE

log = logging.getLogger(__name__)

COMMUNITY_1 = "pyannote/speaker-diarization-community-1"

# Community-1 defaults (retuning them on real meetings did not help)
_PARAMS = {"segmentation": {"min_duration_off": 0.0},
           "clustering": {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}}

# Stage weights for the progress bar (embedding extraction dominates).
_STAGES = {"segmentation": (0.0, 0.3), "embeddings": (0.3, 0.97)}


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[Turn]
    speakers: list[str]
    embeddings: np.ndarray | None  # one centroid per speaker, in `speakers` order
    cache: "DiarizationCache | None" = None


class _CachingPipeline(SpeakerDiarization):
    """SpeakerDiarization that reuses segmentations / embeddings passed in the file dict, so
    re-clustering (another speaker count) takes seconds instead of minutes."""

    def get_segmentations(self, file, hook=None):
        if file.get("cached_segmentations") is not None:
            return file["cached_segmentations"]
        return super().get_segmentations(file, hook=hook)

    def get_embeddings(self, file, binary_segmentations, exclude_overlap=False, hook=None):
        """pyannote embeds every (chunk, local speaker) pair, silent ones too, and the embedding
        model always reads the whole 10 s chunk. With the Khmer model's 5 local speakers about
        half the pairs are silent (53% on a 44-min meeting), and their embeddings are never used:
        clustering drops them and reconstruction marks them inactive. So embed only the active
        pairs and leave NaN, which clustering already skips: the same result in half the time."""
        if file.get("cached_embeddings") is not None:
            return file["cached_embeddings"]
        seg = np.nan_to_num(binary_segmentations.data, nan=0.0).astype(np.float32)
        num_chunks, num_frames, num_speakers = seg.shape
        if exclude_overlap:  # as pyannote: frames with one speaker, unless that leaves too few
            duration = binary_segmentations.sliding_window.duration
            min_num_frames = math.ceil(num_frames * self._embedding.min_num_samples
                                       / (duration * self._embedding.sample_rate))
            clean = seg * (seg.sum(axis=2, keepdims=True) < 2)
        else:
            min_num_frames, clean = -1, seg
        pairs = np.argwhere(seg.sum(axis=1) > 0)  # (chunk, speaker), in chunk order

        def waveforms_and_masks():
            chunk = waveform = None
            for c, s in pairs:
                if c != chunk:
                    chunk = c
                    waveform, _ = self._audio.crop(file, binary_segmentations.sliding_window[c],
                                                   mode="pad")
                mask = clean[c, :, s] if clean[c, :, s].sum() > min_num_frames else seg[c, :, s]
                yield waveform[None], torch.from_numpy(np.ascontiguousarray(mask))[None]

        embeddings = np.full((num_chunks, num_speakers, self._embedding.dimension), np.nan,
                             dtype=np.float32)
        size = self.embedding_batch_size
        total = math.ceil(len(pairs) / size)
        items = waveforms_and_masks()
        if hook is not None:
            hook("embeddings", None, total=total, completed=0)
        for i in range(total):
            waveforms, masks = zip(*itertools.islice(items, size))
            batch = self._embedding(torch.vstack(waveforms), masks=torch.vstack(masks))
            done = pairs[i * size: i * size + len(batch)]
            embeddings[done[:, 0], done[:, 1]] = batch
            if hook is not None:
                hook("embeddings", batch, total=total, completed=i + 1)
        return embeddings


def _build(segmentation: str, token: str | None, device: torch.device):
    pipeline = _CachingPipeline(
        segmentation=Model.from_pretrained(segmentation),
        embedding={"checkpoint": COMMUNITY_1, "subfolder": "embedding"},
        plda={"checkpoint": COMMUNITY_1, "subfolder": "plda"},
        clustering="VBxClustering",
        segmentation_batch_size=32,
        embedding_batch_size=32,
        token=token,
    )
    pipeline.instantiate(_PARAMS)
    return pipeline.to(device)


def cpu_budget() -> int:
    """CPUs this process may really use: the affinity mask, capped by a cgroup CPU quota (a
    container started with --cpus N still sees every host core in os.cpu_count(); running one
    torch thread per host core under an 8-CPU quota was throttled to a fraction of the speed)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]  # cgroup v2
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        try:  # cgroup v1
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = min(n, math.ceil(quota / period))
        except (OSError, ValueError):
            pass
    return max(1, n)


class Diarizer:
    """Loads the pipeline lazily, once, and serializes calls (one GPU context)."""

    def __init__(self, segmentation: str, device: str = "auto", token: str | None = None):
        if device in (None, "", "auto"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(cpu_budget())
        self.segmentation = segmentation
        self.token = token
        self._pipeline = None
        self._stamp = None
        self._lock = threading.Lock()

    def _load(self):
        """Build the pipeline once, and again whenever the checkpoint file changes (a retrained
        model is picked up by "Re-run yours" without restarting the server)."""
        stamp = os.stat(self.segmentation).st_mtime_ns if os.path.isfile(self.segmentation) else None
        if self._pipeline is not None and stamp == self._stamp:
            return
        self._pipeline, self._stamp = None, stamp
        try:
            self._pipeline = _build(self.segmentation, self.token, self.device)
        except Exception as e:  # gated and not cached yet, offline, ...
            raise RuntimeError(f"could not load {Path(self.segmentation).name} + Community-1 "
                               f"(set HF_TOKEN once to download Community-1): {e}") from e
        log.info("system: %s + Community-1 VBx on %s", Path(self.segmentation).name, self.device)

    def _run(self, file, hook, kwargs):
        """Run the pipeline; on a broken GPU stack (seen: cuDNN failing after an NVIDIA driver
        update until reboot) retry without cuDNN, then on the CPU, instead of failing."""
        try:
            return self._pipeline(file, hook=hook, **kwargs)
        except Exception as e:
            if self.device.type != "cuda" or not _is_cuda_failure(e):
                raise
            log.warning("diarization: GPU error (%s); retrying without cuDNN", str(e).splitlines()[0][:120])
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = False
        try:
            return self._pipeline(file, hook=hook, **kwargs)
        except Exception as e:
            if not _is_cuda_failure(e):
                raise
            log.warning("diarization: GPU still failing (%s); switching to CPU", str(e).splitlines()[0][:120])
        self.device = torch.device("cpu")
        torch.set_num_threads(cpu_budget())
        self._pipeline.to(self.device)
        return self._pipeline(file, hook=hook, **kwargs)

    def __call__(self, audio: np.ndarray, num_speakers: int | None = None,
                 min_speakers: int | None = None, max_speakers: int | None = None,
                 cache: "DiarizationCache | None" = None, progress=None) -> DiarizationResult:
        """Diarize a whole 16 kHz mono recording. Pass the `cache` of an earlier run of the same
        audio to only re-cluster (another speaker count): seconds, not minutes."""
        captured = {}
        with self._lock:
            self._load()

            def hook(step_name, step_artefact, file=None, total=None, completed=None):
                if step_name in ("segmentation", "embeddings") and total is None:
                    captured[step_name] = step_artefact
                if progress and step_name in _STAGES and total:
                    lo, hi = _STAGES[step_name]
                    progress(lo + (hi - lo) * (completed or 0) / total)

            waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
            file = {"waveform": waveform, "sample_rate": SAMPLE_RATE}
            if cache is not None:
                file["cached_segmentations"] = cache.segmentations
                file["cached_embeddings"] = cache.embeddings
            kwargs = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                            max_speakers=max_speakers).items() if v}
            out = self._run(file, hook, kwargs)
        annotation = getattr(out, "speaker_diarization", out)
        speakers = list(annotation.labels())
        turns = [Turn(round(seg.start, 3), round(seg.end, 3), spk)
                 for seg, _, spk in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: (t.start, t.end))
        emb = getattr(out, "speaker_embeddings", None)
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32)[: len(speakers)]
        if cache is None and "segmentation" in captured and "embeddings" in captured:
            cache = DiarizationCache(captured["segmentation"], captured["embeddings"])
        if progress:
            progress(1.0)
        return DiarizationResult(turns, speakers, emb, cache)


def _is_cuda_failure(e: Exception) -> bool:
    msg = str(e)
    return isinstance(e, RuntimeError) and any(k in msg for k in ("CUDA", "cuDNN", "CUDNN", "NVML"))


@dataclass
class DiarizationCache:
    """The expensive intermediate results: local segmentations and per-chunk embeddings."""
    segmentations: SlidingWindowFeature
    embeddings: np.ndarray

    def save(self, path: str | Path):
        sw = self.segmentations.sliding_window
        np.savez(path, seg=self.segmentations.data.astype(np.float16),
                 window=np.array([sw.start, sw.duration, sw.step]),
                 emb=self.embeddings.astype(np.float32))

    @classmethod
    def load(cls, path: str | Path) -> "DiarizationCache":
        z = np.load(path)
        start, duration, step = z["window"].tolist()
        seg = SlidingWindowFeature(z["seg"].astype(np.float32),
                                   SlidingWindow(start=start, duration=duration, step=step))
        return cls(seg, z["emb"])

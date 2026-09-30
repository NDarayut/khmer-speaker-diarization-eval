"""
Runs the external diarizers as subprocesses, each in its own environment, so they can work on
another GPU while the system under test runs in this process:

    Reference      DiariZen (BUT-FIT), the labels                   .venv-diarizen, reference_worker.py
    SecondOpinion  NVIDIA Nemotron-3-Diarization, the cross-check   .venv-nemo,     nemotron_worker.py
"""

import collections
import logging
import os
import subprocess
import threading
from pathlib import Path

from . import rttm

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
# stdout lines of the workers (and DiariZen's own prints) -> progress
_PROGRESS = {"@stage load": 0.02, "@stage diarize": 0.1, "Extracting segmentations": 0.1,
             "Extracting Embeddings": 0.45, "Clustering": 0.9, "@stage done": 1.0}


def gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def resolve_device(device: str, prefer: str = "last") -> str:
    """auto -> the last GPU (DiariZen: the system under test uses the first) or the first
    (Nemotron: small, it shares the system's GPU), or cpu without a GPU."""
    if device in (None, "", "auto"):
        n = gpu_count()
        return (f"cuda:{n - 1}" if prefer == "last" else "cuda:0") if n else "cpu"
    return device


class ExternalRun:
    def __init__(self, name: str, proc: subprocess.Popen, out: Path, progress=None):
        self.name = name
        self.proc = proc
        self.out = out
        self.progress = progress
        self.tail: collections.deque[str] = collections.deque(maxlen=30)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            self.tail.append(line)
            for key, p in _PROGRESS.items():
                if line.startswith(key) and self.progress:
                    self.progress(p)

    def wait(self) -> list:
        code = self.proc.wait()
        self._reader.join(timeout=5)
        if code != 0 or not self.out.is_file():
            log.error("%s output:\n%s", self.name, "\n".join(self.tail))
            last = next((line for line in reversed(self.tail) if line.strip()), f"exit code {code}")
            raise RuntimeError(f"{self.name} failed: {last[:500]}")
        return rttm.read(self.out)

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()


class External:
    """One external diarizer: its interpreter, worker script, model and device."""
    title = "external diarizer"
    worker = ""
    setup = ""
    prefer = "last"

    def __init__(self, python: str, model: str, device: str = "auto"):
        self.python = python
        self.model = model
        self.device = resolve_device(device, self.prefer)

    @property
    def available(self) -> bool:
        return Path(self.python).is_file()

    def args(self, num_speakers=None, min_speakers=None, max_speakers=None) -> list[str]:
        return []

    def start(self, wav: Path, out: Path, progress=None, **count) -> ExternalRun:
        if not self.available:
            raise RuntimeError(f"{self.title} is not set up ({self.python} not found): "
                               f"run {self.setup}")
        out = Path(out)
        out.unlink(missing_ok=True)
        cmd = [self.python, str(HERE / self.worker), "--audio", str(wav), "--out", str(out),
               "--model", self.model, *self.args(**count)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               # long files fragment GPU memory; this lets the allocator reuse the gaps
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               "CUDA_VISIBLE_DEVICES": self.device.split(":")[1] if self.device.startswith("cuda:")
               else ("" if self.device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "0"))}
        env.pop("VIRTUAL_ENV", None)
        log.info("%s: %s on %s", self.title, self.model, self.device)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=env, cwd=HERE.parent)
        return ExternalRun(self.title, proc, out, progress)


class Reference(External):
    """DiariZen: the labels."""
    title = "reference diarizer (DiariZen)"
    worker = "reference_worker.py"
    setup = "scripts/setup_diarizen.sh"

    def args(self, num_speakers=None, min_speakers=None, max_speakers=None) -> list[str]:
        out = []
        for flag, v in (("--num-speakers", num_speakers), ("--min-speakers", min_speakers),
                        ("--max-speakers", max_speakers)):
            if v:
                out += [flag, str(v)]
        return out


class SecondOpinion(External):
    """Nemotron-3-Diarization: the cross-check. Up to 8 speakers, no speaker-count option."""
    title = "second opinion (Nemotron)"
    worker = "nemotron_worker.py"
    setup = "scripts/setup_nemotron.sh"
    prefer = "first"
    max_speakers = 8

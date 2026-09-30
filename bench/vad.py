"""
Speech activity from Silero VAD, a small dedicated voice-activity model that shares nothing with
the diarizers. The labels use it to settle one kind of disagreement: DiariZen and Nemotron agree
on who could be speaking, but not on *whether* anyone is (see consensus.py).

Stored per recording as vad.npy: the raw speech probability every 32 ms (float16), with none of
Silero's usual padding or merging of short pauses, which would blur the frame-level decision.
"""

import threading

import numpy as np
import torch

from .audio import SAMPLE_RATE

HOP = 0.032  # Silero's frame: 512 samples at 16 kHz


class Vad:
    """Loads Silero lazily, once; runs on the CPU (it is small, and the GPUs are busy)."""

    def __init__(self, threads: int = 8):
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._model is None:
                from silero_vad import load_silero_vad
                self._model = load_silero_vad()
            before = torch.get_num_threads()
            torch.set_num_threads(self.threads)
            try:
                with torch.inference_mode():
                    p = self._model.audio_forward(torch.from_numpy(np.ascontiguousarray(audio, np.float32)),
                                                  SAMPLE_RATE)
                self._model.reset_states()
            finally:
                torch.set_num_threads(before)
        return p.squeeze(0).numpy().astype(np.float16)


def frames(prob: np.ndarray, duration: float, hop: float = 0.01) -> np.ndarray:
    """The 32 ms probabilities resampled to `hop` frames (linear interpolation)."""
    n = max(1, int(round(duration / hop)))
    return np.interp(np.arange(n) * hop, np.arange(len(prob)) * HOP + HOP / 2, prob.astype(np.float32))

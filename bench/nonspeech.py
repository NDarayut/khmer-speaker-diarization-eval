"""
Music detection, so that intros, jingles, background music and songs are kept out of the labels.

Diarizers label who speaks when; on music they disagree or, worse, agree on a "speaker" that is a
song. The Audio Spectrogram Transformer fine-tuned on AudioSet (MIT/ast-finetuned-audioset,
527 sound classes) scores 5 s windows every second; the music-like and speech-like classes'
probabilities are assigned to each window's middle second.

Only *music without speech* is excluded: talk over background music is fine data. On a 4-minute
Khmer video with background music, music >= 0.5 alone flagged 40% of the seconds DiariZen heard
speech in; music >= 0.5 with speech < 0.5 flagged 1% of them, and 75% of the non-speech seconds.
"""

import logging
import threading

import numpy as np
import torch

from .audio import SAMPLE_RATE

log = logging.getLogger(__name__)

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MUSIC_CLASSES = ("Music", "Musical instrument", "Song", "Singing", "Background music",
                 "Theme music", "Jingle (music)", "Soundtrack music", "Pop music", "Choir")
SPEECH_CLASSES = ("Speech", "Conversation", "Narration, monologue", "Male speech, man speaking",
                  "Female speech, woman speaking", "Child speech, kid speaking")
HOP = 1.0          # seconds between scores
WINDOW = 5.0       # seconds of audio behind each score
THRESHOLD = 0.5    # a second is music-only when music >= this and speech < this
MIN_RUN = 2.0      # shorter music runs are ignored (s)
MERGE_GAP = 2.0    # music runs closer than this are joined (s)


def regions(music: np.ndarray, speech: np.ndarray, hop: float = HOP, threshold: float = THRESHOLD,
            min_run: float = MIN_RUN, merge_gap: float = MERGE_GAP) -> list[list[float]]:
    """Per-second music and speech probabilities -> [[start, end], ...] music-only regions."""
    music, speech = np.asarray(music), np.asarray(speech)
    on = np.r_[False, (music >= threshold) & (speech < threshold), False]
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    runs = [[float(a * hop), float(b * hop)] for a, b in zip(edges[::2], edges[1::2])]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [[round(a, 2), round(b, 2)] for a, b in merged if b - a >= min_run]


class MusicDetector:
    """Loads AST lazily, once; one call at a time."""

    def __init__(self, device: str = "auto", batch_size: int = 32):
        if device in (None, "", "auto"):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = self._fe = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self._fe = ASTFeatureExtractor.from_pretrained(MODEL)
        self._model = ASTForAudioClassification.from_pretrained(MODEL).to(self.device).eval()
        names = self._model.config.label2id
        self._music = [names[c] for c in MUSIC_CLASSES if c in names]
        self._speech = [names[c] for c in SPEECH_CLASSES if c in names]
        log.info("music detector: %s on %s", MODEL, self.device)

    def __call__(self, audio: np.ndarray, progress=None) -> dict:
        """16 kHz mono -> {"hop", "music": [p per second], "speech": [...], "regions": [...]}"""
        with self._lock:
            self._load()
            n = max(1, int(np.ceil(len(audio) / SAMPLE_RATE / HOP)))
            half = int(WINDOW * SAMPLE_RATE / 2)
            padded = np.pad(audio.astype(np.float32), (half, half))
            music, speech = np.zeros(n, np.float32), np.zeros(n, np.float32)
            for i in range(0, n, self.batch_size):
                idx = range(i, min(n, i + self.batch_size))
                # window centred on the middle of second k
                clips = [padded[int((k + 0.5) * HOP * SAMPLE_RATE): int((k + 0.5) * HOP * SAMPLE_RATE) + 2 * half]
                         for k in idx]
                feats = self._fe(clips, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                with torch.inference_mode():
                    logits = self._model(feats["input_values"].to(self.device)).logits
                p = torch.sigmoid(logits.float()).cpu().numpy()
                music[i:i + len(idx)] = p[:, self._music].max(1)
                speech[i:i + len(idx)] = p[:, self._speech].max(1)
                if progress:
                    progress(min(1.0, (i + len(idx)) / n))
        return {"hop": HOP, "model": MODEL, "music": np.round(music, 3).tolist(),
                "speech": np.round(speech, 3).tolist(), "regions": regions(music, speech)}

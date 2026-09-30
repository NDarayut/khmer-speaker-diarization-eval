"""
Audio helpers. Every recording is converted once, on import, to a canonical 16 kHz mono
16-bit WAV; everything else (diarization, clips for ASR, waveform, export) reads that file.
"""

import io
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
PEAK_HOP_S = 0.01  # one waveform peak pair per 10 ms


def probe_duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def transcode(src: str | Path, dst: str | Path) -> float:
    """Any audio/video file ffmpeg can read -> 16 kHz mono PCM16 WAV. Returns its duration."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return sf.info(str(dst)).duration


def load(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return audio


def read_span(path: str | Path, start: float, end: float) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        a = max(0, int(round(start * f.samplerate)))
        b = min(f.frames, int(round(end * f.samplerate)))
        f.seek(a)
        return f.read(max(0, b - a), dtype="float32")


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def peaks(audio: np.ndarray, hop_s: float = PEAK_HOP_S) -> np.ndarray:
    """Interleaved (min, max) per hop as int8, for drawing the waveform at any zoom."""
    hop = int(round(hop_s * SAMPLE_RATE))
    n = int(np.ceil(len(audio) / hop))
    padded = np.zeros(n * hop, dtype=np.float32)
    padded[: len(audio)] = audio
    frames = padded.reshape(n, hop)
    # normalise to the 99.9th percentile, not the maximum, so one loud click does not
    # flatten the whole waveform
    scale = max(float(np.percentile(np.abs(audio), 99.9)) if len(audio) else 0.0, 1e-4)
    out = np.empty(2 * n, dtype=np.int8)
    out[0::2] = np.clip(np.round(frames.min(axis=1) / scale * 127), -127, 127)
    out[1::2] = np.clip(np.round(frames.max(axis=1) / scale * 127), -127, 127)
    return out


def energy_db(audio: np.ndarray, hop_s: float = PEAK_HOP_S, win_s: float = 0.03) -> np.ndarray:
    """Short-time RMS in dB per hop (used to find pauses when splitting long utterances)."""
    hop = int(round(hop_s * SAMPLE_RATE))
    win = int(round(win_s * SAMPLE_RATE))
    n = int(np.ceil(len(audio) / hop))
    sq = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    centers = np.arange(n) * hop + hop // 2
    a = np.clip(centers - win // 2, 0, len(audio))
    b = np.clip(centers + win // 2, 0, len(audio))
    rms = np.sqrt((sq[b] - sq[a]) / np.maximum(b - a, 1))
    return 20 * np.log10(np.maximum(rms, 1e-6))

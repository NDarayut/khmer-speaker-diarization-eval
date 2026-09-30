"""
On-disk storage, one folder per recording under the data directory:

    data/<id>/meta.json            name, source, duration, options, status, headline scores
              audio.wav            16 kHz mono PCM16 (the only audio anything reads)
              peaks.bin            int8 (min, max) per 10 ms, for the waveform
              reference.rttm       the reference diarizer's output (DiariZen)
              system.rttm          the system under test, speaker count found by itself
              system_oracle.rttm   the system under test, given the reference's speaker count
              diar_cache.npz       the system's segmentations + embeddings (re-clustering in seconds)
              metrics.json         scores, speaker mapping, per-speaker breakdown, error track

JSON files are written atomically (temp file + rename), so a crash never leaves half a file.
"""

import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

def write_json(path: Path, data, backup: bool = False) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.flush()
        os.fsync(f.fileno())
    if backup and path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    return s[:48] or "recording"


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def lock(self, rec_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(rec_id, threading.Lock())

    def dir(self, rec_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", rec_id):
            raise KeyError(rec_id)
        return self.root / rec_id

    def exists(self, rec_id: str) -> bool:
        try:
            return (self.dir(rec_id) / "meta.json").is_file()
        except KeyError:
            return False

    def create(self, name: str, source: str) -> str:
        rec_id = f"{slugify(Path(name).stem)}-{secrets.token_hex(3)}"
        d = self.dir(rec_id)
        d.mkdir(parents=True)
        self.save_meta(rec_id, {
            "id": rec_id, "name": Path(name).stem, "source": source, "created": time.time(),
            "status": "importing", "duration": None,
        })
        return rec_id

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            meta = read_json(d / "meta.json") if d.is_dir() else None
            if meta:
                out.append(meta)
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def meta(self, rec_id: str) -> dict:
        meta = read_json(self.dir(rec_id) / "meta.json")
        if meta is None:
            raise KeyError(rec_id)
        return meta

    def save_meta(self, rec_id: str, meta: dict) -> None:
        write_json(self.dir(rec_id) / "meta.json", meta)

    def update_meta(self, rec_id: str, **changes) -> dict:
        with self.lock(rec_id):
            meta = self.meta(rec_id)
            meta.update(changes)
            self.save_meta(rec_id, meta)
            return meta

    def audio_path(self, rec_id: str) -> Path:
        return self.dir(rec_id) / "audio.wav"

    def delete(self, rec_id: str) -> None:
        shutil.rmtree(self.dir(rec_id))

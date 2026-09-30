"""
Spot check: estimate how often the pseudo-labels are wrong by listening to a small random sample,
not to everything.

Windows are drawn at random from the kept (UEM) regions of accepted recordings, each piece with
probability proportional to its length, so every labelled second is equally likely. A person
marks each window correct or wrong; the share judged correct, with a 95% Wilson interval, goes
into the dataset card. 30 windows give an interval of about ±12 points, 100 about ±7.

    data/_review.json   {"items": [{id, rec_id, start, end, verdict, note, judged}], "created"}
"""

import math
import random
import secrets
import threading
import time

from .store import Store, read_json, write_json

WINDOW = 8.0       # seconds per window (shorter when the kept piece is shorter)
MIN_WINDOW = 3.0   # kept pieces shorter than this are not sampled
VERDICTS = ("correct", "wrong", "skip")

_lock = threading.Lock()


def _path(store: Store):
    return store.root / "_review.json"


def load(store: Store) -> dict:
    return read_json(_path(store)) or {"items": [], "created": time.time()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def stats(doc: dict) -> dict:
    judged = [i for i in doc["items"] if i.get("verdict") in ("correct", "wrong")]
    k = sum(i["verdict"] == "correct" for i in judged)
    lo, hi = wilson(k, len(judged))
    return {"items": len(doc["items"]), "judged": len(judged), "correct": k,
            "wrong": len(judged) - k,
            "pending": sum(i.get("verdict") is None for i in doc["items"]),
            "accuracy": round(k / len(judged), 4) if judged else None,
            "low": round(lo, 4), "high": round(hi, 4)}


def sample(store: Store, eligible: list[tuple[dict, dict]], n: int, window: float = WINDOW,
           seed: int | None = None) -> dict:
    """Add n new random windows (from the kept regions of `eligible` (meta, consensus) pairs)."""
    rng = random.Random(seed)
    pieces = [(meta["id"], a, b) for meta, c in eligible for a, b in c["uem"] if b - a >= MIN_WINDOW]
    if not pieces:
        raise ValueError("no labelled regions to sample from yet")
    weights = [b - a for _, a, b in pieces]
    with _lock:
        doc = load(store)
        taken = {(i["rec_id"], round(i["start"])) for i in doc["items"]}
        added, tries = 0, 0
        while added < n and tries < 50 * n:
            tries += 1
            rec_id, a, b = rng.choices(pieces, weights)[0]
            w = min(window, b - a)
            start = round(rng.uniform(a, b - w), 2)
            if (rec_id, round(start)) in taken:
                continue
            taken.add((rec_id, round(start)))
            doc["items"].append({"id": secrets.token_hex(4), "rec_id": rec_id, "start": start,
                                 "end": round(start + w, 2), "verdict": None, "note": ""})
            added += 1
        write_json(_path(store), doc)
    return doc


def judge(store: Store, item_id: str, verdict: str | None, note: str = "") -> dict:
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    with _lock:
        doc = load(store)
        item = next((i for i in doc["items"] if i["id"] == item_id), None)
        if item is None:
            raise KeyError(item_id)
        item.update(verdict=verdict, note=note[:500], judged=time.time() if verdict else None)
        write_json(_path(store), doc)
    return item


def forget_recording(store: Store, rec_id: str) -> None:
    """Drop the windows of a deleted recording."""
    with _lock:
        doc = load(store)
        doc["items"] = [i for i in doc["items"] if i["rec_id"] != rec_id]
        write_json(_path(store), doc)

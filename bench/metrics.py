"""
Scoring the system against the reference.

`evaluate` gives, for each scoring condition (forgiveness collar around reference boundaries,
with or without the overlapped speech), the standard pyannote.metrics numbers:

    DER = (missed + false alarm + confusion) / reference speech
    JER   Jaccard error rate: the mean, over reference speakers, of how badly each is matched

plus the optimal one-to-one speaker mapping (system -> reference), a per-reference-speaker
breakdown and an error track for the timeline. The breakdown and the track are computed on a
10 ms grid without a collar; the DER on that grid agrees with pyannote's collar-0 DER.

Several recordings are pooled by summing the components (`aggregate`), so a long recording
counts more than a short one, as in the usual corpus-level DER.
"""

import numpy as np
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from . import rttm

HOP = 0.01
# (collar: seconds forgiven on EACH side of every reference boundary, as in NIST md-eval's
# "±0.25 s"; pyannote's `collar` is the total width, so it gets twice this; skip overlap)
CONDITIONS = [(0.0, False), (0.25, False), (0.0, True), (0.25, True)]


def condition_key(collar: float, skip_overlap: bool) -> str:
    return f"collar={collar:g}" + (",no-overlap" if skip_overlap else "")


def _uem(duration: float, uem=None) -> Timeline:
    """The scored regions: [[start, end], ...], or the whole recording."""
    return Timeline([Segment(a, b) for a, b in uem] if uem is not None else [Segment(0, duration)])


def score(ref_turns, hyp_turns, duration: float, collar: float = 0.0,
          skip_overlap: bool = False, uem=None) -> dict:
    """Standard scores; with `uem` ([[start, end], ...]) only those regions are scored."""
    ref, hyp = rttm.to_annotation(ref_turns), rttm.to_annotation(hyp_turns)
    uem = _uem(duration, uem)
    der = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip_overlap)
    d = der(ref, hyp, uem=uem, detailed=True)
    total = d["total"]
    jer = (JaccardErrorRate(collar=2 * collar, skip_overlap=skip_overlap)(ref, hyp, uem=uem)
           if ref.labels() else None)
    return {
        "collar": collar, "skip_overlap": skip_overlap,
        "speech": round(total, 3),
        "missed": round(d["missed detection"], 3),
        "false_alarm": round(d["false alarm"], 3),
        "confusion": round(d["confusion"], 3),
        "der": round(d["diarization error rate"], 5) if total else None,
        "jer": round(float(jer), 5) if jer is not None else None,
    }


def speech(turns, uem=None) -> float:
    """Seconds with at least one speaker (overlap counted once), within `uem` if given."""
    tl = rttm.to_annotation(turns).get_timeline().support()
    if uem is not None:
        tl = tl.crop(_uem(0, uem), mode="intersection")
    return tl.duration()


def mapping(ref_turns, hyp_turns, duration: float, uem=None) -> dict[str, str]:
    """Optimal one-to-one mapping system speaker -> reference speaker (unmatched extras absent)."""
    ref, hyp = rttm.to_annotation(ref_turns), rttm.to_annotation(hyp_turns)
    return dict(DiarizationErrorRate().optimal_mapping(ref, hyp, uem=_uem(duration, uem)))


def grid(turns, labels, n):
    m = np.zeros((n, len(labels)), dtype=bool)
    col = {s: i for i, s in enumerate(labels)}
    for s, e, spk in turns:
        a, b = int(round(s / HOP)), min(n, int(round(e / HOP)))
        if b > a:
            m[a:b, col[spk]] = True
    return m


def mask(duration: float, uem=None) -> np.ndarray:
    """10 ms frames inside the scored regions."""
    n = max(1, int(round(duration / HOP)))
    if uem is None:
        return np.ones(n, dtype=bool)
    m = np.zeros(n, dtype=bool)
    for a, b in uem:
        m[int(round(a / HOP)): min(n, int(round(b / HOP)))] = True
    return m


def frame_analysis(ref_turns, hyp_turns, duration: float, hyp_to_ref: dict[str, str],
                   uem=None) -> dict:
    """Frame-level (10 ms, no collar) error track and per-reference-speaker breakdown.

    Each system speaker is renamed to its mapped reference speaker; unmatched system speakers
    stay separate columns. Per frame (as in DER): miss = max(0, n_ref - n_hyp),
    false alarm = max(0, n_hyp - n_ref), confusion = min(n_ref, n_hyp) - correct."""
    n = max(1, int(round(duration / HOP)))
    ref_labels = rttm.speakers(ref_turns)
    extras = [s for s in rttm.speakers(hyp_turns) if s not in hyp_to_ref]
    cols = ref_labels + extras
    R = np.zeros((n, len(cols)), dtype=bool)
    R[:, :len(ref_labels)] = grid(ref_turns, ref_labels, n)
    H = grid([[s, e, hyp_to_ref.get(spk, spk)] for s, e, spk in hyp_turns], cols, n)
    if uem is not None:  # outside the scored regions nothing counts
        keep = mask(duration, uem)[:, None]
        R &= keep
        H &= keep
    n_ref, n_hyp = R.sum(1), H.sum(1)
    correct = (R & H).sum(1)
    miss = np.maximum(0, n_ref - n_hyp)
    fa = np.maximum(0, n_hyp - n_ref)
    conf = np.minimum(n_ref, n_hyp) - correct

    # the dominant error of each frame (confusion first on ties), as runs [start, end, kind]
    kind = np.zeros(n, dtype=np.int8)  # 0 = no error
    worst = np.maximum.reduce([conf, miss, fa])
    kind[(worst > 0) & (fa == worst)] = 3
    kind[(worst > 0) & (miss == worst)] = 2
    kind[(worst > 0) & (conf == worst)] = 1
    names = {1: "confusion", 2: "miss", 3: "fa"}
    track = []
    change = np.flatnonzero(np.diff(kind)) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, n]):
        if kind[a]:
            track.append([round(a * HOP, 2), round(b * HOP, 2), names[int(kind[a])]])

    # per reference speaker: its speech found under its mapped system speaker, given to another
    # system speaker (confusion), or not covered by any extra system speaker (missed)
    per_speaker = []
    spare = H & ~R  # system activity with no reference speaker of that label
    inv = {v: h for h, v in hyp_to_ref.items()}
    for k, spk in enumerate(ref_labels):
        mine = R[:, k]
        lost = mine & ~H[:, k]
        confused = lost & spare.any(1)
        per_speaker.append({
            "speaker": spk, "mapped_to": inv.get(spk),
            "speech": round(mine.sum() * HOP, 2),
            "found": round((mine & H[:, k]).sum() * HOP, 2),
            "confused": round(confused.sum() * HOP, 2),
            "missed": round((lost & ~confused).sum() * HOP, 2),
        })
    speech = n_ref.sum()
    frame_der = float((miss.sum() + fa.sum() + conf.sum()) / speech) if speech else None
    return {"errors": track, "per_speaker": per_speaker,
            "frame_der": round(frame_der, 5) if frame_der is not None else None}


def evaluate(ref_turns, hyp_turns, duration: float, uem=None) -> dict:
    """Everything the compare view shows for one system on one recording (within `uem`)."""
    hyp_to_ref = mapping(ref_turns, hyp_turns, duration, uem)
    return {
        "speakers": len(rttm.speakers(hyp_turns)),
        "mapping": hyp_to_ref,
        "scores": {condition_key(c, s): score(ref_turns, hyp_turns, duration, c, s, uem)
                   for c, s in CONDITIONS},
        **frame_analysis(ref_turns, hyp_turns, duration, hyp_to_ref, uem),
    }


def pooled_der(scores: list[dict], key: str) -> float | None:
    """Corpus-level DER of some recordings' scores for one condition."""
    rows = [s[key] for s in scores if s.get(key)]
    speech = sum(r["speech"] for r in rows)
    return sum(r["missed"] + r["false_alarm"] + r["confusion"] for r in rows) / speech if speech else None


def cluster_bootstrap(groups: dict[str, list[dict]], key: str, n: int = 2000, seed: int = 0,
                      paired: dict[str, list[dict]] | None = None) -> dict | None:
    """95% interval of the pooled DER when whole groups (channels) are resampled with
    replacement: recordings of one channel are not independent, channels roughly are.
    With `paired` (the same groups scored for another system), the interval of the difference
    DER(paired) - DER(groups) instead, and how often the other system came out better."""
    names = [g for g in groups if groups[g]]
    if len(names) < 2:
        return None
    rng = np.random.default_rng(seed)
    stat = []
    for _ in range(n):
        pick = [names[i] for i in rng.integers(0, len(names), len(names))]
        a = pooled_der([s for g in pick for s in groups[g]], key)
        if paired is None:
            stat.append(a)
        else:
            stat.append(pooled_der([s for g in pick for s in paired[g]], key) - a)
    stat = np.array(stat, dtype=float)
    out = {"low": round(float(np.percentile(stat, 2.5)), 5), "high": round(float(np.percentile(stat, 97.5)), 5),
           "groups": len(names), "resamples": n}
    if paired is not None:
        out["better_share"] = round(float((stat < 0).mean()), 4)  # other system lower DER
    return out


def aggregate(results: list[dict]) -> dict:
    """Pool per-recording `scores` dicts (one per recording, same keys): sums of components
    divided by the total reference speech. JER is averaged, weighted by reference speech."""
    out = {}
    for key in (results[0] if results else {}):
        rows = [r[key] for r in results if r.get(key)]
        speech = sum(r["speech"] for r in rows)
        err = {c: sum(r[c] for r in rows) for c in ("missed", "false_alarm", "confusion")}
        jer_rows = [r for r in rows if r["jer"] is not None]
        jer_w = sum(r["speech"] for r in jer_rows)
        out[key] = {
            "recordings": len(rows), "speech": round(speech, 3),
            **{c: round(v, 3) for c, v in err.items()},
            "der": round(sum(err.values()) / speech, 5) if speech else None,
            "jer": round(sum(r["jer"] * r["speech"] for r in jer_rows) / jer_w, 5) if jer_w else None,
        }
    return out

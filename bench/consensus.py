"""
Pseudo-labels without listening: keep only what two independent diarizers agree on.

    labels    DiariZen's turns (reference.rttm)
    scored    a UEM: the parts of the recording where
                - Nemotron-3-Diarization (second.rttm), after the best one-to-one mapping of its
                  speakers to DiariZen's, has exactly the same speakers active (both silent counts),
                  except for boundary jitter: a disagreement shorter than JITTER where both models
                  change state (the same boundary, a little shifted) is tolerated, and DiariZen's
                  boundary is used,
                  and where the two disagree only on *whether* someone speaks (DiariZen: one
                  speaker, Nemotron: nobody, or the reverse), a third judge, Silero VAD (vad.py),
                  settles it when it clearly sides with DiariZen (speech >= VAD_SPEECH, or
                  silence <= VAD_SILENCE); the labels stay DiariZen's either way,
                - at least COLLAR away from any other disagreement (its surroundings are suspect),
                - not music without speech (nonspeech.py),
              in pieces of at least MIN_PIECE.

Training and evaluation use the labels only inside the UEM, as pyannote does with any UEM.

**Bias:** what two models agree on is the easier speech. Overlapped speech almost never survives
(both must agree on every speaker at once), and turn changes survive less often than the middle
of a turn; `coverage` reports both. So a score on these labels is a score on the easier part of
the audio: compare systems with it, but don't read it as the error on the whole recording.

A recording is **accepted** as data when the second opinion could judge it (DiariZen found at most
8 speakers, Nemotron's limit) and the agreed part keeps at least MIN_SPEECH_KEPT of DiariZen's
speech; otherwise it is rejected, with the reason.
"""

import numpy as np
from scipy.ndimage import binary_dilation

from . import metrics, rttm

HOP = metrics.HOP
COLLAR = 0.25           # seconds dropped on each side of every disagreement
JITTER = 0.25           # a shorter disagreement where both models change state is tolerated
VAD_SPEECH = 0.5        # Silero probability that confirms DiariZen's speech (set from data, see README)
VAD_SILENCE = 0.5       # ... and below which it confirms DiariZen's silence
MIN_PIECE = 1.0         # agreed pieces shorter than this are dropped (s)
MIN_SPEECH_KEPT = 0.5   # accept a recording when this share of the reference speech is kept
MAX_SPEAKERS = 8        # the second opinion's limit


def runs(on: np.ndarray, hop: float = HOP) -> list[list[float]]:
    """Boolean frames -> [[start, end], ...] in seconds."""
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    return [[round(a * hop, 2), round(b * hop, 2)] for a, b in zip(edges[::2], edges[1::2])]


def _changes(M: np.ndarray) -> np.ndarray:
    """Frames where the set of active speakers differs from the previous frame's."""
    return np.r_[False, (M[1:] != M[:-1]).any(axis=1)]


def build(ref_turns, second_turns, duration: float, music_regions=(),
          collar: float = COLLAR, min_piece: float = MIN_PIECE,
          min_speech_kept: float = MIN_SPEECH_KEPT, jitter: float = JITTER,
          vad: np.ndarray | None = None, vad_speech: float = VAD_SPEECH,
          vad_silence: float = VAD_SILENCE) -> dict:
    """`vad`: speech probability per 10 ms frame (vad.frames), to settle speech/silence disputes."""
    n = max(1, int(round(duration / HOP)))
    ref_speech = metrics.speech(ref_turns)
    ref_spk = rttm.speakers(ref_turns)
    base = {"duration": round(duration, 3), "reference_speakers": len(ref_spk),
            "reference_speech": round(ref_speech, 3),
            "settings": {"collar": collar, "min_piece": min_piece, "min_speech_kept": min_speech_kept,
                         "jitter": jitter, "vad": vad is not None, "vad_speech": vad_speech,
                         "vad_silence": vad_silence}}

    if second_turns is None:
        return {**base, "status": "rejected", "reason": "no second opinion", "uem": [],
                "excluded": [[0.0, round(duration, 2), "unchecked"]], "kept": 0.0, "speech_kept": 0.0}
    if len(ref_spk) > MAX_SPEAKERS:
        return {**base, "status": "rejected",
                "reason": f"{len(ref_spk)} speakers: more than the second opinion's {MAX_SPEAKERS}",
                "uem": [], "excluded": [[0.0, round(duration, 2), "unchecked"]],
                "kept": 0.0, "speech_kept": 0.0}

    to_ref = metrics.mapping(ref_turns, second_turns, duration)
    extras = [s for s in rttm.speakers(second_turns) if s not in to_ref]
    cols = ref_spk + extras
    R = np.zeros((n, len(cols)), dtype=bool)
    R[:, :len(ref_spk)] = metrics.grid(ref_turns, ref_spk, n)
    S = metrics.grid([[s, e, to_ref.get(spk, spk)] for s, e, spk in second_turns], cols, n)
    agree = (R == S).all(axis=1)

    # boundary jitter: short disagreements where both models change state nearby
    tolerated = np.zeros(n, dtype=bool)
    J = int(round(jitter / HOP))
    if J:
        cR, cS = _changes(R), _changes(S)
        for a, b in runs(~agree):
            fa, fb = int(round(a / HOP)), int(round(b / HOP))
            lo, hi = max(0, fa - J), min(n, fb + J + 1)
            if fb - fa < J and cR[lo:hi].any() and cS[lo:hi].any():
                tolerated[fa:fb] = True

    # speech/silence disputes the VAD settles in DiariZen's favour
    resolved = np.zeros(n, dtype=bool)
    if vad is not None:
        v = np.zeros(n, dtype=np.float32)
        v[:min(n, len(vad))] = vad[:n]
        r_n, s_n = R[:, :len(ref_spk)].sum(1), S.sum(1)
        resolved = ~agree & ~tolerated & (((r_n == 1) & (s_n == 0) & (v >= vad_speech))
                                          | ((r_n == 0) & (s_n >= 1) & (v <= vad_silence)))

    pad = int(round(collar / HOP))
    disagree = ~agree & ~tolerated & ~resolved
    near = binary_dilation(disagree, iterations=pad) if pad and disagree.any() else disagree
    music = metrics.mask(duration, [r[:2] for r in music_regions]) if music_regions else np.zeros(n, bool)
    keep = ~near & ~music
    # drop short pieces: they are mostly slivers between two disagreements
    short = np.zeros(n, dtype=bool)
    for a, b in runs(keep):
        if b - a < min_piece:
            short[int(round(a / HOP)): int(round(b / HOP))] = True
    keep &= ~short

    # why each dropped frame was dropped (first reason that applies)
    reason = np.full(n, "", dtype=object)
    reason[short] = "short"
    reason[near & ~disagree] = "near disagreement"
    reason[disagree] = "disagreement"
    reason[music] = "music"
    excluded = []
    for kind in ("music", "disagreement", "near disagreement", "short"):
        excluded += [[a, b, kind] for a, b in runs(reason == kind)]
    excluded.sort()

    uem = runs(keep)
    speech_kept = metrics.speech(ref_turns, uem) if uem else 0.0
    # what kind of speech survives: overlap, and speech near a turn change (±0.5 s)
    cnt = R[:, :len(ref_spk)].sum(1)
    sp, ov = cnt >= 1, cnt >= 2
    near_turn = binary_dilation(_changes(R[:, :len(ref_spk)]), iterations=50) & sp
    share = lambda m: round(float((keep & m).sum() / m.sum()), 4) if m.any() else None  # noqa: E731
    kept_share = speech_kept / ref_speech if ref_speech else 0.0
    status, why = "accepted", None
    if not uem or ref_speech == 0:
        status, why = "rejected", "nothing agreed"
    elif kept_share < min_speech_kept:
        status, why = "rejected", f"only {kept_share:.0%} of the reference speech agreed (< {min_speech_kept:.0%})"
    return {
        **base, "status": status, "reason": why,
        "second_speakers": len(rttm.speakers(second_turns)), "mapping": to_ref,
        "uem": uem, "excluded": excluded,
        "kept": round(sum(b - a for a, b in uem), 3),
        "speech_kept": round(speech_kept, 3),
        "speech_kept_share": round(kept_share, 4),
        "agreement": round(float(agree.mean()), 4),
        "coverage": {"speech": share(sp), "overlap": share(ov), "near_turn_change": share(near_turn),
                     "overlap_seconds": round(float(ov.sum() * HOP), 2)},
        "jitter_tolerated": round(float(tolerated.sum() * HOP), 2),
        "vad_resolved": round(float(resolved.sum() * HOP), 2),
        "dropped": {k: round(float((reason == k).sum() * HOP), 2)
                    for k in ("disagreement", "near disagreement", "music", "short")},
    }


def write_uem(uem, path, uri: str = "rec") -> None:
    with open(path, "w") as f:
        for a, b in uem:
            f.write(f"{uri} 1 {a:.3f} {b:.3f}\n")


def read_uem(path) -> list[list[float]]:
    out = []
    for line in open(path).read().splitlines():
        f = line.split()
        if len(f) >= 4:
            out.append([float(f[2]), float(f[3])])
    return out

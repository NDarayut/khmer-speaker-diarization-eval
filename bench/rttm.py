"""
Turns <-> RTTM files <-> pyannote Annotations.

A turn is `[start, end, speaker]` (JSON-friendly); lists of turns are what the job stores and
the web page draws. Both systems' speakers are renamed by talk time, most first: R1, R2, ... for
the reference and S1, S2, ... for the system, so labels are stable and short.
"""

from collections import Counter
from pathlib import Path

from pyannote.core import Annotation, Segment

Turn = list  # [start: float, end: float, speaker: str]


def relabel(turns, prefix: str) -> list[Turn]:
    """Rename speakers to <prefix>1.. by total talk time; sort by start, drop empty turns."""
    talk = Counter()
    for s, e, spk in turns:
        talk[spk] += e - s
    names = {spk: f"{prefix}{i + 1}" for i, (spk, _) in enumerate(talk.most_common())}
    out = [[round(float(s), 3), round(float(e), 3), names[spk]] for s, e, spk in turns if e > s]
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return out


def speakers(turns) -> list[str]:
    """Speaker labels, most talk time first."""
    talk = Counter()
    for s, e, spk in turns:
        talk[spk] += e - s
    return [spk for spk, _ in talk.most_common()]


def to_annotation(turns, uri: str = "rec") -> Annotation:
    ann = Annotation(uri=uri)
    for i, (s, e, spk) in enumerate(turns):
        ann[Segment(s, e), i] = spk
    return ann


def write(turns, path: str | Path, uri: str = "rec") -> None:
    lines = [f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n"
             for s, e, spk in turns]
    Path(path).write_text("".join(lines))


def read(path: str | Path) -> list[Turn]:
    turns = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            turns.append([round(start, 3), round(start + dur, 3), f[7]])
    turns.sort(key=lambda t: (t[0], t[1], t[2]))
    return turns

"""
The pseudo-labelled recordings as a dataset: built locally, optionally pushed to a private
Hugging Face dataset repo, and read back for evaluation (evaluate.py).

    <name>/README.md              dataset card: method, sizes, spot-check accuracy, licenses
          database.yml            pyannote.database protocol KhmerPseudo.SpeakerDiarization.YouTube
          metadata.jsonl          one row per recording
          audio/<uri>.flac        16 kHz mono (unless built without audio)
          rttm/<uri>.rttm         labels: DiariZen's turns
          uem/<uri>.uem           the regions to use: where Nemotron agrees and there is no music
          pyannote/<split>.{lst,rttm,uem}   the same, per split, as pyannote.database wants them
          review.json             the spot-check verdicts

Splits are fixed by a hash of the recording's YouTube channel (80/10/10), or of its uri when the
channel is unknown: every video of a channel lands in the same split, so a podcast's hosts are
never in both train and test. A recording never changes split when the dataset grows; its meta
may pin one ("split": "test").
"""

import hashlib
import json
import shutil
import time
from pathlib import Path

import soundfile as sf

from . import audio, consensus, rttm, youtube
from .store import Store, read_json

SPLITS = ("train", "development", "test")
PROTOCOL = "KhmerPseudo.SpeakerDiarization.YouTube"


def uri_of(meta: dict) -> str:
    vid = youtube.video_id(meta.get("source") or "")
    return f"yt-{vid}" if vid else meta["id"]


def group_of(uri: str, meta: dict | None = None) -> str:
    """What a split is chosen by: the YouTube channel when known, else the recording."""
    channel = ((meta or {}).get("youtube") or {}).get("channel_id")
    return f"channel:{channel}" if channel else uri


def split_of(uri: str, meta: dict | None = None) -> str:
    if meta and meta.get("split") in SPLITS:
        return meta["split"]
    x = int(hashlib.sha1(group_of(uri, meta).encode()).hexdigest()[:8], 16) / 0x100000000
    return "train" if x < 0.8 else "development" if x < 0.9 else "test"


def eligible(store: Store) -> list[tuple[dict, dict]]:
    """(meta, consensus) of every recording whose labels were accepted."""
    out = []
    for meta in store.list():
        c = read_json(store.dir(meta["id"]) / "consensus.json")
        if meta.get("status") == "ready" and c and c.get("status") == "accepted":
            out.append((meta, c))
    return out


def summary(store: Store) -> dict:
    """Sizes of what an export would contain, and why the rest is left out."""
    per_split = {s: {"recordings": 0, "labelled_hours": 0.0, "speech_hours": 0.0} for s in SPLITS}
    rejected = []
    for meta in store.list():
        c = read_json(store.dir(meta["id"]) / "consensus.json")
        if meta.get("status") != "ready" or not c:
            continue
        if c["status"] != "accepted":
            rejected.append({"id": meta["id"], "name": meta["name"], "reason": c.get("reason")})
            continue
        s = per_split[split_of(uri_of(meta), meta)]
        s["recordings"] += 1
        s["labelled_hours"] += c["kept"] / 3600
        s["speech_hours"] += c["speech_kept"] / 3600
    for s in per_split.values():
        s["labelled_hours"] = round(s["labelled_hours"], 3)
        s["speech_hours"] = round(s["speech_hours"], 3)
    return {"splits": per_split, "rejected": rejected}


def export(store: Store, out_root: str | Path, name: str, include_audio: bool = True,
           review: dict | None = None, models: dict | None = None, eval_only: bool = False) -> dict:
    """Build the dataset folder <out_root>/<name> (replacing an earlier build of that name).
    With eval_only, every recording goes to the test split (a pure evaluation set)."""
    if not name or any(c in name for c in "/\\") or name.startswith("."):
        raise ValueError("dataset name must be a plain folder name")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f".{name}.building"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ("rttm", "uem", "pyannote", *(("audio",) if include_audio else ())):
        (tmp / sub).mkdir(parents=True)

    rows = []
    per_split = {s: {"uris": [], "rttm": [], "uem": []} for s in SPLITS}
    for meta, c in eligible(store):
        d = store.dir(meta["id"])
        uri = uri_of(meta)
        split = "test" if eval_only else split_of(uri, meta)
        turns = rttm.read(d / "reference.rttm")
        rttm.write(turns, tmp / "rttm" / f"{uri}.rttm", uri=uri)
        consensus.write_uem(c["uem"], tmp / "uem" / f"{uri}.uem", uri=uri)
        if include_audio:
            sf.write(tmp / "audio" / f"{uri}.flac", audio.load(d / "audio.wav"), audio.SAMPLE_RATE,
                     format="FLAC")
        p = per_split[split]
        p["uris"].append(uri)
        p["rttm"].append((tmp / "rttm" / f"{uri}.rttm").read_text())
        p["uem"].append((tmp / "uem" / f"{uri}.uem").read_text())
        rows.append({
            "uri": uri, "split": split, "title": meta["name"], "source": meta.get("source"),
            "youtube_id": youtube.video_id(meta.get("source") or ""),
            "channel": (meta.get("youtube") or {}).get("channel"),
            "channel_id": (meta.get("youtube") or {}).get("channel_id"),
            "duration": meta["duration"], "labelled": c["kept"], "labelled_speech": c["speech_kept"],
            "speech_kept_share": c["speech_kept_share"], "agreement": c["agreement"],
            "speakers": c["reference_speakers"], "dropped": c["dropped"],
            "coverage": c.get("coverage"),
        })
    if not rows:
        shutil.rmtree(tmp)
        raise ValueError("no accepted recordings to export yet")

    for split, p in per_split.items():
        if eval_only and split != "test":
            continue
        (tmp / "pyannote" / f"{split}.lst").write_text("".join(u + "\n" for u in p["uris"]))
        (tmp / "pyannote" / f"{split}.rttm").write_text("".join(p["rttm"]))
        (tmp / "pyannote" / f"{split}.uem").write_text("".join(p["uem"]))
    with open(tmp / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (tmp / "database.yml").write_text(_database_yml(include_audio, ("test",) if eval_only else SPLITS))
    if review:
        (tmp / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1))
    stats = _stats(rows)
    stats["eval_only"] = eval_only
    stats["channels"] = len({r["channel_id"] or r["uri"] for r in rows})
    (tmp / "README.md").write_text(card(name, stats, include_audio, review, models or {}))

    final = out_root / name
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return {"path": str(final), "recordings": len(rows), "include_audio": include_audio, **stats}


def _stats(rows) -> dict:
    """Hours per split, and how much of the hard speech survived (pooled over recordings)."""
    ov = sum((r.get("coverage") or {}).get("overlap_seconds") or 0 for r in rows)
    ov_kept = sum(((r.get("coverage") or {}).get("overlap") or 0) * ((r.get("coverage") or {}).get("overlap_seconds") or 0)
                  for r in rows)
    out = {"coverage": {"overlap_seconds": round(ov, 1), "overlap_kept": round(ov_kept / ov, 4) if ov else None}}
    out["splits"] = {}
    for s in SPLITS:
        mine = [r for r in rows if r["split"] == s]
        out["splits"][s] = {"recordings": len(mine),
                  "labelled_hours": round(sum(r["labelled"] for r in mine) / 3600, 3),
                  "speech_hours": round(sum(r["labelled_speech"] for r in mine) / 3600, 3)}
    return out


def _database_yml(include_audio: bool, splits=SPLITS) -> str:
    db, task, proto = PROTOCOL.split(".")
    subsets = "".join(
        f"        {s}:\n"
        f"          uri: pyannote/{s}.lst\n"
        f"          annotation: pyannote/{s}.rttm\n"
        f"          annotated: pyannote/{s}.uem\n" for s in splits)
    head = f"Databases:\n  {db}: audio/{{uri}}.flac\n\n" if include_audio else (
        f"# no audio in this copy: fetch it with  python -m bench.evaluate --fetch-audio <this folder>\n"
        f"Databases:\n  {db}: audio/{{uri}}.flac\n\n")
    # speaker labels are per recording (R1 in one file is not R1 in another)
    return head + f"Protocols:\n  {db}:\n    {task}:\n      {proto}:\n        scope: file\n" + subsets


def card(name: str, stats: dict, include_audio: bool, review: dict | None, models: dict) -> str:
    rows = "\n".join(f"| {s} | {v['recordings']} | {v['labelled_hours']:.2f} | {v['speech_hours']:.2f} |"
                     for s, v in stats["splits"].items() if v["recordings"] or not stats.get("eval_only"))
    purpose = (f"**An evaluation set:** every recording is in `test` ({stats.get('channels', '?')} YouTube "
               "channels). Score systems with it; do not train on it." if stats.get("eval_only") else
               "Splits are fixed per YouTube channel (hash of the channel id), so the same hosts are never "
               "in two splits and a recording never changes split. Don't train on `test`.")
    rv = (review or {}).get("stats") or {}
    if rv.get("judged"):
        check = (f"{rv['correct']} of {rv['judged']} randomly sampled labelled windows were judged "
                 f"correct by ear ({rv['accuracy']:.0%}, 95% interval {rv['low']:.0%}–{rv['high']:.0%}).")
    else:
        check = "Not spot-checked yet."
    return f"""---
pretty_name: {name}
language:
- km
tags:
- speaker-diarization
- pseudo-labels
license: other
---

# {name}

Khmer speaker-diarization labels made **without manual annotation**, from YouTube videos.

- **Labels:** {models.get('reference', 'DiariZen')} turns (`rttm/`).
- **Kept regions** (`uem/`): only where a second, independent diarizer
  ({models.get('second', 'Nemotron-3-Diarization')}) has exactly the same speakers active after
  speaker mapping, at least 0.25 s away from any disagreement, and not music without speech
  (AudioSet AST classifier). Use the labels **only inside the UEM**.
- A recording is kept when at least half of the reference speech survives this filter, and it has
  at most 8 speakers (the second opinion's limit).

| split | recordings | labelled hours | speech hours |
|---|---|---|---|
{rows}

**Spot check:** {check}

**What the labels cover.** Two models agree most on the easy speech. Of the overlapped speech
DiariZen found ({stats['coverage']['overlap_seconds'] / 60:.1f} min), {f"{stats['coverage']['overlap_kept']:.0%}" if stats['coverage']['overlap_kept'] is not None else "none"} is kept, and speech right at
a turn change survives less often than the middle of a turn (per recording in
`metadata.jsonl`, `coverage`). A DER on these labels is a DER on the easier part of the audio:
use it to compare systems, not as the error on whole recordings.

{purpose}

**Don't score DiariZen or Nemotron-3-Diarization on it**: the labels come from them. Also check that
no system you score was trained on these videos.

## Use

```python
from pyannote.database import registry, FileFinder
registry.load_database("database.yml")
protocol = registry.get_protocol("{PROTOCOL}", preprocessors={{"audio": FileFinder()}})
for file in protocol.test():
    file["annotation"], file["annotated"]   # labels, and the regions to score
```

To evaluate a diarization system on the test split (from the Khmer diarization bench repo):

```bash
python -m bench.evaluate {name} --split test
```

{"Audio is included (16 kHz mono FLAC)." if include_audio else "Audio is **not** included: `python -m bench.evaluate --fetch-audio` downloads it from YouTube by `youtube_id` (metadata.jsonl)."}

## Licenses and caveats

- The audio belongs to its YouTube owners; this dataset is for private research. Keep the
  repository private.
- The labels come from DiariZen, whose weights are CC BY-NC 4.0: non-commercial use.
- Pseudo-labels have errors. The UEM removes the doubtful parts, but where both models make the
  same mistake the mistake stays; the spot check above estimates how often.

Built {time.strftime('%Y-%m-%d')}.
"""


def push(folder: str | Path, repo_id: str, token: str | None = None) -> str:
    """Upload a built dataset to a PRIVATE Hugging Face dataset repo. Refuses a public repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError(f"{repo_id} is public: this tool only pushes to private repos")
    # files of recordings no longer in the dataset are removed; audio only when this copy has it
    stale = ["rttm/*", "uem/*", "pyannote/*", *(["audio/*"] if (Path(folder) / "audio").is_dir() else [])]
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
                      commit_message=f"Update from the Khmer diarization bench ({time.strftime('%Y-%m-%d %H:%M')})",
                      delete_patterns=stale)
    return f"https://huggingface.co/datasets/{repo_id}"


def hf_user(token: str | None = None) -> str | None:
    """The logged-in Hugging Face user, or None (no token, or an invalid one)."""
    try:
        from huggingface_hub import whoami
        return whoami(token=token)["name"]
    except Exception:
        return None

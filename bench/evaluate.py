"""
Evaluate a diarization system on a dataset built by this tool (dataset.py), inside its UEM.

    python -m bench.evaluate exports/khmer-diarization-eval --split test
    python -m bench.evaluate your-user/khmer-diarization-eval --split test      # from the HF Hub
    python -m bench.evaluate DATASET --segmentation weights/new.ckpt              # another checkpoint
    python -m bench.evaluate DATASET --oracle-count                               # told the speaker count
    python -m bench.evaluate DATASET --hyp-dir some/rttms/                        # any system's RTTMs (<uri>.rttm)
    python -m bench.evaluate DATASET --fetch-audio                                # only download missing audio
    python -m bench.evaluate DATASET --hyp-dir new/ --compare eval-old.json       # is new better than old?

Prints per-recording, per-channel and pooled DER / JER (no collar and ±0.25 s), and writes them as
JSON (--out). The system's RTTMs are saved next to it, so they can be re-scored without re-running.

Recordings of one YouTube channel share hosts, studio and style, so they are not independent: the
unit of evidence is the channel. Next to the pooled DER (long recordings weigh more) it reports
the average over channels (each show counts once), and a 95% interval from resampling whole
channels. --compare gives the interval of the difference to an earlier run on the same
recordings: if it excludes 0, the difference is more than channel-to-channel luck.
"""

import argparse
import json
import logging
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

import soundfile as sf  # noqa: E402

from bench import audio, consensus, metrics, rttm, youtube  # noqa: E402
from bench.dataset import SPLITS  # noqa: E402

log = logging.getLogger("evaluate")


def resolve(dataset: str, token: str | None = None) -> Path:
    """A local dataset folder, or a Hugging Face dataset repo id downloaded to the HF cache."""
    p = Path(dataset).expanduser()
    if p.is_dir():
        return p
    if dataset.count("/") == 1 and not p.exists():
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(dataset, repo_type="dataset", token=token))
    raise SystemExit(f"no dataset at {dataset} (a folder, or a Hugging Face dataset repo id)")


def rows(root: Path, split: str | None) -> list[dict]:
    out = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines() if line]
    return [r for r in out if split in (None, "all", r["split"])]


def ensure_audio(root: Path, row: dict) -> Path:
    """The recording's audio; downloaded from YouTube when this copy of the dataset has none."""
    path = root / "audio" / f"{row['uri']}.flac"
    if path.is_file():
        return path
    if not row.get("youtube_id"):
        raise FileNotFoundError(f"{path} is missing and {row['uri']} has no YouTube id")
    path.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src, _ = youtube.download(row["youtube_id"], Path(tmp))
        wav = Path(tmp) / "audio.wav"
        duration = audio.transcode(src, wav)
        if abs(duration - row["duration"]) > 1.0:
            log.warning("%s: downloaded audio lasts %.1f s, the labels were made on %.1f s: "
                        "the video may have changed", row["uri"], duration, row["duration"])
        sf.write(path, audio.load(wav), audio.SAMPLE_RATE, format="FLAC")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="dataset folder or Hugging Face dataset repo id")
    p.add_argument("--split", default="test", choices=[*SPLITS, "all"])
    p.add_argument("--hyp-dir", help="score these RTTMs (<uri>.rttm) instead of running a system")
    p.add_argument("--segmentation", help="segmentation checkpoint (default: DIAR_SEGMENTATION)")
    p.add_argument("--device", help="default: DIAR_DEVICE")
    p.add_argument("--oracle-count", action="store_true", help="give the system the labels' speaker count")
    p.add_argument("--fetch-audio", action="store_true", help="only download missing audio, then stop")
    p.add_argument("--out", help="results JSON (default: eval-<split>-<time>.json in the current folder)")
    p.add_argument("--compare", help="an earlier results JSON: is this run better, beyond channel noise?")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from config import Settings
    settings = Settings.from_env()
    root = resolve(args.dataset, settings.hf_token)
    todo = rows(root, "all" if args.fetch_audio else args.split)
    if not todo:
        raise SystemExit(f"no recordings in split {args.split!r}")
    if args.fetch_audio:
        for r in todo:
            ensure_audio(root, r)
        print(f"audio ready for {len(todo)} recordings in {root / 'audio'}")
        return 0

    diarizer = None
    if not args.hyp_dir:
        from bench.diarize import Diarizer
        diarizer = Diarizer(args.segmentation or settings.segmentation,
                            args.device or settings.device, settings.hf_token)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else Path.cwd() / f"eval-{args.split}-{stamp}.json"
    hyp_out = out.with_suffix("")
    hyp_out.mkdir(parents=True, exist_ok=True)

    per_file, pooled_in = [], []
    for r in todo:
        uri = r["uri"]
        ref = rttm.read(root / "rttm" / f"{uri}.rttm")
        uem = consensus.read_uem(root / "uem" / f"{uri}.uem")
        if args.hyp_dir:
            hyp = rttm.read(Path(args.hyp_dir) / f"{uri}.rttm")
        else:
            t = time.time()
            wav = audio.load(ensure_audio(root, r))
            n = len(rttm.speakers(ref)) if args.oracle_count else None
            result = diarizer(wav, num_speakers=n)
            hyp = rttm.relabel([[x.start, x.end, x.speaker] for x in result.turns], "S")
            rttm.write(hyp, hyp_out / f"{uri}.rttm", uri=uri)
            log.info("%s: diarized in %.0f s", uri, time.time() - t)
        scores = {metrics.condition_key(c, s): metrics.score(ref, hyp, r["duration"], c, s, uem)
                  for c, s in metrics.CONDITIONS}
        pooled_in.append(scores)
        per_file.append({"uri": uri, "title": r["title"], "channel": r.get("channel") or uri,
                         "group": r.get("channel_id") or uri,
                         "ref_speakers": len(rttm.speakers(ref)),
                         "hyp_speakers": len(rttm.speakers(hyp)), "scores": scores})

    pooled = metrics.aggregate(pooled_in)
    stats = channel_stats(per_file)
    result = {"dataset": str(root), "split": args.split, "recordings": len(per_file),
              "system": args.hyp_dir or {"segmentation": args.segmentation or settings.segmentation,
                                         "oracle_count": args.oracle_count},
              "pooled": pooled, **stats, "per_file": per_file, "created": time.time()}
    if args.compare:
        result["compare"] = compare(per_file, json.loads(Path(args.compare).read_text()), args.compare)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    report(result, per_file, pooled, out, hyp_out if not args.hyp_dir else None)
    return 0


KEYS = (metrics.condition_key(0, False), metrics.condition_key(0.25, False))


def _groups(per_file) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for f in per_file:
        groups.setdefault(f.get("group") or f["uri"], []).append(f["scores"])
    return groups


def channel_stats(per_file) -> dict:
    """Per-channel pooled DER, the mean over channels, and channel-bootstrap intervals."""
    groups = _groups(per_file)
    names = {f.get("group") or f["uri"]: f.get("channel") or f["uri"] for f in per_file}
    out = {"per_channel": [], "channel_mean": {}, "interval": {}}
    for g, scores in groups.items():
        out["per_channel"].append({"group": g, "channel": names[g], "recordings": len(scores),
                                   "speech": round(sum(s[KEYS[0]]["speech"] for s in scores), 1),
                                   **{k: metrics.pooled_der(scores, k) for k in KEYS}})
    for k in KEYS:
        vals = [c[k] for c in out["per_channel"] if c[k] is not None]
        out["channel_mean"][k] = round(sum(vals) / len(vals), 5) if vals else None
        out["interval"][k] = metrics.cluster_bootstrap(groups, k)
    return out


def compare(per_file, earlier: dict, path: str) -> dict:
    """Paired channel bootstrap of DER(this) - DER(earlier), on the recordings both scored."""
    before = {f["uri"]: f for f in earlier["per_file"]}
    both = [f for f in per_file if f["uri"] in before]
    now_g = _groups(both)
    then_g = _groups([{**before[f["uri"]], "group": f.get("group")} for f in both])
    out = {"against": path, "recordings": len(both)}
    for k in KEYS:
        out[k] = {"this": metrics.pooled_der([f["scores"] for f in both], k),
                  "earlier": metrics.pooled_der([before[f["uri"]]["scores"] for f in both], k),
                  "difference": metrics.cluster_bootstrap(then_g, k, paired=now_g)}
    return out


def report(result, per_file, pooled, out, hyp_out):
    key, key25 = KEYS
    pct = lambda x: "   –  " if x is None else f"{x * 100:5.1f}%"  # noqa: E731
    print(f"\n{'recording':<40} {'spk ref/hyp':>11} {'DER':>7} {'±0.25s':>7} {'miss':>6} {'FA':>6} {'conf':>6}")
    for f in per_file:
        s, s25 = f["scores"][key], f["scores"][key25]
        sp = s["speech"] or 1
        print(f"{f['uri'][:40]:<40} {f['ref_speakers']:>5}/{f['hyp_speakers']:<5} {pct(s['der'])} {pct(s25['der'])} "
              f"{pct(s['missed'] / sp)} {pct(s['false_alarm'] / sp)} {pct(s['confusion'] / sp)}")
    s, s25 = pooled[key], pooled[key25]
    sp = s["speech"] or 1
    label = f"POOLED ({len(per_file)} recordings, {s['speech'] / 3600:.2f} h speech)"
    print(f"{label:<52} {pct(s['der'])} {pct(s25['der'])} {pct(s['missed'] / sp)} {pct(s['false_alarm'] / sp)} "
          f"{pct(s['confusion'] / sp)}   JER {pct(s['jer'])}")

    print(f"\n{'channel':<40} {'recs':>4} {'speech':>8} {'DER':>7} {'±0.25s':>7}")
    for c in sorted(result["per_channel"], key=lambda c: -c["speech"]):
        print(f"{c['channel'][:40]:<40} {c['recordings']:>4} {c['speech'] / 3600:6.2f} h {pct(c[key])} {pct(c[key25])}")
    cm = result["channel_mean"]
    print(f"{'MEAN OVER ' + str(len(result['per_channel'])) + ' CHANNELS (each show counts once)':<55} {pct(cm[key])} {pct(cm[key25])}")

    def ci(x):
        return f"95% {pct(x['low']).strip()} – {pct(x['high']).strip()}" if x else "needs 2+ channels"
    iv = result["interval"]
    print(f"\nPooled DER {pct(s['der']).strip()} ({ci(iv[key])}), ±0.25 s {pct(s25['der']).strip()} ({ci(iv[key25])})"
          f"  [channels resampled]")
    if len(result["per_channel"]) < 5:
        print(f"Only {len(result['per_channel'])} channels: the interval is rough; aim for 5+ shows.")
    if "compare" in result:
        c = result["compare"]
        print(f"\nVs {c['against']} ({c['recordings']} common recordings):")
        for k, name in ((key, "no collar"), (key25, "±0.25 s")):
            d = c[k]
            diff = d["difference"]
            verdict = ("" if not diff else " -> better" if diff["high"] < 0 else " -> worse" if diff["low"] > 0
                       else " -> no clear difference")
            span = (f"difference 95% {diff['low'] * 100:+.1f} to {diff['high'] * 100:+.1f} points, "
                    f"better in {diff['better_share']:.0%} of resamples" if diff else "needs 2+ channels")
            print(f"  {name:9}: {pct(d['earlier']).strip()} -> {pct(d['this']).strip()}  ({span}){verdict}")
    print(f"\nresults: {out}" + (f"\nsystem RTTMs: {hyp_out}" if hyp_out else ""))


if __name__ == "__main__":
    sys.exit(main())

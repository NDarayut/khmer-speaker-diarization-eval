"""Fast tests: no GPU, no network, no models."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bench import audio, metrics, rttm
from bench.diarize import Turn
from bench.store import Store
from config import Settings

SR = audio.SAMPLE_RATE


# ---------------------------------------------------------------- audio
def test_peaks_and_energy_shapes():
    a = np.sin(np.linspace(0, 200, SR * 2)).astype(np.float32) * 0.5
    p = audio.peaks(a)
    assert p.dtype == np.int8 and len(p) == 2 * 200
    assert p.max() == 127 and p.min() == -127
    assert len(audio.energy_db(a)) == 200


def test_transcode_and_read_span(tmp_path):
    import soundfile as sf
    src = tmp_path / "in.wav"
    sf.write(src, np.random.default_rng(0).normal(0, .1, (44100 * 2, 2)).astype(np.float32), 44100)
    dur = audio.transcode(src, tmp_path / "out.wav")
    assert abs(dur - 2.0) < 0.01
    assert len(audio.read_span(tmp_path / "out.wav", 0.5, 1.0)) == SR // 2


def test_embeddings_skip_silent_local_speakers():
    import torch
    from types import SimpleNamespace
    from pyannote.core import SlidingWindow, SlidingWindowFeature
    from bench.diarize import _CachingPipeline

    seg = np.zeros((3, 10, 4), dtype=np.float32)  # 3 chunks, 10 frames, 4 local speakers
    seg[0, :, 0] = 1
    seg[0, :4, 1] = seg[1, 2:, 1] = seg[2, :, 3] = 1
    seg[0, 1, 2] = np.nan  # partial stitching
    seen = []

    def embed(waveforms, masks):
        seen.append(len(masks))
        return np.stack([waveforms[:, 0, 0].numpy(), masks.sum(1).numpy()], axis=1)

    pipe = object.__new__(_CachingPipeline)
    pipe.embedding_batch_size = 2
    pipe._embedding = type("E", (), {"dimension": 2, "min_num_samples": 100, "sample_rate": 100,
                                     "__call__": lambda self, w, masks: embed(w, masks)})()
    pipe._audio = SimpleNamespace(crop=lambda file, chunk, mode: (torch.full((1, 1000), chunk.start), 1))
    windows = SlidingWindowFeature(seg, SlidingWindow(start=0, duration=10, step=5))
    emb = pipe.get_embeddings({}, windows)
    active = [(0, 0), (0, 1), (1, 1), (2, 3)]
    assert seen == [2, 2]  # 4 active pairs of 12, in batches of 2
    for c in range(3):
        for s in range(4):
            if (c, s) in active:
                assert emb[c, s].tolist() == [c * 5, np.nansum(seg[c, :, s])]  # its chunk, its mask
            else:
                assert np.isnan(emb[c, s]).all()
    # exclude_overlap: speaker 0 in chunk 0 keeps its 6 single-speaker frames (>= 1 frame needed)
    emb = pipe.get_embeddings({}, windows, exclude_overlap=True)
    assert emb[0, 0, 1] == 6 and emb[0, 1, 1] == 4  # speaker 1 only overlaps: whole mask


# ---------------------------------------------------------------- rttm
def test_rttm_round_trip_and_relabel(tmp_path):
    turns = rttm.relabel([(5.0, 6.0, "spk_b"), (0.0, 1.0, "spk_a"), (1.0, 4.0, "spk_b"), (2, 2, "x")], "R")
    assert turns == [[0.0, 1.0, "R2"], [1.0, 4.0, "R1"], [5.0, 6.0, "R1"]]  # by talk time, empty dropped
    rttm.write(turns, tmp_path / "a.rttm")
    assert rttm.read(tmp_path / "a.rttm") == turns
    assert rttm.speakers(turns) == ["R1", "R2"]


# ---------------------------------------------------------------- metrics
def _der(ref, hyp, dur=10.0, **kw):
    return metrics.score(ref, hyp, dur, **kw)


def test_perfect_and_swapped_labels_score_zero():
    ref = [[0, 5, "A"], [5, 10, "B"]]
    assert _der(ref, ref)["der"] == 0
    swapped = [[0, 5, "B"], [5, 10, "A"]]
    assert _der(ref, swapped)["der"] == 0
    assert metrics.mapping(ref, swapped, 10) == {"B": "A", "A": "B"}


def test_missed_and_false_alarm():
    s = _der([[0, 10, "A"]], [[0, 5, "x"]])
    assert s["missed"] == pytest.approx(5) and s["der"] == pytest.approx(0.5)
    s = _der([[0, 5, "A"]], [[0, 5, "x"], [5, 7, "y"]])
    assert s["false_alarm"] == pytest.approx(2) and s["der"] == pytest.approx(0.4)


def test_confusion_track_and_per_speaker():
    ref = [[0, 4, "A"], [4, 8, "B"]]
    hyp = [[0, 6, "X"], [6, 8, "Y"]]
    m = metrics.evaluate(ref, hyp, 10)
    assert m["mapping"] == {"X": "A", "Y": "B"}
    s = m["scores"]["collar=0"]
    assert s["confusion"] == pytest.approx(2) and s["der"] == pytest.approx(0.25)
    assert m["errors"] == [[4.0, 6.0, "confusion"]]
    assert m["frame_der"] == pytest.approx(0.25)
    b = next(p for p in m["per_speaker"] if p["speaker"] == "B")
    assert b == {"speaker": "B", "mapped_to": "Y", "speech": 4.0, "found": 2.0, "confused": 2.0, "missed": 0.0}


def test_error_track_kinds():
    ref = [[0, 2, "A"], [4, 6, "A"]]
    hyp = [[0, 1, "x"], [3, 6, "x"]]
    assert metrics.evaluate(ref, hyp, 8)["errors"] == [[1.0, 2.0, "miss"], [3.0, 4.0, "fa"]]


def test_collar_forgives_boundary_shifts():
    ref = [[0, 5, "A"], [5, 10, "B"]]
    hyp = [[0, 5.2, "A"], [5.2, 10, "B"]]
    assert _der(ref, hyp)["der"] == pytest.approx(0.02)
    assert _der(ref, hyp, collar=0.25)["der"] == pytest.approx(0)


def test_frame_der_matches_pyannote():
    rng = np.random.default_rng(1)
    def rand_turns(labels):
        out, t = [], 0.0
        while t < 55:
            d = float(rng.uniform(0.5, 4))
            out.append([round(t, 2), round(t + d, 2), str(rng.choice(labels))])
            t += d + float(rng.uniform(-0.3, 1.0))  # overlaps and pauses
        return out
    ref, hyp = rand_turns(["A", "B", "C"]), rand_turns(["x", "y"])
    m = metrics.evaluate(ref, hyp, 60)
    assert m["frame_der"] == pytest.approx(m["scores"]["collar=0"]["der"], abs=0.01)


def test_aggregate_weights_by_speech():
    a = {"collar=0": _der([[0, 10, "A"]], [[0, 5, "x"]])}           # 5 s missed of 10
    b = {"collar=0": _der([[0, 30, "A"]], [[0, 30, "x"]], dur=30)}  # perfect, 30 s
    pooled = metrics.aggregate([a, b])["collar=0"]
    assert pooled["der"] == pytest.approx(5 / 40) and pooled["recordings"] == 2

# ---------------------------------------------------------------- consensus
def test_consensus_keeps_only_agreement_away_from_music():
    from bench import consensus
    ref = [[0, 4, "R1"], [4, 8, "R2"]]
    second = [[0, 4, "N1"], [4, 6, "N2"], [6, 8, "N1"]]  # disagrees 6-8
    c = consensus.build(ref, second, 10, music_regions=[[0, 1]])
    assert c["mapping"] == {"N1": "R1", "N2": "R2"}
    assert c["uem"] == [[1.0, 5.75], [8.25, 10.0]]
    assert c["dropped"]["music"] == 1.0 and c["dropped"]["disagreement"] == 2.0
    assert c["status"] == "accepted" and c["speech_kept"] == pytest.approx(4.75)
    # little agreement -> rejected; too many speakers -> rejected; no second opinion -> rejected
    assert consensus.build(ref, [[0, 8, "N1"]], 10)["status"] == "rejected"
    many = [[i, i + 1, f"R{i}"] for i in range(9)]
    assert "more than" in consensus.build(many, many, 10)["reason"]
    assert consensus.build(ref, None, 10)["reason"] == "no second opinion"


def test_consensus_tolerates_boundary_jitter_only():
    from bench import consensus
    ref = [[0, 4, "R1"], [4, 8, "R2"]]
    shifted = [[0, 4.1, "N1"], [4.1, 8, "N2"]]  # the same turn change, 0.1 s later
    c = consensus.build(ref, shifted, 10)
    assert c["uem"] == [[0.0, 10.0]] and c["jitter_tolerated"] == pytest.approx(0.1)
    # a 0.2 s turn only the second opinion has (no change in the reference there): dropped
    extra = [[0, 2, "N1"], [2, 2.2, "N2"], [2.2, 4, "N1"], [4, 8, "N2"]]
    c = consensus.build(ref, extra, 10)
    assert c["uem"] == [[0.0, 1.75], [2.45, 10.0]]
    assert c["coverage"]["overlap"] is None and c["coverage"]["speech"] == pytest.approx(7.3 / 8, abs=1e-3)


def test_vad_settles_speech_silence_disputes_for_diarizen_only():
    from bench import consensus
    ref = [[0, 4, "R1"], [4, 8, "R2"]]
    second = [[0, 2, "N1"], [4, 8, "N2"], [8, 9, "N2"]]  # silent 2-4 s; speech 8-9 s (DiariZen: none)
    vad = np.zeros(1000, np.float32)
    vad[200:400] = 0.9   # 2-4 s: the VAD hears speech -> DiariZen's R1 confirmed
    vad[800:900] = 0.1   # 8-9 s: the VAD hears none -> DiariZen's silence confirmed
    c = consensus.build(ref, second, 10, vad=vad)
    assert c["uem"] == [[0.0, 10.0]] and c["vad_resolved"] == pytest.approx(3.0)
    # the VAD siding with Nemotron settles nothing: those frames are dropped as before
    c = consensus.build(ref, second, 10, vad=np.where(vad > 0.5, 0.1, 0.9).astype(np.float32))
    assert c["vad_resolved"] == 0 and c["uem"] == [[0.0, 1.75], [4.25, 7.75]]
    # a speaker dispute (both speak, different people) is never settled by the VAD
    c = consensus.build(ref, [[0, 8, "N1"]], 10, vad=np.full(1000, 0.9, np.float32))
    assert c["vad_resolved"] == 0


def test_consensus_drops_short_pieces():
    from bench import consensus
    ref = [[0, 10, "R1"]]
    second = [[0, 3, "N1"], [4, 10, "N1"]]  # silent 3-4 -> disagreement; a 0.5 s island 3.0-3.5?
    c = consensus.build(ref, second, 10, min_piece=3.0)
    assert c["uem"] == [[4.25, 10.0]]  # 0-2.75 is shorter than 3 s
    assert c["dropped"]["short"] == pytest.approx(2.75)


def test_music_only_regions():
    from bench.nonspeech import regions
    music = np.array([.9, .9, .9, .1, .9, .9, .9, .9, .2, .1])
    speech = np.array([.1, .1, .1, .1, .1, .9, .1, .1, .1, .1])
    # second 5 has speech over music: not music-only; runs 0-3 and 4-5, 6-8 merge across short gaps
    assert regions(music, speech, merge_gap=1.5) == [[0.0, 8.0]]
    assert regions(music, speech, merge_gap=0.5, min_run=2.0) == [[0.0, 3.0], [6.0, 8.0]]


def test_splits_follow_the_channel():
    from bench import dataset
    a = {"youtube": {"channel_id": "UCabc"}}
    splits = {dataset.split_of(f"yt-video{i}", a) for i in range(30)}
    assert len(splits) == 1  # every video of a channel in one split
    assert dataset.split_of("yt-x", {**a, "split": "test"}) == "test"  # pinned
    assert len({dataset.split_of(f"yt-video{i}", {}) for i in range(60)}) == 3  # no channel: per video


def test_review_samples_kept_regions_uniformly(tmp_path):
    from bench import review
    st = Store(tmp_path)
    eligible = [({"id": "a"}, {"uem": [[0, 10], [20, 21], [30, 130]]})]  # 1 s piece: too short
    doc = review.sample(st, eligible, 40, window=2, seed=1)
    assert len(doc["items"]) == 40
    for i in doc["items"]:
        assert (0 <= i["start"] and i["end"] <= 10) or (30 <= i["start"] and i["end"] <= 130)
    # ~100/110 of the windows fall in the long piece
    assert sum(i["start"] >= 30 for i in doc["items"]) >= 30
    review.judge(st, doc["items"][0]["id"], "wrong")
    assert review.stats(review.load(st))["wrong"] == 1
    review.forget_recording(st, "a")
    assert review.load(st)["items"] == []


def test_wilson_interval():
    from bench.review import wilson
    lo, hi = wilson(27, 30)
    assert 0.73 < lo < 0.75 and 0.96 < hi < 0.98
    assert wilson(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------- server
class FakeExternal:
    """Stands in for DiariZen / Nemotron: returns fixed turns (or raises)."""
    device = "cpu"
    available = True

    def __init__(self, turns, model="fake/model"):
        self.turns, self.model, self.calls = turns, model, []

    def start(self, wav, out, progress=None, **count):
        self.calls.append(count)
        turns = self.turns

        def wait():
            if isinstance(turns, Exception):
                raise turns
            return turns
        return SimpleNamespace(wait=wait, kill=lambda: None)


REF_TURNS = [[0, 4, "a"], [4, 8, "b"]]
SECOND_TURNS = [[0, 4, "n0"], [4, 6, "n1"], [6, 8, "n0"]]  # disagrees with the reference 6-8 s


def fake_system(wav, num_speakers=None, cache=None, progress=None, **kw):
    """Stands in for the system: with its own count it hears one speaker too long (confusion
    4-6 s); told there are 2 speakers it is right."""
    turns = ([Turn(0, 4, "x"), Turn(4, 8, "y")] if num_speakers == 2
             else [Turn(0, 6, "x"), Turn(6, 8, "y")])
    if progress:
        progress(1.0)
    return SimpleNamespace(turns=turns, cache=None)


def fake_music(wav, progress=None):
    return {"hop": 1.0, "music": [], "speech": [], "regions": []}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import soundfile as sf
    from bench import youtube
    for k, sub in (("BENCH_DATA_DIR", "data"), ("BENCH_INBOX_DIR", "inbox"), ("BENCH_EXPORT_DIR", "exports")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(k, str(tmp_path / sub))
    monkeypatch.setenv("DIAR_DEVICE", "cpu")
    monkeypatch.setenv("DIARIZEN_DEVICE", "cpu")
    monkeypatch.setenv("NEMOTRON_DEVICE", "cpu")

    def fake_download(vid, dest_dir, progress=None, info=None):
        path = dest_dir / "source.wav"
        sf.write(path, np.zeros(SR * 10, dtype=np.float32), SR)
        if info is not None:
            info.update(channel="Khmer Talk", channel_id=f"UC{vid[:3]}", language=None)
        return path, f"Khmer talk {vid}"

    monkeypatch.setattr(youtube, "download", fake_download)
    from server.app import app
    from server import app as server
    with TestClient(app) as c:
        jobs = server.state.jobs  # no models in these tests
        monkeypatch.setattr(jobs, "diarizer", fake_system)
        monkeypatch.setattr(jobs, "reference", FakeExternal(REF_TURNS, "fake/diarizen"))
        monkeypatch.setattr(jobs, "second", FakeExternal(SECOND_TURNS, "fake/nemotron"))
        monkeypatch.setattr(jobs, "music", fake_music)
        monkeypatch.setattr(jobs, "vad", None)
        c.jobs = jobs
        yield c


def test_inbox_lists_media_files_only(client, tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "a.m4a").write_bytes(b"x")
    (inbox / "sub").mkdir()
    (inbox / "sub" / "b.WAV").write_bytes(b"xy")
    (inbox / "notes.txt").write_text("no")
    (inbox / ".hidden.mp3").write_bytes(b"no")
    files = client.get("/api/inbox").json()["files"]
    assert [f["name"] for f in files] == ["a.m4a", "sub/b.WAV"]
    assert files[1]["path"] == str(inbox / "sub" / "b.WAV") and files[1]["bytes"] == 2


@pytest.mark.parametrize("url, vid", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "jNQXAC9IVRw"),
    ("https://youtube.com/watch?feature=share&v=jNQXAC9IVRw&t=30s", "jNQXAC9IVRw"),
    ("  https://youtu.be/jNQXAC9IVRw?si=abc ", "jNQXAC9IVRw"),
    ("https://m.youtube.com/shorts/jNQXAC9IVRw", "jNQXAC9IVRw"),
    ("https://www.youtube.com/live/jNQXAC9IVRw", "jNQXAC9IVRw"),
    ("https://www.youtube.com/playlist?list=PL123", None),
    ("https://www.youtube.com/watch?v=short", None),
    ("https://evil.com/watch?v=jNQXAC9IVRw", None),
    ("https://youtube.com.evil.com/watch?v=jNQXAC9IVRw", None),
    ("file:///etc/passwd", None),
])
def test_youtube_video_id(url, vid):
    from bench import youtube
    assert youtube.video_id(url) == vid



def _wait(client, job_id):
    for _ in range(400):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _import(client, vid="jNQXAC9IVRw", **data):
    r = client.post("/api/recordings", data={"url": f"https://youtu.be/{vid}", **data})
    assert r.status_code == 200, r.text
    r = r.json()
    job = _wait(client, r["job"]["id"])
    assert job["stage"] == "done", job["error"]
    return r["id"]


def test_youtube_import_runs_all_systems_labels_and_scores(client, tmp_path):
    assert client.post("/api/recordings", data={"url": "https://evil.com/x"}).status_code == 400
    rid = _import(client, max_speakers="4")
    assert client.jobs.reference.calls == [{"num_speakers": None, "min_speakers": None, "max_speakers": 4}]
    assert client.post("/api/recordings", data={"url": "https://youtu.be/jNQXAC9IVRw"}).status_code == 409

    rec = client.get(f"/api/recordings/{rid}").json()
    meta, m = rec["meta"], rec["metrics"]
    assert meta["youtube"] == {"channel": "Khmer Talk", "channel_id": "UCjNQ"}
    assert meta["name"] == "Khmer talk jNQXAC9IVRw" and meta["duration"] == 10.0 and meta["status"] == "ready"
    d = tmp_path / "data" / rid
    assert (d / "audio.wav").is_file() and not (d / "source.wav").exists()
    assert rec["turns"]["reference"] == [[0.0, 4.0, "R1"], [4.0, 8.0, "R2"]]  # relabelled
    assert rec["turns"]["second"][0][2] == "N1"
    # against DiariZen, whole file
    assert m["system"]["scores"]["collar=0"]["der"] == pytest.approx(0.25)
    assert m["system_oracle"]["scores"]["collar=0"]["der"] == 0
    assert m["second"]["scores"]["collar=0"]["der"] == pytest.approx(0.25)
    # labels: agreement 0-5.75 and 8.25-10; the system is scored there only
    assert rec["labels"]["status"] == "accepted" and rec["labels"]["uem"] == [[0.0, 5.75], [8.25, 10.0]]
    vs = m["vs_labels"]["system"]["scores"]["collar=0"]
    assert vs["speech"] == pytest.approx(5.75) and vs["confusion"] == pytest.approx(1.75)
    assert meta["scores"]["labels"]["system"]["der"] == pytest.approx(1.75 / 5.75, abs=1e-4)
    assert client.get(f"/api/recordings/{rid}/rttm/system").text.startswith("SPEAKER rec 1 0.000 6.000")
    assert client.get(f"/api/recordings/{rid}/rttm/labels.uem").text.startswith("rec 1 0.000 5.750")
    assert client.get(f"/api/recordings/{rid}/rttm/../meta.json").status_code == 404

    s = client.get("/api/summary").json()
    assert s["recordings"] == 1 and s["system"]["collar=0"]["der"] == pytest.approx(0.25)
    assert s["labels"]["accepted"] == 1 and s["labels"]["system"]["collar=0"]["der"] == pytest.approx(1.75 / 5.75, abs=1e-4)

    # re-running the reference redoes the oracle and the labels, with the import's options
    job = _wait(client, client.post(f"/api/recordings/{rid}/rerun", json={"system": "reference"}).json()["job"]["id"])
    assert job["stage"] == "done", job["error"]
    assert client.jobs.reference.calls[-1]["max_speakers"] == 4
    assert client.post(f"/api/recordings/{rid}/rerun", json={"system": "x"}).status_code == 400
    q = client.post("/api/rerun", json={"system": "second", "only_missing": True}).json()
    assert q["queued"] == []  # it has one


def test_reference_failure_is_reported(client):
    client.jobs.reference = FakeExternal(RuntimeError("reference diarizer (DiariZen) failed: CUDA out of memory"))
    r = client.post("/api/recordings", data={"url": "https://youtu.be/jNQXAC9IVRw"}).json()
    job = _wait(client, r["job"]["id"])
    assert job["stage"] == "error" and "out of memory" in job["error"]
    rec = client.get(f"/api/recordings/{r['id']}").json()
    assert rec["meta"]["status"] == "error" and rec["turns"]["system"] is not None  # system result kept


def test_second_opinion_failure_only_rejects_the_labels(client):
    client.jobs.second = FakeExternal(RuntimeError("second opinion (Nemotron) failed: boom"))
    rid = _import(client)
    rec = client.get(f"/api/recordings/{rid}").json()
    assert rec["meta"]["status"] == "ready" and "boom" in rec["meta"]["warnings"][0]
    assert rec["labels"]["status"] == "rejected" and rec["labels"]["reason"] == "no second opinion"
    assert "vs_labels" not in rec["metrics"]
    # once Nemotron works, only_missing re-runs just the recordings without it
    client.jobs.second = FakeExternal(SECOND_TURNS)
    q = client.post("/api/rerun", json={"system": "second", "only_missing": True}).json()
    assert q["queued"] == [rid]
    assert _wait(client, client.jobs.latest_for(rid)["id"])["stage"] == "done"
    assert client.get(f"/api/recordings/{rid}").json()["labels"]["status"] == "accepted"


def test_playlist_import_skips_known_short_and_long(client, monkeypatch):
    from bench import youtube
    _import(client, vid="aaaaaaaaaaa")
    monkeypatch.setattr(youtube, "expand", lambda url, limit=50: [
        {"id": "aaaaaaaaaaa", "title": "known", "duration": 600},
        {"id": "bbbbbbbbbbb", "title": "ok", "duration": 600},
        {"id": "ccccccccccc", "title": "short", "duration": 20},
        {"id": "ddddddddddd", "title": "long", "duration": 5 * 3600},
    ])
    r = client.post("/api/recordings", data={"url": "https://www.youtube.com/@SomeKhmerChannel"}).json()
    assert r["collection"] == "https://www.youtube.com/@SomeKhmerChannel/videos"
    assert [s["reason"] for s in r["skipped"]] == ["already in the library", "too short", "too long"]
    assert len(r["created"]) == 1
    assert _wait(client, client.jobs.latest_for(r["created"][0])["id"])["stage"] == "done"


def _wait_task(client):
    for _ in range(400):
        t = client.get("/api/dataset").json()["task"]
        if t and t["state"] != "running":
            return t
        time.sleep(0.05)
    raise AssertionError("dataset task did not finish")


def test_review_export_and_evaluate(client, tmp_path):
    from bench import evaluate, rttm as rttm_mod
    rid = _import(client)
    client.patch(f"/api/recordings/{rid}", json={"split": "test"})
    assert client.patch(f"/api/recordings/{rid}", json={"split": "dev"}).status_code == 400

    # spot check: only one distinct 8 s window fits this 10 s recording's kept regions
    doc = client.post("/api/review/sample", json={"n": 4}).json()
    assert [(i["start"], i["end"]) for i in doc["items"]] == [(0.0, 5.75)]
    assert client.put(f"/api/review/{doc['items'][0]['id']}", json={"verdict": "maybe"}).status_code == 400
    client.put(f"/api/review/{doc['items'][0]['id']}", json={"verdict": "correct"})
    st = client.get("/api/review").json()["stats"]
    assert st["judged"] == 1 and st["correct"] == 1

    info = client.get("/api/dataset").json()
    assert info["splits"]["test"]["recordings"] == 1 and info["review"]["judged"] == 1
    assert client.post("/api/dataset/export", json={"name": "../x"}).status_code == 200
    assert "plain folder name" in _wait_task(client)["error"]
    client.post("/api/dataset/export", json={"name": "khm", "include_audio": True})
    t = _wait_task(client)
    assert t["state"] == "done", t["error"]
    root = tmp_path / "exports" / "khm"
    uri = "yt-jNQXAC9IVRw"
    assert (root / "audio" / f"{uri}.flac").is_file()
    assert (root / "uem" / f"{uri}.uem").read_text() == f"{uri} 1 0.000 5.750\n{uri} 1 8.250 10.000\n"
    assert "1 of 1" in (root / "README.md").read_text()
    assert client.post("/api/dataset/push", json={"name": "nope"}).status_code == 400

    # the pyannote protocol loads
    from pyannote.database import registry, FileFinder
    registry.load_database(str(root / "database.yml"))
    proto = registry.get_protocol("KhmerPseudo.SpeakerDiarization.YouTube", preprocessors={"audio": FileFinder()})
    files = list(proto.test())
    assert [f["uri"] for f in files] == [uri] and files[0]["annotated"].duration() == pytest.approx(7.5)

    # evaluate: precomputed RTTMs, scored inside the UEM
    hyp = tmp_path / "hyp"
    hyp.mkdir()
    rttm_mod.write([[0, 6, "x"], [6, 8, "y"]], hyp / f"{uri}.rttm", uri=uri)
    out = tmp_path / "eval.json"
    assert evaluate.main([str(root), "--split", "test", "--hyp-dir", str(hyp), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["recordings"] == 1
    assert res["pooled"]["collar=0"]["der"] == pytest.approx(1.75 / 5.75, abs=1e-4)



# ---------------------------------------------------------------- store / settings
def test_store_rejects_bad_ids(tmp_path):
    st = Store(tmp_path)
    rid = st.create("meeting one.m4a", "/x/meeting one.m4a")
    assert st.meta(rid)["name"] == "meeting one" and st.exists(rid)
    with pytest.raises(KeyError):
        st.dir("../etc")
    assert not st.exists("../etc")


def test_settings_never_print_secrets(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_secret\nDIARIZEN_DEVICE=cuda:1\n")
    for k in ("HF_TOKEN", "DIARIZEN_DEVICE"):
        monkeypatch.delenv(k, raising=False)
    s = Settings.from_env(env)
    assert s.hf_token == "hf_secret" and s.diarizen_device == "cuda:1"
    assert "hf_secret" not in repr(s)


def test_settings_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_INBOX_DIR", "/work/inbox")
    monkeypatch.setenv("BENCH_HOST", "0.0.0.0")
    s = Settings.from_env(tmp_path / "none.env")
    assert s.inbox_dir == "/work/inbox" and s.host == "0.0.0.0"


def test_env_example_has_no_values():
    from config import ROOT
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.startswith("HF_TOKEN"):
            assert line.split("=", 1)[1] == ""


def test_reference_device_and_command(tmp_path, monkeypatch):
    import subprocess
    from bench import reference
    monkeypatch.setattr(reference, "gpu_count", lambda: 2)
    assert reference.resolve_device("auto") == "cuda:1"
    monkeypatch.setattr(reference, "gpu_count", lambda: 0)
    assert reference.resolve_device("auto") == "cpu"
    seen = {}

    class P:
        stdout = iter(["@stage load\n", "Extracting Embeddings.\n", "@stage done\n"])
        def __init__(self, cmd, env, **kw):
            seen.update(cmd=cmd, env=env)
            (tmp_path / "out.rttm").write_text("SPEAKER rec 1 1.000 2.000 <NA> <NA> spk0 <NA> <NA>\n")
        def wait(self): return 0
        def poll(self): return 0

    python = tmp_path / "python"
    python.write_text("")
    monkeypatch.setattr(subprocess, "Popen", P)
    progress = []
    run = reference.Reference(str(python), "m", "cuda:1").start(
        tmp_path / "a.wav", tmp_path / "out.rttm", num_speakers=3, progress=progress.append)
    assert run.wait() == [[1.0, 3.0, "spk0"]]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert seen["cmd"][-2:] == ["--num-speakers", "3"]
    assert progress == [0.02, 0.45, 1.0]


# ---------------------------------------------------------------- evaluation statistics
def _s(speech, err):
    return {"collar=0": {"speech": speech, "missed": err, "false_alarm": 0.0, "confusion": 0.0},
            "collar=0.25": {"speech": speech, "missed": err / 2, "false_alarm": 0.0, "confusion": 0.0}}


def test_cluster_bootstrap_and_paired_difference():
    groups = {f"ch{i}": [_s(100, 10 + i)] for i in range(6)}  # DER 10-15% per channel
    iv = metrics.cluster_bootstrap(groups, "collar=0")
    assert 0.10 <= iv["low"] <= 0.125 <= iv["high"] <= 0.15 and iv["groups"] == 6
    assert metrics.cluster_bootstrap({"one": [_s(100, 10)]}, "collar=0") is None
    better = {g: [_s(100, s[0]["collar=0"]["missed"] - 3)] for g, s in groups.items()}  # 3 points better everywhere
    d = metrics.cluster_bootstrap(groups, "collar=0", paired=better)
    assert d["high"] < 0 and d["better_share"] == 1.0


def test_evaluate_reports_channels_and_compares(tmp_path, capsys):
    import json as js
    from bench import evaluate
    root = tmp_path / "ds"
    (root / "rttm").mkdir(parents=True)
    (root / "uem").mkdir()
    rows = []
    for i in range(4):
        uri = f"yt-v{i}"
        rttm.write([[0, 5, "A"], [5, 10, "B"]], root / "rttm" / f"{uri}.rttm", uri=uri)
        (root / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 10.000\n")
        rows.append({"uri": uri, "split": "test", "title": uri, "duration": 10.0,
                     "channel": f"Show {i % 2}", "channel_id": f"UC{i % 2}"})
    (root / "metadata.jsonl").write_text("".join(js.dumps(r) + "\n" for r in rows))
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    for r in rows:
        rttm.write([[0, 7, "x"], [7, 10, "y"]], old / f"{r['uri']}.rttm")   # 2 s confusion
        rttm.write([[0, 5.5, "x"], [5.5, 10, "y"]], new / f"{r['uri']}.rttm")  # 0.5 s confusion
    evaluate.main([str(root), "--hyp-dir", str(old), "--out", str(tmp_path / "old.json")])
    res = js.loads((tmp_path / "old.json").read_text())
    assert len(res["per_channel"]) == 2 and res["channel_mean"]["collar=0"] == pytest.approx(0.2)
    evaluate.main([str(root), "--hyp-dir", str(new), "--out", str(tmp_path / "new.json"),
                   "--compare", str(tmp_path / "old.json")])
    c = js.loads((tmp_path / "new.json").read_text())["compare"]
    assert c["recordings"] == 4 and c["collar=0"]["difference"]["high"] < 0
    assert "-> better" in capsys.readouterr().out


def test_eval_only_export_puts_everything_in_test(client, tmp_path):
    from bench import dataset
    from server import app as server
    _import(client)
    _import(client, vid="bbbbbbbbbbb")
    res = dataset.export(server.state.store, tmp_path / "ex", "evalset", include_audio=False, eval_only=True)
    assert res["splits"]["test"]["recordings"] == 2 and res["splits"]["train"]["recordings"] == 0
    root = tmp_path / "ex" / "evalset"
    assert not (root / "pyannote" / "train.lst").exists() and "train:" not in (root / "database.yml").read_text()
    assert "An evaluation set" in (root / "README.md").read_text()


def test_interrupted_imports_resume_on_startup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import soundfile as sf
    from bench import youtube
    from bench.store import Store
    for k, sub in (("BENCH_DATA_DIR", "data"), ("BENCH_INBOX_DIR", "inbox"), ("BENCH_EXPORT_DIR", "exports")):
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(k, str(tmp_path / sub))
    st = Store(tmp_path / "data")
    a = st.create("Some talk", youtube.canonical_url("aaaaaaaaaaa"))            # interrupted
    b = st.create("Other talk", youtube.canonical_url("bbbbbbbbbbb"))
    st.update_meta(b, status="error", error="DownloadError: HTTP Error 403")      # failed download
    c = st.create("gone.wav", "upload:gone.wav")                                  # nothing to resume from
    seen = []
    monkeypatch.setattr(youtube, "download", lambda vid, d, progress=None, info=None: (
        seen.append(vid), sf.write(d / "s.wav", np.zeros(SR, dtype=np.float32), SR), (d / "s.wav", "t"))[2])
    from server.app import app
    with TestClient(app):
        from server import app as server
        assert {server.state.jobs.latest_for(r)["rec_id"] for r in (a, b)} == {a, b}
        assert server.state.jobs.latest_for(c) is None
        assert st.meta(a)["name"] == "Some talk"  # a batch title is kept

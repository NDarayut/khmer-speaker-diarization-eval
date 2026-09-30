# Khmer Speaker Diarization Eval

Build a Khmer speaker-diarization **evaluation set from YouTube without listening to all of it**,
and score diarization systems on it.

1. Paste a YouTube video, playlist or channel link (or pick local files).
2. **Labels:** [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
   (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`) finds who speaks when.
3. **Cross-check:** [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
   (NVIDIA's Sortformer successor, up to 8 speakers) diarizes the same audio independently, and
   an AudioSet classifier finds music.
4. **Kept:** only the regions where the two diarizers agree and there is no music. When they
   disagree only on *whether* someone speaks, [Silero VAD](https://github.com/snakers4/silero-vad)
   breaks the tie in DiariZen's favour or the region is dropped. That is a UEM next to DiariZen's
   RTTM, and a recording is accepted when most of its speech is kept.
5. **Your system** is scored against those labels (and against DiariZen on the whole file):
   `weights/segmentation_khmer.ckpt`, with everything else from pyannote Community-1
   (embedding, PLDA, VBx clustering, its default parameters).
6. **Spot check:** listen to about 30 random labelled windows to estimate how often the labels
   are wrong.
7. **Dataset:** build it (audio + RTTM + UEM + pyannote protocol), push it to a private Hugging
   Face repo, and evaluate any checkpoint on it with one command. The scores come per recording,
   per channel and pooled, with channel-bootstrap intervals and a paired comparison between two
   checkpoints.

Everything except the Hugging Face upload runs on this machine. With two GPUs, DiariZen runs on
one GPU, and Nemotron plus your system run on the other, all at the same time.

## What to expect from the labels

**Current build:** 40 videos from 15 channels (podcasts, interviews, a talk show, panels, news).
38 were accepted, with **20.0 h of labelled speech**; 86.7% of DiariZen's speech is kept.
- Agreement: on accepted videos, DiariZen and Nemotron are typically 1–6% DER apart.
- A rejected example: a 6-speaker news report where the two models were 41% DER apart, and only
  49% of the speech agreed. That is the filter doing its job.

**Where the two labellers disagree, and who is right.** Most disagreement is about *whether*
someone speaks, not *who*: Nemotron marks 6% of DiariZen's speech as silence, the reverse is
2.4%, and speaker mix-ups are 1.3%. Silero VAD (a dedicated speech detector, independent of
all three systems) was measured on 16 recordings from 14 channels. It hears speech in 96.9% of
the frames both call speech, and in 16.2% of the frames both call silence:

| Disputed frames | Silero hears speech | Reading |
|---|---|---|
| DiariZen: speech, Nemotron: silence | 55.0% | about half is real speech: Nemotron misses speech |
| Nemotron: speech, DiariZen: silence | 29.1% | ~84% is really silence: DiariZen is right |

So Silero settles these disputes, only when it sides with DiariZen:
- Silence confirmed (Silero < 0.5): about **99%** reliable.
- Speech confirmed (Silero >= 0.5): about **85%** reliable. It recovered 36 minutes, at an
  estimated cost of ~0.4 DER points of label noise over the set.

**The labels are biased toward easy speech.** Two models rarely agree on every speaker at once,
so **overlapped speech mostly drops out**: about 30% of it survived on a 43-minute panel talk.
Speech right at a turn change survives less often than the middle of a turn (58% there).
- A DER against these labels is a DER on the easier part of the audio. Use it to *compare*
  systems and checkpoints, not as the error on whole recordings.
- The same panel talk shows the gap: your system scores 3.0% against the labels, but 43%
  against DiariZen in the dropped regions.
- The page and the dataset card report this coverage for every recording.

Small boundary disagreements are not held against the labels. If both models change speaker
within 0.25 s of each other, that is the same boundary a little shifted, and DiariZen's timing
is used. The ±0.25 s collar in standard scoring forgives the same thing.

## Why these two models

| DER %, no collar | pyannote 3.1 | DiariZen-large-v2 |
|---|---|---|
| AMI (SDM) | 22.4 | 13.9 |
| AISHELL-4 | 12.2 | 10.1 |
| AliMeeting (far) | 24.4 | 10.8 |
| MSDWild | 25.3 | 15.8 |
| DIHARD 3 (full) | 21.7 | 14.5 |
| VoxConverse | 11.3 | 9.1 |

- **DiariZen** has the lowest published error of the open diarizers. These numbers are from
  its README; it also beats Community-1 on these sets. It has no speaker limit.
- **Nemotron-3-Diarization** (Sep 2026, OpenMDW license) is an end-to-end Sortformer. Nothing in
  it is shared with DiariZen (WavLM segmentation + WeSpeaker embedding + VBx) or with your
  system, so their agreement is real evidence. NVIDIA reports DIHARD III DER 12.7%, against 19.1%
  for Sortformer v2.1. It handles at most 8 speakers, so recordings with more are rejected.

**Caveat:** DiariZen's clustering uses a WeSpeaker ResNet34 embedding with VBx, close kin to
Community-1's. So against DiariZen alone, the speaker-assignment half of your system is not
fully independent of the reference. The Nemotron cross-check removes most of that concern for
the labels.

## Setup

You need a Linux machine with an NVIDIA GPU, `ffmpeg` on the PATH, and
[uv](https://docs.astral.sh/uv/) or Python 3.10 + 3.12.

```bash
# the app, your system and the music detector (Python 3.12, pyannote.audio 4)
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# the labeller, in its own environment (Python 3.10, torch 2.1.1 + DiariZen's pyannote fork)
scripts/setup_diarizen.sh
# the cross-check, in its own environment (Python 3.12, NeMo from its main branch)
scripts/setup_nemotron.sh
python serve.py                         # http://localhost:8010
```

- Community-1 is gated on Hugging Face. Accept its terms at
  https://hf.co/pyannote/speaker-diarization-community-1 and set `HF_TOKEN` in `.env` for the
  first run. After that it is read from the cache.
- Both setup scripts pin what they install (a DiariZen commit, a NeMo commit) and are safe to
  run again.
  - Nemotron-3 uses RoPE attention, which the latest NeMo release (3.0.0) cannot build. That is
    why NeMo comes from GitHub (`NEMO_REF` picks another commit or tag).
- Recordings imported before Nemotron was set up get it with *Run Nemotron where missing*, on
  the Dataset page.
- Uploading needs a Hugging Face token **with write access**: `hf auth login`, then restart the
  server.

**Speed and memory** (2× RTX 4070 Ti SUPER, 16 GB each):
- A 3–4 minute video takes about 30 s from link to labels, download included.
- A 43-minute panel talk:

  | Step | Time | Where |
  |---|---|---|
  | Your system | about 1 min | GPU 0 |
  | Music detection | 54 s | GPU 0 |
  | Nemotron | 21 s | GPU 0 |
  | DiariZen | 87 s | GPU 1, peak 8.7 GB |

- If DiariZen runs out of GPU memory, it retries by itself with half the batch size.

## Using it

**Add recordings.**
- Paste a YouTube link:
  - a video (watch, youtu.be, shorts, live);
  - a **playlist**;
  - or a **channel** (`/@name`, `/channel/…`), in the order YouTube lists its uploads.
- For a playlist or channel, choose how many videos to take and the length range. Videos
  already in the library are skipped.
- You can also upload a file, or enter a path. Files in `inbox/` are offered in the path field.
- *Speakers* (automatic, exact, or a range) applies to DiariZen and to your system. Nemotron has
  no such option.

Status steps in the table:
1. Download, then convert to 16 kHz mono.
2. **Diarizing:** DiariZen, Nemotron, your system, then music detection.
3. **Re-clustering:** your system told DiariZen's speaker count. This takes seconds, because it
   reuses the cached embeddings.
4. **Labelling:** the agreement filter.
5. **Scoring.**

**Recordings table:** speakers, share of speech kept as labels (or why the recording was
rejected), your DER against the labels, and your DER against DiariZen. The card above it pools
all recordings.

**Comparison page:**
- **Timeline:** DiariZen, Nemotron (colored by the DiariZen speaker each maps to), the *Kept*
  lane, your speakers, and the disagreement lane. When scoring against the labels, the dropped
  regions are hatched.
- **Vs. labels / Vs. DiariZen:** score against the labels (kept regions only) or against
  DiariZen on the whole file.
- **Own count / Reference count:** your system with the speaker count it found, or told
  DiariZen's count.
- **Side panel:**
  - scores: no collar or ±0.25 s, with or without overlap;
  - label statistics: kept, agreement, overlap and turn-change coverage, what was dropped and
    why;
  - the recording's dataset split;
  - re-run buttons for each system, plus *Rebuild labels*;
  - RTTM / UEM downloads.
- **Largest disagreements:** longest first, with who each system hears there. Double-click on
  the timeline, or **N** / **⇧N**, plays them.

**Spot check page:**
- *Draw more windows* samples random 8 s windows from the kept regions. Each labelled second is
  equally likely.
- Listen (**Space**), then mark **Y** correct, **N** wrong or **S** can't tell. A window is
  wrong when:
  - a speaker is missing, or labelled while silent, for more than about half a second;
  - a turn goes to the wrong speaker;
  - one person is split into two.
- The accuracy and its 95% interval go into the dataset card.

| Windows judged | Accuracy within about |
|---|---|
| 30 | ±12 points |
| 100 | ±7 points |

**Dataset page:** sizes per split, rejected recordings, build, push, and the evaluation
commands.

## The dataset

```
exports/<name>/README.md            dataset card: method, sizes, coverage, spot check, licenses
               database.yml         pyannote protocol KhmerPseudo.SpeakerDiarization.YouTube
               metadata.jsonl       one row per recording (YouTube id, split, kept, coverage…)
               audio/<uri>.flac     16 kHz mono (unless built without audio)
               rttm/<uri>.rttm      labels (DiariZen)
               uem/<uri>.uem        where to use them
               pyannote/<split>.{lst,rttm,uem}
               review.json          spot-check verdicts
```

- **Splits:** 80/10/10 by a hash of the video id, so a recording never changes split when the
  dataset grows. You can pin a recording's split on its page, for example to get a test split
  going early.
- **Pushing:** it goes to a **private** Hugging Face dataset repo. The tool creates the repo
  private and refuses to push to a public one.
- **Without audio:** build with *Include audio* off to push only the labels and YouTube ids.
  `python -m bench.evaluate DATASET --fetch-audio` downloads the audio again. It warns if a
  video's length has changed since labelling.

```python
from pyannote.database import registry, FileFinder
registry.load_database("exports/khmer-diarization-eval/database.yml")
protocol = registry.get_protocol("KhmerPseudo.SpeakerDiarization.YouTube",
                                 preprocessors={"audio": FileFinder()})
```

## Evaluating on it

The dataset is built for **evaluation only** by default: every recording is in `test`.

```bash
python -m bench.evaluate exports/khmer-diarization-eval --out eval-current.json   # your checkpoint
python -m bench.evaluate exports/khmer-diarization-eval --segmentation weights/new.ckpt \
       --compare eval-current.json            # is the new checkpoint really better?
python -m bench.evaluate your-user/khmer-diarization-eval    # from the Hugging Face Hub
python -m bench.evaluate DATASET --oracle-count              # told the labels' speaker count
python -m bench.evaluate DATASET --hyp-dir other_system/     # any system's <uri>.rttm files
```

It prints:
- per-recording DER (no collar and ±0.25 s), with missed / false alarm / confusion;
- **per-channel DER, and the mean over channels.** Each show counts once, while pooled DER lets
  long podcasts dominate;
- the pooled DER with a **95% interval from resampling whole channels**. Recordings of one show
  are not independent evidence; different shows roughly are;
- with `--compare`, the **paired difference** to an earlier run on the same recordings. It says
  "better", "worse" or "no clear difference".

Results go to a JSON file, and the system's RTTMs are saved next to it.

**Don't evaluate DiariZen or Nemotron-3-Diarization on it:** the labels come from them. Also
check that the system you score was not trained on these videos.

## How it is scored

- **DER** = (missed + false alarm + confusion) ÷ reference speech. It is computed with
  `pyannote.metrics` after the optimal one-to-one mapping of your speakers to the reference's.
  - **Missed:** the reference has more speakers active than yours.
  - **False alarm:** yours has more speakers active than the reference.
  - **Confusion:** speech is detected but given to the wrong speaker.
- **Scored region:** against the labels, only the UEM is scored. Against DiariZen, the whole
  file is.
- **JER** (Jaccard error rate) averages the error per reference speaker.
- **Conditions:** no collar or ±0.25 s around each reference boundary (NIST convention: pyannote
  gets `collar=0.5`), each with and without overlapped speech. The headline number is no collar,
  overlap included.
- **Pooled scores** sum each component over the recordings and then divide. A long recording
  weighs more.
- The error track and per-speaker breakdown use a 10 ms grid without a collar. Their total
  matches pyannote's no-collar DER.

## Files per recording

```
data/<id>/meta.json            name, source, duration, options, status, split, headline scores
          audio.wav, peaks.bin
          reference.rttm       DiariZen (R1, R2, … by talk time)
          second.rttm          Nemotron (N1, N2, …)
          system.rttm          yours, own speaker count (S1, S2, …)
          system_oracle.rttm   yours, told DiariZen's speaker count
          nonspeech.json       music / speech probability per second, music-only regions
          vad.npy              Silero speech probability every 32 ms
          consensus.json       kept regions, dropped regions and why, coverage, acceptance
          labels.uem           the kept regions
          diar_cache.npz       your system's segmentations + embeddings
          metrics.json         everything the page shows
data/_review.json              spot-check windows and verdicts
```

## Configuration

Set these in `.env` or as environment variables (see `.env.example`):

| Variable | Default | |
|---|---|---|
| `DIAR_SEGMENTATION` | weights/segmentation_khmer.ckpt | your segmentation checkpoint |
| `DIAR_DEVICE` | auto (`cuda:0`) | your system and the music detector |
| `HF_TOKEN` | – | download Community-1 once; push datasets (write token) |
| `DIARIZEN_PYTHON` | .venv-diarizen/bin/python | |
| `DIARIZEN_MODEL` | BUT-FIT/diarizen-wavlm-large-s80-md-v2 | |
| `DIARIZEN_DEVICE` | auto (the last GPU) | |
| `NEMOTRON_PYTHON` | .venv-nemo/bin/python | |
| `NEMOTRON_MODEL` | nvidia/Nemotron-3-Diarization | |
| `NEMOTRON_DEVICE` | auto (the first GPU) | |
| `BENCH_EXPORT_DIR` | exports | where datasets are built |
| `HF_DATASET_REPO` | `<you>/khmer-diarization-eval` | default upload target |
| `BENCH_DATA_DIR` | data | recordings and results |
| `BENCH_INBOX_DIR` | inbox | files offered in the import form |
| `BENCH_HOST` | 127.0.0.1 | address the server binds to |

The label thresholds are constants at the top of `bench/consensus.py` and
`bench/nonspeech.py`: the collar, the jitter, the shortest kept piece, the acceptance share,
and the music threshold. After changing them, use *Rebuild all labels*.

## Tests

```bash
pytest          # no GPU, no network, no models: all diarizers and detectors are faked
```

## Licenses

- This code: see the repository.
- pyannote.audio: MIT. Community-1 weights: CC-BY-4.0.
- Nemotron-3-Diarization: OpenMDW 1.1.
- AST AudioSet classifier (`MIT/ast-finetuned-audioset-10-10-0.4593`): BSD-3-Clause.
- Silero VAD: MIT.
- `weights/segmentation_khmer.ckpt`, the fine-tuned Khmer segmentation model: published with this
  repository.
- **DiariZen weights: CC BY-NC 4.0, non-commercial.** Labels made with DiariZen inherit that in
  practice: use the dataset for research.
- **Audio from YouTube belongs to its owners.** Keep datasets with audio private.

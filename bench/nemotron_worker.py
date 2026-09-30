"""
The second opinion, NVIDIA Nemotron-3-Diarization (a streaming Sortformer, up to 8 speakers), run
as a separate process in its own environment (.venv-nemo, see scripts/setup_nemotron.sh); it must
not import anything from this repo.

    .venv-nemo/bin/python bench/nemotron_worker.py --audio audio.wav --out second.rttm \
        [--model nvidia/Nemotron-3-Diarization]

The GPU is chosen by the caller with CUDA_VISIBLE_DEVICES. The model has no speaker-count
option. Progress goes to stdout as lines "@stage <name>".
"""

import argparse
import os
import sys
import warnings

# this script's folder is first on sys.path: keep the app's modules from shadowing NeMo's imports
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
warnings.filterwarnings("ignore")


def stage(name: str) -> None:
    print(f"@stage {name}", flush=True)


def parse(segment) -> tuple[float, float, str]:
    """NeMo returns "start end speaker_N" strings (older versions: tuples/lists)."""
    if isinstance(segment, str):
        start, end, spk = segment.split()[:3]
    else:
        start, end, spk = segment[:3]
    return float(start), float(end), str(spk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="nvidia/Nemotron-3-Diarization")
    args = p.parse_args()

    stage("load")
    import logging
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(args.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    # the model card's offline setting (chunked, so any length works)
    m = model.sortformer_modules
    m.chunk_len, m.chunk_right_context, m.fifo_len, m.spkcache_update_period = 340, 40, 40, 300
    if hasattr(model, "_check_streaming_parameters"):
        model._check_streaming_parameters()

    stage("diarize")
    with torch.inference_mode():
        segments = model.diarize(audio=[args.audio], batch_size=1)[0]
    with open(args.out, "w") as f:
        for seg in segments:
            start, end, spk = parse(seg)
            if end > start:
                f.write(f"SPEAKER rec 1 {start:.3f} {end - start:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    stage("done")


if __name__ == "__main__":
    sys.exit(main())

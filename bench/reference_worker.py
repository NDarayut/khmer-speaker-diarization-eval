"""
The reference diarizer, DiariZen, run as a separate process in its own environment
(.venv-diarizen, see scripts/setup_diarizen.sh); it must not import anything from this repo.

    .venv-diarizen/bin/python bench/reference_worker.py --audio audio.wav --out reference.rttm \
        [--model BUT-FIT/diarizen-wavlm-large-s80-md-v2] [--num-speakers N | --min-speakers A --max-speakers B]

The GPU is chosen by the caller with CUDA_VISIBLE_DEVICES (DiariZen always uses cuda:0 when a GPU
is visible). Progress goes to stdout as lines "@stage <name>", next to DiariZen's own prints
("Extracting segmentations." ...), which the caller reads as progress too.
"""

import argparse
import os
import sys
import warnings

# this script's folder is first on sys.path: keep the app's modules (audio, rttm, ...) from
# shadowing DiariZen's imports
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
warnings.filterwarnings("ignore")


def stage(name: str) -> None:
    print(f"@stage {name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="BUT-FIT/diarizen-wavlm-large-s80-md-v2")
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--batch-size", type=int, default=16,
                   help="segmentation batch (WavLM-Large, 16 s chunks); halved on out-of-memory")
    args = p.parse_args()

    stage("load")
    from diarizen.pipelines.inference import DiariZenPipeline
    pipeline = DiariZenPipeline.from_pretrained(args.model)
    if args.num_speakers:
        pipeline.min_speakers = pipeline.max_speakers = args.num_speakers
    else:
        if args.min_speakers:
            pipeline.min_speakers = args.min_speakers
        if args.max_speakers:
            pipeline.max_speakers = args.max_speakers

    # DiariZen prints "Extracting segmentations." / "Extracting Embeddings." / "Clustering."
    # itself; the caller maps those to progress too
    stage("diarize")
    import torch
    batch = args.batch_size
    while True:
        pipeline.segmentation_batch_size = batch
        try:
            result = pipeline(args.audio, sess_name="rec")
            break
        except (MemoryError, torch.cuda.OutOfMemoryError):
            if batch <= 1:
                raise
            batch //= 2
            torch.cuda.empty_cache()
            print(f"@stage out of GPU memory: retrying with batch size {batch}", flush=True)
    with open(args.out, "w") as f:
        for seg, _, spk in result.itertracks(yield_label=True):
            f.write(f"SPEAKER rec 1 {seg.start:.3f} {seg.duration:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    stage("done")


if __name__ == "__main__":
    sys.exit(main())

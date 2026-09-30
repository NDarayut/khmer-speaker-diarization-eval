#!/usr/bin/env bash
# Sets up the reference diarizer, DiariZen (BUT-FIT), in its own environment:
#
#   third_party/DiariZen     the toolkit at a pinned commit, with its pyannote-audio fork
#   .venv-diarizen           Python 3.10, torch 2.1.1 (CUDA 12.1), as DiariZen requires
#
# It cannot share the app's environment (pyannote.audio 4, torch 2.11): the app runs it as a
# subprocess (bench/reference_worker.py). Re-running this script is safe.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT=844f5555b0a98acd0931511fc641a8c5b8ba92c7
MODEL="${DIARIZEN_MODEL:-BUT-FIT/diarizen-wavlm-large-s80-md-v2}"
SRC="$ROOT/third_party/DiariZen"
VENV="$ROOT/.venv-diarizen"

if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/BUTSpeechFIT/DiariZen.git "$SRC"
fi
git -C "$SRC" fetch -q origin "$COMMIT"
git -C "$SRC" checkout -q "$COMMIT"

if command -v uv >/dev/null; then
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.10 "$VENV"
    PIP=(uv pip install -q --python "$VENV/bin/python")
else
    [ -x "$VENV/bin/python" ] || python3.10 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    PIP=("$VENV/bin/pip" install -q)
fi

"${PIP[@]}" torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu121
# DiariZen's requirements, without the notebook/training extras the worker never uses
grep -vE '^(jupyterlab|tensorboard|pre-commit|pesq|pystoi|openpyxl|onnxruntime-gpu)\b' \
    "$SRC/requirements.txt" > "$VENV/diarizen-requirements.txt"
"${PIP[@]}" -r "$VENV/diarizen-requirements.txt" -c "$SRC/constraints.txt"
"${PIP[@]}" -e "$SRC" -c "$SRC/constraints.txt"
"${PIP[@]}" -e "$SRC/pyannote-audio" -c "$SRC/constraints.txt"

# download the model once (then it is read from the Hugging Face cache)
"$VENV/bin/python" - "$MODEL" <<'EOF'
import sys
from diarizen.pipelines.inference import DiariZenPipeline
DiariZenPipeline.from_pretrained(sys.argv[1])
print("DiariZen ready:", sys.argv[1])
EOF

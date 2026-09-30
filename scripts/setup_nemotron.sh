#!/usr/bin/env bash
# Sets up the second opinion, NVIDIA Nemotron-3-Diarization (Sortformer, up to 8 speakers),
# in its own environment:
#
#   .venv-nemo      Python 3.12 + NeMo toolkit (ASR extras)
#
# NeMo pins many packages, so it is kept apart from the app's environment; the app runs it as a
# subprocess (bench/nemotron_worker.py). Re-running this script is safe.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${NEMOTRON_MODEL:-nvidia/Nemotron-3-Diarization}"
VENV="$ROOT/.venv-nemo"

if command -v uv >/dev/null; then
    [ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
    PIP=(uv pip install -q --python "$VENV/bin/python")
else
    [ -x "$VENV/bin/python" ] || python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    PIP=("$VENV/bin/pip" install -q)
fi

"${PIP[@]}" Cython packaging
# Nemotron-3-Diarization uses RoPE attention, which the latest NeMo release (3.0.0, Sep 2026)
# cannot build yet: install NeMo from its main branch, at the commit this was tested with
# (NEMO_REF=main for the newest, or a release tag once one supports it)
"${PIP[@]}" "nemo-toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF:-00278b0bd95bb2b8174b88012aa21b003c59d2e9}"

# download the model once (then it is read from the Hugging Face cache)
"$VENV/bin/python" - "$MODEL" <<'EOF'
import sys
from nemo.collections.asr.models import SortformerEncLabelModel
SortformerEncLabelModel.from_pretrained(sys.argv[1])
print("Nemotron ready:", sys.argv[1])
EOF

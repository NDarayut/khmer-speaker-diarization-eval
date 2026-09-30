"""
Runtime settings from environment variables, plus an optional `.env` file whose values never
override variables already set in the environment.

  Your system (in this process):
    DIAR_SEGMENTATION    segmentation checkpoint                       (weights/segmentation_khmer.ckpt)
    DIAR_DEVICE          auto | cuda | cuda:0 | cpu                    (auto: cuda:0 if there is a GPU)
    HF_TOKEN             only to download gated Community-1 once; then it is read from the HF cache
  Reference (DiariZen, a subprocess in its own environment: scripts/setup_diarizen.sh):
    DIARIZEN_PYTHON      its interpreter                               (.venv-diarizen/bin/python)
    DIARIZEN_MODEL       Hugging Face model id                         (BUT-FIT/diarizen-wavlm-large-s80-md-v2)
    DIARIZEN_DEVICE      auto | cuda:N | cpu                           (auto: the last GPU, so both run at once)
  Second opinion (Nemotron-3-Diarization, a subprocess: scripts/setup_nemotron.sh):
    NEMOTRON_PYTHON      its interpreter                               (.venv-nemo/bin/python)
    NEMOTRON_MODEL       Hugging Face model id                         (nvidia/Nemotron-3-Diarization)
    NEMOTRON_DEVICE      auto | cuda:N | cpu                           (auto: the first GPU)
  Dataset:
    BENCH_EXPORT_DIR     where datasets are built                      (exports)
    HF_DATASET_REPO      Hugging Face dataset repo to push to          (<your user>/khmer-diarization-eval)
  App:
    BENCH_DATA_DIR       where recordings and results are kept         (data)
    BENCH_INBOX_DIR      audio files here are offered in the import form (inbox)
    BENCH_HOST           address the server binds to                   (127.0.0.1)
"""

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, # comments, optional quotes."""
    values = {}
    p = Path(path)
    if not p.is_file():
        return values
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


ENV = {
    "segmentation": "DIAR_SEGMENTATION",
    "device": "DIAR_DEVICE",
    "hf_token": "HF_TOKEN",
    "diarizen_python": "DIARIZEN_PYTHON",
    "diarizen_model": "DIARIZEN_MODEL",
    "diarizen_device": "DIARIZEN_DEVICE",
    "nemotron_python": "NEMOTRON_PYTHON",
    "nemotron_model": "NEMOTRON_MODEL",
    "nemotron_device": "NEMOTRON_DEVICE",
    "export_dir": "BENCH_EXPORT_DIR",
    "hf_dataset_repo": "HF_DATASET_REPO",
    "data_dir": "BENCH_DATA_DIR",
    "inbox_dir": "BENCH_INBOX_DIR",
    "host": "BENCH_HOST",
}


@dataclass(frozen=True)
class Settings:
    segmentation: str = str(ROOT / "weights" / "segmentation_khmer.ckpt")
    device: str = "auto"
    hf_token: str | None = None
    diarizen_python: str = str(ROOT / ".venv-diarizen" / "bin" / "python")
    diarizen_model: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    diarizen_device: str = "auto"
    nemotron_python: str = str(ROOT / ".venv-nemo" / "bin" / "python")
    nemotron_model: str = "nvidia/Nemotron-3-Diarization"
    nemotron_device: str = "auto"
    export_dir: str = str(ROOT / "exports")
    hf_dataset_repo: str | None = None
    data_dir: str = str(ROOT / "data")
    inbox_dir: str = str(ROOT / "inbox")
    host: str = "127.0.0.1"

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = ROOT / ".env", **overrides) -> "Settings":
        env = {**(read_env_file(env_file) if env_file else {}), **os.environ}
        values = {}
        for f in fields(cls):
            raw = env.get(ENV[f.name])
            if raw:
                values[f.name] = raw
        return replace(cls(**values), **overrides)

    def __repr__(self) -> str:  # never print secrets
        shown = {f.name: getattr(self, f.name) for f in fields(self)}
        shown["hf_token"] = "<set>" if shown["hf_token"] else None
        return f"Settings({shown})"

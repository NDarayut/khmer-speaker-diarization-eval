"""
Start the diarization benchmark:

    python serve.py                     # http://localhost:8010
    python serve.py --port 9000
    python serve.py --host 0.0.0.0      # other machines too
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

import uvicorn  # noqa: E402

from config import Settings  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default=Settings.from_env().host,
                   help="127.0.0.1 = this machine only (default: $BENCH_HOST or 127.0.0.1)")
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    uvicorn.run("server.app:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

"""Start the TiMMInsight demo: python -m demo [--port 8765] [--data-root runs/demo]."""
import argparse
from pathlib import Path

import uvicorn

from demo.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="listen address (local by default)")
    ap.add_argument("--data-root", default="runs/demo", help="uploaded datasets")
    args = ap.parse_args()
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"TiMMInsight demo: http://{args.host}:{args.port}/")
    uvicorn.run(create_app(root), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

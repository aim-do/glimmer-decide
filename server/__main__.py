"""Run the web app:  python -m server [--mock] [--port 8000] [--host 127.0.0.1] [--no-browser]

Creates .env from .env.example on first run, prints which services are configured,
opens the browser and serves on localhost. --mock needs no keys, accounts or GPU.
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def ensure_env_file(root: Path = ROOT) -> bool:
    """Copy .env.example to .env if there is none. Returns True when it created one."""
    env = root / ".env"
    if env.exists():
        return False
    shutil.copy(root / ".env.example", env)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m server", description="Glimmer Decide web app")
    ap.add_argument("--mock", action="store_true", help="offline demo: fake decisions, no keys or GPU")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 opens the app to your network - and your API credits with it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args()

    if ensure_env_file():
        print("Created .env from .env.example - add API keys there for real models (see README).")
    if args.mock:
        os.environ["MOCK"] = "1"   # wins over .env: load_dotenv never overrides set variables

    import uvicorn

    from .app import app, settings

    def state(on: bool, hint: str) -> str:
        return "configured" if on else f"not configured - {hint}"

    print("Glimmer Decide" + ("  [MOCK: fake decisions]" if settings.mock else ""))
    print(f"  Glimmer (reasoning):  {state(settings.glimmer_enabled, 'GLIMMER_API_KEY in .env')}")
    print(f"  MetaEncoder (GPU):    {state(settings.encoder_enabled, 'python deploy/setup_modal.py all')}")
    print(f"  SAM 3.1 highlight:    {state(settings.sam_enabled, 'MODEL_API_KEY in .env')}")
    if args.host not in LOCAL_HOSTS:
        print(f"WARNING: listening on {args.host}: anyone who reaches this port can spend your API credits.\n"
              "  Phones on plain http cannot use the camera (Photo upload still works); prefer `tailscale serve`.")

    url = f"http://{'localhost' if args.host in LOCAL_HOSTS + ('0.0.0.0',) else args.host}:{args.port}"
    print(f"  Open {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

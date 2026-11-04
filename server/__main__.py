"""Run the web app:  python -m server [--mock] [--https] [--host 127.0.0.1] [--port 8000] [--no-browser]

Creates .env from .env.example on first run, prints which services are configured,
opens the browser and serves on localhost. --mock needs no keys, accounts or GPU.
--https serves the page to phones on the same Wi-Fi with a self-signed certificate, so their
browsers allow the camera (they warn about the certificate once).
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
ALL_INTERFACES = ("0.0.0.0", "::")


def ensure_env_file(root: Path = ROOT) -> bool:
    """Copy .env.example to .env if there is none. Returns True when it created one."""
    env = root / ".env"
    if env.exists():
        return False
    shutil.copy(root / ".env.example", env)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m server", description="Glimmer Decide web app")
    ap.add_argument("--mock", action="store_true", help="offline demo: fake decisions, no keys or GPU")
    ap.add_argument("--https", action="store_true",
                    help="for phones on your Wi-Fi: HTTPS with a self-signed certificate, so the camera "
                         "works; listens on 0.0.0.0 unless --host is given")
    ap.add_argument("--host", default=None,
                    help="default 127.0.0.1 (0.0.0.0 with --https); 0.0.0.0 opens the app to your "
                         "network - and your API credits with it")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)
    host = args.host or ("0.0.0.0" if args.https else "127.0.0.1")

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

    tls: dict[str, str] = {}
    lan: list[str] = []
    if args.https:
        from .certs import ensure_cert, lan_ips

        lan = lan_ips() if host in ALL_INTERFACES else [h for h in (host,) if h not in LOCAL_HOSTS]
        cert, key = ensure_cert(lan)
        tls = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    scheme = "https" if args.https else "http"

    if host not in LOCAL_HOSTS:
        print(f"WARNING: listening on {host}: anyone who reaches this port can spend your API credits.")
        if args.https:
            for ip in lan:
                print(f"  Phone on the same Wi-Fi: https://{ip}:{args.port}")
            print("  The certificate is self-signed, so browsers warn once: Advanced -> Proceed\n"
                  "  (iPhone: Show Details -> visit this website). After that the camera works.")
        else:
            print("  Phones on plain http cannot use the camera (Photo still works): add --https.")

    url = f"{scheme}://{'localhost' if host in LOCAL_HOSTS + ALL_INTERFACES else host}:{args.port}"
    print(f"  Open {url}  (Ctrl+C to quit)", flush=True)   # before uvicorn's log, also when piped
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=args.port, log_level="info", **tls)


if __name__ == "__main__":
    main()

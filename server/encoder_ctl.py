"""Start/stop the MetaEncoder GPU service on Modal from the web UI.

Start = `modal deploy` (seconds; the app is undeployed between sessions) and then a
/health call that waits through the cold start (~2 min) so the first decision is fast.
Stop  = `modal app stop`: terminates the GPU container and undeploys the app, so no
stray request or page load can wake it and spend credits.

Modal is driven through its CLI in a subprocess (same venv), so the web server needs
no Modal SDK import and every operation matches what a user would type.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from .config import Settings

APP_NAME = "glimmer-decide-metaencoder"
ROOT = Path(__file__).resolve().parent.parent
DEPLOY_FILE = ROOT / "deploy" / "modal_metaencoder.py"
STATUS_TTL_S = 5.0
SETUP_HINT = "Not set up yet: run `python deploy/setup_modal.py all` once (see README)."

log = logging.getLogger("glimmer-decide")

Runner = Callable[[list[str], float], subprocess.CompletedProcess]


class ControlError(Exception):
    status = 400


class Busy(ControlError):
    status = 409


def run_modal(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-m", "modal", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)


def modal_logged_in() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    cfg = Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")
    try:
        return "token_id" in cfg.read_text(encoding="utf-8")
    except OSError:
        return False


def _deploy_error(out: str) -> str:
    low = out.lower()
    if "secret" in low and ("not found" in low or "could not find" in low):
        return "Modal secrets are missing: run `python deploy/setup_modal.py all`."
    if "token" in low and ("invalid" in low or "unauthenticated" in low or "expired" in low):
        return "Modal login failed: run `modal token new`."
    return "Deploy failed - see the server log."


def _health_error(code: int) -> str:
    if code == 401:
        return "ENCODER_TOKEN does not match the Modal secret: run `python deploy/setup_modal.py all`."
    if code == 404:
        return "MetaEncoder service not found after deploy."
    return (f"GPU container failed to start (HTTP {code}); see "
            f"`modal app logs {APP_NAME}` (often: no Hugging Face access to the gated weights).")


class EncoderControl:
    """One operation at a time; each start/stop bumps a generation counter so a
    superseded job (a warm-up cut short by Stop) can never overwrite newer state."""

    def __init__(self, get_settings: Callable[[], Settings], reload_settings: Callable[[], None] = lambda: None,
                 run: Runner = run_modal, http_get=httpx.get,
                 spawn: Callable[..., None] | None = None, logged_in: Callable[[], bool] = modal_logged_in):
        self._settings = get_settings
        self._reload = reload_settings
        self._run = run
        self._get = http_get
        self._spawn = spawn or (lambda fn, *a: threading.Thread(target=fn, args=a, daemon=True).start())
        self._logged_in = logged_in
        self._lock = threading.Lock()
        self._remote_lock = threading.Lock()
        self._gen = 0
        self.op = "idle"            # idle | starting | warming | stopping
        self.op_started = 0.0
        self.error: str | None = None
        self._remote: dict | None = None
        self._remote_at = 0.0

    # ---- status ---------------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        s = self._settings()
        if s.mock:
            return "Mock mode: no GPU is used."
        if importlib.util.find_spec("modal") is None:
            return "Modal CLI is not installed: pip install -r requirements.txt"
        if not (s.encoder_url and s.encoder_token):
            self._reload()   # setup_modal.py may have written them while the server ran
            s = self._settings()
        if not (s.encoder_url and s.encoder_token) or not self._logged_in():
            return SETUP_HINT
        return None

    def remote(self, force: bool = False) -> dict:
        """Deployed? How many GPU containers run? Cached briefly: each read is a CLI call."""
        with self._remote_lock:
            if not force and self._remote is not None and time.monotonic() - self._remote_at < STATUS_TTL_S:
                return self._remote
            remote: dict = {"deployed": None, "containers": None}
            try:
                r = self._run(["app", "list", "--json"], 60)
                if r.returncode == 0:
                    mine = [a for a in json.loads(r.stdout or "[]")
                            if a.get("description") == APP_NAME and a.get("state") == "deployed"]
                    remote = {"deployed": bool(mine), "containers": sum(int(a.get("tasks") or 0) for a in mine)}
                else:
                    log.warning("modal app list failed: %s", (r.stderr or r.stdout)[-500:])
            except (subprocess.TimeoutExpired, OSError, ValueError, TypeError) as e:
                log.warning("modal app list failed: %s", e)
            self._remote, self._remote_at = remote, time.monotonic()
            return remote

    def status(self) -> dict:
        reason = self.unavailable_reason()
        if reason:
            return {"available": False, "reason": reason}
        remote = self.remote()
        with self._lock:
            op, started, error = self.op, self.op_started, self.error
        return {"available": True, "op": op, "error": error,
                "elapsed_s": int(time.monotonic() - started) if op != "idle" else 0, **remote}

    # ---- operations -----------------------------------------------------------

    def _begin(self, op: str, refuse: tuple[str, ...]) -> int:
        reason = self.unavailable_reason()
        if reason:
            raise ControlError(reason)
        with self._lock:
            if self.op in refuse:
                raise Busy(f"MetaEncoder is busy ({self.op}).")
            self._gen += 1
            self.op, self.op_started, self.error = op, time.monotonic(), None
            return self._gen

    def _current(self, gen: int) -> bool:
        with self._lock:
            return gen == self._gen

    def _finish(self, gen: int, error: str | None) -> None:
        with self._lock:
            if gen != self._gen:
                return
            self.op, self.error = "idle", error
        self._remote_at = 0.0   # next status read asks Modal again

    def start(self) -> dict:
        gen = self._begin("starting", refuse=("starting", "warming", "stopping"))
        self._spawn(self._start_job, gen)
        return {"started": True}

    def stop(self) -> dict:
        # Allowed while warming (cuts a cold start short), not while a deploy is mid-flight.
        gen = self._begin("stopping", refuse=("starting", "stopping"))
        self._spawn(self._stop_job, gen)
        return {"stopping": True}

    def _start_job(self, gen: int) -> None:
        s = self._settings()
        try:
            r = self._run(["deploy", str(DEPLOY_FILE)], 900)
            if r.returncode != 0:
                log.warning("modal deploy failed:\n%s", (r.stdout + r.stderr)[-3000:])
                return self._finish(gen, _deploy_error(r.stdout + r.stderr))
            if not self._current(gen):
                return
            with self._lock:
                self.op = "warming"
            resp = self._get(f"{s.encoder_url}/health", timeout=900, follow_redirects=True,
                             headers={"Authorization": f"Bearer {s.encoder_token}"})
            self._finish(gen, None if resp.status_code == 200 else _health_error(resp.status_code))
        except (subprocess.TimeoutExpired, OSError, httpx.HTTPError) as e:
            log.warning("MetaEncoder start failed: %s", e)
            self._finish(gen, f"Start failed ({type(e).__name__}) - see the server log.")

    def _stop_job(self, gen: int) -> None:
        try:
            r = self._run(["app", "stop", "--yes", APP_NAME], 120)
            out = (r.stdout + r.stderr).lower()
            done = r.returncode == 0 or "already stopped" in out or "not found" in out or "could not find" in out
            if not done:
                log.warning("modal app stop failed:\n%s", (r.stdout + r.stderr)[-2000:])
            self._finish(gen, None if done else "Stop failed - see the server log.")
        except (subprocess.TimeoutExpired, OSError) as e:
            log.warning("MetaEncoder stop failed: %s", e)
            self._finish(gen, f"Stop failed ({type(e).__name__}) - see the server log.")

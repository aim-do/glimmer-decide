"""HTTP server: serves the camera web app, the decision API and MetaEncoder GPU control.

Run:  python -m server            (or: python -m uvicorn server.app:app --host 127.0.0.1 --port 8000)
(localhost only: the API spends paid credits; reach it from a phone via `tailscale serve`).
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import replace
from pathlib import Path
from typing import Literal

import httpx
from dotenv import dotenv_values, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from .config import DECIDERS as DECIDER_NAMES, load_settings  # noqa: E402
from .decision import decide  # noqa: E402
from .encoder import decide_encoder  # noqa: E402
from .encoder_ctl import ControlError, EncoderControl  # noqa: E402
from .sam import segment  # noqa: E402
from .tasks import list_domains, load_domain  # noqa: E402
from .types import finish  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("glimmer-decide")

settings = load_settings()
CAPTURES = ROOT / "data" / "captures"
DATA_URL_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,")
DECIDERS = {"encoder": decide_encoder, "glimmer": decide}
MAX_IMAGE_CHARS = 8_000_000
MAX_IMAGE_BYTES = 5_000_000
RATE_LIMITS = {"/api/decide": (30, 60.0), "/api/warmup": (3, 60.0), "/api/capture": (60, 60.0),
               "/api/encoder/start": (6, 60.0), "/api/encoder/stop": (12, 60.0)}
UI_HEADER = ("x-requested-with", "glimmer-decide")


def _reload_encoder_env() -> None:
    """Pick up ENCODER_URL/ENCODER_TOKEN that deploy/setup_modal.py wrote while the server ran."""
    global settings
    vals = dotenv_values(ROOT / ".env")
    url = (vals.get("ENCODER_URL") or "").strip().rstrip("/")
    token = (vals.get("ENCODER_TOKEN") or "").strip()
    if url.startswith("https://") and token and (url, token) != (settings.encoder_url, settings.encoder_token):
        settings = replace(settings, encoder_url=url, encoder_token=token)
        log.info("MetaEncoder endpoint loaded from .env: %s", url)


encoder_ctl = EncoderControl(lambda: settings, _reload_encoder_env)

@asynccontextmanager
async def _lifespan(_app):
    yield
    pool.shutdown(wait=False, cancel_futures=True)


pool = ThreadPoolExecutor(max_workers=4)
app = FastAPI(title="glimmer-decide", lifespan=_lifespan)
_capture_lock = threading.Lock()
_warmup_lock = threading.Lock()
_warmup_running = False
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


@app.middleware("http")
async def _guard(request: Request, call_next):
    limit = RATE_LIMITS.get(request.url.path)
    if limit and request.method == "POST":
        max_hits, window = limit
        now = time.monotonic()
        with _rate_lock:
            hits = [t for t in _rate_hits.get(request.url.path, []) if now - t < window]
            if len(hits) >= max_hits:
                return JSONResponse({"detail": "rate limit: slow down"}, status_code=429)
            hits.append(now)
            _rate_hits[request.url.path] = hits
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "style-src 'self'; script-src 'self'")
    return resp


def _clean_goal(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("goal must not be blank")
    return v


class DecideRequest(BaseModel):
    domain: str = "desk"
    goal: str = Field(min_length=1, max_length=500)
    image: str | None = Field(default=None, max_length=MAX_IMAGE_CHARS)
    decider: Literal["encoder", "glimmer", "both"] | None = None
    highlight: bool = True

    @field_validator("goal", mode="after")
    @classmethod
    def strip_goal(cls, v: str) -> str:
        return _clean_goal(v)


class CaptureRequest(BaseModel):
    domain: str = "desk"
    goal: str = Field(min_length=1, max_length=500)
    image: str = Field(max_length=MAX_IMAGE_CHARS)
    predicted: str | None = None
    gold: list[str] = Field(default_factory=list)

    @field_validator("goal", mode="after")
    @classmethod
    def strip_goal(cls, v: str) -> str:
        return _clean_goal(v)


def _domain(domain_id: str):
    try:
        return load_domain(domain_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown domain {domain_id!r}") from None
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"invalid domain {domain_id!r}: {type(e).__name__}") from None


def _decode_image(image: str | None) -> bytes | None:
    if image is None:
        return None
    m = DATA_URL_RE.match(image)
    if not m:
        raise HTTPException(400, "image must be a base64 data URL (jpeg/png/webp)")
    try:
        raw = base64.b64decode(image.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "image is not valid base64") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "image is empty or too large")
    return raw


def _check_image(image: str | None) -> None:
    _decode_image(image)


def _require_ui(request: Request) -> None:
    """Body-less POSTs that spend money need a custom header: browsers only send one
    cross-origin after a CORS preflight, which this server never grants, so another
    site open in the same browser cannot trigger them (localhost CSRF)."""
    name, value = UI_HEADER
    if request.headers.get(name) != value:
        raise HTTPException(403, "missing X-Requested-With header")


@app.get("/api/status")
def status():
    return {"mock": settings.mock, "default_decider": settings.decider,
            "encoder": settings.encoder_enabled, "glimmer": settings.glimmer_enabled,
            "model": settings.glimmer_model, "sam": settings.sam_enabled}


@app.get("/api/encoder")
def encoder_status():
    """MetaEncoder GPU state for the page's Start/Stop panel."""
    return encoder_ctl.status()


@app.post("/api/encoder/start", dependencies=[Depends(_require_ui)])
def encoder_start():
    try:
        return encoder_ctl.start()
    except ControlError as e:
        raise HTTPException(e.status, str(e)) from None


@app.post("/api/encoder/stop", dependencies=[Depends(_require_ui)])
def encoder_stop():
    try:
        return encoder_ctl.stop()
    except ControlError as e:
        raise HTTPException(e.status, str(e)) from None


@app.post("/api/warmup", dependencies=[Depends(_require_ui)])
def warmup():
    """Wake the MetaEncoder GPU container in the background (cold start takes minutes)."""
    global _warmup_running
    if settings.mock or not settings.encoder_url:
        return {"started": False}
    with _warmup_lock:
        if _warmup_running:
            return {"started": False, "already_running": True}
        _warmup_running = True

    def ping():
        global _warmup_running
        try:
            httpx.get(f"{settings.encoder_url}/health", timeout=900, follow_redirects=True,
                      headers={"Authorization": f"Bearer {settings.encoder_token}"})
        except httpx.HTTPError as e:
            log.warning("warmup ping failed: %s", e)
        finally:
            with _warmup_lock:
                _warmup_running = False
    threading.Thread(target=ping, daemon=True).start()
    return {"started": True}


@app.get("/api/domains")
def domains():
    out = []
    for d in list_domains():
        try:
            dom = load_domain(d["id"])
        except Exception as e:  # one bad domain must not dead-end the UI
            log.warning("skipping domain %s: %s", d.get("id"), e)
            continue
        out.append({**d, "actions": [{"id": a.id, "label": a.label, "name": a.name} for a in dom.actions]})
    return out


@app.post("/api/decide")
def api_decide(req: DecideRequest):
    _check_image(req.image)
    dom = _domain(req.domain)
    which = req.decider or settings.decider
    if which not in DECIDER_NAMES:
        raise HTTPException(400, f"unknown decider {which!r}")
    names = ["encoder", "glimmer"] if which == "both" else [which]
    futures = [pool.submit(DECIDERS[n], settings, dom, req.goal, req.image) for n in names]
    # One shared deadline for the whole request, not one per decider. Answers
    # that already finished are kept; only the stragglers become errors.
    deadline = time.monotonic() + settings.request_timeout_s
    decisions = []
    for n, f in zip(names, futures):
        try:
            decisions.append(f.result(timeout=max(0.0, deadline - time.monotonic())))
        except FuturesTimeout:
            f.cancel()   # only helps if still queued; running threads finish in background
            log.warning("decide timed out after %ss (waiting on %s)", settings.request_timeout_s, n)
            decisions.append(finish(dom, None, "", 0, req.image is None,
                                    "decision timed out - try again", n))
    primary = next((d for d in decisions if d.action), decisions[0])
    highlight = None
    if req.highlight and primary.concept and req.image and settings.sam_enabled:
        try:
            highlight = segment(settings, primary.concept, req.image)
        except Exception as e:  # SAM must never lose a good decision
            log.warning("SAM failed: %s", e)
            highlight = {"concept": primary.concept, "boxes": [], "error": "highlight unavailable",
                         "latency_ms": 0}
    return {"decisions": [d.to_dict() for d in decisions], "highlight": highlight,
            "highlight_decider": primary.decider}


@app.post("/api/capture")
def api_capture(req: CaptureRequest):
    """Save frame + goal for later labeling (eval set / LoRA data in step 2)."""
    raw = _decode_image(req.image)
    _domain(req.domain)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    ext = DATA_URL_RE.match(req.image).group(1).replace("jpeg", "jpg")
    stem = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
    img_path = CAPTURES / f"{stem}.{ext}"
    tmp = CAPTURES / f".{stem}.tmp"
    tmp.write_bytes(raw)
    tmp.replace(img_path)   # atomic: readers never see a partial frame
    rec = {"image": f"captures/{img_path.name}", "domain": req.domain, "goal": req.goal,
           "gold": req.gold, "predicted": req.predicted}
    with _capture_lock:
        with (CAPTURES.parent / "captures.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": rec["image"]}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

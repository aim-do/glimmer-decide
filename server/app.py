"""HTTP server: serves the camera web app and the decision API.

Run:  python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
(localhost only: the API spends paid credits; reach it from a phone via `tailscale serve`).
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from .config import load_settings  # noqa: E402
from .decision import decide  # noqa: E402
from .encoder import decide_encoder  # noqa: E402
from .sam import segment  # noqa: E402
from .tasks import list_domains, load_domain  # noqa: E402

settings = load_settings()
CAPTURES = ROOT / "data" / "captures"
DATA_URL_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,")
DECIDERS = {"encoder": decide_encoder, "glimmer": decide}

app = FastAPI(title="glimmer-decide")
pool = ThreadPoolExecutor(max_workers=4)


class DecideRequest(BaseModel):
    domain: str = "desk"
    goal: str = Field(min_length=1, max_length=500)
    image: str | None = None      # data URL; None -> blind ablation
    decider: Literal["encoder", "glimmer", "both"] | None = None
    highlight: bool = True


class CaptureRequest(BaseModel):
    domain: str = "desk"
    goal: str = Field(min_length=1, max_length=500)
    image: str
    predicted: str | None = None
    gold: list[str] = []          # accepted action ids, filled in by the labeler


def _domain(domain_id: str):
    try:
        return load_domain(domain_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown domain {domain_id!r}")


def _check_image(image: str | None) -> None:
    if image is not None and not DATA_URL_RE.match(image):
        raise HTTPException(400, "image must be a base64 data URL (jpeg/png/webp)")


@app.get("/api/status")
def status():
    return {"mock": settings.mock, "default_decider": settings.decider,
            "encoder": settings.encoder_enabled, "glimmer": settings.glimmer_enabled,
            "model": settings.glimmer_model, "sam": settings.sam_enabled}


@app.post("/api/warmup")
def warmup():
    """Wake the MetaEncoder GPU container in the background (cold start takes minutes)."""
    if settings.mock or not settings.encoder_url:
        return {"started": False}

    def ping():
        try:
            httpx.get(f"{settings.encoder_url}/health", timeout=900, follow_redirects=True,
                      headers={"Authorization": f"Bearer {settings.encoder_token}"})
        except httpx.HTTPError:
            pass
    threading.Thread(target=ping, daemon=True).start()
    return {"started": True}


@app.get("/api/domains")
def domains():
    out = []
    for d in list_domains():
        dom = load_domain(d["id"])
        out.append({**d, "actions": [{"id": a.id, "label": a.label, "name": a.name} for a in dom.actions]})
    return out


@app.post("/api/decide")
def api_decide(req: DecideRequest):
    _check_image(req.image)
    dom = _domain(req.domain)
    which = req.decider or settings.decider
    names = ["encoder", "glimmer"] if which == "both" else [which]
    futures = [pool.submit(DECIDERS[n], settings, dom, req.goal, req.image) for n in names]
    decisions = [f.result() for f in futures]
    primary = next((d for d in decisions if d.action), decisions[0])
    highlight = None
    if req.highlight and primary.concept and req.image and settings.sam_enabled:
        highlight = segment(settings, primary.concept, req.image)
    return {"decisions": [d.to_dict() for d in decisions], "highlight": highlight}


@app.post("/api/capture")
def api_capture(req: CaptureRequest):
    """Save frame + goal for later labeling (eval set / LoRA data in step 2)."""
    _check_image(req.image)
    _domain(req.domain)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    ext = DATA_URL_RE.match(req.image).group(1).replace("jpeg", "jpg")
    stem = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    img_path = CAPTURES / f"{stem}.{ext}"
    img_path.write_bytes(base64.b64decode(req.image.split(",", 1)[1]))
    rec = {"image": f"captures/{img_path.name}", "domain": req.domain, "goal": req.goal,
           "gold": req.gold, "predicted": req.predicted}
    with (CAPTURES.parent / "captures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": rec["image"]}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

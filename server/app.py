"""HTTP server: serves the camera web app and the decision API.

Run:  python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from .config import load_settings  # noqa: E402
from .decision import decide  # noqa: E402
from .sam import segment  # noqa: E402
from .tasks import list_domains, load_domain  # noqa: E402

settings = load_settings()
CAPTURES = ROOT / "data" / "captures"
DATA_URL_RE = re.compile(r"^data:image/(jpeg|png|webp);base64,")

app = FastAPI(title="glimmer-decide")


class DecideRequest(BaseModel):
    domain: str = "desk"
    goal: str = Field(min_length=1, max_length=500)
    image: str | None = None      # data URL; None -> blind ablation
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
    return {"mock": settings.mock, "model": settings.glimmer_model,
            "provider": settings.glimmer_base_url, "sam": settings.sam_enabled}


@app.get("/api/domains")
def domains():
    out = []
    for d in list_domains():
        dom = load_domain(d["id"])
        out.append({**d, "actions": [{"id": a.id, "label": a.label} for a in dom.actions]})
    return out


@app.post("/api/decide")
def api_decide(req: DecideRequest):
    _check_image(req.image)
    dom = _domain(req.domain)
    dec = decide(settings, dom, req.goal, req.image)
    out = dec.to_dict()
    out["highlight"] = None
    if req.highlight and dec.concept and req.image and settings.sam_enabled:
        out["highlight"] = segment(settings, dec.concept, req.image)
    return out


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

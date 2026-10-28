"""SAM 3.1 via Meta Model API: highlight the object Glimmer picked.

The API returns one text line per frame, e.g.
  <0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600;c=0.91|><|mask;...|>,1<|box;...|>...
We only use boxes for now; masks are skipped.
"""
from __future__ import annotations

import re
import time

import httpx

from .config import Settings

BOX_RE = re.compile(r"(\d+)<\|box;([^|]*)\|>")


def parse_boxes(line: str) -> list[dict]:
    boxes = []
    for obj_id, fields in BOX_RE.findall(line or ""):
        kv = dict(f.split("=", 1) for f in fields.split(";") if "=" in f)
        try:
            box = {"id": int(obj_id),
                   "x1": int(kv["x1"]), "y1": int(kv["y1"]),
                   "x2": int(kv["x2"]), "y2": int(kv["y2"]),
                   "w": int(kv["w"]), "h": int(kv["h"])}
        except (KeyError, ValueError):
            continue
        if "c" in kv:
            box["c"] = float(kv["c"])
        boxes.append(box)
    return boxes


def _output_text(resp: dict) -> str:
    if resp.get("output_text"):
        return resp["output_text"]
    parts = []
    for item in resp.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text"):
                parts.append(c.get("text", ""))
    return "".join(parts)


def segment(settings: Settings, concept: str, image_data_url: str,
            client: httpx.Client | None = None) -> dict:
    t0 = time.perf_counter()
    body = {
        "model": settings.sam_model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": image_data_url},
        ]}],
        "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
    }
    headers = {"Authorization": f"Bearer {settings.meta_api_key}"}
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        r = client.post(f"{settings.meta_base_url}/responses", json=body, headers=headers)
        r.raise_for_status()
        boxes, error = parse_boxes(_output_text(r.json())), None
    except (httpx.HTTPError, ValueError) as e:
        detail = getattr(getattr(e, "response", None), "text", "")
        boxes, error = [], f"{type(e).__name__}: {e} {detail[:300]}".strip()
    finally:
        if own:
            client.close()
    return {"concept": concept, "boxes": boxes, "error": error,
            "latency_ms": int((time.perf_counter() - t0) * 1000)}

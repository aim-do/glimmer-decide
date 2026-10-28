"""MetaEncoder ("Glimmer decision", System 1): frame + goal -> probability over actions.

MetaEncoder (facebook/meta-encoder, arXiv 2610.11316) is Muse Glimmer 30B contrastively
fine-tuned into a bi-encoder. The task (camera frame + goal text) and every candidate
action are embedded separately (last token, L2-normalised) and ranked by cosine
similarity; a softmax over the scores gives a distribution. One forward pass for the
task, no text generation, and candidate embeddings are computed once and cached.

The weights run on our own GPU service (deploy/modal_metaencoder.py), which exposes
POST /embed: {"items": [{"text": ..., "image": <data URL>}]} ->
{"dim": D, "embeddings": [<base64 float32>, ...]}.
"""
from __future__ import annotations

import base64
import hashlib
import threading
import time
from collections.abc import Callable

import httpx
import numpy as np

from .config import Settings
from .decision import Decision, _mock_letter, finish
from .tasks import Domain

# Closed-set prompt format recommended on the MetaEncoder model card:
# "{state}\n\nTask: {instruction}\nCriteria:\n- {label}: {what the label means}".
# The camera frame is the state; the goal and the criteria are text.
INSTRUCTION = ("Select the correct option: which object visible in the image "
               "should the user use to achieve the goal?")

EmbedFn = Callable[[list[dict]], np.ndarray]

_cache: dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()


def task_text(domain: Domain, goal: str) -> str:
    criteria = "\n".join(f"- {a.name}: {a.detail}" for a in domain.actions)
    return f"User goal: {goal.strip()}\n\nTask: {INSTRUCTION}\nCriteria:\n{criteria}"


def candidate_texts(domain: Domain, style: str) -> list[str]:
    # "label": names only, descriptions live in the task's criteria block (card's JEV style).
    # "label_detail": "name: description" per candidate (best rendering in the paper's ablation).
    if style == "label_detail":
        return [f"{a.name}: {a.detail}" for a in domain.actions]
    return [a.name for a in domain.actions]


def remote_embed(settings: Settings, items: list[dict]) -> np.ndarray:
    headers = {"Authorization": f"Bearer {settings.encoder_token}"}
    # A cold GPU container takes minutes to load 60 GB of weights; Modal answers long
    # requests with a redirect to poll, hence the generous timeout and follow_redirects.
    r = httpx.post(f"{settings.encoder_url}/embed", json={"items": items}, headers=headers,
                   timeout=900, follow_redirects=True)
    r.raise_for_status()
    data = r.json()
    return np.stack([np.frombuffer(base64.b64decode(e), dtype=np.float32) for e in data["embeddings"]])


def _hash_vec(text: str, dim: int = 64) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def mock_embed_factory(domain: Domain, goal: str, has_image: bool, style: str) -> EmbedFn:
    """Offline stand-in: candidates get hash vectors, the task points at the keyword-stub
    choice. Exercises the real cosine/softmax/caching path without a GPU."""
    target = dict(domain.letters())[_mock_letter(domain, goal, has_image)]
    target_text = candidate_texts(domain, style)[domain.actions.index(target)]

    def embed(items: list[dict]) -> np.ndarray:
        out = []
        for it in items:
            if it.get("text", "").startswith("User goal:"):
                v = 3 * _hash_vec(target_text) + _hash_vec(it["text"])
                out.append(v / np.linalg.norm(v))
            else:
                out.append(_hash_vec(it["text"]))
        return np.stack(out)
    return embed


def _candidate_matrix(embed: EmbedFn, texts: list[str], namespace: str) -> np.ndarray:
    keys = [f"{namespace}|{t}" for t in texts]
    with _cache_lock:
        missing = [t for t, k in zip(texts, keys) if k not in _cache]
    if missing:
        vecs = embed([{"text": t} for t in missing])
        with _cache_lock:
            for t, v in zip(missing, vecs):
                _cache[f"{namespace}|{t}"] = v
    with _cache_lock:
        return np.stack([_cache[k] for k in keys])


def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def decide_encoder(settings: Settings, domain: Domain, goal: str, image_data_url: str | None,
                   embed: EmbedFn | None = None) -> Decision:
    blind = image_data_url is None
    if embed is not None:
        namespace = f"custom:{id(embed)}"   # never mix vectors from different embedders
    elif not settings.encoder_enabled:
        return finish(domain, None, "", 0, blind, "MetaEncoder is not configured (ENCODER_URL)", "encoder")
    elif settings.mock:
        embed, namespace = mock_embed_factory(domain, goal, not blind, settings.encoder_style), "mock"
    else:
        embed, namespace = (lambda items: remote_embed(settings, items)), settings.encoder_url

    t0 = time.perf_counter()
    try:
        cands = _candidate_matrix(embed, candidate_texts(domain, settings.encoder_style), namespace)
        task = {"text": task_text(domain, goal)}
        if image_data_url is not None:
            task["image"] = image_data_url
        else:
            # Blind ablation: tell the encoder the frame is missing, as the Glimmer prompt does.
            task["text"] = task["text"].replace("\n\nTask:", "\n(No camera frame is available.)\n\nTask:", 1)
        q = embed([task])[0]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        detail = getattr(getattr(e, "response", None), "text", "")
        return finish(domain, None, "", int((time.perf_counter() - t0) * 1000), blind,
                      f"{type(e).__name__}: {e} {detail[:300]}".strip(), "encoder")
    latency_ms = int((time.perf_counter() - t0) * 1000)

    scores = cands @ q
    probs = softmax(scores, settings.encoder_tau)
    order = np.argsort(-scores)
    raw = "\n".join(f"{domain.actions[i].name:14s} cos={scores[i]:+.4f} p={probs[i]:.3f}" for i in order)
    best = domain.actions[int(order[0])]
    return finish(domain, best.id, raw, latency_ms, blind, None, "encoder",
                  {a.id: round(float(p), 4) for a, p in zip(domain.actions, probs)})

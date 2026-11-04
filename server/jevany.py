"""JevAny (System 1 decision head on Muse Glimmer 30B): frame + goal -> probability over actions.

JevAny-Muse-Glimmer-30B (SimpleJev, Apache-2.0) is a LoRA plus a small pointer head on Meta's
Muse Glimmer 30B. One backbone prefill reads the state (goal + camera frame), the question and
the labelled options; the head scores the options. No text is generated, like MetaEncoder,
but the options are read together with the task instead of being embedded separately.

The weights run on our own GPU service (deploy/modal_jevany.py), which exposes
POST /decide {"state", "instructions", "criteria": {label: meaning}, "image"?}
  -> {"choice": label, "probabilities": {label: p}, "latency_ms": model time}.

Its training data was text decisions only; how well it reads camera frames is exactly what
the labeled eval set measures.
"""
from __future__ import annotations

import time

import httpx

from .config import Settings
from .tasks import Domain
from .types import Decision, finish, mock_letter, public_error

INSTRUCTIONS = "Which object visible in the image should the user use to achieve the goal?"
MODEL_ID = "SimpleJev/JevAny-Muse-Glimmer-30B-LoRA"


def request_body(domain: Domain, goal: str, image_data_url: str | None) -> dict:
    """The decision in JevAny's own shape: a state, an instruction and {option label: meaning}.
    Labels are the domain's short names ("binder clip", "none", "unclear view")."""
    state = {"user goal": goal.strip(),
             "camera": ("the attached photo shows what is in front of the user" if image_data_url
                        else "no camera frame is available")}
    body = {"state": state, "instructions": INSTRUCTIONS,
            "criteria": {a.name: a.detail for a in domain.actions}}
    if image_data_url is not None:
        body["image"] = image_data_url
    return body


def _remote(settings: Settings, body: dict) -> dict:
    headers = {"Authorization": f"Bearer {settings.encoder_token}"}   # one token for both GPU services
    # A cold GPU container loads 60 GB of weights first; Modal answers long requests with a
    # redirect to poll, hence the generous timeout and follow_redirects.
    r = httpx.post(f"{settings.jevany_url}/decide", json=body, headers=headers,
                   timeout=900, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def _mock(domain: Domain, goal: str, has_image: bool) -> dict:
    target = dict(domain.letters())[mock_letter(domain, goal, has_image)]
    rest = (1 - 0.9) / (len(domain.actions) - 1)
    return {"choice": target.name, "latency_ms": 0,
            "probabilities": {a.name: 0.9 if a is target else rest for a in domain.actions}}


def decide_jevany(settings: Settings, domain: Domain, goal: str, image_data_url: str | None) -> Decision:
    blind = image_data_url is None
    if not settings.jevany_enabled:
        return finish(domain, None, "", 0, blind, "JevAny is not configured (JEVANY_URL)", "jevany")
    t0 = time.perf_counter()
    try:
        data = (_mock(domain, goal, not blind) if settings.mock
                else _remote(settings, request_body(domain, goal, image_data_url)))
        by_name = {a.name: a for a in domain.actions}
        chosen = by_name[data["choice"]]          # an unknown label is an upstream error
        dist = data["probabilities"]
        probs = {a.id: round(float(dist.get(a.name, 0.0)), 4) for a in domain.actions}
    except (httpx.HTTPError, KeyError, ValueError, TypeError, AttributeError) as e:
        return finish(domain, None, "", int((time.perf_counter() - t0) * 1000), blind,
                      public_error(e), "jevany")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    order = sorted(domain.actions, key=lambda a: -probs[a.id])
    raw = "\n".join(f"{a.name:14s} p={probs[a.id]:.3f}" for a in order)
    if data.get("latency_ms") is not None:
        raw += f"\n(model time on the GPU: {data['latency_ms']} ms)"
    # The model's own pick, no margin gate: its probabilities come from a trained head,
    # and a refusal threshold belongs on the labeled eval set, not here.
    return finish(domain, chosen.id, raw, latency_ms, blind, None, "jevany", probs)

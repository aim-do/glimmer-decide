"""Generative Glimmer ("System 2"): camera frame + goal -> one action from a closed set.

Comparison baseline and fallback for the MetaEncoder decider (encoder.py). Hosted
Glimmer always reasons before answering, so this path is slower; it is prompted with
lettered options and must end with a single `ANSWER: <letter>` line. Anything else is
a parse failure, which we report instead of silently guessing.
"""
from __future__ import annotations

import re
import time

import httpx

from . import upstream
from .config import Settings
from .tasks import Domain
from .types import Decision, finish, public_error
from .types import mock_letter as _mock_letter

__all__ = ["Decision", "finish", "build_messages", "parse_letter", "decide"]

SYSTEM_TEMPLATE = """Reasoning strength: {reasoning}
You are the decision module of a camera assistant for a desk workspace.
You receive one camera frame and the user's goal. Choose exactly one option.

Rules:
- Only choose an object you can actually see in the frame. Do not rely on what usually is on a desk.
- If no visible object fits the goal (including its constraints), choose the "none" option.
- If the frame is too blurred, dark, cropped or occluded to decide, choose the "ask for another angle" option.
- If several visible objects fit, choose the one that fits the constraints best.

Output format: after any thinking, the final line must be exactly
ANSWER: <letter>
and nothing else."""

ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*\(?([A-Z])\)?", re.IGNORECASE)


def build_messages(domain: Domain, goal: str, image_data_url: str | None,
                   reasoning: str) -> list[dict]:
    options = "\n".join(f"{letter}) {a.label}" for letter, a in domain.letters())
    text = f"Goal: {goal.strip()}\n\nOptions:\n{options}"
    if image_data_url is None:
        # Blind ablation: same question, no frame. Measures how much the camera matters.
        text = "(No camera frame is available.)\n" + text
        user_content: list[dict] | str = text
    else:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": text},
        ]
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(reasoning=reasoning)},
        {"role": "user", "content": user_content},
    ]


def parse_letter(text: str, n_options: int) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return None
    letter = matches[-1].upper()
    return letter if ord(letter) - ord("A") < n_options else None


def decide(settings: Settings, domain: Domain, goal: str,
           image_data_url: str | None, client: httpx.Client | None = None,
           *, deadline: float | None = None) -> Decision:
    letters = dict(domain.letters())
    blind = image_data_url is None
    t0 = time.perf_counter()

    if not settings.glimmer_enabled:
        return finish(domain, None, "", 0, blind, "Glimmer is not configured (GLIMMER_API_KEY)", "glimmer")
    if settings.mock:
        letter: str | None = _mock_letter(domain, goal, not blind)
        raw = f"(mock)\nANSWER: {letter}"
        error = None
    else:
        body = {
            "model": settings.glimmer_model,
            "messages": build_messages(domain, goal, image_data_url, settings.glimmer_reasoning),
            "max_tokens": settings.glimmer_max_tokens,
            "temperature": settings.glimmer_temperature,
            "top_p": 0.95,
        }
        if "openrouter.ai" in settings.glimmer_base_url:
            # Reasoning can't be disabled on hosted Glimmer, but effort shortens it.
            body["reasoning"] = {"effort": settings.glimmer_reasoning}
        headers = {"Authorization": f"Bearer {settings.glimmer_api_key}"}
        own = client is None
        client = client or upstream.client(deadline, 120)    # 120 s, or less near the request deadline
        try:
            r = client.post(f"{settings.glimmer_base_url}/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            raw = msg.get("content") or ""
            if not ANSWER_RE.search(raw):
                # Some providers split the channels; the answer may sit in reasoning.
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                raw = (raw + "\n" + reasoning).strip()
            error = None
        except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError, AttributeError) as e:
            raw, error = "", public_error(e)
        finally:
            if own:
                client.close()
        letter = parse_letter(raw, len(letters)) if error is None else None
        if error is None and letter is None:
            error = "unparseable answer"

    latency_ms = int((time.perf_counter() - t0) * 1000)
    action_id = letters[letter].id if letter else None
    return finish(domain, action_id, raw, latency_ms, blind, error, "glimmer")

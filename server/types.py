"""Shared decision types: one return contract for every decider.

Kept separate from the Glimmer client so the MetaEncoder does not import
from the module it is compared against.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import httpx

from .tasks import ASK_VIEW, NONE, Domain

log = logging.getLogger("glimmer-decide")


@dataclass
class Decision:
    action: str | None       # action id, None on failure
    letter: str | None
    hint: str
    concept: str | None      # what SAM should highlight, if anything
    raw: str
    latency_ms: int
    blind: bool = False
    error: str | None = None
    decider: str = "glimmer"
    probs: dict[str, float] | None = None   # per-action probability (encoder only)

    def to_dict(self) -> dict:
        return asdict(self)


def public_error(e: BaseException) -> str:
    """Client-safe error: class + upstream status, never the response body."""
    log.warning("decider error: %s: %s", type(e).__name__, e)
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status:
        return f"upstream error (HTTP {status})"
    if isinstance(e, httpx.TimeoutException):
        return "upstream timed out - try again"
    return f"upstream error ({type(e).__name__})"


def finish(domain: Domain, action_id: str | None, raw: str, latency_ms: int, blind: bool,
           error: str | None, decider: str, probs: dict[str, float] | None = None) -> Decision:
    if action_id is None:
        return Decision(None, None, "Could not get a decision - try again.", None,
                        raw, latency_ms, blind, error, decider, probs)
    letter = next(key for key, a in domain.letters() if a.id == action_id)
    action = domain.by_id(action_id)
    concept = action.concept if action.id not in (NONE, ASK_VIEW) else None
    return Decision(action.id, letter, action.hint, concept, raw, latency_ms, blind, error, decider, probs)


# Keyword -> object-name fragment. Matched against the domain's own action
# names, so a new domain gets sensible mocks without code edits.
_MOCK_KEYWORDS = [("label", "marker"), ("write", "marker"), ("cut", "scissors"),
                  ("seal", "tape"), ("without punching", "binder clip"),
                  ("staple", "stapler"), ("mail", "envelope"), ("glue", "glue")]


def mock_letter(domain: Domain, goal: str, has_image: bool) -> str:
    """Deterministic keyword stub so the UI and tests run without an API key."""
    g = goal.lower()
    target = ASK_VIEW if not has_image else NONE
    for kw, name_part in _MOCK_KEYWORDS:
        if kw in g:
            for a in domain.actions:
                if name_part in a.name:
                    target = a.id
                    break
            break
    for letter, a in domain.letters():
        if a.id == target:
            return letter
    return domain.letters()[-1][0]

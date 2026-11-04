"""Upstream HTTP calls bounded by one request deadline, redirect hops included.

/api/decide gives every decider and SAM the same deadline (REQUEST_TIMEOUT_S). Each call gets the
time that is left as its timeout, and a request hook re-applies the remaining time before every
hop: Modal answers long requests with a redirect to poll, and each hop would otherwise restart the
clock. A stuck GPU or API therefore frees its worker thread at the deadline, not minutes later
(the services' own 900 s budgets for cold starts still apply when no deadline is given, e.g. eval).
"""
from __future__ import annotations

import time

import httpx

MIN_S = 0.5   # never hand httpx a zero or negative timeout; the hook refuses expired requests


def _now() -> float:
    return time.monotonic()


def time_left(deadline: float | None, cap: float) -> float:
    """Seconds this call may take: the cap, or less when the request deadline is closer."""
    if deadline is None:
        return cap
    return max(MIN_S, min(cap, deadline - _now()))


def client(deadline: float | None, cap: float, *, follow_redirects: bool = False,
           transport: httpx.BaseTransport | None = None) -> httpx.Client:
    def bound(request: httpx.Request) -> None:
        if deadline is None:
            return
        left = deadline - _now()
        if left <= 0:
            raise httpx.TimeoutException("request deadline passed", request=request)
        request.extensions["timeout"] = httpx.Timeout(min(cap, left)).as_dict()
    return httpx.Client(timeout=time_left(deadline, cap), follow_redirects=follow_redirects,
                        event_hooks={"request": [bound]}, transport=transport)


def post(url: str, *, json: dict, headers: dict, deadline: float | None, cap: float) -> httpx.Response:
    """POST to one of our Modal web endpoints: follows its poll redirects, within the deadline."""
    with client(deadline, cap, follow_redirects=True) as c:
        return c.post(url, json=json, headers=headers)

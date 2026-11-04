"""Review fixes: one hard deadline for every upstream call of a request; the encoder cache race."""
import os
import time
from dataclasses import replace

os.environ["MOCK"] = "1"

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server.app as appmod  # noqa: E402
from server import encoder, upstream  # noqa: E402
from server.types import public_error  # noqa: E402
from tests.test_jevany import PIXEL  # noqa: E402


def test_time_left_is_capped_by_the_deadline():
    assert upstream.time_left(None, 900) == 900
    now = time.monotonic()
    assert 9 < upstream.time_left(now + 10, 900) <= 10
    assert upstream.time_left(now + 5000, 900) == 900
    assert upstream.time_left(now - 5, 900) == upstream.MIN_S


def test_an_expired_deadline_sends_nothing():
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200))
    with upstream.client(time.monotonic() - 1, 60, transport=transport) as c:
        with pytest.raises(httpx.TimeoutException):
            c.post("https://x.test/decide", json={})
    assert calls == []
    assert public_error(httpx.TimeoutException("t")) == "upstream timed out - try again"


def test_each_redirect_hop_gets_only_the_time_left(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(upstream, "_now", lambda: clock[0])
    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"]["read"])
        if request.url.path == "/decide":
            clock[0] += 20                  # Modal held the first hop 20 s, then redirects to poll
            return httpx.Response(303, headers={"location": "https://x.test/poll"})
        return httpx.Response(200, json={"ok": True})
    with upstream.client(1030.0, 900, follow_redirects=True, transport=httpx.MockTransport(handler)) as c:
        assert c.post("https://x.test/decide", json={}).json() == {"ok": True}
    assert seen == [30.0, 10.0]             # 30 s left, then 10 s: no hop restarts the clock
    clock[0] = 2000.0                       # past the deadline: the poll would not even start
    with upstream.client(1030.0, 900, follow_redirects=True, transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(httpx.TimeoutException):
            c.post("https://x.test/decide", json={})


def test_decide_hands_one_deadline_to_its_deciders(monkeypatch):
    got = {}

    def fake(settings, dom, goal, image, *, deadline):
        got["deadline"] = deadline
        return appmod.finish(dom, "TAKE_MARKER", "", 1, image is None, None, "glimmer")
    monkeypatch.setitem(appmod.DECIDERS, "glimmer", fake)
    t0 = time.monotonic()
    r = TestClient(appmod.app).post("/api/decide", json={"goal": "Label this box", "image": PIXEL,
                                                          "decider": "glimmer", "highlight": False})
    assert r.status_code == 200 and r.json()["decisions"][0]["action"] == "TAKE_MARKER"
    budget = appmod.settings.request_timeout_s
    assert t0 + budget <= got["deadline"] <= time.monotonic() + budget


def test_highlight_is_skipped_when_no_time_is_left(monkeypatch):
    def fake(settings, dom, goal, image, *, deadline):
        return appmod.finish(dom, "TAKE_SCISSORS", "", 1, False, None, "glimmer")

    def no_sam(*a, **k):
        raise AssertionError("SAM must not start without time left")
    monkeypatch.setitem(appmod.DECIDERS, "glimmer", fake)
    monkeypatch.setattr(appmod, "segment", no_sam)
    monkeypatch.setattr(appmod, "settings", replace(appmod.settings, mock=False, meta_api_key="k",
                                                    request_timeout_s=appmod.SAM_MIN_S / 2))
    r = TestClient(appmod.app).post("/api/decide", json={"goal": "Cut the paper", "image": PIXEL,
                                                          "decider": "glimmer", "highlight": True}).json()
    assert r["decisions"][0]["action"] == "TAKE_SCISSORS"
    assert r["highlight"]["error"] == "highlight skipped: out of time"


def vectors(items):
    return np.stack([encoder._hash_vec(item["text"]) for item in items])


def test_a_full_cache_never_breaks_the_current_request(monkeypatch):
    monkeypatch.setattr(encoder, "_cache", {})
    monkeypatch.setattr(encoder, "_cache_max", 3)            # smaller than one domain's options
    texts = [f"option {i}" for i in range(10)]
    m = encoder._candidate_matrix(vectors, texts, "ns")
    assert m.shape == (10, 64) and np.allclose(m[7], encoder._hash_vec("option 7"))
    assert list(encoder._cache) == ["ns|option 7", "ns|option 8", "ns|option 9"]   # oldest went first


def test_cache_hits_survive_an_eviction_by_another_request(monkeypatch):
    monkeypatch.setattr(encoder, "_cache", {})
    encoder._candidate_matrix(vectors, ["a", "b"], "ns")

    def racing(items):                                     # another request empties the cache meanwhile
        encoder._cache.clear()
        return vectors(items)
    m = encoder._candidate_matrix(racing, ["a", "b", "c"], "ns")
    assert m.shape == (3, 64) and np.allclose(m[0], encoder._hash_vec("a"))
    with pytest.raises(ValueError):
        encoder._candidate_matrix(lambda items: vectors(items)[:1], ["x", "y"], "ns")

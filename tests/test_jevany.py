"""JevAny decider (decision head on Muse Glimmer 30B), its API routes and its GPU controller."""
import os
from dataclasses import replace

os.environ["MOCK"] = "1"

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import server.jevany as jevany  # noqa: E402
from server.app import app  # noqa: E402
from server.config import load_settings  # noqa: E402
from server.encoder_ctl import JEVANY, EncoderControl  # noqa: E402
from server.tasks import load_domain  # noqa: E402

PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
         "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
LIVE = replace(load_settings(), mock=False, jevany_url="https://j.modal.run", encoder_token="t")
URL = "https://j.modal.run/decide"


def test_request_body_is_jevanys_closed_set_shape():
    d = load_domain("desk")
    body = jevany.request_body(d, "  Cut the paper  ", PIXEL)
    assert body["state"]["user goal"] == "Cut the paper" and "photo" in body["state"]["camera"]
    assert body["instructions"] == jevany.INSTRUCTIONS and body["image"] == PIXEL
    assert list(body["criteria"]) == [a.name for a in d.actions]
    assert {"none", "unclear view"} <= set(body["criteria"])
    blind = jevany.request_body(d, "Cut the paper", None)
    assert "image" not in blind and "no camera frame" in blind["state"]["camera"]


def fake_post(monkeypatch, response):
    sent = {}

    def post(url, *, json, headers, deadline, cap):
        sent.update(url=url, body=json, auth=headers["Authorization"], deadline=deadline, cap=cap)
        return response
    monkeypatch.setattr(jevany.upstream, "post", post)
    return sent


def test_answer_maps_labels_back_to_actions(monkeypatch):
    d = load_domain("desk")
    probs = {a.name: 0.01 for a in d.actions} | {"scissors": 0.9}
    sent = fake_post(monkeypatch, httpx.Response(
        200, json={"choice": "scissors", "probabilities": probs, "latency_ms": 41.5},
        request=httpx.Request("POST", URL)))
    dec = jevany.decide_jevany(LIVE, d, "Cut the paper", PIXEL)
    assert sent["url"] == URL and sent["auth"] == "Bearer t" and sent["body"]["image"] == PIXEL
    assert dec.action == "TAKE_SCISSORS" and dec.decider == "jevany" and dec.error is None
    assert dec.probs["TAKE_SCISSORS"] == 0.9 and set(dec.probs) == {a.id for a in d.actions}
    assert "model time on the GPU: 41.5 ms" in dec.raw


def test_unknown_label_and_http_errors_become_errors(monkeypatch):
    d = load_domain("desk")
    fake_post(monkeypatch, httpx.Response(200, json={"choice": "hammer", "probabilities": {}},
                                          request=httpx.Request("POST", URL)))
    assert jevany.decide_jevany(LIVE, d, "Cut", PIXEL).action is None
    fake_post(monkeypatch, httpx.Response(503, request=httpx.Request("POST", URL)))
    dec = jevany.decide_jevany(LIVE, d, "Cut", PIXEL)
    assert dec.action is None and dec.error == "upstream error (HTTP 503)"


def test_api_offers_jevany_and_compares_it():
    c = TestClient(app)
    assert c.get("/api/status").json()["jevany"] is True
    r = c.post("/api/decide", json={"goal": "Label this box", "image": PIXEL, "decider": "jevany"}).json()
    assert r["decisions"][0]["decider"] == "jevany" and r["decisions"][0]["action"] == "TAKE_MARKER"


def test_gpu_routes_cover_both_services():
    c = TestClient(app)
    assert c.get("/api/gpu/jevany").json() == {"available": False, "reason": "Mock mode: no GPU is used."}
    assert c.get("/api/gpu/nope").status_code == 404
    assert c.post("/api/gpu/jevany/start").status_code == 403          # UI header required
    hdr = {"X-Requested-With": "glimmer-decide"}
    assert c.post("/api/gpu/jevany/start", headers=hdr).status_code == 400   # mock: nothing to start


class FakeModal:
    def __init__(self):
        self.calls, self.apps = [], []

    def __call__(self, args, timeout):
        import json
        import subprocess
        self.calls.append(args)
        if args[:2] == ["app", "list"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(self.apps), "")
        if args[0] == "deploy":
            self.apps = [{"description": JEVANY.app_name, "state": "deployed", "tasks": "1"}]
        if args[:2] == ["app", "stop"]:
            self.apps = []
        return subprocess.CompletedProcess(args, 0, "", "")


def test_jevany_controller_uses_its_own_app_and_url():
    fake, health = FakeModal(), []

    class Resp:
        status_code = 200
    ctl = EncoderControl(lambda: LIVE, run=fake, spawn=lambda fn, *a: fn(*a), logged_in=lambda: True,
                         http_get=lambda url, **k: health.append((url, k["timeout"])) or Resp(), service=JEVANY)
    ctl.start()
    assert fake.calls[0][1].endswith("modal_jevany.py")
    assert health == [("https://j.modal.run/health", JEVANY.health_timeout_s)]
    assert ctl.status()["containers"] == 1
    ctl.stop()
    assert ["app", "stop", "--yes", "glimmer-decide-jevany"] in fake.calls


def test_jevany_setup_hint_names_its_own_command():
    not_set_up = replace(LIVE, jevany_url="")
    ctl = EncoderControl(lambda: not_set_up, run=FakeModal(), logged_in=lambda: True, service=JEVANY)
    assert "setup_modal.py jevany" in ctl.status()["reason"]

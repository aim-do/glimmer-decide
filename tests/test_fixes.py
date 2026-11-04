"""Regression tests for the deep-review fix pass."""
import os
from dataclasses import replace

os.environ["MOCK"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server import encoder  # noqa: E402
from server.app import app  # noqa: E402
from server.config import load_settings  # noqa: E402
from server.decision import decide  # noqa: E402
from server.sam import parse_boxes, segment  # noqa: E402
from server.tasks import load_domain  # noqa: E402

PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
         "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload

    def post(self, *a, **k):
        return _Resp(self._payload)

    def close(self):
        pass


def test_glimmer_empty_choices_is_error_not_crash(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("GLIMMER_API_KEY", "test-key")
    s = load_settings()
    d = decide(s, load_domain("desk"), "label it", PIXEL, client=_Client({"choices": []}))
    assert d.action is None and d.error


def test_sam_malformed_response_is_error_not_crash(monkeypatch):
    monkeypatch.setenv("MOCK", "0")
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    s = load_settings()
    out = segment(s, "scissors", PIXEL, client=_Client(["not-a-dict"]))
    assert out["boxes"] == [] and out["error"]


def test_parse_boxes_bad_confidence_keeps_good_boxes():
    line = ("0<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10;c=high|>,"
            "1<|box;x1=1.0;y1=2;x2=30;y2=40;w=100;h=100;c=0.5|>")
    boxes = parse_boxes(line)
    assert len(boxes) == 2 and boxes[1]["x2"] == 30


def test_config_rejects_bad_decider_and_tau(monkeypatch):
    monkeypatch.setenv("DECIDER", "encodr")
    with pytest.raises(ValueError):
        load_settings()
    monkeypatch.setenv("DECIDER", "both")
    monkeypatch.setenv("ENCODER_TAU", "0")
    with pytest.raises(ValueError):
        load_settings()
    monkeypatch.setenv("ENCODER_TAU", "0.02")
    assert load_settings().decider == "both"


def test_softmax_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        encoder.softmax(np.array([1.0, 2.0]), 0.0)


def test_domain_traversal_rejected():
    with pytest.raises((FileNotFoundError, ValueError)):
        load_domain("../.claude/launch")


def test_encoder_blind_reports_model_pick():
    # Blind ablation must measure the model, not a hard-coded override.
    d = load_domain("desk")
    s = replace(load_settings(), encoder_tau=1.0)
    out = encoder.decide_encoder(s, d, "Cut the paper", None)
    assert out.action == "TAKE_SCISSORS" and out.blind is True


def test_encoder_near_tie_asks_for_another_view():
    d = load_domain("desk")
    s = replace(load_settings(), encoder_tau=1.0, encoder_min_margin=0.99)
    out = encoder.decide_encoder(s, d, "Cut the paper", PIXEL)
    assert out.action == "ASK_VIEW"


def test_public_error_shows_status_not_body():
    import httpx
    from server.types import public_error
    req = httpx.Request("POST", "https://example.invalid")
    resp = httpx.Response(401, request=req, text="secret detail")
    err = public_error(httpx.HTTPStatusError("boom", request=req, response=resp))
    assert err == "upstream error (HTTP 401)" and "secret" not in err


def test_summarize_excludes_skipped_rows():
    from eval.run_eval import summarize
    rows = [{"gold": ["NONE"], "pred": None, "correct": False, "latency_ms": 0, "skipped": True},
            {"gold": ["NONE"], "pred": "NONE", "correct": True, "latency_ms": 5}]
    s = summarize(rows)
    assert s["n"] == 1 and s["parse_failures"] == 0 and s["accuracy"] == 1.0


def test_api_whitespace_goal_rejected_and_captures_unique(tmp_path, monkeypatch):
    import server.app as srv
    monkeypatch.setattr(srv, "CAPTURES", tmp_path / "captures")
    c = TestClient(app, raise_server_exceptions=False)
    assert c.post("/api/decide", json={"goal": "   ", "image": PIXEL}).status_code == 422
    names = set()
    for _ in range(5):
        r = c.post("/api/capture", json={"goal": "Label this box", "image": PIXEL,
                                         "gold": ["TAKE_MARKER"]})
        assert r.status_code == 200
        names.add(r.json()["saved"])
    assert len(names) == 5
    assert c.post("/api/decide", json={"goal": "x", "domain": "../.claude/launch",
                                       "image": PIXEL}).status_code in (400, 404)


def test_eval_load_records_skips_bad_lines(tmp_path):
    from eval.run_eval import load_records
    p = tmp_path / "caps.jsonl"
    p.write_text('{"image":"captures/a.jpg","domain":"desk","goal":"g","gold":["NONE"]}\n'
                 '{bad json\n\n', encoding="utf-8")
    recs, skipped = load_records(p)
    assert len(recs) == 1 and skipped == 1

import os
from dataclasses import replace

os.environ["MOCK"] = "1"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server import encoder  # noqa: E402
from server.app import app  # noqa: E402
from server.config import load_settings  # noqa: E402
from server.decision import build_messages, parse_letter  # noqa: E402
from server.sam import parse_boxes  # noqa: E402
from server.tasks import ASK_VIEW, NONE, load_domain  # noqa: E402

PIXEL = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
         "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_domain_has_refusal_options():
    d = load_domain("desk")
    ids = [a.id for a in d.actions]
    assert ids[-2:] == [NONE, ASK_VIEW]
    assert "TAKE_BINDER_CLIP" in ids
    assert d.letters()[0][0] == "A"
    assert all(a.name and a.detail for a in d.actions)


def test_parse_letter_takes_last_answer_and_bounds():
    assert parse_letter("maybe ANSWER: B\n...\nANSWER: d", 10) == "D"
    assert parse_letter("Answer = (C)", 10) == "C"
    assert parse_letter("I pick the marker", 10) is None
    assert parse_letter("ANSWER: Z", 10) is None


def test_build_messages_image_and_blind():
    d = load_domain("desk")
    with_img = build_messages(d, "label the box", PIXEL, "low")
    assert with_img[0]["content"].startswith("Reasoning strength: low")
    assert with_img[1]["content"][0]["type"] == "image_url"
    blind = build_messages(d, "label the box", None, "low")
    assert isinstance(blind[1]["content"], str) and "No camera frame" in blind[1]["content"]


def test_parse_sam_boxes():
    line = ("<0f>0<|box;x1=181;y1=365;x2=556;y2=486;w=1516;h=1600;c=0.91|><|mask;x=0;y=0;data=1,1,!0|>,"
            "3<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|><|mask;x=0;y=0;data=1,1,~0|>")
    boxes = parse_boxes(line)
    assert [b["id"] for b in boxes] == [0, 3]
    assert boxes[0]["c"] == 0.91 and boxes[0]["w"] == 1516
    assert parse_boxes("") == []


def test_encoder_task_format_and_candidates():
    d = load_domain("desk")
    t = encoder.task_text(d, " Label this box ")
    assert t.startswith("User goal: Label this box\n\nTask: Select the correct option")
    assert "\nCriteria:\n- binder clip: holds" in t and "- none: " in t
    assert encoder.candidate_texts(d, "label")[0] == "binder clip"
    assert encoder.candidate_texts(d, "label_detail")[0].startswith("binder clip: ")


def test_encoder_cosine_softmax_and_cache():
    d = load_domain("desk")
    names = encoder.candidate_texts(d, "label")
    vecs = {n: v for n, v in zip(names, np.eye(len(names), dtype=np.float32))}
    calls = []

    def fake_embed(items):
        calls.append(len(items))
        out = []
        for it in items:
            if it["text"].startswith("User goal:"):
                assert it.get("image") == PIXEL
                v = vecs["scissors"] + 0.5 * vecs["marker"]
                out.append(v / np.linalg.norm(v))
            else:
                out.append(vecs[it["text"]])
        return np.stack(out)

    s = replace(load_settings(), encoder_tau=1.0)   # soft enough that small probs don't round to 0
    encoder._cache.clear()
    a = encoder.decide_encoder(s, d, "Cut along the line", PIXEL, embed=fake_embed)
    b = encoder.decide_encoder(s, d, "Cut along the line", PIXEL, embed=fake_embed)
    assert a.action == b.action == "TAKE_SCISSORS" and a.decider == "encoder"
    assert a.concept == "scissors"
    assert abs(sum(a.probs.values()) - 1) < 1e-3
    assert a.probs["TAKE_SCISSORS"] > a.probs["TAKE_MARKER"] > a.probs["TAKE_STAPLER"]
    assert calls == [len(names), 1, 1]       # candidates embedded once, then only the task


def test_api_decide_both_and_capture(tmp_path, monkeypatch):
    import server.app as srv
    monkeypatch.setattr(srv, "CAPTURES", tmp_path / "captures")
    c = TestClient(app)
    st = c.get("/api/status").json()
    assert st["mock"] is True and st["encoder"] and st["glimmer"]
    r = c.post("/api/decide", json={"goal": "Label this box", "image": PIXEL, "decider": "both"}).json()
    assert [d["decider"] for d in r["decisions"]] == ["encoder", "glimmer"]
    assert all(d["action"] == "TAKE_MARKER" for d in r["decisions"])
    assert r["decisions"][0]["probs"]["TAKE_MARKER"] == max(r["decisions"][0]["probs"].values())
    blind = c.post("/api/decide", json={"goal": "something odd", "image": None, "decider": "glimmer"}).json()
    assert blind["decisions"][0]["blind"] is True
    assert c.post("/api/decide", json={"goal": "x", "image": "not-a-data-url"}).status_code == 400
    saved = c.post("/api/capture", json={"goal": "Label this box", "image": PIXEL,
                                         "gold": ["TAKE_MARKER"]}).json()
    assert (tmp_path / saved["saved"]).exists()
    assert (tmp_path / "captures.jsonl").read_text().count("TAKE_MARKER") == 1


def test_unconfigured_deciders_report_errors(monkeypatch):
    for k in ("MOCK", "ENCODER_URL", "GLIMMER_API_KEY"):
        monkeypatch.setenv(k, "0" if k == "MOCK" else "")
    s = load_settings()
    d = load_domain("desk")
    assert not s.mock and not s.encoder_enabled and not s.glimmer_enabled
    assert "not configured" in encoder.decide_encoder(s, d, "x", PIXEL).error
    from server.decision import decide
    assert "not configured" in decide(s, d, "x", PIXEL).error

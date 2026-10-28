import os

os.environ["MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402
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


def test_api_decide_and_capture(tmp_path, monkeypatch):
    import server.app as srv
    monkeypatch.setattr(srv, "CAPTURES", tmp_path / "captures")
    c = TestClient(app)
    assert c.get("/api/status").json()["mock"] is True
    r = c.post("/api/decide", json={"goal": "Label this box", "image": PIXEL}).json()
    assert r["action"] == "TAKE_MARKER" and r["concept"] == "marker"
    blind = c.post("/api/decide", json={"goal": "something odd", "image": None}).json()
    assert blind["blind"] is True
    assert c.post("/api/decide", json={"goal": "x", "image": "not-a-data-url"}).status_code == 400
    saved = c.post("/api/capture", json={"goal": "Label this box", "image": PIXEL,
                                         "gold": ["TAKE_MARKER"]}).json()
    assert (tmp_path / saved["saved"]).exists()
    assert (tmp_path / "captures.jsonl").read_text().count("TAKE_MARKER") == 1

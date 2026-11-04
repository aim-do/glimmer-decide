"""JevAny fine-tuning: capture -> training rows, the GPU job's settings, serving a trained run."""
import importlib.util
import json
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import httpx
import pytest

import server.encoder_ctl as ctl
import server.jevany as jevany
from server.tasks import load_domain
from tests.test_jevany import LIVE, PIXEL, URL, fake_post
from train.jevany_data import captures_dataset, from_records, smoke_dataset, smoke_png

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"


def test_training_row_matches_what_the_service_is_asked():
    d = load_domain("desk")
    row = jevany.training_row(d, "  Cut the paper  ", "images/a.jpg", ["TAKE_SCISSORS"])
    asked = jevany.request_body(d, "Cut the paper", PIXEL)
    q = row["questions"][jevany.QUESTION]
    assert row["state"] == asked["state"] and q["instructions"] == asked["instructions"]
    assert q["criteria"] == asked["criteria"] and q["label"] == "scissors" and "target" not in q
    assert row["media"] == [{"type": "image", "uri": "images/a.jpg"}] and q["type"] == "choice"


def test_several_correct_actions_share_a_soft_target():
    d = load_domain("desk")
    q = jevany.training_row(d, "Hold sheets", "images/a.jpg",
                            ["TAKE_BINDER_CLIP", "TAKE_STAPLER", "TAKE_BINDER_CLIP"])["questions"]["action"]
    assert q["label"] == "binder clip" and q["target"] == {"binder clip": 1.0, "stapler": 1.0}
    with pytest.raises(KeyError):
        jevany.training_row(d, "Cut", "images/a.jpg", ["TAKE_HAMMER"])
    with pytest.raises(ValueError):
        jevany.training_row(d, "Cut", "images/a.jpg", [])


def test_bad_capture_lines_are_skipped_with_a_reason():
    reads = []

    def read(rel):
        reads.append(rel)
        return b"img:" + rel.encode()
    ds = from_records(enumerate([
        {"image": "captures/a.jpg", "domain": "desk", "goal": "Cut the paper", "gold": ["TAKE_SCISSORS"]},
        {"image": "captures/a.jpg", "domain": "desk", "goal": "Label this box", "gold": ["TAKE_MARKER"]},
        {"image": "other/a.jpg", "goal": "Seal the package", "gold": ["TAKE_TAPE"]},   # domain defaults to desk
        None,
        {"image": "captures/b.jpg", "goal": "Cut", "gold": []},
        {"image": "captures/b.jpg", "goal": " ", "gold": ["NONE"]},
        {"image": "captures/b.txt", "goal": "Cut", "gold": ["NONE"]},
        {"image": "captures/b.jpg", "domain": "kitchen", "goal": "Cut", "gold": ["NONE"]},
        {"image": "captures/b.jpg", "goal": "Cut", "gold": ["TAKE_HAMMER"]},
    ], 1), read)
    assert len(ds.rows) == 3 and reads == ["captures/a.jpg", "other/a.jpg"]   # a.jpg read once
    assert [r["media"][0]["uri"] for r in ds.rows] == ["images/a.jpg", "images/a.jpg", "images/1-a.jpg"]
    assert set(ds.images) == {"a.jpg", "1-a.jpg"} and ds.images["1-a.jpg"] == b"img:other/a.jpg"
    reasons = " | ".join(ds.skipped)
    assert len(ds.skipped) == 6
    for why in ("not a JSON object", "no correct action", "no goal", "not an image", "kitchen", "TAKE_HAMMER"):
        assert why in reasons


def test_captures_file_reads_images_only_inside_its_folder(tmp_path):
    data = tmp_path / "data"
    (data / "captures").mkdir(parents=True)
    (data / "captures" / "x.jpg").write_bytes(b"jpeg")
    (tmp_path / "secret.jpg").write_bytes(b"not yours")
    lines = [json.dumps({"image": "captures/x.jpg", "domain": "desk", "goal": "Label this box",
                         "gold": ["TAKE_MARKER"], "predicted": "TAKE_TAPE"}),
             "", "{broken",
             json.dumps({"image": "../secret.jpg", "goal": "Cut", "gold": ["NONE"]}),
             json.dumps({"image": "captures/missing.jpg", "goal": "Cut", "gold": ["NONE"]})]
    (data / "captures.jsonl").write_text("\n".join(lines), encoding="utf-8")
    ds = captures_dataset(data / "captures.jsonl")
    assert len(ds.rows) == 1 and ds.images == {"x.jpg": b"jpeg"}
    assert [s.split(":")[0] for s in ds.skipped] == ["line 3", "line 4", "line 5"]   # file line numbers
    assert "not a JSON object" in ds.skipped[0] and "outside" in ds.skipped[1]


def png_pixels(png: bytes):
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    raw = zlib.decompress(png[png.index(b"IDAT") + 4: png.index(b"IEND") - 8])
    stride = 1 + 3 * width
    return width, height, lambda x, y: tuple(raw[y * stride + 1 + 3 * x: y * stride + 4 + 3 * x])


def test_smoke_set_is_one_empty_frame_whose_answer_is_none():
    ds = smoke_dataset()
    assert len(ds.rows) == 1 and not ds.skipped and list(ds.images) == ["smoke-frame.png"]
    q = ds.rows[0]["questions"]["action"]
    assert q["label"] == "none" and ds.rows[0]["state"]["user goal"] == "Cut the paper along a straight line"
    width, height, pixel = png_pixels(smoke_png())
    assert (width, height) == (320, 240)
    assert pixel(150, 120) == (200, 40, 40) and pixel(100, 120) == (128, 128, 128)
    assert pixel(219, 159) == (200, 40, 40) and pixel(220, 120) == pixel(150, 80) == (128, 128, 128)


def test_answer_names_a_fine_tuned_run(monkeypatch):
    d = load_domain("desk")
    probs = {a.name: 0.01 for a in d.actions} | {"none": 0.9}
    fake_post(monkeypatch, httpx.Response(
        200, json={"choice": "none", "probabilities": probs, "latency_ms": 9, "run": "desk-v1"},
        request=httpx.Request("POST", URL)))
    dec = jevany.decide_jevany(LIVE, d, "Cut the paper", PIXEL)
    assert dec.action == "NONE" and "fine-tuned run: desk-v1" in dec.raw


def test_start_reads_the_run_from_env_file_without_restart(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("JEVANY_RUN=desk-v1\nJEVANY_GPU=\nHF_TOKEN=secret\n", encoding="utf-8")
    assert ctl.deploy_env(env) == {"JEVANY_RUN": "desk-v1", "JEVANY_GPU": ""}   # nothing else leaks in
    assert ctl.deploy_env(tmp_path / "missing.env") == {}
    seen = {}

    def run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ctl, "ENV_FILE", env)
    monkeypatch.setattr(ctl.subprocess, "run", run)
    monkeypatch.setenv("ENCODER_GPU", "A100-80GB")
    ctl.run_modal(["deploy", "x.py"], 5)
    assert seen["JEVANY_RUN"] == "desk-v1" and seen["ENCODER_GPU"] == "A100-80GB"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_service_serves_the_run_named_in_jevany_run(monkeypatch):
    pytest.importorskip("modal")
    monkeypatch.delenv("JEVANY_RUN", raising=False)
    released = load_module("modal_jevany_released", DEPLOY / "modal_jevany.py")
    assert released.CHECKPOINT == f"{released.ADAPTER}@{released.ADAPTER_REV}" and released.RUN == ""
    monkeypatch.setenv("JEVANY_RUN", "desk-v1")
    tuned = load_module("modal_jevany_tuned", DEPLOY / "modal_jevany.py")
    assert tuned.CHECKPOINT == "/cache/runs/desk-v1/model"
    monkeypatch.setenv("JEVANY_RUN", "../../etc")
    with pytest.raises(ValueError):
        load_module("modal_jevany_bad", DEPLOY / "modal_jevany.py")


def test_training_job_warm_starts_the_released_architecture(monkeypatch):
    pytest.importorskip("modal")
    import dotenv

    monkeypatch.delenv("JEVANY_RUN", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)   # keep the real .env out of tests
    monkeypatch.setattr(sys, "path", list(sys.path))
    job = load_module("modal_jevany_train_test", DEPLOY / "modal_jevany_train.py")
    args = job.trainer_args("d/train.jsonl", "runs/x/model",
                            {"epochs": 20, "max_steps": 0, "lr": 1e-4, "head_lr": 1e-4, "accum": 1})
    assert "--multimodal" in args
    pairs = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--") and
             not args[i + 1].startswith("--")}
    assert pairs["--init_from"] == f"{job.svc.ADAPTER}@{job.svc.ADAPTER_REV}"
    assert pairs["--base_revision"] == job.svc.BASE_REV and pairs["--lora"] == "8"
    assert pairs["--head_dim"] == pairs["--head_residual_dim"] == "256"
    assert pairs["--branch_mode"] == "rows" and pairs["--weights_dtype"] == "bf16"
    assert pairs["--p_none"] == pairs["--p_distract"] == "0" and pairs["--epochs"] == "20"
    assert job.SMOKE["accum"] == 1 and job.app.name == "glimmer-decide-jevany-train"
    summary = {"run": "r", "gpu": "H100", "records": 1, "optimizer_steps": 2, "peak_gpu_gb": 70.1,
               "seconds": {"train": 3}, "losses": [2.0, 0.5],
               "before": [{"goal": "Cut", "gold": ["none"], "choice": "scissors", "p_gold": 0.04}],
               "after": [{"goal": "Cut", "gold": ["none"], "choice": "none", "p_gold": 0.97}]}
    text = job.report(summary)
    assert "2.000 0.500" in text and "before scissors (p correct 0.040) -> after none (p correct 0.970)" in text


def test_env_example_documents_jevany_run():
    text = (DEPLOY.parent / ".env.example").read_text(encoding="utf-8")
    assert "\nJEVANY_RUN=\n" in text

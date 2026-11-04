"""MetaEncoder GPU Start/Stop: controller logic with a fake Modal CLI, plus the API guard."""
import json
import os
import subprocess
from dataclasses import replace

os.environ["MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from server.__main__ import ensure_env_file  # noqa: E402
from server.app import app  # noqa: E402
from server.config import load_settings  # noqa: E402
from server.encoder_ctl import APP_NAME, Busy, ControlError, EncoderControl  # noqa: E402

LIVE = replace(load_settings(), mock=False, encoder_url="https://x.modal.run", encoder_token="t")


class FakeModal:
    """Records CLI calls; answers `app list` from self.apps."""

    def __init__(self, deploy_rc=0, stop_out=""):
        self.calls, self.apps, self.deploy_rc, self.stop_out = [], [], deploy_rc, stop_out

    def __call__(self, args, timeout):
        self.calls.append(args[:2])
        if args[:2] == ["app", "list"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(self.apps), "")
        if args[0] == "deploy":
            self.apps = [{"description": APP_NAME, "state": "deployed", "tasks": "0"}]
            return subprocess.CompletedProcess(args, self.deploy_rc, "", "Secret 'huggingface' not found"
                                               if self.deploy_rc else "")
        if args[:2] == ["app", "stop"]:
            self.apps = []
            return subprocess.CompletedProcess(args, 1 if self.stop_out else 0, self.stop_out, "")
        raise AssertionError(args)


class Resp:
    def __init__(self, code):
        self.status_code = code


def ctl_with(fake, health=200, settings=LIVE):
    return EncoderControl(lambda: settings, run=fake, http_get=lambda *a, **k: Resp(health),
                          spawn=lambda fn, *a: fn(*a), logged_in=lambda: True)


def test_start_deploys_then_warms_and_reports_running():
    fake = FakeModal()
    ctl = ctl_with(fake)
    assert ctl.start() == {"started": True}
    assert fake.calls[0][0] == "deploy" and fake.calls[0][1].endswith("modal_metaencoder.py")
    fake.apps[0]["tasks"] = "1"       # the warmed container now runs
    s = ctl.status()
    assert s["available"] and s["op"] == "idle" and s["error"] is None
    assert s["deployed"] is True and s["containers"] == 1


def test_stop_undeploys_and_treats_already_stopped_as_success():
    fake = FakeModal(stop_out="App is already stopped.")
    ctl = ctl_with(fake)
    assert ctl.stop() == {"stopping": True}
    assert ["app", "stop"] in fake.calls
    s = ctl.status()
    assert s["error"] is None and s["deployed"] is False and s["containers"] == 0


def test_deploy_failure_explains_missing_secrets():
    ctl = ctl_with(FakeModal(deploy_rc=1))
    ctl.start()
    assert "setup_modal.py all" in ctl.status()["error"]


def test_bad_token_on_health_is_explained():
    ctl = ctl_with(FakeModal(), health=401)
    ctl.start()
    assert "ENCODER_TOKEN" in ctl.status()["error"]


def test_busy_and_superseded_jobs():
    fake = FakeModal()
    jobs = []
    ctl = EncoderControl(lambda: LIVE, run=fake, http_get=lambda *a, **k: Resp(200),
                         spawn=lambda fn, *a: jobs.append((fn, a)), logged_in=lambda: True)
    ctl.start()
    try:
        ctl.start()
        raise AssertionError("second start must be refused")
    except Busy:
        pass
    try:
        ctl.stop()                     # refused while the deploy runs
        raise AssertionError("stop during deploy must be refused")
    except Busy:
        pass
    start_fn, start_args = jobs[0]
    with ctl._lock:
        ctl.op = "warming"             # deploy done, GPU warming: Stop is allowed now
    ctl.stop()
    stop_fn, stop_args = jobs[1]
    stop_fn(*stop_args)
    ctl._finish(start_args[0], "late warm-up error")   # the cut-short start must not overwrite
    assert ctl.op == "idle" and ctl.error is None


def test_unavailable_without_setup_or_in_mock():
    not_set_up = replace(LIVE, encoder_url="", encoder_token="")
    ctl = EncoderControl(lambda: not_set_up, run=FakeModal(), logged_in=lambda: True)
    assert "setup_modal.py all" in ctl.status()["reason"]
    try:
        ctl.start()
        raise AssertionError("start without setup must be refused")
    except ControlError as e:
        assert e.status == 400
    assert ctl_with(FakeModal(), settings=load_settings()).status()["reason"].startswith("Mock")


def test_api_requires_ui_header_for_paid_posts():
    c = TestClient(app)
    for path in ("/api/encoder/start", "/api/encoder/stop", "/api/warmup"):
        assert c.post(path).status_code == 403
    hdr = {"X-Requested-With": "glimmer-decide"}
    assert c.post("/api/encoder/start", headers=hdr).status_code == 400    # mock: nothing to start
    assert c.get("/api/encoder").json() == {"available": False, "reason": "Mock mode: no GPU is used."}


def test_first_run_creates_env_from_example(tmp_path):
    (tmp_path / ".env.example").write_text("MOCK=0\n", encoding="utf-8")
    assert ensure_env_file(tmp_path) is True
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "MOCK=0\n"
    assert ensure_env_file(tmp_path) is False

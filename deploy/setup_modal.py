"""Set up and operate the GPU services (MetaEncoder, JevAny) on your own Modal account. Never prints tokens.

  python deploy/setup_modal.py all       # one-time MetaEncoder setup: login, secrets, deploy, ENCODER_URL/TOKEN in .env
  python deploy/setup_modal.py jevany    # one-time JevAny setup: deploy, ~60 GB of weights on a CPU box, JEVANY_URL
  python deploy/setup_modal.py status    # deployed? GPU containers running? (both services)
  python deploy/setup_modal.py start [encoder|jevany]   # deploy (the page's Start button also warms the GPU)
  python deploy/setup_modal.py stop      # stop both GPUs and undeploy: nothing can wake them or bill

Steps of `all`, also runnable one by one:
  login     `modal token new` in the browser if this machine has no Modal token yet
  secrets   HF token (.env HF_TOKEN or `hf auth login`) + a fresh ENCODER_TOKEN -> Modal secrets and .env
  deploy    `modal deploy deploy/modal_metaencoder.py` (ENCODER_GPU in .env picks the GPU, default H100)
  url       writes the deployed ENCODER_URL into .env
"""
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
DEPLOY_FILE = ROOT / "deploy" / "modal_metaencoder.py"
APP, CLS = "glimmer-decide-metaencoder", "MetaEncoderService"
JEVANY_FILE = ROOT / "deploy" / "modal_jevany.py"
JEVANY_APP, JEVANY_CLS = "glimmer-decide-jevany", "JevAnyService"
APPS = (APP, JEVANY_APP)


def modal_cli(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    # The environment is read per call, after __main__ loaded .env: ENCODER_GPU, JEVANY_GPU and
    # JEVANY_RUN from .env then reach `modal deploy`, which reads them.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-m", "modal", *args], cwd=ROOT, env=env,
                          capture_output=capture, text=True, encoding="utf-8", errors="replace")


def ensure_env_file() -> None:
    if not ENV.exists():
        shutil.copy(ROOT / ".env.example", ENV)
        print("created .env from .env.example")


def read_env() -> dict[str, str]:
    out = {}
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def set_env(key: str, value: str) -> None:
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def put_secret(name: str, env: dict[str, str]) -> None:
    import modal

    try:
        modal.Secret.objects.delete(name)
    except Exception:
        pass
    modal.Secret.objects.create(name, env)
    print(f"modal secret '{name}': written")


def cmd_login() -> None:
    if modal_cli("token", "info", capture=True).returncode == 0:
        print("Modal login: OK")
        return
    print("No Modal token on this machine - opening the browser to log in (free account works).")
    if modal_cli("token", "new").returncode != 0 or modal_cli("token", "info", capture=True).returncode != 0:
        sys.exit("Modal login did not complete. Run `modal token new` and retry.")


def cmd_secrets() -> None:
    from huggingface_hub import HfApi, get_token

    hf = get_token()   # HF_TOKEN from .env (loaded in main) or the `hf auth login` cache
    if not hf:
        sys.exit("No Hugging Face token: put HF_TOKEN=... in .env or run `hf auth login` (see README).")
    try:
        HfApi().auth_check("facebook/meta-encoder", token=hf)
        print("HF access to facebook/meta-encoder: OK")
    except Exception as e:
        msg = (f"HF token has no access to facebook/meta-encoder ({type(e).__name__}). Accept the terms on "
               "https://huggingface.co/facebook/meta-encoder and, for a fine-grained token, enable "
               "'Read access to contents of all public gated repos you can access'.")
        if "--force" not in sys.argv:
            sys.exit(msg)
        # Editing a fine-grained token's permissions keeps its value, so the secret stays valid.
        print("WARNING: " + msg + " Writing the secret anyway (--force).")
    put_secret("huggingface", {"HF_TOKEN": hf})

    token = read_env().get("ENCODER_TOKEN") or secrets.token_urlsafe(32)
    set_env("ENCODER_TOKEN", token)
    put_secret("glimmer-decide-auth", {"ENCODER_TOKEN": token})
    print("ENCODER_TOKEN: stored in .env and Modal")


def cmd_deploy() -> None:
    if modal_cli("deploy", str(DEPLOY_FILE)).returncode != 0:
        sys.exit("modal deploy failed (see above).")


def cmd_url() -> None:
    import modal

    url = modal.Cls.from_name(APP, CLS)().web.get_web_url()
    set_env("ENCODER_URL", url)
    print(f"ENCODER_URL={url}")


def cmd_status() -> None:
    r = modal_cli("app", "list", "--json", capture=True)
    if r.returncode != 0:
        sys.exit(r.stderr or r.stdout)
    listed = json.loads(r.stdout or "[]")
    for app in APPS:
        mine = [a for a in listed if a.get("description") == app and a.get("state") == "deployed"]
        gpus = sum(int(a.get("tasks") or 0) for a in mine)
        print(f"{app}: {'deployed' if mine else 'not deployed (stopped)'}, GPU containers running: {gpus}")


def cmd_stop() -> None:
    for app in APPS:
        r = modal_cli("app", "stop", "--yes", app, capture=True)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0 and not any(s in out.lower() for s in ("already stopped", "not found", "could not find")):
            sys.exit(out or f"modal app stop {app} failed")
        print(out or f"{app} stopped")


def cmd_start() -> None:
    which = sys.argv[2] if len(sys.argv) > 2 else "encoder"
    files = {"encoder": DEPLOY_FILE, "jevany": JEVANY_FILE}
    if which not in files:
        sys.exit("usage: python deploy/setup_modal.py start [encoder|jevany]")
    if modal_cli("deploy", str(files[which])).returncode != 0:
        sys.exit("modal deploy failed (see above).")


def ensure_service_token() -> None:
    """The bearer token both GPU services check; created here if `all` never ran."""
    token = read_env().get("ENCODER_TOKEN")
    if not token:
        token = secrets.token_urlsafe(32)
        set_env("ENCODER_TOKEN", token)
    put_secret("glimmer-decide-auth", {"ENCODER_TOKEN": token})


def cmd_jevany() -> None:
    ensure_env_file()
    cmd_login()
    ensure_service_token()
    if modal_cli("deploy", str(JEVANY_FILE)).returncode != 0:
        sys.exit("modal deploy failed (see above).")
    print("Downloading the JevAny weights (~60 GB: Muse Glimmer 30B + the adapter) into a Modal Volume\n"
          "on a cheap CPU container, so the GPU never waits for them. This takes ~10-20 min, once.")
    if modal_cli("run", f"{JEVANY_FILE}::download").returncode != 0:
        sys.exit("weight download failed (see above); rerun `python deploy/setup_modal.py jevany`.")
    import modal

    url = modal.Cls.from_name(JEVANY_APP, JEVANY_CLS)().web.get_web_url()
    set_env("JEVANY_URL", url)
    print(f"JEVANY_URL={url}")
    print("\nDone. Pick JevAny in the page and press Start in its GPU panel (~3 min, ~$4/h until Stop).")


def cmd_all() -> None:
    ensure_env_file()
    cmd_login()
    cmd_secrets()
    cmd_deploy()
    cmd_url()
    print("\nDone. Run `python -m server`, then use Start/Stop in the page's MetaEncoder panel.\n"
          "The GPU runs only after Start (or a MetaEncoder request); it sleeps after 5 idle minutes,\n"
          "and Stop shuts it down completely. JevAny, the second fast decider: "
          "`python deploy/setup_modal.py jevany`.")


COMMANDS = {"all": cmd_all, "jevany": cmd_jevany, "login": cmd_login, "secrets": cmd_secrets,
            "deploy": cmd_deploy, "start": cmd_start, "url": cmd_url, "status": cmd_status, "stop": cmd_stop}

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ENV)   # HF_TOKEN, ENCODER_GPU, ... from .env
    COMMANDS.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()

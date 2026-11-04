"""Set up and operate the MetaEncoder GPU service on your own Modal account. Never prints tokens.

  python deploy/setup_modal.py all       # one-time setup: login, secrets, deploy, write ENCODER_URL/TOKEN to .env
  python deploy/setup_modal.py status    # deployed? GPU containers running?
  python deploy/setup_modal.py start     # deploy (the web page's Start button also warms the GPU)
  python deploy/setup_modal.py stop      # stop the GPU and undeploy: nothing can wake it or bill

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
CLI_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def modal_cli(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "modal", *args], cwd=ROOT, env=CLI_ENV,
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
    mine = [a for a in json.loads(r.stdout or "[]") if a.get("description") == APP and a.get("state") == "deployed"]
    gpus = sum(int(a.get("tasks") or 0) for a in mine)
    print(f"{APP}: {'deployed' if mine else 'not deployed (stopped)'}, GPU containers running: {gpus}")


def cmd_stop() -> None:
    r = modal_cli("app", "stop", "--yes", APP, capture=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0 and not any(s in out.lower() for s in ("already stopped", "not found", "could not find")):
        sys.exit(out or "modal app stop failed")
    print(out or f"{APP} stopped")


def cmd_all() -> None:
    ensure_env_file()
    cmd_login()
    cmd_secrets()
    cmd_deploy()
    cmd_url()
    print("\nDone. Run `python -m server`, then use Start/Stop in the page's MetaEncoder panel.\n"
          "The GPU runs only after Start (or a MetaEncoder request); it sleeps after 5 idle minutes,\n"
          "and Stop shuts it down completely.")


COMMANDS = {"all": cmd_all, "login": cmd_login, "secrets": cmd_secrets, "deploy": cmd_deploy,
            "start": cmd_deploy, "url": cmd_url, "status": cmd_status, "stop": cmd_stop}

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ENV)   # HF_TOKEN, ENCODER_GPU, ... from .env
    COMMANDS.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()

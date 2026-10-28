"""Modal setup for the MetaEncoder service. Never prints tokens.

  python deploy/setup_modal.py secrets   # HF token (.env HF_TOKEN or `hf auth login`) + a fresh ENCODER_TOKEN -> Modal secrets, .env
  modal deploy deploy/modal_metaencoder.py
  python deploy/setup_modal.py url       # writes the deployed ENCODER_URL into .env
  python deploy/setup_modal.py stop      # stop the GPU container now instead of waiting 5 idle minutes
"""
import json
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
APP, CLS = "glimmer-decide-metaencoder", "MetaEncoderService"


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


def cmd_secrets() -> None:
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, get_token

    load_dotenv(ENV)   # HF_TOKEN may live in .env; get_token() reads it from the environment
    hf = get_token()
    if not hf:
        sys.exit("No Hugging Face token: set HF_TOKEN in .env or run `hf auth login`.")
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


def cmd_url() -> None:
    import modal

    url = modal.Cls.from_name(APP, CLS)().web.get_web_url()
    set_env("ENCODER_URL", url)
    print(f"ENCODER_URL={url}")


def cmd_stop() -> None:
    modal_cli = [sys.executable, "-m", "modal", "container"]
    listing = subprocess.run(modal_cli + ["list", "--json"], capture_output=True, text=True, check=True)
    ids = [c["container_id"] for c in json.loads(listing.stdout) if c.get("app_name") == APP]
    for cid in ids:
        # --yes: without a TTY the CLI aborts at its confirmation prompt.
        subprocess.run(modal_cli + ["stop", "--yes", cid], check=False)
    print(f"stopped {len(ids)} container(s) of {APP}")


if __name__ == "__main__":
    {"secrets": cmd_secrets, "url": cmd_url, "stop": cmd_stop}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()

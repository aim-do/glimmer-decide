"""One-time Modal setup for the MetaEncoder service. Never prints tokens.

  python deploy/setup_modal.py secrets   # HF token (from `hf auth login`) + a fresh ENCODER_TOKEN -> Modal secrets, .env
  modal deploy deploy/modal_metaencoder.py
  python deploy/setup_modal.py url       # writes the deployed ENCODER_URL into .env
"""
import secrets
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
    from huggingface_hub import HfApi, get_token

    hf = get_token()
    if not hf:
        sys.exit("No Hugging Face token. Run `hf auth login` (paste a read token there, not in chat).")
    try:
        HfApi().auth_check("facebook/meta-encoder", token=hf)
    except Exception as e:
        sys.exit(f"HF token has no access to facebook/meta-encoder ({type(e).__name__}). "
                 "Open https://huggingface.co/facebook/meta-encoder and accept the terms.")
    print("HF access to facebook/meta-encoder: OK")
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


if __name__ == "__main__":
    {"secrets": cmd_secrets, "url": cmd_url}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()

"""Settings from environment (.env is loaded by app.py / eval scripts)."""
from __future__ import annotations

import os
from dataclasses import dataclass

DECIDERS = ("encoder", "glimmer", "both")
ENCODER_STYLES = ("label", "label_detail")
REASONING_LEVELS = ("low", "medium", "high", "xhigh")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: str) -> float:
    try:
        return float(_env(name, default))
    except ValueError:
        raise ValueError(f"{name} must be a number, got {_env(name, default)!r}") from None


def _int(name: str, default: str) -> int:
    try:
        return int(_env(name, default))
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {_env(name, default)!r}") from None


@dataclass(frozen=True)
class Settings:
    # MetaEncoder ("Glimmer decision"): our own GPU service (deploy/modal_metaencoder.py).
    encoder_url: str
    encoder_token: str
    encoder_tau: float          # softmax temperature over cosine scores (uncalibrated)
    encoder_style: str          # candidate text: "label" | "label_detail"
    # Generative Glimmer behind any OpenAI-compatible /chat/completions endpoint:
    # OpenRouter (default; routes to DeepInfra/Together), Together directly, a Fireworks
    # on-demand deployment (its serverless Glimmer ended 2026-09-25), or our own vLLM.
    glimmer_base_url: str
    glimmer_api_key: str
    glimmer_model: str
    glimmer_reasoning: str      # low | medium | high | xhigh
    glimmer_max_tokens: int
    glimmer_temperature: float
    # Meta Model API (SAM 3.1 highlight). Optional.
    meta_base_url: str
    meta_api_key: str
    sam_model: str
    # Which decider answers by default: encoder | glimmer | both.
    decider: str
    # MOCK=1 -> no network calls, deterministic fake decisions (UI/dev/tests). Never implicit.
    mock: bool
    # Cosine margin (top1 - top2) below which the encoder refuses with NONE.
    encoder_min_margin: float = 0.05
    # Hard deadline for one /api/decide call, seconds.
    request_timeout_s: float = 240.0

    @property
    def encoder_enabled(self) -> bool:
        return self.mock or bool(self.encoder_url)

    @property
    def glimmer_enabled(self) -> bool:
        return self.mock or bool(self.glimmer_api_key)

    @property
    def sam_enabled(self) -> bool:
        return bool(self.meta_api_key) and not self.mock


def load_settings() -> Settings:
    encoder_url = _env("ENCODER_URL").rstrip("/")
    tau = _float("ENCODER_TAU", "0.02")
    if tau <= 0:
        raise ValueError(f"ENCODER_TAU must be > 0, got {tau}")
    style = _env("ENCODER_STYLE", "label")
    if style not in ENCODER_STYLES:
        raise ValueError(f"ENCODER_STYLE must be one of {ENCODER_STYLES}, got {style!r}")
    reasoning = _env("GLIMMER_REASONING", "low")
    if reasoning not in REASONING_LEVELS:
        raise ValueError(f"GLIMMER_REASONING must be one of {REASONING_LEVELS}, got {reasoning!r}")
    max_tokens = _int("GLIMMER_MAX_TOKENS", "4096")
    if max_tokens <= 0:
        raise ValueError(f"GLIMMER_MAX_TOKENS must be > 0, got {max_tokens}")
    temperature = _float("GLIMMER_TEMPERATURE", "1.0")
    if not 0 <= temperature <= 2:
        raise ValueError(f"GLIMMER_TEMPERATURE must be in [0, 2], got {temperature}")
    decider = _env("DECIDER") or ("encoder" if encoder_url else "glimmer")
    if decider not in DECIDERS:
        raise ValueError(f"DECIDER must be one of {DECIDERS}, got {decider!r}")
    margin = _float("ENCODER_MIN_MARGIN", "0.05")
    if margin < 0:
        raise ValueError(f"ENCODER_MIN_MARGIN must be >= 0, got {margin}")
    timeout_s = _float("REQUEST_TIMEOUT_S", "240")
    if timeout_s <= 0:
        raise ValueError(f"REQUEST_TIMEOUT_S must be > 0, got {timeout_s}")
    for name, url in (("ENCODER_URL", encoder_url),
                      ("GLIMMER_BASE_URL", _env("GLIMMER_BASE_URL", "https://openrouter.ai/api/v1")),
                      ("META_BASE_URL", _env("META_BASE_URL", "https://api.meta.ai/v1"))):
        if url and not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError(f"{name} must use https (or localhost), got {url!r}")
    return Settings(
        encoder_url=encoder_url,
        encoder_token=_env("ENCODER_TOKEN"),
        encoder_tau=tau,
        encoder_style=style,
        glimmer_base_url=_env("GLIMMER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        glimmer_api_key=_env("GLIMMER_API_KEY"),
        glimmer_model=_env("GLIMMER_MODEL", "meta/muse-glimmer-30b"),
        glimmer_reasoning=reasoning,
        glimmer_max_tokens=max_tokens,
        glimmer_temperature=temperature,
        meta_base_url=_env("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/"),
        meta_api_key=_env("MODEL_API_KEY"),
        sam_model=_env("SAM_MODEL", "sam-3.1"),
        decider=decider,
        mock=_env("MOCK", "0") == "1",
        encoder_min_margin=margin,
        request_timeout_s=timeout_s,
    )

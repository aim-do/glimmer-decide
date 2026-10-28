"""Settings from environment (.env is loaded by app.py / eval scripts)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


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
    return Settings(
        encoder_url=encoder_url,
        encoder_token=_env("ENCODER_TOKEN"),
        encoder_tau=float(_env("ENCODER_TAU", "0.02")),
        encoder_style=_env("ENCODER_STYLE", "label"),
        glimmer_base_url=_env("GLIMMER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        glimmer_api_key=_env("GLIMMER_API_KEY"),
        glimmer_model=_env("GLIMMER_MODEL", "meta/muse-glimmer-30b"),
        glimmer_reasoning=_env("GLIMMER_REASONING", "low"),
        glimmer_max_tokens=int(_env("GLIMMER_MAX_TOKENS", "4096")),
        glimmer_temperature=float(_env("GLIMMER_TEMPERATURE", "1.0")),
        meta_base_url=_env("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/"),
        meta_api_key=_env("MODEL_API_KEY"),
        sam_model=_env("SAM_MODEL", "sam-3.1"),
        decider=_env("DECIDER") or ("encoder" if encoder_url else "glimmer"),
        mock=_env("MOCK", "0") == "1",
    )

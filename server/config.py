"""Settings from environment (.env is loaded by app.py / eval scripts)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Glimmer behind any OpenAI-compatible /chat/completions endpoint:
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
    # MOCK=1 -> no network calls, deterministic fake decisions (UI/dev/tests).
    mock: bool

    @property
    def sam_enabled(self) -> bool:
        return bool(self.meta_api_key) and not self.mock


def load_settings() -> Settings:
    return Settings(
        glimmer_base_url=_env("GLIMMER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        glimmer_api_key=_env("GLIMMER_API_KEY"),
        glimmer_model=_env("GLIMMER_MODEL", "meta/muse-glimmer-30b"),
        glimmer_reasoning=_env("GLIMMER_REASONING", "low"),
        glimmer_max_tokens=int(_env("GLIMMER_MAX_TOKENS", "4096")),
        glimmer_temperature=float(_env("GLIMMER_TEMPERATURE", "1.0")),
        meta_base_url=_env("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/"),
        meta_api_key=_env("MODEL_API_KEY"),
        sam_model=_env("SAM_MODEL", "sam-3.1"),
        mock=_env("MOCK", "0") == "1" or not _env("GLIMMER_API_KEY"),
    )

"""Configuration management for GenAI contextual post-processing.

Supports runtime configuration via environment variables without hardcoded
credentials, default API keys, or fixed model identities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenAIConfig:
    """Configuration settings for LLM contextual correction."""

    enabled: bool = False
    """Master toggle for GenAI post-processing. False by default."""

    provider: str = "mock"
    """LLM provider name: 'mock', 'openai', 'anthropic', 'ollama', etc."""

    model: str = "gpt-4o-mini"
    """Model identifier passed to the provider."""

    api_key: Optional[str] = None
    """API key credential. Never hardcoded; read from environment variables."""

    api_base: Optional[str] = None
    """Optional base URL for local LLMs (e.g., Ollama or vLLM endpoints)."""

    temperature: float = 0.0
    """Sampling temperature. Defaults to 0.0 for deterministic grammatical correction."""

    max_tokens: int = 128
    """Maximum output tokens for the short lip-reading sentence correction."""

    timeout: float = 10.0
    """Maximum wait time in seconds for the LLM API call before falling back to raw text."""

    @classmethod
    def from_env(cls) -> GenAIConfig:
        """Loads configuration from environment variables with safe, conservative fallbacks."""
        # Enable toggle: check GENAI_ENABLED or LIPREAD_GENAI_ENABLED
        enabled_str = os.environ.get(
            "GENAI_ENABLED",
            os.environ.get("LIPREAD_GENAI_ENABLED", "false"),
        ).strip().lower()
        enabled = enabled_str in ("1", "true", "yes", "on")

        provider = os.environ.get("LIPREAD_LLM_PROVIDER", "mock").strip().lower()
        model = os.environ.get("LIPREAD_LLM_MODEL", "gpt-4o-mini").strip()

        # Check provider-specific API keys if general key is not set
        api_key = os.environ.get("LIPREAD_LLM_API_KEY")
        if not api_key:
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
            elif provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            elif provider in ("google", "google_genai", "gemini"):
                api_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY"))

        api_base = os.environ.get(
            "LIPREAD_LLM_API_BASE",
            os.environ.get("OPENAI_API_BASE"),
        )

        try:
            temperature = float(os.environ.get("LIPREAD_LLM_TEMPERATURE", "0.0"))
        except ValueError:
            temperature = 0.0

        try:
            max_tokens = int(os.environ.get("LIPREAD_LLM_MAX_TOKENS", "128"))
        except ValueError:
            max_tokens = 128

        try:
            timeout = float(os.environ.get("LIPREAD_LLM_TIMEOUT", "10.0"))
        except ValueError:
            timeout = 10.0

        return cls(
            enabled=enabled,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


def get_genai_config() -> GenAIConfig:
    """Returns the active GenAI configuration resolved from the environment."""
    return GenAIConfig.from_env()

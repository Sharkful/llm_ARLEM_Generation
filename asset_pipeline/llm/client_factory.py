"""Multi-provider instructor client factory.

Every LLM call in this pipeline goes through get_instructor_client() rather
than instantiating a provider SDK directly, so provider/model stays a
config-level choice (see config.py DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL)
instead of being baked into individual pipeline modules. This mirrors the
three provider patterns already documented in the repo root CLAUDE.md
("LLM Integration" section) and used across this codebase.
"""
from __future__ import annotations

from typing import Literal

import instructor

import config

Provider = Literal["anthropic", "openai", "google"]

_SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "google")


def resolve_provider_and_model(
    provider: str | None = None, model: str | None = None
) -> tuple[str, str]:
    """Fill in defaults from config, without ever guessing silently past that."""
    resolved_provider = provider or config.DEFAULT_LLM_PROVIDER
    if resolved_provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider {resolved_provider!r}. "
            f"Supported: {_SUPPORTED_PROVIDERS}"
        )
    resolved_model = model or (
        config.DEFAULT_LLM_MODEL
        if resolved_provider == config.DEFAULT_LLM_PROVIDER
        else config.PROVIDER_DEFAULT_MODELS[resolved_provider]
    )
    return resolved_provider, resolved_model


def get_instructor_client(provider: str | None = None) -> instructor.Instructor:
    """Return an instructor-wrapped client for the requested provider.

    Raises a clear error (not a fallback to another provider) if the
    provider's API key is missing -- never silently substitutes.
    """
    resolved_provider, _ = resolve_provider_and_model(provider)

    if resolved_provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot create an anthropic client.")
        import anthropic

        return instructor.from_anthropic(anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY))

    if resolved_provider == "openai":
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot create an openai client.")
        import openai

        return instructor.from_openai(openai.OpenAI(api_key=config.OPENAI_API_KEY))

    if resolved_provider == "google":
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot create a google client.")
        from google import genai

        return instructor.from_google(
            genai.Client(api_key=config.GEMINI_API_KEY), use_async=False
        )

    raise ValueError(f"Unsupported LLM provider {resolved_provider!r}.")

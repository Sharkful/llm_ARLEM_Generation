import pytest

import config
from llm.client_factory import get_instructor_client, resolve_provider_and_model


def test_resolve_provider_and_model_uses_config_defaults(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "DEFAULT_LLM_MODEL", "claude-sonnet-4-6")
    provider, model = resolve_provider_and_model()
    assert provider == "anthropic"
    assert model == "claude-sonnet-4-6"


def test_resolve_provider_and_model_explicit_overrides():
    provider, model = resolve_provider_and_model(provider="openai", model="gpt-4o-mini")
    assert provider == "openai"
    assert model == "gpt-4o-mini"


def test_resolve_provider_and_model_falls_back_to_provider_default_model():
    provider, model = resolve_provider_and_model(provider="google")
    assert provider == "google"
    assert model == config.PROVIDER_DEFAULT_MODELS["google"]


def test_resolve_provider_and_model_rejects_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        resolve_provider_and_model(provider="not_a_real_provider")


def test_get_instructor_client_raises_clear_error_when_key_missing(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        get_instructor_client(provider="openai")


def test_get_instructor_client_never_silently_falls_back_to_another_provider(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-fake-present")
    # Even though openai has a key, requesting anthropic without its key must
    # raise -- never silently substitute a different provider.
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        get_instructor_client(provider="anthropic")

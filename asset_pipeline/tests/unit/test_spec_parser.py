from types import SimpleNamespace

import pytest

from models.spec_models import AssetSpec
from pipeline import spec_parser


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeCompletions:
    def __init__(self, response, completion):
        self._response = response
        self._completion = completion
        self.last_call_kwargs = None

    def create_with_completion(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response, self._completion


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, response, completion):
        self.chat = _FakeChat(_FakeCompletions(response, completion))


def _fake_spec():
    return AssetSpec(object_id="moon", description="a small gray moon with craters")


def test_parse_description_returns_spec_and_log_entry(monkeypatch):
    fake_response = _fake_spec()
    fake_completion = SimpleNamespace(usage=_FakeUsage(input_tokens=42, output_tokens=17))
    fake_client = _FakeClient(fake_response, fake_completion)

    monkeypatch.setattr(spec_parser, "get_instructor_client", lambda provider: fake_client)

    spec, entry = spec_parser.parse_description(
        "a small gray moon with visible crater texture", provider="anthropic", model="claude-sonnet-4-6"
    )

    assert spec == fake_response
    assert entry.success is True
    assert entry.provider == "anthropic"
    assert entry.model == "claude-sonnet-4-6"
    assert entry.input_tokens == 42
    assert entry.output_tokens == 17
    assert entry.purpose == "spec_parsing"


def test_parse_description_sends_response_model_and_description(monkeypatch):
    fake_response = _fake_spec()
    fake_completion = SimpleNamespace(usage=_FakeUsage())
    fake_client = _FakeClient(fake_response, fake_completion)

    monkeypatch.setattr(spec_parser, "get_instructor_client", lambda provider: fake_client)

    spec_parser.parse_description("a red cube", provider="openai", model="gpt-4o-mini")

    kwargs = fake_client.chat.completions.last_call_kwargs
    assert kwargs["response_model"] is AssetSpec
    assert kwargs["model"] == "gpt-4o-mini"
    assert any("a red cube" in m["content"] for m in kwargs["messages"] if m["role"] == "user")


def test_parse_description_wraps_failure_in_spec_parsing_error(monkeypatch):
    # Patch the retry-wrapped call directly so this test doesn't burn through
    # tenacity's real exponential backoff (5 attempts, up to 60s each).
    def _always_raise(client, llm_model, description):
        raise RuntimeError("boom")

    monkeypatch.setattr(spec_parser, "get_instructor_client", lambda provider: _FakeClient(None, None))
    monkeypatch.setattr(spec_parser, "_call_with_usage", _always_raise)

    with pytest.raises(spec_parser.SpecParsingError) as exc_info:
        spec_parser.parse_description("anything", provider="anthropic", model="claude-sonnet-4-6")

    assert exc_info.value.log_entry.success is False
    assert "boom" in exc_info.value.log_entry.error_message

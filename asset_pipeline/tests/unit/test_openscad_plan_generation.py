from types import SimpleNamespace

import pytest

from models.generation_models import OpenSCADPlan
from models.spec_models import AssetSpec
from pipeline import openscad_generator as gen


class _FakeUsage:
    def __init__(self, input_tokens=30, output_tokens=200):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


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


def _fake_plan():
    return OpenSCADPlan(
        parameters={"width": 0.05},
        scad_source="cube([0.05, 0.05, 0.05]);",
        expected_bounds_m=[0.05, 0.05, 0.05],
    )


def test_generate_openscad_plan_returns_plan_and_log_entry(monkeypatch):
    fake_response = _fake_plan()
    fake_completion = SimpleNamespace(usage=_FakeUsage(input_tokens=100, output_tokens=400))
    fake_client = _FakeClient(fake_response, fake_completion)

    monkeypatch.setattr(gen, "get_instructor_client", lambda provider: fake_client)

    plan, entry = gen.generate_openscad_plan(
        AssetSpec(object_id="x", description="a bracket"), provider="anthropic", model="claude-sonnet-4-6"
    )

    assert plan == fake_response
    assert entry.purpose == "openscad_generation"
    assert entry.input_tokens == 100
    assert entry.output_tokens == 400


def test_generate_openscad_plan_uses_repair_prompt_when_context_given(monkeypatch):
    fake_response = _fake_plan()
    fake_completion = SimpleNamespace(usage=_FakeUsage())
    fake_client = _FakeClient(fake_response, fake_completion)

    monkeypatch.setattr(gen, "get_instructor_client", lambda provider: fake_client)

    plan, entry = gen.generate_openscad_plan(
        AssetSpec(object_id="x", description="a bracket"),
        provider="anthropic",
        model="claude-sonnet-4-6",
        repair_context=("cube([broken syntax", "ERROR: expected )"),
    )

    assert entry.purpose == "openscad_repair"
    kwargs = fake_client.chat.completions.last_call_kwargs
    user_msg = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
    assert "ERROR: expected )" in user_msg
    assert "broken syntax" in user_msg


def test_generate_openscad_plan_wraps_failure(monkeypatch):
    def _always_raise(client, llm_model, spec, repair_context, revision_context=None):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(gen, "get_instructor_client", lambda provider: _FakeClient(None, None))
    monkeypatch.setattr(gen, "_call_with_usage", _always_raise)

    with pytest.raises(gen.OpenSCADGenerationError) as exc_info:
        gen.generate_openscad_plan(
            AssetSpec(object_id="x", description="a bracket"), provider="anthropic", model="claude-sonnet-4-6"
        )

    assert exc_info.value.log_entry.success is False
    assert "rate limited" in exc_info.value.log_entry.error_message

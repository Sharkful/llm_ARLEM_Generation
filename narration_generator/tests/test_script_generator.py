import pytest

from models.log_models import LLMCallEntry
from narration_definitions import (
    ARLabNarrationScript,
    NarrationClip,
    NarrationModule,
    SaySegment,
    TTSRuntimeConfig,
)
from pipeline.script_generator import generate_narration_script
from script_definitions import ARSceneNarrationBrief, SceneClipBrief, SceneModuleBrief

_SAMPLE_BRIEF = ARSceneNarrationBrief(
    scene="A 3D bowl-shaped loss surface in AR space.",
    objectives=["Explain what the loss surface represents."],
    modules=[
        SceneModuleBrief(clips=[
            SceneClipBrief(
                change="The loss surface appears.",
                meaning="Students understand height = prediction error.",
            )
        ])
    ],
)

_SAMPLE_SCRIPT = ARLabNarrationScript(
    modules=[NarrationModule(clips=[
        NarrationClip(
            title="Loss Surface Introduction",
            segments=[SaySegment(type="say", text="What you see is a loss surface.")],
        )
    ])]
)

_RUNTIME = TTSRuntimeConfig(allowed_voices=["alloy", "nova"], default_voice="alloy")


class _FakeCompletion:
    class _Usage:
        input_tokens = 300
        output_tokens = 500
        cache_read_input_tokens = 0
    usage = _Usage()


def test_generate_script_returns_valid_model(mocker):
    mock = mocker.patch("pipeline.script_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_SCRIPT, _FakeCompletion()
    )
    result, _ = generate_narration_script(_SAMPLE_BRIEF, _RUNTIME, "claude-sonnet-4-6", "fake-key")
    assert isinstance(result, ARLabNarrationScript)
    assert result.modules[0].clips[0].title == "Loss Surface Introduction"


def test_generate_script_log_entry_fields(mocker):
    mock = mocker.patch("pipeline.script_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_SCRIPT, _FakeCompletion()
    )
    _, entry = generate_narration_script(_SAMPLE_BRIEF, _RUNTIME, "claude-sonnet-4-6", "fake-key")
    assert isinstance(entry, LLMCallEntry)
    assert entry.purpose == "script_generation"
    assert entry.input_tokens == 300
    assert entry.output_tokens == 500
    assert entry.success is True


def test_generate_script_calls_correct_model(mocker):
    mock = mocker.patch("pipeline.script_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_SCRIPT, _FakeCompletion()
    )
    generate_narration_script(_SAMPLE_BRIEF, _RUNTIME, "claude-sonnet-4-6", "fake-key")
    kwargs = mock.return_value.chat.completions.create_with_completion.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["response_model"] is ARLabNarrationScript

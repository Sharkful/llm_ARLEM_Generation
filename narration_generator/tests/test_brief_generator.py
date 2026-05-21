import pytest

from models.log_models import LLMCallEntry
from pipeline.brief_generator import generate_scene_brief
from script_definitions import ARSceneNarrationBrief, SceneClipBrief, SceneModuleBrief

SAMPLE_DESCRIPTION = "A lab about gradient descent on a 3D loss surface."

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


class _FakeCompletion:
    class _Usage:
        input_tokens = 100
        output_tokens = 200
        cache_read_input_tokens = 0
    usage = _Usage()


def test_generate_brief_returns_valid_model(mocker):
    mock = mocker.patch("pipeline.brief_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_BRIEF, _FakeCompletion()
    )
    result, _ = generate_scene_brief(SAMPLE_DESCRIPTION, "claude-sonnet-4-6", "fake-key")
    assert isinstance(result, ARSceneNarrationBrief)
    assert result.scene == _SAMPLE_BRIEF.scene


def test_generate_brief_log_entry_fields(mocker):
    mock = mocker.patch("pipeline.brief_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_BRIEF, _FakeCompletion()
    )
    _, entry = generate_scene_brief(SAMPLE_DESCRIPTION, "claude-sonnet-4-6", "fake-key")
    assert isinstance(entry, LLMCallEntry)
    assert entry.purpose == "brief_generation"
    assert entry.model == "claude-sonnet-4-6"
    assert entry.input_tokens == 100
    assert entry.output_tokens == 200
    assert entry.success is True


def test_generate_brief_calls_correct_model(mocker):
    mock = mocker.patch("pipeline.brief_generator.instructor.from_anthropic")
    mock.return_value.chat.completions.create_with_completion.return_value = (
        _SAMPLE_BRIEF, _FakeCompletion()
    )
    generate_scene_brief(SAMPLE_DESCRIPTION, "claude-sonnet-4-6", "fake-key")
    kwargs = mock.return_value.chat.completions.create_with_completion.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["response_model"] is ARSceneNarrationBrief

import pytest

from narration_definitions import (
    ARLabNarrationScript,
    NarrationClip,
    NarrationModule,
    PauseSegment,
    SaySegment,
    TTSRuntimeConfig,
)
from pipeline.validation import validate_narration_script


def _make_clip(text: str = "This is a normal narration segment.", title: str | None = "My Clip") -> NarrationClip:
    return NarrationClip(title=title, segments=[SaySegment(type="say", text=text)])


def _make_script(clips: list[NarrationClip] | None = None) -> ARLabNarrationScript:
    if clips is None:
        clips = [_make_clip()]
    return ARLabNarrationScript(modules=[NarrationModule(clips=clips)])


def _runtime(voices: list[str] | None = None) -> TTSRuntimeConfig:
    return TTSRuntimeConfig(
        allowed_voices=voices if voices is not None else ["alloy", "nova", "echo"],
        default_voice="alloy",
    )


class TestValidateNarrationScript:
    def test_valid_script_no_errors(self):
        results = validate_narration_script(_make_script(), _runtime())
        assert not [r for r in results if r.level == "error"]

    def test_unknown_clip_voice_is_error(self):
        clip = NarrationClip(
            title="Test",
            voice="unknown_voice",
            segments=[SaySegment(type="say", text="Hello there.")],
        )
        results = validate_narration_script(_make_script([clip]), _runtime())
        errors = [r for r in results if r.level == "error"]
        assert len(errors) == 1
        assert "unknown_voice" in errors[0].message

    def test_unknown_default_voice_is_error(self):
        script = ARLabNarrationScript(
            default_voice="bad_voice",
            modules=[NarrationModule(clips=[_make_clip()])],
        )
        results = validate_narration_script(script, _runtime())
        assert any("bad_voice" in r.message for r in results if r.level == "error")

    def test_missing_clip_title_is_warning(self):
        results = validate_narration_script(_make_script([_make_clip(title=None)]), _runtime())
        assert any("no title" in r.message for r in results if r.level == "warning")

    def test_long_say_segment_is_warning(self):
        clip = _make_clip(text="word " * 200)
        results = validate_narration_script(_make_script([clip]), _runtime())
        assert any("long" in r.message.lower() for r in results if r.level == "warning")

    def test_long_pause_is_warning(self):
        clip = NarrationClip(
            title="Test",
            segments=[
                SaySegment(type="say", text="Hello there."),
                PauseSegment(type="pause", seconds=5.0),
            ],
        )
        results = validate_narration_script(_make_script([clip]), _runtime())
        assert any("5.0" in r.message for r in results if r.level == "warning")

    def test_short_clip_is_warning(self):
        results = validate_narration_script(_make_script([_make_clip(text="Hi.")]), _runtime())
        assert any("short" in r.message.lower() for r in results if r.level == "warning")

    def test_empty_allowed_voices_skips_voice_check(self):
        clip = NarrationClip(
            title="Test",
            voice="any_voice",
            segments=[SaySegment(type="say", text="Hello there.")],
        )
        results = validate_narration_script(_make_script([clip]), _runtime(voices=[]))
        assert not [r for r in results if r.level == "error"]

    def test_clip_ref_format(self):
        results = validate_narration_script(_make_script([_make_clip(title=None)]), _runtime())
        clip_refs = [r.clip_ref for r in results if r.clip_ref]
        assert "m01_c001" in clip_refs

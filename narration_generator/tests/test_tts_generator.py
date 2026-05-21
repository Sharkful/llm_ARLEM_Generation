import io
from unittest.mock import MagicMock

import pytest
from pydub import AudioSegment

from narration_definitions import NarrationClip, PauseSegment, SaySegment
from pipeline.tts_generator import OpenAITTSProvider, TTSGenerator, TTSProvider


def _silent_mp3_bytes(duration_ms: int = 100) -> bytes:
    buf = io.BytesIO()
    AudioSegment.silent(duration=duration_ms).export(buf, format="mp3")
    return buf.getvalue()


def _mock_provider(mp3_bytes: bytes | None = None) -> MagicMock:
    p = MagicMock(spec=TTSProvider)
    p.provider_name = "mock"
    p.model_name = "mock-tts"
    p.synthesize.return_value = mp3_bytes or _silent_mp3_bytes()
    return p


def _clip(*segments) -> NarrationClip:
    return NarrationClip(segments=list(segments))


class TestTTSGenerator:
    def test_say_segment_calls_provider(self):
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="alloy")
        gen.generate_clip_audio(_clip(SaySegment(type="say", text="Hello.")), effective_voice="alloy")
        provider.synthesize.assert_called_once_with("Hello.", "alloy", instructions=None)

    def test_pause_segment_does_not_call_provider(self):
        # A clip must have at least one say; the pause should not trigger a provider call
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="alloy")
        clip = _clip(SaySegment(type="say", text="Hello."), PauseSegment(type="pause", seconds=0.5))
        gen.generate_clip_audio(clip, effective_voice="alloy")
        # Only one call for the say segment, none for the pause
        assert provider.synthesize.call_count == 1

    def test_mixed_segments_calls_provider_once_per_say(self):
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="alloy")
        clip = _clip(
            SaySegment(type="say", text="First."),
            PauseSegment(type="pause", seconds=0.3),
            SaySegment(type="say", text="Second."),
        )
        gen.generate_clip_audio(clip, effective_voice="alloy")
        assert provider.synthesize.call_count == 2

    def test_returns_audio_segment(self):
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="alloy")
        result = gen.generate_clip_audio(_clip(SaySegment(type="say", text="Hello.")))
        assert isinstance(result, AudioSegment)

    def test_uses_default_voice_when_no_effective_voice(self):
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="nova")
        gen.generate_clip_audio(_clip(SaySegment(type="say", text="Hello.")))
        provider.synthesize.assert_called_once_with("Hello.", "nova", instructions=None)

    def test_effective_voice_overrides_default(self):
        provider = _mock_provider()
        gen = TTSGenerator(provider, default_voice="nova")
        gen.generate_clip_audio(_clip(SaySegment(type="say", text="Hello.")), effective_voice="echo")
        provider.synthesize.assert_called_once_with("Hello.", "echo", instructions=None)

    def test_save_clip_creates_file(self, tmp_path):
        gen = TTSGenerator(_mock_provider(), default_voice="alloy")
        filepath = tmp_path / "audio" / "m01_c001.mp3"
        duration = gen.save_clip(AudioSegment.silent(duration=500), filepath)
        assert filepath.exists()
        assert abs(duration - 0.5) < 0.1

    def test_count_characters_only_counts_say(self):
        gen = TTSGenerator(_mock_provider(), default_voice="alloy")
        clip = _clip(
            SaySegment(type="say", text="Hello world."),
            PauseSegment(type="pause", seconds=0.5),
            SaySegment(type="say", text="Goodbye."),
        )
        assert gen.count_characters(clip) == len("Hello world.") + len("Goodbye.")


class TestOpenAITTSProvider:
    def test_provider_name(self, mocker):
        mocker.patch("pipeline.tts_generator.OpenAI")
        assert OpenAITTSProvider(api_key="fake", model="tts-1").provider_name == "openai"

    def test_model_name(self, mocker):
        mocker.patch("pipeline.tts_generator.OpenAI")
        assert OpenAITTSProvider(api_key="fake", model="tts-1-hd").model_name == "tts-1-hd"

    def test_synthesize_calls_openai(self, mocker):
        mock_openai = mocker.patch("pipeline.tts_generator.OpenAI")
        mock_response = MagicMock()
        mock_response.content = _silent_mp3_bytes()
        mock_openai.return_value.audio.speech.create.return_value = mock_response

        result = OpenAITTSProvider(api_key="fake", model="tts-1").synthesize("Hello.", "alloy")

        mock_openai.return_value.audio.speech.create.assert_called_once_with(
            model="tts-1", voice="alloy", input="Hello.", response_format="mp3"
        )
        assert isinstance(result, bytes)

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from openai import OpenAI
from pydub import AudioSegment

from narration_definitions import ClipStyle, NarrationClip


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, text: str, voice: str, **kwargs) -> bytes:
        """Return raw MP3 bytes for the given text and voice."""
        ...

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...


class OpenAITTSProvider:
    def __init__(self, api_key: str, model: str = "tts-1") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def synthesize(self, text: str, voice: str, **kwargs) -> bytes:
        response = self._client.audio.speech.create(
            model=self._model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        return response.content


class TTSGenerator:
    def __init__(self, provider: TTSProvider, default_voice: str) -> None:
        self._provider = provider
        self._default_voice = default_voice

    def generate_clip_audio(
        self,
        clip: NarrationClip,
        effective_voice: Optional[str] = None,
        style: Optional[ClipStyle] = None,
    ) -> AudioSegment:
        voice = effective_voice or self._default_voice
        style_instructions = style.instructions if style else None

        parts: list[AudioSegment] = []
        for seg in clip.segments:
            if seg.type == "say":
                mp3_bytes = self._provider.synthesize(
                    seg.text, voice, instructions=style_instructions
                )
                parts.append(AudioSegment.from_mp3(io.BytesIO(mp3_bytes)))
            elif seg.type == "pause":
                silence_ms = int(seg.seconds * 1000)
                parts.append(AudioSegment.silent(duration=silence_ms))

        if not parts:
            return AudioSegment.silent(duration=0)

        result = parts[0]
        for part in parts[1:]:
            result = result + part
        return result

    def save_clip(self, audio: AudioSegment, filepath: Path) -> float:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        audio.export(str(filepath), format="mp3")
        return len(audio) / 1000.0

    @property
    def provider(self) -> TTSProvider:
        return self._provider

    def count_characters(self, clip: NarrationClip) -> int:
        return sum(
            len(seg.text) for seg in clip.segments if seg.type == "say"
        )

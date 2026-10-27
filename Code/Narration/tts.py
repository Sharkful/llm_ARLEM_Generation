"""
Text-to-speech providers for lab narration (#93).

Each provider turns one clip's spoken text into raw PCM in the bundle format
(24 kHz, 16-bit, mono; see bundle.py), plus the token usage and served model
the provider reports. Neither provider is asked for a lossy format, so audio is
never re-encoded.

Paper defaults (NARRATION.md, "Defaults (frozen for paper runs)") live in
``PROVIDERS`` and ``STYLE_INSTRUCTION``. Changing any of them means a new run
group.

The SDK clients never retry on their own (OpenAI ``max_retries=0``, Gemini one
attempt): retries are the caller's job, transient errors only, counted.
"""

from __future__ import annotations

import base64
import io
import json
import re
import wave
from dataclasses import dataclass

from bundle import WAV_CHANNELS, WAV_SAMPLE_RATE, WAV_SAMPLE_WIDTH

STYLE_INSTRUCTION = (
    "Speak as a clear, friendly science instructor guiding a student through a "
    "lab, at a calm, moderate pace."
)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str            # pinned model id, also the tracking/pricing.py key
    voice: str
    max_chars: int        # a longer clip is an error, never split
    api_key_env: tuple[str, ...]
    sdk_module: str
    sdk_package: str


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        model="gpt-4o-mini-tts-2025-12-15",
        voice="marin",
        max_chars=4096,                       # the speech endpoint's input limit
        api_key_env=("OPENAI_API_KEY",),
        sdk_module="openai",
        sdk_package="openai",
    ),
    "gemini": ProviderSpec(
        name="gemini",
        model="gemini-3.8-flash-tts",
        voice="Charon",
        max_chars=4000,                       # our cap; the longest wave-2 clip is 966
        api_key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        sdk_module="google.genai",
        sdk_package="google-genai",
    ),
}


@dataclass
class Synthesis:
    """One clip's audio and what the provider reported about it."""
    pcm: bytes                    # 24 kHz, 16-bit, mono, little-endian
    input_tokens: int | None      # text tokens billed (None: not reported)
    output_tokens: int | None     # audio tokens billed (None: not reported)
    served_model: str | None      # model the provider says answered (None: not reported)

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (WAV_SAMPLE_WIDTH * WAV_CHANNELS * WAV_SAMPLE_RATE)


class TTSFormatError(RuntimeError):
    """The provider answered, but not with audio in the format we asked for."""


def pcm_from_audio(data: bytes, mime_type: str | None) -> bytes:
    """Raw PCM from a WAV file or a raw-PCM payload, checked against the bundle
    format. Nothing is resampled: a different rate or width is an error."""
    if data[:4] == b"RIFF":
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                fmt = (w.getframerate(), w.getsampwidth(), w.getnchannels(), w.getcomptype())
                frames = w.readframes(w.getnframes())
        except (wave.Error, EOFError) as exc:
            raise TTSFormatError(f"unreadable WAV from the provider ({exc})") from exc
        if fmt != (WAV_SAMPLE_RATE, WAV_SAMPLE_WIDTH, WAV_CHANNELS, "NONE"):
            raise TTSFormatError(f"provider WAV is {fmt}, expected 24 kHz 16-bit mono PCM")
        return frames
    mime = (mime_type or "").lower()
    if mime.startswith(("audio/l16", "audio/pcm")):
        rate = re.search(r"rate=(\d+)", mime)
        if rate and int(rate.group(1)) != WAV_SAMPLE_RATE:
            raise TTSFormatError(f"provider PCM is {rate.group(1)} Hz, expected {WAV_SAMPLE_RATE}")
        if len(data) % WAV_SAMPLE_WIDTH:
            raise TTSFormatError("provider PCM has an odd number of bytes")
        return data
    raise TTSFormatError(f"unexpected audio type {mime_type!r} from the provider")


# ── OpenAI ────────────────────────────────────────────────────────────


def parse_openai_sse(body: bytes) -> tuple[bytes, int | None, int | None]:
    """(pcm, input_tokens, output_tokens) from a /v1/audio/speech SSE body:
    ``speech.audio.delta`` events carry base64 audio, and the closing
    ``speech.audio.done`` event carries the token usage."""
    pcm = bytearray()
    usage = None
    saw_event = False
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except ValueError as exc:
            raise TTSFormatError(f"malformed speech event ({exc})") from exc
        saw_event = True
        kind = event.get("type")
        if kind == "speech.audio.delta":
            pcm += base64.b64decode(event.get("audio") or "")
        elif kind == "speech.audio.done":
            usage = event.get("usage") or {}
    if not saw_event:
        raise TTSFormatError("expected server-sent speech events, got none")
    if usage is None:
        raise TTSFormatError("the speech stream ended without speech.audio.done")
    if len(pcm) % WAV_SAMPLE_WIDTH:
        raise TTSFormatError("speech audio has an odd number of bytes")
    return bytes(pcm), usage.get("input_tokens"), usage.get("output_tokens")


class OpenAITTS:
    """OpenAI ``/v1/audio/speech`` with ``response_format="pcm"`` (24 kHz, 16-bit,
    mono) and ``stream_format="sse"``, the format that reports token usage."""

    def __init__(self, spec: ProviderSpec, voice: str, style: str,
                 api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI

        self.spec, self.voice, self.style = spec, voice, style
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    def synthesize(self, text: str) -> Synthesis:
        raw = self.client.audio.speech.with_raw_response.create(
            model=self.spec.model,
            voice=self.voice,
            input=text,
            instructions=self.style,
            response_format="pcm",
            stream_format="sse",
        )
        pcm, tokens_in, tokens_out = parse_openai_sse(raw.content)
        # The speech endpoint has no model field in its body; use the header when sent.
        served = raw.headers.get("openai-model") or None
        return Synthesis(pcm, tokens_in, tokens_out, served)


# ── Gemini ────────────────────────────────────────────────────────────


class GeminiTTS:
    """Gemini ``generateContent`` with audio output. The style goes in the part's
    ``speech_metadata.style``, so it is never read aloud as part of the text."""

    def __init__(self, spec: ProviderSpec, voice: str, style: str,
                 api_key: str | None = None, base_url: str | None = None):
        from google import genai
        from google.genai import types

        self.spec, self.voice, self.style, self.types = spec, voice, style, types
        http_options = types.HttpOptions(
            base_url=base_url,
            retry_options=types.HttpRetryOptions(attempts=1),  # no SDK retries
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def synthesize(self, text: str) -> Synthesis:
        t = self.types
        resp = self.client.models.generate_content(
            model=self.spec.model,
            contents=[t.Content(role="user", parts=[
                t.Part(text=text, speech_metadata=t.SpeechMetadata(style=self.style)),
            ])],
            config=t.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=t.SpeechConfig(voice_config=t.VoiceConfig(voice=self.voice)),
            ),
        )
        inline = next(
            (p.inline_data for c in (resp.candidates or [])
             for p in ((c.content.parts if c.content else None) or [])
             if p.inline_data and p.inline_data.data),
            None,
        )
        if inline is None:
            reason = resp.candidates[0].finish_reason if resp.candidates else None
            raise TTSFormatError(f"no audio in the response (finish reason: {reason})")
        pcm = pcm_from_audio(inline.data, inline.mime_type)
        um = resp.usage_metadata
        served = (resp.model_version or "").removeprefix("models/") or None
        return Synthesis(
            pcm,
            getattr(um, "prompt_token_count", None) if um else None,
            getattr(um, "candidates_token_count", None) if um else None,
            served,
        )


def make_provider(name: str, voice: str | None = None, style: str = STYLE_INSTRUCTION,
                  api_key: str | None = None, base_url: str | None = None):
    spec = PROVIDERS[name]
    cls = {"openai": OpenAITTS, "gemini": GeminiTTS}[name]
    return cls(spec, voice or spec.voice, style, api_key=api_key, base_url=base_url)

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

Every request has a hard time limit (``REQUEST_TIME_LIMIT``, 90 s; the longest
wave-2 clip normally takes under 12 s). Past it the request is abandoned with
``TTSTimeout``, which the caller retries. OpenAI streams its audio, so a watchdog
closes a stream that is still open at the limit, whether it has gone silent or is
trickling; the client timeout covers the wait for the stream to start. Gemini
answers in one piece. Its limit is the SDK timeout, which httpx applies to each
wait (connect, each read) rather than to the whole request, plus the same 90 s
sent to the server as a deadline (``X-Server-Timeout``). A Gemini reply that
trickled in could in principle run past 90 s, but its one-piece replies make that
unlikely.
"""

from __future__ import annotations

import base64
import importlib.metadata
import io
import json
import re
import threading
import time
import wave
from dataclasses import dataclass

from bundle import WAV_CHANNELS, WAV_SAMPLE_RATE, WAV_SAMPLE_WIDTH

REQUEST_TIME_LIMIT = 90.0   # seconds per request, start to last byte

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
    min_sdk_version: str | None = None   # first SDK release with every field we send


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
        # Part.speech_metadata (the style) and VoiceConfig.voice first appear in
        # google-genai 2.25.0 (built 2026-09-22, the day before Gemini 3.8 Flash TTS).
        min_sdk_version="2.25.0",
    ),
}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


def sdk_problem(spec: ProviderSpec) -> str | None:
    """Why the installed SDK can't run this provider (missing or too old), or None."""
    try:
        have = importlib.metadata.version(spec.sdk_package)
    except importlib.metadata.PackageNotFoundError:
        return (f"Python package '{spec.sdk_package}' is not installed ({spec.name} needs it). "
                "Activate the project venv and run: pip install -r requirements.txt")
    if spec.min_sdk_version and _version_tuple(have) < _version_tuple(spec.min_sdk_version):
        return (f"{spec.sdk_package} {have} is too old for {spec.model}: it needs "
                f"{spec.min_sdk_version} or newer. Run: "
                f'pip install -U "{spec.sdk_package}>={spec.min_sdk_version}"')
    return None


@dataclass
class Synthesis:
    """One clip's audio and what the provider reported about it."""
    pcm: bytes                    # 24 kHz, 16-bit, mono, little-endian
    input_tokens: int | None      # text tokens billed (None: not reported)
    output_tokens: int | None     # audio tokens billed (None: not reported)
    served_model: str | None      # model the provider says answered (None: not reported)
    request_id: str | None = None # the provider's id for the request, to match billing records
    first_audio_seconds: float | None = None  # request start to the first audio received
    max_gap_seconds: float | None = None      # longest wait between audio pieces (OpenAI)

    @property
    def seconds(self) -> float:
        return len(self.pcm) / (WAV_SAMPLE_WIDTH * WAV_CHANNELS * WAV_SAMPLE_RATE)


class TTSTimeout(Exception):
    """The request passed ``REQUEST_TIME_LIMIT`` and was abandoned (retryable)."""


class TTSNoAudio(Exception):
    """The provider answered this clip without any audio (retryable, clip-specific)."""


def _is_timeout(exc: BaseException) -> bool:
    """True if an httpx timeout is the error or one of its causes (SDKs may wrap it)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ in ("ReadTimeout", "WriteTimeout", "ConnectTimeout",
                                  "PoolTimeout", "TimeoutException"):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


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


class SpeechEvents:
    """Reads a /v1/audio/speech SSE stream line by line: ``speech.audio.delta``
    events carry base64 audio, and the closing ``speech.audio.done`` event carries
    the token usage. Also notes when audio first arrived and the longest gap."""

    def __init__(self, t0: float | None = None):
        self.t0 = time.monotonic() if t0 is None else t0
        self.pcm = bytearray()
        self.usage = None
        self.saw_event = False
        self.first_audio = None
        self.max_gap = 0.0
        self._last = None

    def feed(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except ValueError as exc:
            raise TTSFormatError(f"malformed speech event ({exc})") from exc
        self.saw_event = True
        kind = event.get("type")
        if kind == "speech.audio.delta":
            now = time.monotonic()
            if self.first_audio is None:
                self.first_audio = now - self.t0
            else:
                self.max_gap = max(self.max_gap, now - self._last)
            self._last = now
            self.pcm += base64.b64decode(event.get("audio") or "")
        elif kind == "speech.audio.done":
            self.usage = event.get("usage") or {}

    def result(self) -> tuple[bytes, int | None, int | None]:
        if not self.saw_event:
            raise TTSFormatError("expected server-sent speech events, got none")
        if self.usage is None:
            raise TTSFormatError("the speech stream ended without speech.audio.done")
        if len(self.pcm) % WAV_SAMPLE_WIDTH:
            raise TTSFormatError("speech audio has an odd number of bytes")
        return bytes(self.pcm), self.usage.get("input_tokens"), self.usage.get("output_tokens")


def parse_openai_sse(body: bytes) -> tuple[bytes, int | None, int | None]:
    """(pcm, input_tokens, output_tokens) from a whole /v1/audio/speech SSE body."""
    events = SpeechEvents()
    for line in body.decode("utf-8", errors="replace").splitlines():
        events.feed(line)
    return events.result()


class OpenAITTS:
    """OpenAI ``/v1/audio/speech`` with ``response_format="pcm"`` (24 kHz, 16-bit,
    mono) and ``stream_format="sse"``, the format that reports token usage."""

    def __init__(self, spec: ProviderSpec, voice: str, style: str,
                 api_key: str | None = None, base_url: str | None = None,
                 time_limit: float = REQUEST_TIME_LIMIT):
        import httpx
        from openai import OpenAI

        self.spec, self.voice, self.style, self.time_limit = spec, voice, style, time_limit
        # The timeout bounds the wait for the stream to start; the watchdog in
        # synthesize() bounds the whole request.
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                             timeout=httpx.Timeout(time_limit, connect=min(10.0, time_limit)))

    def synthesize(self, text: str) -> Synthesis:
        t0 = time.monotonic()
        with self.client.audio.speech.with_streaming_response.create(
            model=self.spec.model,
            voice=self.voice,
            input=text,
            instructions=self.style,
            response_format="pcm",
            stream_format="sse",
        ) as resp:
            expired = threading.Event()

            def expire():
                expired.set()
                resp.close()   # unblocks a read that is waiting on a stalled stream

            watchdog = threading.Timer(max(0.0, self.time_limit - (time.monotonic() - t0)), expire)
            watchdog.daemon = True
            watchdog.start()
            events = SpeechEvents(t0)
            try:
                for line in resp.iter_lines():
                    events.feed(line)
                    if events.usage is not None:   # speech.audio.done ends the reply
                        break
                    if expired.is_set():
                        break
            except Exception as exc:
                if events.usage is None:
                    if expired.is_set():
                        raise self._timeout(events) from exc
                    raise
            finally:
                watchdog.cancel()
            # A reply that completed is kept even if the watchdog fired in the same instant.
            if events.usage is None and expired.is_set():
                raise self._timeout(events)
            # The speech endpoint reports no model (no body field, and no openai-model
            # header in the 2026-10-27 live run); keep the header in case it is ever sent.
            served = resp.headers.get("openai-model") or None
            request_id = resp.request_id or resp.headers.get("x-request-id") or None
        pcm, tokens_in, tokens_out = events.result()
        return Synthesis(pcm, tokens_in, tokens_out, served, request_id,
                         first_audio_seconds=round(events.first_audio, 3)
                         if events.first_audio is not None else None,
                         max_gap_seconds=round(events.max_gap, 3))

    def _timeout(self, events: SpeechEvents) -> TTSTimeout:
        got = (f"first audio after {events.first_audio:.1f} s, {len(events.pcm)} bytes so far"
               if events.first_audio is not None else "no audio yet")
        return TTSTimeout(f"no complete reply within {self.time_limit:g} s ({got})")


# ── Gemini ────────────────────────────────────────────────────────────


class GeminiTTS:
    """Gemini ``generateContent`` with audio output. The style goes in the part's
    ``speech_metadata.style``, so it is never read aloud as part of the text."""

    def __init__(self, spec: ProviderSpec, voice: str, style: str,
                 api_key: str | None = None, base_url: str | None = None,
                 time_limit: float = REQUEST_TIME_LIMIT):
        problem = sdk_problem(spec)
        if problem:
            raise RuntimeError(problem)
        from google import genai
        from google.genai import types

        self.spec, self.voice, self.style, self.types = spec, voice, style, types
        self.time_limit = time_limit
        http_options = types.HttpOptions(
            base_url=base_url,
            retry_options=types.HttpRetryOptions(attempts=1),  # no SDK retries
            # Milliseconds. httpx applies it to each wait, not the whole request; the
            # SDK also sends it to the server as a deadline (X-Server-Timeout).
            timeout=int(time_limit * 1000),
        )
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def synthesize(self, text: str) -> Synthesis:
        t0 = time.monotonic()
        try:
            resp = self._generate(text)
        except Exception as exc:
            if _is_timeout(exc):
                raise TTSTimeout(f"no complete reply within {self.time_limit:g} s") from exc
            raise
        first_audio = round(time.monotonic() - t0, 3)
        inline = next(
            (p.inline_data for c in (resp.candidates or [])
             for p in ((c.content.parts if c.content else None) or [])
             if p.inline_data and p.inline_data.data),
            None,
        )
        if inline is None:
            # Clip-specific (e.g. finish reason OTHER or SAFETY), so retried, not fatal.
            reason = resp.candidates[0].finish_reason if resp.candidates else None
            raise TTSNoAudio(f"no audio in the response (finish reason: {reason})")
        pcm = pcm_from_audio(inline.data, inline.mime_type)
        um = resp.usage_metadata
        served = (resp.model_version or "").removeprefix("models/") or None
        return Synthesis(
            pcm,
            getattr(um, "prompt_token_count", None) if um else None,
            getattr(um, "candidates_token_count", None) if um else None,
            served,
            getattr(resp, "response_id", None) or None,
            first_audio_seconds=first_audio,   # the whole reply arrives at once
        )

    def _generate(self, text: str):
        t = self.types
        return self.client.models.generate_content(
            model=self.spec.model,
            contents=[t.Content(role="user", parts=[
                t.Part(text=text, speech_metadata=t.SpeechMetadata(style=self.style)),
            ])],
            config=t.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=t.SpeechConfig(voice_config=t.VoiceConfig(voice=self.voice)),
            ),
        )


def make_provider(name: str, voice: str | None = None, style: str = STYLE_INSTRUCTION,
                  api_key: str | None = None, base_url: str | None = None,
                  time_limit: float = REQUEST_TIME_LIMIT):
    spec = PROVIDERS[name]
    cls = {"openai": OpenAITTS, "gemini": GeminiTTS}[name]
    return cls(spec, voice or spec.voice, style, api_key=api_key, base_url=base_url,
               time_limit=time_limit)

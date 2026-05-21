import pytest

from models.manifest_models import AudioClipRecord
from narration_definitions import ClipStyle, NarrationClip, PauseSegment, SaySegment
from pipeline.staleness import compute_clip_hash, is_stale


def _make_clip(text: str = "Hello.", pause: float | None = None, style: ClipStyle | None = None) -> NarrationClip:
    segments: list = [SaySegment(type="say", text=text)]
    if pause is not None:
        segments.append(PauseSegment(type="pause", seconds=pause))
    return NarrationClip(segments=segments, style=style)


def _make_record(clip: NarrationClip, voice: str = "alloy", tts_model: str = "tts-1") -> AudioClipRecord:
    return AudioClipRecord(
        module_index=1,
        clip_index=1,
        filename="m01_c001.mp3",
        voice=voice,
        tts_model=tts_model,
        source_hash=compute_clip_hash(clip, voice, tts_model),
    )


class TestComputeClipHash:
    def test_deterministic(self):
        clip = _make_clip("Hello.")
        h1 = compute_clip_hash(clip, "alloy", "tts-1")
        h2 = compute_clip_hash(clip, "alloy", "tts-1")
        assert h1 == h2

    def test_different_text_different_hash(self):
        assert compute_clip_hash(_make_clip("Hello."), "alloy", "tts-1") != \
               compute_clip_hash(_make_clip("Goodbye."), "alloy", "tts-1")

    def test_different_voice_different_hash(self):
        clip = _make_clip("Hello.")
        assert compute_clip_hash(clip, "alloy", "tts-1") != compute_clip_hash(clip, "nova", "tts-1")

    def test_different_model_different_hash(self):
        clip = _make_clip("Hello.")
        assert compute_clip_hash(clip, "alloy", "tts-1") != compute_clip_hash(clip, "alloy", "tts-1-hd")

    def test_pause_change_different_hash(self):
        assert compute_clip_hash(_make_clip("Hello.", pause=0.5), "alloy", "tts-1") != \
               compute_clip_hash(_make_clip("Hello.", pause=1.0), "alloy", "tts-1")

    def test_returns_64_char_hex(self):
        h = compute_clip_hash(_make_clip(), "alloy", "tts-1")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestIsStale:
    def test_unchanged_clip_not_stale(self):
        clip = _make_clip("Hello.")
        assert not is_stale(clip, _make_record(clip), "tts-1")

    def test_text_change_marks_stale(self):
        clip = _make_clip("Hello.")
        record = _make_record(clip)
        assert is_stale(_make_clip("Goodbye."), record, "tts-1")

    def test_pause_seconds_change_marks_stale(self):
        clip = _make_clip("Hello.", pause=0.5)
        record = _make_record(clip)
        assert is_stale(_make_clip("Hello.", pause=1.5), record, "tts-1")

    def test_different_tts_model_marks_stale(self):
        clip = _make_clip("Hello.")
        record = _make_record(clip, tts_model="tts-1")
        assert is_stale(clip, record, "tts-1-hd")

    def test_voice_mismatch_marks_stale(self):
        clip = _make_clip("Hello.")
        # Record was generated with alloy hash, but record.voice says nova
        record = AudioClipRecord(
            module_index=1,
            clip_index=1,
            filename="m01_c001.mp3",
            voice="nova",
            tts_model="tts-1",
            source_hash=compute_clip_hash(clip, "alloy", "tts-1"),
        )
        assert is_stale(clip, record, "tts-1")

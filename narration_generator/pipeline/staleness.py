from __future__ import annotations

import hashlib
import json

from models.manifest_models import AudioClipRecord
from narration_definitions import NarrationClip


def compute_clip_hash(
    clip: NarrationClip,
    effective_voice: str,
    tts_model: str,
) -> str:
    segments = []
    for seg in clip.segments:
        if seg.type == "say":
            segments.append({"type": "say", "text": seg.text})
        else:
            segments.append({"type": "pause", "seconds": seg.seconds})

    payload = {
        "segments": segments,
        "voice": effective_voice,
        "tts_model": tts_model,
        "style": clip.style.model_dump() if clip.style else None,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def is_stale(clip: NarrationClip, record: AudioClipRecord, tts_model: str) -> bool:
    current = compute_clip_hash(clip, record.voice, tts_model)
    return current != record.source_hash

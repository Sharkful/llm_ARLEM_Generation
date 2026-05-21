from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProjectManifest(BaseModel):
    model_config = {"extra": "forbid"}

    project_id: str
    project_title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_model: str
    tts_provider: str
    tts_model: str
    default_voice: str
    workflow_mode: Literal["auto", "review"] = "auto"
    status: Literal[
        "created",
        "brief_generated",
        "script_generated",
        "audio_generated",
    ] = "created"


class AudioClipRecord(BaseModel):
    model_config = {"extra": "forbid"}

    module_index: int
    clip_index: int
    clip_title: Optional[str] = None
    filename: str
    voice: str
    tts_model: str
    source_hash: str
    status: Literal["pending", "generated", "failed", "stale"] = "pending"
    is_accepted: bool = False
    generated_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    character_count: int = 0
    error_message: Optional[str] = None


class AudioManifest(BaseModel):
    model_config = {"extra": "forbid"}

    project_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tts_provider: str
    clips: list[AudioClipRecord] = Field(default_factory=list)

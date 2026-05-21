from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class LLMCallEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["llm_call"] = "llm_call"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    provider: str = "anthropic"
    purpose: Literal["brief_generation", "script_generation"]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class TTSCallEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["tts_call"] = "tts_call"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model: str
    voice: str
    character_count: int
    clip_ref: str  # e.g. "m01_c001"
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class ValidationEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["validation"] = "validation"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: Literal["info", "warning", "error"]
    message: str
    clip_ref: Optional[str] = None


class SystemEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["system"] = "system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str


LogEntry = Annotated[
    Union[LLMCallEntry, TTSCallEntry, ValidationEntry, SystemEntry],
    Field(discriminator="event_type"),
]

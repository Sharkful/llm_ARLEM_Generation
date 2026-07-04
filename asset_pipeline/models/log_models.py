from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

LLMPurpose = Literal[
    "spec_parsing",
    "classification",
    "openscad_generation",
    "openscad_repair",
    "openscad_revision",
    "material_generation",
    "composite_decomposition",
    "source_relevance_text",
    "source_relevance_image",
]


class LLMCallEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["llm_call"] = "llm_call"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model: str
    purpose: LLMPurpose
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
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
    asset_id: Optional[str] = None


class SystemEntry(BaseModel):
    model_config = {"extra": "forbid"}

    event_type: Literal["system"] = "system"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: str

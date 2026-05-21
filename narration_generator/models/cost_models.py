from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class LLMCostRecord(BaseModel):
    model_config = {"extra": "forbid"}

    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float


class TTSCostRecord(BaseModel):
    model_config = {"extra": "forbid"}

    provider: str
    model: str
    voice: str
    character_count: int
    clip_ref: str
    cost_usd: float


class CostReport(BaseModel):
    model_config = {"extra": "forbid"}

    project_id: str
    run_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    llm_calls: list[LLMCostRecord] = Field(default_factory=list)
    tts_calls: list[TTSCostRecord] = Field(default_factory=list)
    total_llm_cost_usd: float = 0.0
    total_tts_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_clips_generated: int = 0

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from models.cost_models import CostReport, LLMCostRecord, TTSCostRecord
from models.log_models import LLMCallEntry, TTSCallEntry


class CostTracker:
    def __init__(self, price_config: dict) -> None:
        self._claude_prices: dict = price_config.get("claude", {})
        self._tts_prices: dict = price_config.get("tts", {})
        self._llm_entries: list[LLMCallEntry] = []
        self._tts_entries: list[TTSCallEntry] = []

    def record_llm_call(
        self,
        model: str,
        provider: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        duration_seconds: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> LLMCallEntry:
        prices = self._claude_prices.get(model, {})
        cost = (
            (input_tokens / 1000) * prices.get("input", 0)
            + (output_tokens / 1000) * prices.get("output", 0)
            + (cached_tokens / 1000) * prices.get("cached", 0)
        )
        entry = LLMCallEntry(
            model=model,
            provider=provider,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=round(cost, 6),
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
        )
        self._llm_entries.append(entry)
        return entry

    def record_tts_call(
        self,
        provider: str,
        model: str,
        voice: str,
        character_count: int,
        clip_ref: str,
        duration_seconds: float,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> TTSCallEntry:
        prices = self._tts_prices.get(model, {})
        cost = (character_count / 1000) * prices.get("per_1k_chars", 0)
        entry = TTSCallEntry(
            provider=provider,
            model=model,
            voice=voice,
            character_count=character_count,
            clip_ref=clip_ref,
            cost_usd=round(cost, 6),
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
        )
        self._tts_entries.append(entry)
        return entry

    def build_cost_report(self, project_id: str, run_id: Optional[str] = None) -> CostReport:
        if run_id is None:
            run_id = str(uuid.uuid4())

        llm_records = [
            LLMCostRecord(
                model=e.model,
                purpose=e.purpose,
                input_tokens=e.input_tokens,
                output_tokens=e.output_tokens,
                cached_tokens=e.cached_tokens,
                cost_usd=e.cost_usd,
            )
            for e in self._llm_entries
            if e.success
        ]
        tts_records = [
            TTSCostRecord(
                provider=e.provider,
                model=e.model,
                voice=e.voice,
                character_count=e.character_count,
                clip_ref=e.clip_ref,
                cost_usd=e.cost_usd,
            )
            for e in self._tts_entries
            if e.success
        ]

        total_llm = sum(r.cost_usd for r in llm_records)
        total_tts = sum(r.cost_usd for r in tts_records)

        return CostReport(
            project_id=project_id,
            run_id=run_id,
            llm_calls=llm_records,
            tts_calls=tts_records,
            total_llm_cost_usd=round(total_llm, 6),
            total_tts_cost_usd=round(total_tts, 6),
            total_cost_usd=round(total_llm + total_tts, 6),
            total_clips_generated=sum(1 for e in self._tts_entries if e.success),
        )

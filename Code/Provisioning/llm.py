"""
The LLM calls of a provisioning run (spec: PROVISIONING.md, "Engineering rules").

One ``TrackedLLM`` per run: one client and one tracker, shared by every step (the
reuse judge, #108, and describe and route, #106). Each call goes through
``benchmark.create_instructor_client`` and ``benchmark.guarded_create`` (the #44
Gemini and Anthropic guards inside the bounded transient retry), tracked by
``InstructorTracker``, like every LLM entry point. Each step gets its own record
(calls, tokens, cost, retries, time, served models, any error); ``summary()`` covers
the whole run.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parent / "Benchmark"
DEFAULT_MODEL = "claude-haiku-5.5"
DEFAULT_MAX_RETRIES = 3

# (response_model, messages, max_retries) -> a validated response_model instance
CallFn = Callable[[type, list, int], BaseModel]

API_KEY_ENV = {"anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",),
               "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def _benchmark():
    if str(BENCHMARK_DIR) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_DIR))
    import benchmark
    return benchmark


def model_config_for(model_key: str):
    """The ``benchmark_config.MODELS`` entry for ``model_key`` (KeyError if unknown)."""
    bm = _benchmark()
    if model_key not in bm.MODELS:
        raise KeyError(f"unknown model {model_key!r} (see benchmark.py --list-models)")
    return bm.MODELS[model_key]


def check_environment(model_key: str) -> None:
    """Exit-worthy problems before any call: unknown model, SDK missing, or no key."""
    cfg = model_config_for(model_key)
    _benchmark().require_environment([cfg])
    names = API_KEY_ENV.get(cfg.provider.value, ())
    if names and not any(os.environ.get(n) for n in names):
        raise KeyError(f"no API key for {model_key}: set {' or '.join(names)} in .env")


class TrackedLLM:
    """The run's LLM: one tracked client (or ``call_fn``, for tests)."""

    def __init__(self, model_key: str = DEFAULT_MODEL, *, call_fn: Optional[CallFn] = None,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.model_key = model_key
        self.max_retries = max_retries
        self.stats: dict = {}       # transient_retries, reask_restarts (benchmark.guarded_create)
        self.tracker = None
        self.model_config = None
        if call_fn is not None:
            self._call_fn = call_fn
            return
        bm = _benchmark()
        from tracking import InstructorTracker
        self.model_config = model_config_for(model_key)
        self.tracker = InstructorTracker(
            model=self.model_config.model_id, provider=self.model_config.provider.value,
            pricing_config=self.model_config.pricing_config)
        client = bm.create_instructor_client(self.model_config, bm.SpecType.JSON_LAB)
        self._client = self.tracker.wrap(client)
        self._call_fn = self._llm_call

    def _llm_call(self, response_model, messages, max_retries):
        return _benchmark().guarded_create(self._client, self.tracker, self.model_config,
                                           response_model, messages, max_retries, self.stats)

    def describe(self) -> dict:
        """The model fields every step record starts with."""
        record: dict = {"model": self.model_key}
        if self.model_config is not None:
            record.update(model_id=self.model_config.model_id,
                          provider=self.model_config.provider.value,
                          reasoning_effort=self.model_config.reasoning_effort,
                          tool_choice=_benchmark().tool_choice_for(self.model_config))
        return record

    def call(self, response_model: type, prompt: str) -> tuple[Optional[BaseModel], dict]:
        """(result, record). The caller enters any validation scope around this call.
        On failure the result is None and the record carries the error."""
        record = self.describe()
        record["prompt_characters"] = len(prompt)
        before = dict(self.stats)
        first = len(self.tracker.aggregate.calls) if self.tracker is not None else 0
        t0 = time.monotonic()
        result, error = None, None
        try:
            result = self._call_fn(response_model, [{"role": "user", "content": prompt}],
                                   self.max_retries)
        except Exception as exc:  # recorded; the CLI decides the exit code
            error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:2000]}"
        record["seconds"] = round(time.monotonic() - t0, 2)
        for key in ("transient_retries", "reask_restarts"):
            record[key] = self.stats.get(key, 0) - before.get(key, 0)
        if self.tracker is not None:
            calls = self.tracker.aggregate.calls[first:]
            record["calls"] = len(calls)
            record["prompt_tokens"] = sum(c.total_usage.prompt_tokens for c in calls)
            record["completion_tokens"] = sum(c.total_usage.completion_tokens for c in calls)
            record["cost_usd"] = round(sum(self.tracker.pricing.calculate_cost(c.total_usage)
                                           for c in calls), 6)
            # This step's calls only (tracker.served_models() covers the whole run)
            record["served_models"] = sorted({a.served_model for c in calls for a in c.attempts
                                              if a.served_model})
        else:
            record["calls"] = 1
        if error:
            record["error"] = error
        return result, record

    def summary(self) -> Optional[dict]:
        """The whole run's tracker summary (None with ``call_fn``)."""
        return self.tracker.summary() if self.tracker is not None else None

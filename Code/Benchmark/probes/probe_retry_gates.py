"""
Live check that a validation reask is accepted by each provider (issue #70; gate 2 of #44).

One small call per model, built exactly like a benchmark call
(``create_instructor_client`` + ``provider_create_kwargs`` + the tracker + the
transient retryer). The response model's validator rejects attempt 1 on purpose,
so instructor has to send a reask. The model passes only if the provider accepts
that reask and attempt 2 parses. This is the failure #44 was about: a malformed
reask that the API rejects with a 400 before the model sees the correction.

It also covers the Anthropic tool_choice "auto" path (issue #70): a thinking
model like Claude Haiku 5.5 returns thinking blocks, which the reask must replay
unchanged. Haiku 5.5 accepts a forced choice, so the forced-choice 400 itself is
only exercised by probing Claude Sonnet 5.5, Opus 5.5 or Fable 5.1 (once one is
registered, #72). A plain-text reply instead of a tool call shows up as an extra parse
error, not a failure.

Costs a few cents per model. Needs the provider API keys in .env.

Usage:
    python "Code/Benchmark/probes/probe_retry_gates.py" claude-haiku-5.5 claude-haiku-4.5 \\
        claude-sonnet-5 gemini-3.1-flash-lite gemini-3.5-flash gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for sub in ("Code/Benchmark", "Code/Schemas", ""):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

_validations = [0]   # validations in the current call (reset per call)
_rejections = [0]    # deliberate rejections across the probe's calls


class CityFact(BaseModel):
    """A city, its approximate population and one fact about it."""

    city: str
    country: str
    population_millions: float = Field(description="Approximate population in millions")
    fact: str = Field(description="One short sentence about the city")

    @model_validator(mode="after")
    def _reject_first_attempt(self):
        _validations[0] += 1
        if _validations[0] == 1:
            _rejections[0] += 1
            raise ValueError(
                "Deliberate probe check: the first answer is always rejected. "
                "Call the tool again with the same city and a different fact."
            )
        return self


def probe(model_key: str, max_retries: int) -> dict:
    from benchmark import (
        _is_transient_api_error,
        _transient_retryer,
        create_instructor_client,
        provider_create_kwargs,
        tool_choice_for,
    )
    from benchmark_config import MODELS, SpecType
    from tracking import InstructorTracker

    mc = MODELS[model_key]
    tracker = InstructorTracker(model=mc.model_id, provider=mc.provider.value,
                                pricing_config=mc.pricing_config)
    client = tracker.wrap(create_instructor_client(mc, SpecType.JSON_LAB))
    kwargs = dict(
        response_model=CityFact, max_retries=max_retries,
        messages=[{"role": "system", "content": "You answer with structured data."},
                  {"role": "user", "content": "Pick a large city and describe it."}],
    )
    kwargs.update(provider_create_kwargs(mc, CityFact))

    def create():
        # Reset per call: when the transient retryer restarts a call (e.g. its reask
        # hit a 503), the new call must get its own deliberate rejection and reask,
        # so a PASS always means the final call's reask was accepted.
        _validations[0] = 0
        return client.create(**kwargs)

    _rejections[0] = 0
    out = {"model": model_key, "tool_choice": tool_choice_for(mc), "ok": False, "error": None}
    full_error = ""
    error: Exception | None = None
    retryer = _transient_retryer()
    try:
        result = retryer(create)
        out["result"] = f"{result.city}, {result.country}"
    except Exception as e:  # noqa: BLE001 — report every failure the same way
        error = e
        full_error = f"{type(e).__name__}: {e}"
        # instructor's retry exception opens with "<failed_attempts>"; show the
        # last line that says something instead.
        lines = [ln.strip() for ln in str(e).splitlines()
                 if ln.strip() and not ln.strip().startswith(("<", "For further information"))]
        out["error"] = f"{type(e).__name__}: {(lines[-1] if lines else str(e))[:200]}"
    out["transient_retries"] = retryer.statistics.get("attempt_number", 1) - 1
    calls = tracker.aggregate.calls
    out["parse_errors"] = sum(c.parse_errors for c in calls)
    out["api_errors"] = sum(c.api_errors for c in calls)
    # Only the final call can show the #44 shape: every earlier call ended in a
    # transient error the retryer re-sent (a 503, not a rejected reask).
    final_api_errors = calls[-1].api_errors if calls else 0
    api_failure = final_api_errors > 0 or any(
        s in full_error for s in ("Error code: 400", "BadRequestError", "INVALID_ARGUMENT"))
    # Requests sent: one tracker attempt per request (issue #71 fixed).
    out["requests"] = sum(c.total_attempts for c in calls)
    out["tokens"] = sum(c.total_usage.total_tokens for c in calls)
    out["cost_usd"] = sum(tracker.pricing.calculate_cost(c.total_usage) for c in calls)
    # Parse errors other than our deliberate rejections (one per call): plain-text
    # replies (tool_choice "auto") or the model failing validation on its own.
    out["other_parse_errors"] = out["parse_errors"] - _rejections[0]
    # PASS: our deliberate rejection was reasked, the provider accepted the reask
    # and a later attempt parsed, all within the final call.
    # TRANSIENT: the transient retryer ran out of re-sends (429/5xx/connection);
    # the provider was unavailable, which says nothing about the reask path.
    # FAIL: the provider rejected a request (the #44 shape: a malformed reask
    # 400s before the model sees the correction).
    # EXHAUSTED: no API error, but the retries ran out (e.g. plain-text replies),
    # so the reask path worked; rerun or raise --max-retries.
    if out["error"] is None and _validations[0] >= 2:
        out["status"] = "PASS"
    elif error is not None and _is_transient_api_error(error):
        out["status"] = "TRANSIENT"
    elif api_failure:
        out["status"] = "FAIL"
    else:
        out["status"] = "EXHAUSTED"
    out["ok"] = out["status"] == "PASS"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="+", help="Model keys from benchmark_config.MODELS")
    p.add_argument("--max-retries", type=int, default=2,
                   help="instructor max_retries per call (default 2: one deliberate "
                        "rejection plus room for one plain-text reply)")
    args = p.parse_args()
    logging.getLogger("instructor").setLevel(logging.CRITICAL)

    from dotenv import load_dotenv
    from benchmark import require_environment
    from benchmark_config import MODELS

    load_dotenv(PROJECT_ROOT / ".env")
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {', '.join(unknown)} (see benchmark.py --list-models)")
    require_environment([MODELS[m] for m in args.models])

    results = []
    for m in args.models:
        print(f"  probing {m} ...", flush=True)
        results.append(probe(m, args.max_retries))

    print(f"\n  {'model':<24} {'tool_choice':<11} {'result':<9} {'requests':>8} "
          f"{'other parse err':>15} {'api err':>7} {'tokens':>7} {'cost':>9}")
    for r in results:
        print(f"  {r['model']:<24} {r['tool_choice']:<11} {r['status']:<9} "
              f"{r['requests']:>8} {r['other_parse_errors']:>15} {r['api_errors']:>7} "
              f"{r['tokens']:>7,} ${r['cost_usd']:.5f}")
        if r["error"]:
            print(f"      {r['error']}")
        if r["transient_retries"]:
            print(f"      {r['transient_retries']} transient re-send(s) (429/5xx/connection); "
                  "the counts above include the restarted call(s)")
    print("\n  PASS      attempt 1 was rejected on purpose, the provider accepted the reask,")
    print("            and a later attempt parsed.")
    print("  TRANSIENT the provider stayed unavailable (429/5xx/connection) through every")
    print("            transient re-send; says nothing about the reask path. Rerun later.")
    print("  FAIL      the provider rejected a request (API error): the #44 reask failure.")
    print("  EXHAUSTED no API error, but the retries ran out before a valid answer (e.g. the")
    print("            model kept answering in plain text). The reask path itself worked;")
    print("            rerun, or raise --max-retries.")
    print("  other parse err = plain-text replies or the model's own validation failures,")
    print("  beyond the deliberate rejection (one per call).")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()

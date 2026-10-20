"""
Cost calculation utilities for LLM API usage.

Provides pricing data for common models and a calculator
to estimate costs from token usage.
"""

import re
from datetime import date
from typing import Dict, Optional

from .metrics import TokenUsage


# Default pricing per 1M tokens (USD), base/standard tier (no batch). See COST.md.
# Optional "cached_input" (cache read) and "cache_write" rates price the part of
# the input the provider reports as cached or written; an entry without them
# bills all input at "input" (how every wave-2 run was costed). Only the current
# roster (issue #72) carries cache rates.
# Optional "long_context_above": the per-request prompt size above which the
# provider bills a higher tier. Tiers aren't priced (our prompts stay far below);
# a request over it is counted and warned about, since its cost is understated.
# Optional "intro_until": the last day of an introductory discount. The stored
# rate is the list (post-discount) rate; a run before that day carries a note.
# Entries marked "verified 2026-07" (issue #32) or "verified 2026-10" (issue #72)
# were re-checked against the official provider pricing pages
# (platform.claude.com, developers.openai.com, ai.google.dev). Entries marked "legacy"/"retired" were superseded or removed
# from the current page and are NOT in the benchmark roster; the stored value is
# retained for historical-artifact cost comparison — treat with caution.
DEFAULT_PRICING: Dict[str, Dict] = {
    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-5.5":       {"input": 5.00,  "output": 30.00, "long_context_above": 272_000},   # verified 2026-07 (current flagship; >272k in = 2x/1.5x)
    "gpt-5.4":       {"input": 2.50,  "output": 15.00},   # verified 2026-07 (current mid)
    "gpt-5.4-mini":  {"input": 0.75,  "output":  4.50},   # verified 2026-07
    "gpt-5.4-nano":  {"input": 0.20,  "output":  1.25},   # verified 2026-07
    # GPT-5.6 and later: automatic caching; cache writes billed at 1.25x input.
    "gpt-6.1-sol":   {"input": 2.00,  "output": 10.00, "cached_input": 0.10, "cache_write": 2.50, "long_context_above": 272_000},   # verified 2026-10 (current mid; >272k in = 2x/1.5x)
    "gpt-6-luna":    {"input": 0.10,  "output":  0.50, "cached_input": 0.01, "cache_write": 0.125},  # verified 2026-10 (current small)
    "gpt-5-mini":    {"input": 0.25,  "output":  2.00},   # 2026-07: superseded by 5.4-mini (was mis-stored 0.15/0.60); not in roster
    "gpt-5-nano":    {"input": 0.05,  "output":  0.40},   # retired 2026-07 — removed from current page; not in roster
    "gpt-5":         {"input": 2.50,  "output": 10.00},
    "gpt-4.1":       {"input": 2.00,  "output":  8.00},
    "gpt-4.1-mini":  {"input": 0.40,  "output":  1.60},
    "gpt-4.1-nano":  {"input": 0.10,  "output":  0.40},
    "gpt-4o":        {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15,  "output":  0.60},   # verified 2026-07: legacy/grandfathered; dropped from roster (issue #32)
    "gpt-4-turbo":   {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50,  "output":  1.50},

    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-opus-4-8":          {"input":  5.00, "output": 25.00},   # verified 2026-07 (current flagship Opus)
    # 2026-08-14: Anthropic made the $2/$10 introductory rate permanent, so it is
    # now the standard rate — not a discount with an expiry. Was stored at the
    # $3/$15 standard rate on the assumption the intro would lapse 2026-08-31.
    # Note this is *also* the as-billed rate for every run in the repo: the
    # wave-2 sweep (2026-07-10) ran inside the old intro window, so $2/$10 is
    # correct both as "what it cost" and "what it costs now".
    "claude-sonnet-5":          {"input":  2.00, "output": 10.00},   # current mid
    "claude-sonnet-4-6":        {"input":  3.00, "output": 15.00},   # verified 2026-07: now legacy (superseded by Sonnet 5); dropped from roster
    # Haiku 4.5 — keyed by its full versioned model ID used in API calls
    "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00},   # verified 2026-07 (previous small)
    # Haiku 5.5: prompts up to 100K tokens. Longer prompts bill $0.50 / $2.50, which
    # isn't modelled here (benchmark and hierarchical prompts stay well under 100K).
    # Cache reads 0.1x input (0.05x on Opus 5.5), writes 1.25x (5-minute TTL).
    # Nothing is cached until the pipeline sends cache_control markers (#73).
    "claude-haiku-5-5":         {"input":  0.10, "output":  0.50, "cached_input": 0.01, "cache_write": 0.125, "long_context_above": 100_000},  # verified 2026-10 (current small; #70)
    "claude-sonnet-5-5":        {"input":  2.00, "output": 10.00, "cached_input": 0.20, "cache_write": 2.50},   # verified 2026-10 (current mid)
    "claude-opus-5-5":          {"input":  4.00, "output": 20.00, "cached_input": 0.20, "cache_write": 5.00},   # verified 2026-10 (current large)
    "claude-fable-5":           {"input": 10.00, "output": 50.00},   # verified 2026-07 (most capable; GA 2026-06-09; not in roster)
    "claude-3-haiku-20240307":   {"input":  0.25, "output":  1.25},   # legacy/deprecated — not on current page
    # Legacy / alternate keys
    "claude-opus-4":      {"input": 15.00, "output": 75.00},
    "claude-sonnet-4":    {"input":  3.00, "output": 15.00},
    "claude-3-opus":      {"input": 15.00, "output": 75.00},
    "claude-3.5-sonnet":  {"input":  3.00, "output": 15.00},
    "claude-3-sonnet":    {"input":  3.00, "output": 15.00},
    "claude-3-haiku":     {"input":  0.25, "output":  1.25},
    "claude-3.5-haiku":   {"input":  0.80, "output":  4.00},

    # ── Google Gemini ─────────────────────────────────────────────────────
    # Pro models are context-tiered; our runs are well under 200k so the <=200k tier applies.
    # Introductory discounts: these two are stored at the rate that applies once the
    # discount ends, so costs reflect what a run will cost going forward. If a
    # provider makes the introductory rate permanent (as Anthropic did for Sonnet 5,
    # above), update the entry and re-tabulate. Code/Analysis/reprice_runs.py doesn't
    # apply cache rates yet, so re-price current-roster runs only after #84.
    #   gemini-4-argon:   $2.00 / $10.00, cached $0.10,  until 2026-11-30
    #   gemini-3.8-flash: $0.75 / $3.75,  cached $0.075, until 2026-12-31
    # Implicit caching (automatic; Argon from 4,096-token prompts): no write charge
    # and no storage fee, so these carry a read rate only.
    "gemini-4-argon":         {"input": 4.00,  "output": 20.00, "cached_input": 0.20,
                               "intro_until": "2026-11-30", "long_context_above": 200_000},  # verified 2026-10 (current large; from 2026-12-01; 200k tier to confirm)
    "gemini-3.8-flash":       {"input": 1.50,  "output":  7.50, "cached_input": 0.15,
                               "intro_until": "2026-12-31"},  # verified 2026-10 (current mid; from 2027-01-01)
    # Serves gemini-3.5-flash requests since 2026-10-08 (served-model pricing, #84).
    # Intro $0.75 / $3.75, cached $0.075, until 2026-12-31.
    "gemini-3.6-flash":       {"input": 1.50,  "output":  7.50, "cached_input": 0.15,
                               "intro_until": "2026-12-31"},  # 2026-10 (ai.google.dev pricing page; to verify)
    "gemini-3.5-flash-lite":  {"input": 0.30,  "output":  2.50, "cached_input": 0.03},  # verified 2026-10 (current small)
    "gemini-3.1-pro-preview": {"input": 2.00,  "output": 12.00, "long_context_above": 200_000},  # verified 2026-07 (keyed by model_id; v1beta serves only -preview, not the GA-renamed plain id — issue #42; <=200k tier)
    "gemini-3.1-flash-lite":  {"input": 0.25,  "output":  1.50},  # verified 2026-07 (GA'd 2026-05-07, renamed from -preview)
    "gemini-3.5-flash":       {"input": 1.50,  "output":  9.00},  # verified 2026-07 (current text flash)
    "gemini-2.5-pro":         {"input": 1.25,  "output": 10.00, "long_context_above": 200_000},  # verified 2026-07 (<=200k tier; current but dropped from roster)
    "gemini-2.5-flash":       {"input": 0.30,  "output":  2.50},  # verified 2026-07 (current but dropped from roster)
    "gemini-2.5-flash-lite":  {"input": 0.10,  "output":  0.40},  # verified 2026-07 (cheapest; kept in roster)
    "gemini-2.0-flash":       {"input": 0.10,  "output":  0.40},
    "gemini-1.5-pro":         {"input": 1.25,  "output":  5.00},
    "gemini-1.5-flash":       {"input": 0.075, "output":  0.30},

    # ── Mistral ───────────────────────────────────────────────────────────
    "mistral-large":  {"input": 2.00, "output": 6.00},
    "mistral-medium": {"input": 2.70, "output": 8.10},
    "mistral-small":  {"input": 0.20, "output": 0.60},

    # ── Other ─────────────────────────────────────────────────────────────
    "llama-3.1-70b": {"input": 0.88, "output": 0.88},
    "llama-3.1-8b":  {"input": 0.10, "output": 0.10},
}


_DATE_SUFFIX = re.compile(r"(-\d{4}-\d{2}-\d{2}|-\d{8})$")


def pricing_key(model_id: Optional[str]) -> Optional[str]:
    """The DEFAULT_PRICING key for a model id: the id itself, or the id without a
    dated-snapshot suffix (``gpt-6-luna-2026-09-01`` -> ``gpt-6-luna``). None when
    neither has an entry."""
    if not model_id:
        return None
    model_id = model_id.removeprefix("models/")
    if model_id in DEFAULT_PRICING:
        return model_id
    base = _DATE_SUFFIX.sub("", model_id)
    return base if base in DEFAULT_PRICING else None


class PricingCalculator:
    """
    Calculate costs based on token usage and model pricing.

    Can be initialized with explicit pricing or will look up
    pricing for known models.
    """

    def __init__(
        self,
        config: Optional[Dict[str, float]] = None,
        model: Optional[str] = None
    ):
        """
        Initialize pricing calculator.

        Args:
            config: Dict with 'input' and 'output' prices per 1M tokens, and
                optional 'cached_input' / 'cache_write' rates (default: 'input')
            model: Model name to look up in default pricing

        If both are provided, config takes precedence.
        If neither provides valid pricing, costs will be 0.
        """
        # An override (ModelConfig.pricing_config: input/output only) is merged onto
        # the model's entry, so its cache rates survive.
        pricing = {**(DEFAULT_PRICING.get(model, {}) if model else {}), **(config or {})}
        self.input_price = pricing.get("input", 0.0)
        self.output_price = pricing.get("output", 0.0)
        self.cached_input_price = pricing.get("cached_input", self.input_price)
        self.cache_write_price = pricing.get("cache_write", self.input_price)
        self.long_context_above: Optional[int] = pricing.get("long_context_above")
        self.intro_until: Optional[str] = pricing.get("intro_until")

    def intro_note(self, model: str, on: date) -> Optional[str]:
        """A note when ``on`` falls inside the model's introductory discount."""
        if self.intro_until and on <= date.fromisoformat(self.intro_until):
            return (f"{model}: introductory price in effect until {self.intro_until}; "
                    "cost uses the list rate")
        return None

    def calculate_cost(self, usage: TokenUsage) -> float:
        """
        Calculate cost in USD for given token usage.

        Args:
            usage: TokenUsage object with prompt and completion tokens

        Returns:
            Total cost in USD
        """
        return self.cost_breakdown(usage)["total"]

    def format_cost(self, usage: TokenUsage) -> str:
        """
        Return formatted cost string.

        Uses more decimal places for small costs.
        """
        cost = self.calculate_cost(usage)
        if cost < 0.01:
            return f"${cost:.6f}"
        return f"${cost:.4f}"

    def cost_breakdown(self, usage: TokenUsage) -> Dict[str, float]:
        """
        Get detailed cost breakdown.

        Returns dict with input_cost (all input, cached and written parts at
        their own rates), output_cost, and total.
        """
        uncached = max(usage.prompt_tokens - usage.cached_tokens - usage.cache_write_tokens, 0)
        input_cost = (uncached * self.input_price
                      + usage.cached_tokens * self.cached_input_price
                      + usage.cache_write_tokens * self.cache_write_price) / 1_000_000
        output_cost = (usage.completion_tokens * self.output_price) / 1_000_000
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total": input_cost + output_cost,
        }

    @classmethod
    def get_pricing_for_model(cls, model: str) -> Optional[Dict[str, float]]:
        """Look up pricing for a known model."""
        return DEFAULT_PRICING.get(model)

    @classmethod
    def list_known_models(cls) -> list:
        """Return list of models with known pricing."""
        return list(DEFAULT_PRICING.keys())

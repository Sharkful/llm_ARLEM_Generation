"""
Cost calculation utilities for LLM API usage.

Provides pricing data for common models and a calculator
to estimate costs from token usage.
"""

from typing import Dict, Optional

from .metrics import TokenUsage


# Default pricing per 1M tokens (USD), base/standard tier (no caching/batch).
# Entries marked "verified 2026-07" were re-checked against the official provider
# pricing pages (platform.claude.com, developers.openai.com, ai.google.dev) as
# part of issue #32. Entries marked "legacy"/"retired" were superseded or removed
# from the current page and are NOT in the benchmark roster; the stored value is
# retained for historical-artifact cost comparison — treat with caution.
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-5.5":       {"input": 5.00,  "output": 30.00},   # verified 2026-07 (current flagship; >272k in = 2x/1.5x)
    "gpt-5.4":       {"input": 2.50,  "output": 15.00},   # verified 2026-07 (current mid)
    "gpt-5.4-mini":  {"input": 0.75,  "output":  4.50},   # verified 2026-07
    "gpt-5.4-nano":  {"input": 0.20,  "output":  1.25},   # verified 2026-07
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
    "claude-sonnet-5":          {"input":  3.00, "output": 15.00},   # verified 2026-07 (current mid; intro $2/$10 thru 2026-08-31 — standard rate stored)
    "claude-sonnet-4-6":        {"input":  3.00, "output": 15.00},   # verified 2026-07: now legacy (superseded by Sonnet 5); dropped from roster
    # Haiku 4.5 — keyed by its full versioned model ID used in API calls
    "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00},   # verified 2026-07 (current small)
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
    "gemini-3.1-pro-preview": {"input": 2.00,  "output": 12.00},  # verified 2026-07 (keyed by model_id; v1beta serves only -preview, not the GA-renamed plain id — issue #42; <=200k tier)
    "gemini-3.1-flash-lite":  {"input": 0.25,  "output":  1.50},  # verified 2026-07 (GA'd 2026-05-07, renamed from -preview)
    "gemini-3.5-flash":       {"input": 1.50,  "output":  9.00},  # verified 2026-07 (current text flash)
    "gemini-2.5-pro":         {"input": 1.25,  "output": 10.00},  # verified 2026-07 (<=200k tier; current but dropped from roster)
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
            config: Dict with 'input' and 'output' prices per 1M tokens
            model: Model name to look up in default pricing

        If both are provided, config takes precedence.
        If neither provides valid pricing, costs will be 0.
        """
        if config:
            self.input_price = config.get("input", 0.0)
            self.output_price = config.get("output", 0.0)
        elif model and model in DEFAULT_PRICING:
            pricing = DEFAULT_PRICING[model]
            self.input_price = pricing["input"]
            self.output_price = pricing["output"]
        else:
            self.input_price = 0.0
            self.output_price = 0.0

    def calculate_cost(self, usage: TokenUsage) -> float:
        """
        Calculate cost in USD for given token usage.

        Args:
            usage: TokenUsage object with prompt and completion tokens

        Returns:
            Total cost in USD
        """
        input_cost = (usage.prompt_tokens * self.input_price) / 1_000_000
        output_cost = (usage.completion_tokens * self.output_price) / 1_000_000
        return input_cost + output_cost

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

        Returns dict with input_cost, output_cost, and total.
        """
        input_cost = (usage.prompt_tokens * self.input_price) / 1_000_000
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

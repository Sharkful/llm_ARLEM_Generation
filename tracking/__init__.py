"""
Instructor Token & Retry Tracking Module

Provides comprehensive tracking of token usage and retry attempts
for Instructor-based LLM calls using the hooks system.

Usage:
    from tracking import InstructorTracker

    tracker = InstructorTracker(model="gpt-5-nano", provider="openai")
    tracked_client = tracker.wrap(client)

    result = tracked_client.create_with_completion(...)
    tracker.print_summary()
"""

from .metrics import (
    TokenUsage,
    ErrorType,
    ErrorRecord,
    AttemptMetrics,
    CallMetrics,
    AggregateMetrics,
)
from .tracker import InstructorTracker, TrackedClient
from .pricing import PricingCalculator, DEFAULT_PRICING
from .export import BenchmarkExporter

__all__ = [
    # Main tracker
    "InstructorTracker",
    "TrackedClient",
    # Data structures
    "TokenUsage",
    "ErrorType",
    "ErrorRecord",
    "AttemptMetrics",
    "CallMetrics",
    "AggregateMetrics",
    # Utilities
    "PricingCalculator",
    "DEFAULT_PRICING",
    "BenchmarkExporter",
]

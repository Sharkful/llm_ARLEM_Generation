"""
Export and reporting utilities for benchmark data.

Provides functions to export metrics in various formats
for analysis and documentation.
"""

import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .metrics import AggregateMetrics, TokenUsage
from .pricing import PricingCalculator


class BenchmarkExporter:
    """Export tracking data in various formats for analysis."""

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Recursively serialize objects for JSON compatibility."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, TokenUsage):
            return {
                "prompt_tokens": obj.prompt_tokens,
                "completion_tokens": obj.completion_tokens,
                "total_tokens": obj.total_tokens,
                "cached_tokens": obj.cached_tokens,
                "reasoning_tokens": obj.reasoning_tokens,
            }
        if hasattr(obj, '__dict__'):
            return {
                k: BenchmarkExporter._serialize(v)
                for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
        if isinstance(obj, list):
            return [BenchmarkExporter._serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: BenchmarkExporter._serialize(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def to_dict(aggregate: AggregateMetrics) -> Dict[str, Any]:
        """Convert aggregate metrics to a serializable dictionary."""
        return BenchmarkExporter._serialize(aggregate)

    @staticmethod
    def to_json(
        aggregate: AggregateMetrics,
        filepath: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Export to JSON format.

        Args:
            aggregate: AggregateMetrics to export
            filepath: Optional path to save JSON file
            indent: Indentation level for formatting

        Returns:
            JSON string
        """
        data = BenchmarkExporter.to_dict(aggregate)
        json_str = json.dumps(data, indent=indent, default=str)

        if filepath:
            Path(filepath).write_text(json_str, encoding='utf-8')

        return json_str

    @staticmethod
    def to_csv_summary(
        aggregates: List[AggregateMetrics],
        filepath: str,
        include_pricing: bool = True
    ):
        """
        Export multiple benchmark runs to CSV for comparison.

        Args:
            aggregates: List of AggregateMetrics from different runs
            filepath: Path to save CSV file
            include_pricing: Whether to include cost columns
        """
        headers = [
            "session_id",
            "model",
            "provider",
            "timestamp",
            "total_calls",
            "successful_calls",
            "failed_calls",
            "success_rate",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "total_retries",
            "calls_with_retries",
            "retry_rate",
            "avg_retries_per_call",
            "max_retries_single_call",
            "parse_errors",
            "api_errors",
            "completion_errors",
            "total_duration_ms",
            "avg_duration_ms",
        ]

        if include_pricing:
            headers.extend(["estimated_cost_usd"])

        rows = []
        for agg in aggregates:
            row = [
                agg.session_id,
                agg.model,
                agg.provider,
                agg.start_time.isoformat(),
                agg.total_calls,
                agg.successful_calls,
                agg.failed_calls,
                f"{agg.success_rate:.4f}",
                agg.total_usage.prompt_tokens,
                agg.total_usage.completion_tokens,
                agg.total_usage.total_tokens,
                agg.total_retries,
                agg.calls_with_retries,
                f"{agg.retry_rate:.4f}",
                f"{agg.avg_retries_per_call:.4f}",
                agg.max_retries_single_call,
                agg.total_parse_errors,
                agg.total_api_errors,
                agg.total_completion_errors,
                f"{agg.total_duration_ms:.2f}",
                f"{agg.avg_duration_ms:.2f}",
            ]

            if include_pricing:
                pricing = PricingCalculator(model=agg.model)
                cost = pricing.calculate_cost(agg.total_usage)
                row.append(f"{cost:.6f}")

            rows.append(row)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    @staticmethod
    def to_markdown_report(
        aggregate: AggregateMetrics,
        pricing_calculator: Optional[PricingCalculator] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Generate a markdown report for documentation/GitHub.

        Args:
            aggregate: AggregateMetrics to report on
            pricing_calculator: Optional calculator for cost section
            title: Optional custom title

        Returns:
            Markdown formatted string
        """
        agg = aggregate
        title = title or "Benchmark Report"

        cost_section = ""
        if pricing_calculator:
            total_cost = pricing_calculator.calculate_cost(agg.total_usage)
            cost_per_call = total_cost / max(agg.total_calls, 1)
            cost_per_1k_tokens = (
                (total_cost / max(agg.total_usage.total_tokens, 1)) * 1000
            )
            cost_section = f"""
## Cost Analysis

| Metric | Value |
|--------|-------|
| Total Cost | ${total_cost:.6f} |
| Cost per Call | ${cost_per_call:.6f} |
| Cost per 1K Tokens | ${cost_per_1k_tokens:.6f} |
"""

        return f"""# {title}

**Model:** {agg.model}
**Provider:** {agg.provider}
**Session ID:** {agg.session_id}
**Timestamp:** {agg.start_time.isoformat()}

## Summary

| Metric | Value |
|--------|-------|
| Total Calls | {agg.total_calls} |
| Successful | {agg.successful_calls} |
| Failed | {agg.failed_calls} |
| Success Rate | {agg.success_rate:.1%} |

## Token Usage

| Token Type | Count |
|------------|-------|
| Prompt Tokens | {agg.total_usage.prompt_tokens:,} |
| Completion Tokens | {agg.total_usage.completion_tokens:,} |
| **Total Tokens** | **{agg.total_usage.total_tokens:,}** |

## Retry Statistics

| Metric | Value |
|--------|-------|
| Total Retries | {agg.total_retries} |
| Calls Requiring Retries | {agg.calls_with_retries} |
| Retry Rate | {agg.retry_rate:.1%} |
| Avg Retries per Call | {agg.avg_retries_per_call:.2f} |
| Max Retries (Single Call) | {agg.max_retries_single_call} |

## Error Breakdown

| Error Type | Count |
|------------|-------|
| Parse Errors | {agg.total_parse_errors} |
| API Errors | {agg.total_api_errors} |
| Completion Errors | {agg.total_completion_errors} |

## Performance

| Metric | Value |
|--------|-------|
| Total Duration | {agg.total_duration_ms:.1f}ms |
| Avg Duration per Call | {agg.avg_duration_ms:.1f}ms |
{cost_section}
---
*Generated by InstructorTracker*
"""

    @staticmethod
    def compare_models(
        aggregates: List[AggregateMetrics],
        include_pricing: bool = True
    ) -> str:
        """
        Generate a markdown comparison table for multiple models.

        Args:
            aggregates: List of AggregateMetrics to compare
            include_pricing: Whether to include cost comparison

        Returns:
            Markdown formatted comparison table
        """
        if not aggregates:
            return "No data to compare."

        lines = [
            "# Model Comparison",
            "",
            "| Model | Calls | Success Rate | Tokens | Retries | Parse Errors | Avg Duration |",
            "|-------|-------|--------------|--------|---------|--------------|--------------|",
        ]

        if include_pricing:
            lines[2] = lines[2].rstrip(" |") + " Cost |"
            lines[3] = lines[3].rstrip("|") + "------|"

        for agg in aggregates:
            row = (
                f"| {agg.model} | {agg.total_calls} | {agg.success_rate:.1%} | "
                f"{agg.total_usage.total_tokens:,} | {agg.total_retries} | "
                f"{agg.total_parse_errors} | {agg.avg_duration_ms:.1f}ms |"
            )

            if include_pricing:
                pricing = PricingCalculator(model=agg.model)
                cost = pricing.calculate_cost(agg.total_usage)
                row = row.rstrip(" |") + f" ${cost:.4f} |"

            lines.append(row)

        return "\n".join(lines)

"""
AR Lab Generation Benchmark Runner

Benchmarks LLM providers on structured AR lab generation quality,
tracking tokens, cost, retries, and output metrics.

Usage:
    # Single model, default topic (JSON Lab spec)
    python benchmark.py --model gpt-4o-mini

    # Multiple models, custom topic
    python benchmark.py --model gpt-4o-mini claude-3-haiku gemini-2.0-flash \
        --topic "Volcanic Eruption Mechanics"

    # ARLEM spec instead of JSON Lab
    python benchmark.py --model claude-sonnet-4 --spec arlem

    # Quick benchmark suite (all default topics × cheap models)
    python benchmark.py --suite quick

    # Full benchmark suite (all default topics × all models)
    python benchmark.py --suite full
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path so we can import from Code/Tools and tracking/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Tools"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Local imports
from benchmark_config import (
    ARLEM_PROMPT_TEMPLATE,
    BenchmarkRunConfig,
    DEFAULT_TOPICS,
    FULL_BENCHMARK_MODELS,
    MODELS,
    ModelConfig,
    Provider,
    QUICK_BENCHMARK_MODELS,
    SYSTEM_PROMPT,
    SpecType,
)
from lab_metrics import analyze_json_lab, analyze_arlem

from tracking import InstructorTracker, BenchmarkExporter


# ── Client Factory ───────────────────────────────────────────────────

def create_instructor_client(model_config: ModelConfig, max_retries: int = 3):
    """
    Create an instructor-patched client for the given provider.

    For Gemini models, uses alternate Pydantic models without Union types
    (loaded via get_response_model with use_gemini_models=True).
    """
    import instructor

    if model_config.provider == Provider.OPENAI:
        from openai import OpenAI
        return instructor.from_openai(OpenAI(), mode=instructor.Mode.TOOLS)

    elif model_config.provider == Provider.ANTHROPIC:
        import anthropic
        return instructor.from_anthropic(anthropic.Anthropic())

    elif model_config.provider == Provider.GOOGLE:
        import google.generativeai as genai
        return instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model_config.model_id),
            use_async=False,
        )

    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")


# ── Model Selection ──────────────────────────────────────────────────

def get_response_model(spec_type: SpecType, use_gemini_models: bool = False):
    """
    Return the appropriate Pydantic response model class.

    Args:
        spec_type: Which spec to generate (JSON_LAB or ARLEM)
        use_gemini_models: If True, import from Gemini-compatible models
                           that avoid Union/discriminated-union types.
                           User will supply these alternate model files.
    """
    if use_gemini_models:
        # Gemini-compatible models (no Union types)
        # These files are expected to be provided by the user at:
        #   Code/Tools/pydantic_json_lab_gemini.py  (for JSON_LAB)
        #   Code/Tools/arlem_simplified.py          (for ARLEM)
        if spec_type == SpecType.JSON_LAB:
            try:
                from pydantic_json_lab_gemini import Lab as GeminiLab
                return GeminiLab
            except ImportError:
                print(
                    "WARNING: pydantic_json_lab_gemini.py not found. "
                    "Falling back to standard models. Gemini may fail on Union types."
                )
                from pydantic_json_lab_claude import Lab
                return Lab
        else:
            try:
                from arlem_simplified import ARLEMScenario as SimplifiedARLEM
                return SimplifiedARLEM
            except ImportError:
                print(
                    "WARNING: arlem_simplified.py not found. "
                    "Falling back to full ARLEM. Gemini may fail on Union types."
                )
                from arlem_full import ARLEMScenario
                return ARLEMScenario
    else:
        # Standard models (OpenAI / Anthropic)
        if spec_type == SpecType.JSON_LAB:
            from pydantic_json_lab_claude import Lab
            return Lab
        else:
            from arlem_full import ARLEMScenario
            return ARLEMScenario


# ── Single Run ───────────────────────────────────────────────────────

def run_single_benchmark(
    model_config: ModelConfig,
    run_config: BenchmarkRunConfig,
) -> dict:
    """
    Execute a single benchmark run: one model × one topic × one spec.

    Returns a dict with all metrics, the generated output, and lab analysis.
    """
    is_gemini = model_config.provider == Provider.GOOGLE
    response_model = get_response_model(run_config.spec_type, use_gemini_models=is_gemini)

    # Create instructor client
    client = create_instructor_client(model_config, run_config.max_retries)

    # Set up tracking
    tracker = InstructorTracker(
        model=model_config.model_id,
        provider=model_config.provider.value,
        pricing_config=model_config.pricing_config,
    )
    tracked_client = tracker.wrap(client)

    # Build messages
    user_prompt = run_config.get_prompt()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # For Anthropic, system prompt goes separately
    create_kwargs = dict(
        response_model=response_model,
        messages=messages,
        max_retries=run_config.max_retries,
    )

    # Anthropic and Gemini don't use model in create() - it's set on the client
    if model_config.provider == Provider.OPENAI:
        create_kwargs["model"] = model_config.model_id
    elif model_config.provider == Provider.ANTHROPIC:
        create_kwargs["model"] = model_config.model_id
        create_kwargs["max_tokens"] = 16384

    # Execute generation
    print(f"\n  Generating with {model_config.display_name}...")
    print(f"  Spec: {run_config.spec_type.value} | Topic: {run_config.topic[:50]}...")

    start_time = time.time()
    result = None
    generation_error = None

    try:
        result = tracked_client.create(**create_kwargs)
    except Exception as e:
        generation_error = str(e)
        print(f"  ERROR: {type(e).__name__}: {generation_error[:200]}")

    wall_time_s = time.time() - start_time

    # Analyze the generated lab
    lab_metrics = {}
    output_json = None
    if result is not None:
        try:
            output_json = result.model_dump(mode="json", exclude_none=True)
            if run_config.spec_type == SpecType.JSON_LAB:
                lab_metrics = analyze_json_lab(output_json)
            else:
                lab_metrics = analyze_arlem(output_json)
        except Exception as e:
            lab_metrics = {"analysis_error": str(e)}

    # Collect tracking metrics
    tracker_summary = tracker.summary()

    # Build result record
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model_config.model_id,
        "provider": model_config.provider.value,
        "display_name": model_config.display_name,
        "spec_type": run_config.spec_type.value,
        "topic": run_config.topic,
        "success": result is not None,
        "error": generation_error,
        "wall_time_seconds": round(wall_time_s, 2),
        "tracking": tracker_summary,
        "lab_metrics": lab_metrics,
    }

    # Save generated output
    if run_config.save_output and output_json is not None:
        output_path = _save_output(model_config, run_config, output_json, record)
        record["output_path"] = str(output_path)

    # Print summary
    _print_run_summary(record)

    return record


# ── Output Saving ────────────────────────────────────────────────────

def _save_output(
    model_config: ModelConfig,
    run_config: BenchmarkRunConfig,
    output_json: dict,
    record: dict,
) -> Path:
    """Save generated JSON and metrics to the benchmark output directory."""
    output_dir = PROJECT_ROOT / run_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filename: {model}_{spec}_{timestamp}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_config.model_id.replace("/", "_").replace(".", "-")
    base_name = f"{safe_model}_{run_config.spec_type.value}_{ts}"

    # Save the generated lab JSON
    lab_path = output_dir / f"{base_name}_output.json"
    lab_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")

    # Save the metrics record
    metrics_path = output_dir / f"{base_name}_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print(f"  Saved: {lab_path.relative_to(PROJECT_ROOT)}")
    return lab_path


def _print_run_summary(record: dict):
    """Print a concise summary of a benchmark run."""
    tracking = record["tracking"]
    tokens = tracking["tokens"]
    cost = tracking["cost"]
    retries = tracking["retries"]
    lab = record.get("lab_metrics", {})

    status = "OK" if record["success"] else "FAILED"
    print(f"\n  [{status}] {record['display_name']} — {record['spec_type']}")
    print(f"  Tokens: {tokens['prompt']:,} in / {tokens['completion']:,} out / {tokens['total']:,} total")
    print(f"  Cost: {cost['formatted']} | Wall time: {record['wall_time_seconds']}s")
    print(f"  Retries: {retries['total']} (parse errors: {tracking['errors']['parse_errors']})")

    if lab:
        # Print key lab metrics
        metric_parts = []
        for key in ["num_objects", "num_clips", "num_components", "num_things",
                     "num_places", "num_actions", "num_predicates"]:
            if key in lab:
                label = key.replace("num_", "").replace("_", " ")
                metric_parts.append(f"{label}: {lab[key]}")
        if metric_parts:
            print(f"  Lab: {' | '.join(metric_parts)}")


# ── Suite Runner ─────────────────────────────────────────────────────

def run_benchmark_suite(
    model_ids: list[str],
    topics: list[str],
    spec_type: SpecType = SpecType.JSON_LAB,
    max_retries: int = 3,
    save_output: bool = True,
) -> list[dict]:
    """
    Run benchmarks across multiple models and topics.

    Returns list of all result records.
    """
    results = []
    total_runs = len(model_ids) * len(topics)
    run_num = 0

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK SUITE: {len(model_ids)} models × {len(topics)} topics = {total_runs} runs")
    print(f"Spec: {spec_type.value}")
    print(f"{'=' * 70}")

    for topic in topics:
        for model_id in model_ids:
            run_num += 1
            print(f"\n--- Run {run_num}/{total_runs} ---")

            if model_id not in MODELS:
                print(f"  SKIPPED: Unknown model '{model_id}'")
                continue

            model_config = MODELS[model_id]
            run_config = BenchmarkRunConfig(
                spec_type=spec_type,
                topic=topic,
                max_retries=max_retries,
                save_output=save_output,
            )

            try:
                record = run_single_benchmark(model_config, run_config)
                results.append(record)
            except Exception as e:
                print(f"  FATAL ERROR: {type(e).__name__}: {e}")
                results.append({
                    "model": model_id,
                    "topic": topic,
                    "success": False,
                    "error": f"Fatal: {e}",
                })

    # Print comparison summary
    _print_suite_summary(results)

    # Save combined results
    if save_output:
        output_dir = PROJECT_ROOT / "Artifacts" / "Data" / "Benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"suite_results_{ts}.json"
        summary_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSuite results saved: {summary_path.relative_to(PROJECT_ROOT)}")

    return results


def _print_suite_summary(results: list[dict]):
    """Print a comparison table of all benchmark runs."""
    print(f"\n{'=' * 70}")
    print("SUITE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Spec':<10} {'Status':<8} {'Tokens':>8} {'Cost':>10} {'Time':>7} {'Retries':>8}")
    print("-" * 80)

    for r in results:
        if "tracking" not in r:
            print(f"{r.get('model', '?'):<25} {'?':<10} {'FATAL':<8}")
            continue

        t = r["tracking"]
        print(
            f"{r['display_name']:<25} "
            f"{r['spec_type']:<10} "
            f"{'OK' if r['success'] else 'FAIL':<8} "
            f"{t['tokens']['total']:>8,} "
            f"{t['cost']['formatted']:>10} "
            f"{r['wall_time_seconds']:>6.1f}s "
            f"{t['retries']['total']:>8}"
        )


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM providers on AR lab generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Model ID(s) to benchmark (e.g., gpt-4o-mini claude-3-haiku)",
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Topic for the AR lab (default: Solar System)",
    )
    parser.add_argument(
        "--spec", "-s",
        type=str,
        choices=["json_lab", "arlem"],
        default="json_lab",
        help="Specification type to generate (default: json_lab)",
    )
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default=None,
        help="Run a pre-defined benchmark suite",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max instructor retries per call (default: 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output files",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model IDs and exit",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for model_id, config in MODELS.items():
            print(f"  {model_id:<25} {config.provider.value:<12} {config.display_name}")
        return

    spec_type = SpecType(args.spec)
    save_output = not args.no_save

    if args.suite:
        # Run pre-defined suite
        model_ids = (
            QUICK_BENCHMARK_MODELS if args.suite == "quick"
            else FULL_BENCHMARK_MODELS
        )
        topics = [args.topic] if args.topic else DEFAULT_TOPICS
        run_benchmark_suite(
            model_ids=model_ids,
            topics=topics,
            spec_type=spec_type,
            max_retries=args.max_retries,
            save_output=save_output,
        )
    elif args.model:
        # Run specific models
        topics = [args.topic or DEFAULT_TOPICS[0]]
        run_benchmark_suite(
            model_ids=args.model,
            topics=topics,
            spec_type=spec_type,
            max_retries=args.max_retries,
            save_output=save_output,
        )
    else:
        print("Error: Specify --model or --suite. Use --list-models to see options.")
        sys.exit(1)


if __name__ == "__main__":
    main()

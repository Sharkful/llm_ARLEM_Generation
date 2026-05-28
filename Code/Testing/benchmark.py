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
from prompt_builder import (
    LabDescription,
    Level,
    Structure,
    build_prompt,
    discover_labs,
)
from lab_metrics import analyze_json_lab, analyze_arlem

from tracking import InstructorTracker, BenchmarkExporter


# ── Client Factory ───────────────────────────────────────────────────

def create_instructor_client(model_config: ModelConfig):
    """
    Create an instructor-patched client for the given provider
    using ``instructor.from_provider("provider/model")``.

    For Gemini models, uses alternate Pydantic models without Union types
    (loaded via get_response_model with use_gemini_models=True).
    """
    import instructor

    provider_prefix = {
        Provider.OPENAI: "openai",
        Provider.ANTHROPIC: "anthropic",
        Provider.GOOGLE: "google",
    }
    prefix = provider_prefix.get(model_config.provider)
    if prefix is None:
        raise ValueError(f"Unknown provider: {model_config.provider}")

    kwargs = {}
    # from_provider expects GOOGLE_API_KEY but our .env uses GEMINI_API_KEY.
    # Use GENAI_STRUCTURED_OUTPUTS mode so Pydantic enums validate correctly.
    if model_config.provider == Provider.GOOGLE:
        kwargs["api_key"] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        kwargs["mode"] = instructor.Mode.GENAI_STRUCTURED_OUTPUTS

    return instructor.from_provider(
        f"{prefix}/{model_config.model_id}",
        **kwargs,
    )


# ── Model Selection ──────────────────────────────────────────────────

def get_response_model(spec_type: SpecType, use_gemini_models: bool = False):
    """
    Return the appropriate Pydantic response model class.

    Args:
        spec_type: Which spec to generate (JSON_LAB, ARLEM, or ARLEM_SIMPLIFIED)
        use_gemini_models: If True, import from Gemini-compatible models
                           that avoid Union/discriminated-union types.

    Gemini-compatible files (no Union / discriminated-union types):
        json_lab_gemini.py            → JSON_LAB
        arlem_full_gemini.py          → ARLEM
        arlem_simplified_gemini.py    → ARLEM_SIMPLIFIED
    """
    if use_gemini_models:
        if spec_type == SpecType.JSON_LAB:
            from json_lab_gemini import Lab as GeminiLab
            return GeminiLab
        elif spec_type == SpecType.ARLEM:
            from arlem_full_gemini import ARLEMScenario as GeminiFullARLEM
            return GeminiFullARLEM
        else:  # ARLEM_SIMPLIFIED
            from arlem_simplified_gemini import ARLEMScenario as GeminiSimpleARLEM
            return GeminiSimpleARLEM
    else:
        # Standard models (OpenAI / Anthropic)
        if spec_type == SpecType.JSON_LAB:
            from json_lab import Lab
            return Lab
        elif spec_type == SpecType.ARLEM:
            from arlem_full import ARLEMScenario
            return ARLEMScenario
        else:  # ARLEM_SIMPLIFIED
            from arlem_simplified import ARLEMScenario as SimpleARLEM
            return SimpleARLEM


# ── Single Run ───────────────────────────────────────────────────────

def run_single_benchmark(
    model_config: ModelConfig,
    run_config: BenchmarkRunConfig,
    *,
    save_prompts: bool = True,
    overwrite_prompts: bool = False,
) -> dict:
    """
    Execute a single benchmark run: one model × one topic × one spec.

    Returns a dict with all metrics, the generated output, and lab analysis.
    """
    is_gemini = model_config.provider == Provider.GOOGLE

    # Build prompt + response model. YAML-driven path takes precedence.
    prompt_file_rel: Optional[str] = None
    if run_config.lab_name and run_config.level:
        labs = discover_labs()
        if run_config.lab_name not in labs:
            raise ValueError(
                f"Unknown lab '{run_config.lab_name}'. "
                f"Available: {sorted(labs.keys())}"
            )
        lab = LabDescription.from_yaml(labs[run_config.lab_name])
        user_prompt, response_model = build_prompt(
            lab,
            run_config.level,
            run_config.spec_type.value,
            structure=run_config.structure,
            use_gemini_models=is_gemini,
            min_objects=run_config.min_objects,
            min_clips=run_config.min_clips,
        )
        if save_prompts:
            prompt_file_rel = _save_prompt_artifact(
                run_config, user_prompt, overwrite=overwrite_prompts
            )
    else:
        response_model = get_response_model(run_config.spec_type, use_gemini_models=is_gemini)
        user_prompt = run_config.get_prompt()

    # Create instructor client (model is baked into the client via from_provider)
    client = create_instructor_client(model_config)

    # Set up tracking
    tracker = InstructorTracker(
        model=model_config.model_id,
        provider=model_config.provider.value,
        pricing_config=model_config.pricing_config,
    )
    tracked_client = tracker.wrap(client)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Model is set on the client; create() just needs response_model + messages
    create_kwargs = dict(
        response_model=response_model,
        messages=messages,
        max_retries=run_config.max_retries,
    )

    # Anthropic requires an explicit max_tokens
    if model_config.provider == Provider.ANTHROPIC:
        create_kwargs["max_tokens"] = 8192

    # Execute generation
    print(f"\n  Generating with {model_config.display_name}...")
    if run_config.lab_name and run_config.level:
        print(
            f"  Spec: {run_config.spec_type.value} | Lab: {run_config.lab_name} "
            f"| Level: {run_config.level.value} | Structure: {run_config.structure.value}"
        )
    else:
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
        "lab_name": run_config.lab_name,
        "level": run_config.level.value if run_config.level else None,
        "structure": (
            run_config.structure.value
            if run_config.lab_name and run_config.level else None
        ),
        "prompt_file": prompt_file_rel,
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

def _save_prompt_artifact(
    run_config: BenchmarkRunConfig,
    user_prompt: str,
    *,
    overwrite: bool = False,
) -> str:
    """Write the assembled prompt to Artifacts/Data/Benchmark/prompts/, deduped.

    The same prompt is shared across all models for a given
    (lab, level, spec, structure) tuple, so we write it once and let
    every run's metrics record reference the same file by relative path.

    Returns the prompt file path relative to PROJECT_ROOT.
    """
    prompts_dir = PROJECT_ROOT / run_config.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    parts = [run_config.lab_name, run_config.level.value, run_config.spec_type.value]
    if run_config.level != Level.L1:
        parts.append(run_config.structure.value)
    fname = "_".join(parts) + ".txt"

    path = prompts_dir / fname
    if overwrite or not path.exists():
        path.write_text(user_prompt, encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


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
    *,
    lab_name: Optional[str] = None,
    level: Optional[Level] = None,
    structure: Structure = Structure.SINGLE_MODULE,
    save_prompts: bool = True,
    overwrite_prompts: bool = False,
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
    if lab_name and level:
        print(f"Lab: {lab_name} | Level: {level.value} | Structure: {structure.value}")
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
                lab_name=lab_name,
                level=level,
                structure=structure,
                max_retries=max_retries,
                save_output=save_output,
            )

            try:
                record = run_single_benchmark(
                    model_config,
                    run_config,
                    save_prompts=save_prompts,
                    overwrite_prompts=overwrite_prompts,
                )
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
        choices=["json_lab", "arlem", "arlem_simple"],
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
    parser.add_argument(
        "--lab",
        type=str,
        default=None,
        help="topic_name of a lab YAML in Artifacts/Lab Descriptions/",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["L1", "L2", "L3", "L4"],
        default=None,
        help="Specificity level for the YAML-driven prompt (requires --lab)",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["single-module", "multi-module", "module-only"],
        default="single-module",
        help="Output structure for L2-L4 (ignored for L1)",
    )
    parser.add_argument(
        "--list-labs",
        action="store_true",
        help="List available lab YAMLs and exit",
    )
    parser.add_argument(
        "--no-save-prompts",
        action="store_true",
        help="Skip writing the prompt artifact alongside benchmark outputs",
    )
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Rewrite the prompt artifact file even if it already exists",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for model_id, config in MODELS.items():
            print(f"  {model_id:<25} {config.provider.value:<12} {config.display_name}")
        return

    if args.list_labs:
        labs = discover_labs()
        if not labs:
            print(f"\nNo lab YAMLs found in Artifacts/Lab Descriptions/")
        else:
            print("\nAvailable labs:")
            for topic_name, path in labs.items():
                print(f"  {topic_name:<40} {path.relative_to(PROJECT_ROOT)}")
        return

    if args.lab and not args.level:
        print("Error: --lab requires --level (L1, L2, L3, or L4).")
        sys.exit(1)
    if args.level and not args.lab:
        print("Error: --level requires --lab.")
        sys.exit(1)

    spec_type = SpecType(args.spec)
    save_output = not args.no_save
    lab_name = args.lab
    level = Level(args.level) if args.level else None
    structure = Structure(args.structure)
    save_prompts = not args.no_save_prompts

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
            lab_name=lab_name,
            level=level,
            structure=structure,
            save_prompts=save_prompts,
            overwrite_prompts=args.overwrite_prompts,
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
            lab_name=lab_name,
            level=level,
            structure=structure,
            save_prompts=save_prompts,
            overwrite_prompts=args.overwrite_prompts,
        )
    else:
        print("Error: Specify --model or --suite. Use --list-models or --list-labs to see options.")
        sys.exit(1)


if __name__ == "__main__":
    main()

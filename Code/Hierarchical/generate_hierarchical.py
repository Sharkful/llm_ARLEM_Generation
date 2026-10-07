"""
Hierarchical JSON Lab generation — CLI.

Generates a lab one module at a time (plan → per-module asset plan → module),
auto-approving every checkpoint. See HIERARCHICAL.md.

Usage:
    python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini \
        --lab apparent_retrograde_motion --level L4

    # several models / every lab, tighter module retry budget, no new components
    python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini claude-haiku-4.5 \
        --all-labs --level L3 --module-retries 2 --no-new-components

    # continue a failed or interrupted run from its first missing step
    python "Code/Hierarchical/generate_hierarchical.py" \
        --resume "Artifacts/Data/Hierarchical/Runs/<run name>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _paths  # noqa: F401
from _paths import PROJECT_ROOT

from benchmark_config import MODELS
from prompt_builder import Level, discover_labs

from pipeline import DEFAULT_OUTPUT_DIR, HierarchicalPipeline, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate JSON Labs hierarchically (plan → asset plan → module)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", nargs="+", help="Model ID(s) from benchmark_config.MODELS")
    p.add_argument("--lab", nargs="+", help="Lab topic_name(s) in Artifacts/Lab Descriptions/")
    p.add_argument("--all-labs", action="store_true", help="Run every discovered lab YAML")
    p.add_argument("--level", choices=["L3", "L4"], default="L4",
                   help="Input specificity for the lab context (default: L4)")
    p.add_argument("--plan-retries", type=int, default=3,
                   help="instructor max_retries for the lab-plan step (default: 3)")
    p.add_argument("--asset-plan-retries", type=int, default=3,
                   help="instructor max_retries for each asset-plan step (default: 3)")
    p.add_argument("--module-retries", type=int, default=3,
                   help="instructor max_retries for each module step (default: 3)")
    p.add_argument("--step-attempts", type=int, default=2,
                   help="Fresh regenerations of a step after it fails terminally (default: 2)")
    p.add_argument("--max-revisions", type=int, default=3,
                   help="Reviewer revise rounds per step before accepting (default: 3)")
    new = p.add_mutually_exclusive_group()
    new.add_argument("--allow-new-components", dest="allow_new_components",
                     action="store_true", default=True,
                     help="Let asset plans propose new components (default)")
    new.add_argument("--no-new-components", dest="allow_new_components", action="store_false",
                     help="Restrict asset plans to the component catalog")
    p.add_argument("--min-objects", type=int, default=4, help="Lab-wide object minimum (default: 4)")
    p.add_argument("--min-clips", type=int, default=5, help="Lab-wide clip minimum (default: 5)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output root, relative to the project (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--resume", type=Path, help="Run directory to continue")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-labs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        for key, cfg in MODELS.items():
            print(f"  {key:<25} {cfg.provider.value:<10} {cfg.display_name}")
        return
    if args.list_labs:
        for name, path in discover_labs().items():
            print(f"  {name:<40} {path.relative_to(PROJECT_ROOT)}")
        return

    if args.resume:
        run_dir = args.resume if args.resume.is_absolute() else PROJECT_ROOT / args.resume
        record = HierarchicalPipeline.resume(run_dir).run()
        sys.exit(0 if record["success"] else 1)

    if not args.model:
        sys.exit("Error: --model is required (see --list-models).")
    labs = sorted(discover_labs()) if args.all_labs else args.lab
    if not labs:
        sys.exit("Error: pass --lab or --all-labs (see --list-labs).")

    config = PipelineConfig(
        level=Level(args.level),
        plan_retries=args.plan_retries,
        asset_plan_retries=args.asset_plan_retries,
        module_retries=args.module_retries,
        step_attempts=args.step_attempts,
        max_revisions=args.max_revisions,
        allow_new_components=args.allow_new_components,
        min_objects=args.min_objects,
        min_clips=args.min_clips,
    )

    results = []
    for lab_name in labs:
        for model_key in args.model:
            try:
                pipe = HierarchicalPipeline(model_key, lab_name, config, output_dir=args.output_dir)
                results.append(pipe.run())
            except Exception as e:  # noqa: BLE001 — keep the sweep going
                print(f"  FATAL ({model_key}, {lab_name}): {type(e).__name__}: {e}")
                results.append({"model": model_key, "lab_name": lab_name, "success": False})

    if len(results) > 1:
        print("\nSUMMARY")
        for r in results:
            t = r.get("totals", {})
            status = "OK" if r["success"] else f"FAIL {r.get('failed_step') or ''}".strip()
            print(f"  {r.get('display_name', r['model']):<22} {r['lab_name']:<36} {status:<18} "
                  f"{t.get('total_tokens', 0):>9,} tok  ${t.get('cost_usd', 0):.4f}")


if __name__ == "__main__":
    main()

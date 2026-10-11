"""
Hierarchical JSON Lab generation — CLI.

Generates a lab one module at a time (plan → per-module asset plan → module),
auto-approving every checkpoint. See HIERARCHICAL.md.

Usage:
    python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini \
        --lab apparent_retrograde_motion --level L4

    # several models / every lab, into a named run group, tighter module retry
    # budget, no new components
    python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini claude-haiku-4.5 \
        --all-labs --level L3 --run-group pilot --module-retries 2 --no-new-components

    # continue a failed or interrupted run from its first missing step
    # (retry budgets may be changed on resume)
    python "Code/Hierarchical/generate_hierarchical.py" \
        --resume "Artifacts/Data/Hierarchical/pilot/Runs/<run name>" --module-retries 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _paths  # noqa: F401
from _paths import PROJECT_ROOT

from benchmark import require_environment
from benchmark_config import MODELS
from prompt_builder import Level, discover_labs

from pipeline import HierarchicalPipeline, PipelineConfig, output_dir_for

# Settings that may be changed when resuming. The rest (level, minimums, scene
# cap, the new-component switch) shape what earlier steps already produced, so a resumed
# run keeps the values it started with.
RESUMABLE_SETTINGS = (
    "plan_retries", "asset_plan_retries", "module_retries", "step_attempts", "max_revisions",
)
FIXED_SETTINGS = ("level", "allow_new_components", "min_objects", "min_clips", "max_scenes")


def parse_args():
    defaults = PipelineConfig()
    p = argparse.ArgumentParser(
        description="Generate JSON Labs hierarchically (plan → asset plan → module)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", nargs="+", help="Model ID(s) from benchmark_config.MODELS")
    p.add_argument("--lab", nargs="+", help="Lab topic_name(s) in Artifacts/Lab Descriptions/")
    p.add_argument("--all-labs", action="store_true", help="Run every discovered lab YAML")
    p.add_argument("--run-group",
                   help="Folder under Artifacts/Data/Hierarchical/ for these runs, e.g. 'testing' "
                        "or 'pilot_oct' (letters, digits, '_', '-'). Default: the "
                        "Hierarchical/ folder itself")
    # Settings default to None so --resume can tell which ones were given; unset
    # ones fall back to the PipelineConfig defaults shown in the help.
    p.add_argument("--level", choices=["L3", "L4"],
                   help=f"Input specificity for the lab context (default: {defaults.level.value})")
    p.add_argument("--plan-retries", type=int,
                   help=f"instructor max_retries for the lab-plan step (default: {defaults.plan_retries})")
    p.add_argument("--asset-plan-retries", type=int,
                   help=f"instructor max_retries for each asset-plan step (default: {defaults.asset_plan_retries})")
    p.add_argument("--module-retries", type=int,
                   help=f"instructor max_retries for each module step (default: {defaults.module_retries})")
    p.add_argument("--step-attempts", type=int,
                   help="Fresh regenerations of a step after it fails terminally "
                        f"(default: {defaults.step_attempts})")
    p.add_argument("--max-revisions", type=int,
                   help=f"Reviewer revise rounds per step before accepting (default: {defaults.max_revisions})")
    new = p.add_mutually_exclusive_group()
    new.add_argument("--allow-new-components", dest="allow_new_components",
                     action="store_true", default=None,
                     help="Let asset plans propose new components (default)")
    new.add_argument("--no-new-components", dest="allow_new_components", action="store_false",
                     help="Restrict asset plans to the component catalog")
    p.add_argument("--min-objects", type=int,
                   help=f"Minimum scene objects per module (default: {defaults.min_objects})")
    p.add_argument("--min-clips", type=int,
                   help=f"Minimum clips per module (default: {defaults.min_clips})")
    p.add_argument("--max-scenes", type=int,
                   help=f"Maximum scenes (= modules) in the lab plan (default: {defaults.max_scenes})")
    p.add_argument("--no-save", action="store_true",
                   help="Don't write any files (run dir, outputs, metrics); runs can't be resumed")
    p.add_argument("--resume", type=Path, help="Run directory to continue")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--list-labs", action="store_true")
    return p.parse_args()


def _given(args, names) -> dict:
    """The settings among ``names`` that were passed on the command line."""
    return {n: getattr(args, n) for n in names if getattr(args, n) is not None}


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
        fixed = _given(args, FIXED_SETTINGS)
        if fixed or args.model or args.lab or args.all_labs or args.run_group or args.no_save:
            sys.exit(
                "Error: --resume continues a run with its original model, lab, run group "
                f"and settings; only these may change: {', '.join(RESUMABLE_SETTINGS)}."
            )
        run_dir = args.resume if args.resume.is_absolute() else PROJECT_ROOT / args.resume
        meta = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if meta["model_key"] in MODELS:
            require_environment([MODELS[meta["model_key"]]])
        pipe = HierarchicalPipeline.resume(
            run_dir, config_overrides=_given(args, RESUMABLE_SETTINGS)
        )
        sys.exit(0 if pipe.run()["success"] else 1)

    if not args.model:
        sys.exit("Error: --model is required (see --list-models).")
    labs = sorted(discover_labs()) if args.all_labs else args.lab
    if not labs:
        sys.exit("Error: pass --lab or --all-labs (see --list-labs).")

    require_environment([MODELS[m] for m in args.model if m in MODELS])

    try:
        output_dir_for(args.run_group)  # validate before any run starts
    except ValueError as e:
        sys.exit(f"Error: {e}")

    settings = _given(args, RESUMABLE_SETTINGS + FIXED_SETTINGS)
    if "level" in settings:
        settings["level"] = Level(settings["level"])
    config = PipelineConfig(**settings)

    results = []
    for lab_name in labs:
        for model_key in args.model:
            try:
                pipe = HierarchicalPipeline(
                    model_key, lab_name, config,
                    run_group=args.run_group, save=not args.no_save,
                )
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

    failed = sum(not r["success"] for r in results)
    if failed:
        print(f"\n{failed} of {len(results)} run(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

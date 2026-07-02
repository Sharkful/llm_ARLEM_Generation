"""Asset Pipeline CLI.

Usage (run from the asset_pipeline/ directory, or `python -m asset_pipeline.cli`
from the repo root once __main__ wiring is added in a later stage):
    python cli.py catalog list
    python cli.py catalog seed
    python cli.py spec "a small gray moon with visible crater texture, about 15cm across"
    python cli.py spec "a small gray moon..." --out spec_moon.json --provider openai
    python cli.py resolve "a small gray moon with visible crater texture"
    python cli.py resolve --spec-file spec_moon.json
    python cli.py generate-parametric bracket_01 "an L-shaped mounting bracket, 5cm wide"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from models.spec_models import AssetSpec
from pipeline.catalog_writer import load_catalog
from pipeline.classifier import ClassificationError
from pipeline.openscad_generator import OpenSCADGenerationError, generate_parametric_asset
from pipeline.resolver import resolve
from pipeline.seed_catalog import main as seed_catalog_main
from pipeline.spec_parser import SpecParsingError, parse_description


def cmd_catalog_list(args: argparse.Namespace) -> int:
    entries = load_catalog()
    if not entries:
        print(f"Catalog is empty ({config.CATALOG_PATH}). Run `catalog seed` first.")
        return 0
    print(f"{'asset_id':<20} {'class':<11} {'review':<7} {'bounds_m':<20} tags")
    print("-" * 90)
    for e in entries:
        bounds = "x".join(f"{v:g}" for v in e.canonical_bounds_m)
        tags = ", ".join(e.tags[:5])
        print(f"{e.asset_id:<20} {e.asset_class:<11} {e.review_level:<7} {bounds:<20} {tags}")
    print(f"\n{len(entries)} asset(s) in {config.CATALOG_PATH}")
    return 0


def cmd_catalog_seed(args: argparse.Namespace) -> int:
    seed_catalog_main()
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    try:
        spec, log_entry = parse_description(
            args.description, provider=args.provider, model=args.model
        )
    except SpecParsingError as exc:
        print(f"ERROR: spec parsing failed after retries: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        # Raised by llm/client_factory.py before any call is attempted,
        # e.g. a missing API key -- report it cleanly rather than a traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    spec_json = json.dumps(spec.model_dump(mode="json"), indent=2)
    print(spec_json)
    print(
        f"\n[{log_entry.provider}/{log_entry.model}] "
        f"{log_entry.input_tokens}in/{log_entry.output_tokens}out tokens, "
        f"{log_entry.duration_seconds}s",
        file=sys.stderr,
    )

    if args.out:
        Path(args.out).write_text(spec_json, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    if args.spec_file:
        spec = AssetSpec.model_validate(json.loads(Path(args.spec_file).read_text(encoding="utf-8")))
    elif args.description:
        try:
            spec, _ = parse_description(args.description, provider=args.provider, model=args.model)
        except SpecParsingError as exc:
            print(f"ERROR: spec parsing failed after retries: {exc}", file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print("ERROR: provide either a description or --spec-file", file=sys.stderr)
        return 1

    catalog = load_catalog()
    try:
        record, log_entries = resolve(spec, catalog)
    except ClassificationError as exc:
        print(f"ERROR: classification failed after retries: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(record.model_dump(mode="json"), indent=2))
    for entry in log_entries:
        print(
            f"[{entry.provider}/{entry.model}] {entry.purpose} "
            f"{entry.input_tokens}in/{entry.output_tokens}out tokens, {entry.duration_seconds}s",
            file=sys.stderr,
        )
    if record.requires_author_review:
        print(f"\nFLAGGED FOR REVIEW: {record.review_reason}", file=sys.stderr)
    return 0


def cmd_generate_parametric(args: argparse.Namespace) -> int:
    if args.spec_file:
        spec = AssetSpec.model_validate(json.loads(Path(args.spec_file).read_text(encoding="utf-8")))
    else:
        spec = AssetSpec(object_id=args.asset_id, description=args.description)

    try:
        result, log_entries = generate_parametric_asset(spec, args.asset_id)
    except OpenSCADGenerationError as exc:
        print(f"ERROR: OpenSCAD plan generation failed after retries: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    for entry in log_entries:
        print(
            f"[{entry.provider}/{entry.model}] {entry.purpose} "
            f"{entry.input_tokens}in/{entry.output_tokens}out tokens, {entry.duration_seconds}s",
            file=sys.stderr,
        )
    if not result.success:
        print(f"\nFAILED after {len(result.repair_attempts)} attempt(s): {result.error_message}", file=sys.stderr)
        return 1
    if result.bounds_diverge:
        print(
            f"\nWARNING: actual bounds {result.actual_bounds_m} diverge from expected "
            f"{result.expected_bounds_m} by more than the tolerance -- requires author review.",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset_pipeline",
        description="Offline AR asset acquisition pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="Inspect or seed the asset catalog")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_sub.add_parser("list", help="List all catalog entries").set_defaults(func=cmd_catalog_list)
    catalog_sub.add_parser(
        "seed", help="(Re)write the seed catalog of core primitives"
    ).set_defaults(func=cmd_catalog_seed)

    spec = sub.add_parser("spec", help="Parse a free-text object description into an AssetSpec")
    spec.add_argument("description", help="Free-text sentence/paragraph describing the object")
    spec.add_argument("--out", metavar="PATH", help="Write the resulting AssetSpec JSON to a file")
    spec.add_argument(
        "--provider", choices=["anthropic", "openai", "google"], default=None,
        help="Override the default LLM provider for this call",
    )
    spec.add_argument("--model", default=None, help="Override the default model for this call")
    spec.set_defaults(func=cmd_spec)

    resolve_parser = sub.add_parser(
        "resolve", help="Classify and resolve an AssetSpec against the catalog"
    )
    resolve_parser.add_argument(
        "description", nargs="?", default=None,
        help="Free-text description (parsed into an AssetSpec first via Stage 1)",
    )
    resolve_parser.add_argument(
        "--spec-file", metavar="PATH",
        help="Use an already-parsed AssetSpec JSON file instead of a raw description",
    )
    resolve_parser.add_argument(
        "--provider", choices=["anthropic", "openai", "google"], default=None,
        help="Override the default LLM provider for this call",
    )
    resolve_parser.add_argument("--model", default=None, help="Override the default model for this call")
    resolve_parser.set_defaults(func=cmd_resolve)

    gen_parser = sub.add_parser(
        "generate-parametric", help="Generate an OpenSCAD-based parametric asset from a description"
    )
    gen_parser.add_argument("asset_id", help="asset_id to generate under library/generated/<asset_id>/")
    gen_parser.add_argument(
        "description", nargs="?", default=None, help="Free-text description of the object to generate"
    )
    gen_parser.add_argument(
        "--spec-file", metavar="PATH", help="Use an AssetSpec JSON file instead of a raw description"
    )
    gen_parser.set_defaults(func=cmd_generate_parametric)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

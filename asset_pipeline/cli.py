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
    python cli.py normalize-mesh bracket_01 library/generated/bracket_01/source.stl --format stl --size 0.05
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
from pipeline.material_generator import MaterialGenerationError, generate_material
from pipeline.mesh_processor import MeshProcessingError, normalize_mesh, save_mesh_meta
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


def cmd_normalize_mesh(args: argparse.Namespace) -> int:
    glb_path = config.LIBRARY_DIR / "generated" / args.asset_id / "model.glb"
    try:
        result = normalize_mesh(
            asset_id=args.asset_id,
            source_path=Path(args.source_path),
            source_format=args.format,
            glb_path=glb_path,
            target_size_m=args.size,
            pivot=args.pivot,
        )
    except MeshProcessingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    if not result.success:
        print(f"\nFAILED: {result.error_message}", file=sys.stderr)
        return 1

    save_mesh_meta(args.asset_id, result)
    print(f"\nWrote {glb_path}", file=sys.stderr)
    if result.decimated:
        print(
            f"NOTE: mesh decimated {result.triangle_count_before} -> "
            f"{result.triangle_count_after} triangles (exceeded MAX_TRIANGLE_COUNT).",
            file=sys.stderr,
        )
    return 0


def cmd_generate_material(args: argparse.Namespace) -> int:
    try:
        result, plan, log_entry = generate_material(
            args.description,
            material_id=args.id,
            provider=args.provider,
            model=args.model,
            force=args.force,
        )
    except MaterialGenerationError as exc:
        print(f"ERROR: material generation failed after retries: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    print(
        f"[{log_entry.provider}/{log_entry.model}] {log_entry.purpose} "
        f"{log_entry.input_tokens}in/{log_entry.output_tokens}out tokens, "
        f"{log_entry.duration_seconds}s",
        file=sys.stderr,
    )
    if not result.success:
        print(f"\nFAILED: {result.error_message}", file=sys.stderr)
        return 1
    if plan.procedural_texture:
        print(
            f"Synthesized {plan.procedural_texture.kind!r} texture -> {result.texture_path}",
            file=sys.stderr,
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report external tool and API key status on this machine.

    Detection paths can lie -- e.g. an MS Store Blender exists on disk but
    Windows denies executing it directly -- so each detected binary is
    actually run with --version rather than just stat'd.
    """
    import subprocess

    ok = True

    def check_tool(name: str, path: str | None, env_var: str) -> bool:
        if not path:
            print(f"  [MISSING] {name}: not found. Install it or set {env_var} in .env")
            return False
        try:
            result = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=30
            )
            version = (result.stdout or result.stderr).strip().splitlines()[0]
            if result.returncode == 0:
                print(f"  [OK]      {name}: {path} ({version})")
                return True
            print(f"  [BROKEN]  {name}: {path} exits {result.returncode}: {version}")
        except Exception as exc:
            print(f"  [BROKEN]  {name}: {path} found but won't run: {exc}")
        return False

    print("External tools:")
    ok &= check_tool("OpenSCAD", config.OPENSCAD_BIN, "ASSET_PIPELINE_OPENSCAD_BIN")
    ok &= check_tool("Blender", config.BLENDER_BIN, "ASSET_PIPELINE_BLENDER_BIN")

    print("\nLLM API keys (only the provider you use needs one):")
    keys = {
        "anthropic": config.ANTHROPIC_API_KEY,
        "openai": config.OPENAI_API_KEY,
        "google": config.GEMINI_API_KEY,
    }
    any_key = False
    for provider, key in keys.items():
        marker = "set" if key else "not set"
        default = "  <- default provider" if provider == config.DEFAULT_LLM_PROVIDER else ""
        print(f"  [{'OK' if key else '--'}]      {provider}: {marker}{default}")
        any_key = any_key or bool(key)
    if not keys.get(config.DEFAULT_LLM_PROVIDER):
        print(
            f"  WARNING: default provider '{config.DEFAULT_LLM_PROVIDER}' has no key -- "
            "LLM stages (spec/resolve/generate-parametric) will fail until it is set in .env"
        )

    print(f"\nCatalog: {config.CATALOG_PATH}"
          f" ({'exists' if config.CATALOG_PATH.exists() else 'missing -- run `catalog seed`'})")
    print(f"Library: {config.LIBRARY_DIR}")

    return 0 if ok else 1


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

    norm_parser = sub.add_parser(
        "normalize-mesh", help="Normalize a source mesh (STL/OBJ/FBX/GLB) into a scaled, pivoted GLB"
    )
    norm_parser.add_argument("asset_id", help="asset_id to write under library/generated/<asset_id>/model.glb")
    norm_parser.add_argument("source_path", help="Path to the source mesh file")
    norm_parser.add_argument(
        "--format", choices=["stl", "obj", "fbx", "glb"], required=True, help="Source mesh format"
    )
    norm_parser.add_argument(
        "--size", type=float, required=True,
        help="Target size in meters for the largest original dimension",
    )
    norm_parser.add_argument(
        "--pivot", choices=["center", "base_center"], default="center",
        help="Pivot convention: center, or base_center for objects that stand on a surface",
    )
    norm_parser.set_defaults(func=cmd_normalize_mesh)

    mat_parser = sub.add_parser(
        "generate-material",
        help="Generate a MaterialDef JSON (+ optional procedural texture) from a description",
    )
    mat_parser.add_argument("description", help="Free-text material/surface description")
    mat_parser.add_argument(
        "--id", default=None, help="material_id to use (overrides the LLM's proposed id)"
    )
    mat_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing material with the same id"
    )
    mat_parser.add_argument(
        "--provider", choices=["anthropic", "openai", "google"], default=None,
        help="Override the default LLM provider for this call",
    )
    mat_parser.add_argument("--model", default=None, help="Override the default model for this call")
    mat_parser.set_defaults(func=cmd_generate_material)

    doctor_parser = sub.add_parser(
        "doctor",
        help="Check external tool binaries (OpenSCAD, Blender) and API keys on this machine",
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

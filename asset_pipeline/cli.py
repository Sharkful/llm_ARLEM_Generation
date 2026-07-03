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
    python cli.py generate-material "rough red rust with visible texture"
    python cli.py intake wooden_stool
    python cli.py validate
    python cli.py doctor
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
from pipeline.external_intake import IntakeError, intake_asset
from pipeline.preview_renderer import (
    PreviewError,
    preview_path,
    render_previews,
    write_contact_sheet,
)
from pipeline.polyhaven import (
    PolyHavenSearchError,
    fetch_to_intake,
    list_model_files,
    search_models,
)
from pipeline.material_generator import MaterialGenerationError, generate_material
from pipeline.mesh_processor import MeshProcessingError, normalize_mesh, save_mesh_meta
from pipeline.openscad_generator import OpenSCADGenerationError, generate_parametric_asset
from pipeline.resolver import resolve
from pipeline.validator import format_json_output, format_screen_text, validate_library
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


def cmd_intake(args: argparse.Namespace) -> int:
    try:
        result = intake_asset(args.asset_id)
    except IntakeError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # catalog_writer uniqueness violation (raced/duplicate id)
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(mode="json"), indent=2))
    if not result.success:
        print(f"\nFAILED: {result.error_message}", file=sys.stderr)
        return 1
    entry = result.catalog_entry
    print(
        f"\nCataloged {entry.asset_id!r} (review_level {entry.review_level}, "
        f"license {entry.provenance.license}, bounds "
        f"{'x'.join(f'{v:g}' for v in entry.canonical_bounds_m)} m)",
        file=sys.stderr,
    )
    return 0


def cmd_search_external(args: argparse.Namespace) -> int:
    try:
        results = search_models(args.query, limit=args.limit)
    except PolyHavenSearchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not results:
        print(f"No Poly Haven models matched {args.query!r}.")
        return 0

    print(f"{'id':<26} {'name':<30} {'license':<8} {'downloads':<10} tags")
    print("-" * 100)
    for r in results:
        tags = ", ".join([*r.categories, *r.tags][:5])
        print(f"{r.asset_id:<26} {r.name[:29]:<30} {r.license:<8} {r.download_count:<10} {tags}")
        print(f"{'':<26} thumbnail: {r.thumbnail_url}")
    print(
        f"\n{len(results)} result(s). Inspect a thumbnail, then download YOUR pick with:\n"
        "  python cli.py fetch-external <id> [--size <meters>]\n"
        "Nothing is downloaded until you run that command."
    )
    return 0


def cmd_fetch_external(args: argparse.Namespace) -> int:
    try:
        if args.list_files:
            files = list_model_files(args.polyhaven_id)
            print(f"{'format':<10} {'resolution':<11} {'size_mb':<9} includes")
            print("-" * 45)
            for f in files:
                print(
                    f"{f.file_format:<10} {f.resolution:<11} "
                    f"{f.size_bytes / 1e6:<9.1f} {f.include_count}"
                )
            return 0
        source_path, downloaded = fetch_to_intake(
            args.polyhaven_id,
            intake_id=args.intake_id,
            file_format=args.format,
            resolution=args.resolution,
            target_size_m=args.size,
        )
    except PolyHavenSearchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    intake_id = args.intake_id or args.polyhaven_id
    total_mb = sum(p.stat().st_size for p in downloaded) / 1e6
    print(f"Downloaded {len(downloaded)} file(s), {total_mb:.1f} MB -> {source_path.parent}")
    print(f"Pre-filled {source_path}")
    if args.size:
        print(f"\nNext: python cli.py intake {intake_id}")
    else:
        print(
            f"\nNext: edit {source_path.name} and set target_size_m "
            f"(real-world largest dimension in meters), then run:\n"
            f"  python cli.py intake {intake_id}"
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


def cmd_preview(args: argparse.Namespace) -> int:
    if not args.all and not args.asset_id:
        print("ERROR: give an asset_id or --all", file=sys.stderr)
        return 1
    entries = load_catalog()
    if not args.all:
        entries = [e for e in entries if e.asset_id == args.asset_id]
        if not entries:
            print(f"ERROR: asset_id {args.asset_id!r} not found in the catalog.", file=sys.stderr)
            return 1

    try:
        written, skipped = render_previews(entries, force=args.force)
    except PreviewError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(f"  rendered {path.name}")
    if skipped:
        print(f"  skipped {len(skipped)} existing preview(s) (use --force to redo)")
    if args.all:
        sheet = write_contact_sheet(load_catalog())
        print(f"\nContact sheet: {sheet}")
    elif written:
        print(f"\nPreview: {preview_path(args.asset_id)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_library(
        records_path=Path(args.records) if args.records else None
    )
    if args.json:
        print(format_json_output(report))
    else:
        print(format_screen_text(report, use_color=not args.no_color))
    return 0 if report.passed else 1


def cmd_review(args: argparse.Namespace) -> int:
    from webapp.app import main as review_main

    print(f"Starting review app on http://127.0.0.1:{args.port}/ (Ctrl+C to stop)")
    review_main(port=args.port, open_browser=not args.no_browser)
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

    intake_parser = sub.add_parser(
        "intake",
        help="Normalize and catalog a manually-provided model file from intake/<asset_id>/",
    )
    intake_parser.add_argument(
        "asset_id", help="Folder name under intake/ holding the model file + source.json"
    )
    intake_parser.set_defaults(func=cmd_intake)

    search_ext = sub.add_parser(
        "search-external",
        help="Search Poly Haven (CC0) models -- read-only, downloads nothing",
    )
    search_ext.add_argument("query", help="Free-text search, e.g. 'wooden table'")
    search_ext.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    search_ext.set_defaults(func=cmd_search_external)

    fetch_ext = sub.add_parser(
        "fetch-external",
        help="Download ONE human-chosen Poly Haven model into intake/<id>/ (your explicit pick is the approval)",
    )
    fetch_ext.add_argument("polyhaven_id", help="Poly Haven asset id from search-external")
    fetch_ext.add_argument(
        "--intake-id", default=None, help="Intake folder/asset id (default: the Poly Haven id)"
    )
    fetch_ext.add_argument("--format", default="gltf", help="File format (default gltf)")
    fetch_ext.add_argument("--resolution", default="1k", help="Texture resolution (default 1k)")
    fetch_ext.add_argument(
        "--size", type=float, default=None,
        help="Real-world largest dimension in meters (pre-fills source.json target_size_m)",
    )
    fetch_ext.add_argument(
        "--list-files", action="store_true",
        help="Only list available format/resolution downloads for this id",
    )
    fetch_ext.set_defaults(func=cmd_fetch_external)

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

    preview_parser = sub.add_parser(
        "preview",
        help="Render preview thumbnail(s) into library/previews/ (+ contact sheet with --all)",
    )
    preview_parser.add_argument(
        "asset_id", nargs="?", default=None, help="Single asset to preview"
    )
    preview_parser.add_argument(
        "--all", action="store_true", help="Preview every catalog asset and write the contact sheet"
    )
    preview_parser.add_argument(
        "--force", action="store_true", help="Re-render even if a preview PNG already exists"
    )
    preview_parser.set_defaults(func=cmd_preview)

    validate_parser = sub.add_parser(
        "validate",
        help="Validate catalog + library (schema, completeness, geometry, licensing); exit 0 only if error-free",
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Machine-readable JSON report instead of screen text"
    )
    validate_parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI colors in screen output"
    )
    validate_parser.add_argument(
        "--records", metavar="PATH", default=None,
        help="Also check a ResolutionRecord JSON file (or list) for unresolved specs (req. doc 12.5)",
    )
    validate_parser.set_defaults(func=cmd_validate)

    review_parser = sub.add_parser(
        "review",
        help="Start the local Flask review/creation app (Library / Create / Worklist)",
    )
    review_parser.add_argument("--port", type=int, default=5173, help="Port (default 5173)")
    review_parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open the browser"
    )
    review_parser.set_defaults(func=cmd_review)

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

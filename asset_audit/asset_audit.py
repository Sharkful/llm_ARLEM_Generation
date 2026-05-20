"""
asset_audit.py — ARLEM Lab Asset Auditor

Parses an ARLEM v2.0 lab JSON and identifies every asset it references
(prefabs, textures, audio clips, behavior scripts), then optionally checks
whether each asset exists on disk.

Usage (from llm_ARLEM_Generation/):
    python asset_audit/asset_audit.py <lab.json> --dedup
    python asset_audit/asset_audit.py <lab.json> --audit
    python asset_audit/asset_audit.py <lab.json> --missing --output json
    python asset_audit/asset_audit.py <lab.json> --module "Synchronous Rotation"
    python asset_audit/asset_audit.py <lab.json> --clip "Synchronous Rotation" clip3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Import Pydantic models from Code/Tools ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "Code" / "Tools"))
try:
    from pydantic_json_lab_claude import (
        CheckAngleComponent,
        Clip,
        DemoModule,
        Lab,
        ObjectChange,
        SceneObject,
    )
    from pydantic import ValidationError
    _PYDANTIC_AVAILABLE = True
except ImportError as _import_err:
    _PYDANTIC_AVAILABLE = False
    _import_err_msg = str(_import_err)


# ── Constants ─────────────────────────────────────────────────────────────────

BUILTIN_COMPONENTS = {"textMeshPro", "rigidBody", "pointerReceiver"}

FALLBACK_TABLE = {
    "texture": "Default gray material - object renders without texture",
    "prefab":  "No prefab: object spawns invisible (Unity) / placeholder geometry (preview)",
    "audio":   "Clip plays silently - no audio heard",
    "script":  "Component not attached - object is static, behavior is skipped",
}

ASSET_TYPE_ORDER = ["prefab", "texture", "audio", "script"]

# ANSI color codes (used only when stdout is a TTY)
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class AssetRef:
    name: str
    asset_type: str          # "texture" | "prefab" | "audio" | "script"
    source: str              # human-readable origin
    found: Optional[bool] = None   # None = not checked; True/False = audit result
    resolved_path: Optional[str] = None
    fallback: Optional[str] = None


# ── Asset Extraction ──────────────────────────────────────────────────────────
# These functions accept either Pydantic model instances or raw dicts,
# using _get() to handle both access patterns transparently.

def _get(obj, key, default=None):
    """Get a field from either a Pydantic model or a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _comp_refs(components, source_prefix: str) -> list[AssetRef]:
    refs = []
    if not components:
        return refs
    for comp in components:
        ct = _get(comp, "componentType", "")
        if ct and ct not in BUILTIN_COMPONENTS:
            refs.append(AssetRef(name=ct, asset_type="script",
                                 source=f"{source_prefix} / script:{ct}"))
        # CheckAngleComponent.audioClipSuccess
        if ct == "checkAngle":
            audio_success = _get(comp, "audioClipSuccess")
            if audio_success:
                refs.append(AssetRef(name=audio_success, asset_type="audio",
                                     source=f"{source_prefix} / checkAngle.audioClipSuccess"))
    return refs


def extract_from_object(obj, module_name: str) -> list[AssetRef]:
    name   = _get(obj, "name", "unknown")
    prefab = _get(obj, "prefab", "")
    prefix = f"Module:{module_name} / Object:{name}"
    refs = []
    if prefab:
        refs.append(AssetRef(name=prefab, asset_type="prefab", source=f"{prefix} / prefab"))
    texture = _get(obj, "texture")
    if texture:
        refs.append(AssetRef(name=texture, asset_type="texture", source=f"{prefix} / texture"))
    refs.extend(_comp_refs(_get(obj, "components"), prefix))
    return refs


def extract_from_clip(clip, module_name: str) -> list[AssetRef]:
    clip_name = _get(clip, "clipName", "unknown")
    prefix = f"Module:{module_name} / Clip:{clip_name}"
    refs = []
    audio = _get(clip, "audioClip")
    if audio:
        refs.append(AssetRef(name=audio, asset_type="audio",
                             source=f"{prefix} / audioClip"))
    changes = _get(clip, "changes") or []
    for change in changes:
        target = _get(change, "target", "unknown")
        refs.extend(_comp_refs(_get(change, "components"),
                               f"{prefix} / change:{target}"))
    return refs


def extract_from_module(module) -> list[AssetRef]:
    mod_name = _get(module, "moduleName", "unknown")
    prefab   = _get(module, "prefab", "")
    refs = []
    if prefab:
        refs.append(AssetRef(name=prefab, asset_type="prefab",
                             source=f"Module:{mod_name} / module-prefab"))
    for obj in (_get(module, "objects") or []):
        refs.extend(extract_from_object(obj, mod_name))
    for clip in (_get(module, "clips") or []):
        refs.extend(extract_from_clip(clip, mod_name))
    return refs


def extract_from_lab(lab) -> list[AssetRef]:
    refs = []
    modules = _get(lab, "modules") or []
    for module in modules:
        refs.extend(extract_from_module(module))
    return refs


def dedup(refs: list[AssetRef]) -> list[AssetRef]:
    seen: set[tuple] = set()
    result = []
    for ref in refs:
        key = (ref.name, ref.asset_type)
        if key not in seen:
            seen.add(key)
            result.append(ref)
    return result


# ── Scope Filtering ───────────────────────────────────────────────────────────

def refs_for_module(lab, module_name: str) -> list[AssetRef]:
    idx = _find_module_idx(lab, module_name)
    modules = _get(lab, "modules") or []
    refs = []
    for module in modules[:idx + 1]:
        refs.extend(extract_from_module(module))
    return dedup(refs)


def refs_for_clip(lab, module_name: str, clip_name: str) -> list[AssetRef]:
    idx = _find_module_idx(lab, module_name)
    modules = _get(lab, "modules") or []
    refs = []
    # All object assets from all modules up through this one
    for module in modules[:idx + 1]:
        mod_name = _get(module, "moduleName", "unknown")
        prefab   = _get(module, "prefab", "")
        if prefab:
            refs.append(AssetRef(name=prefab, asset_type="prefab",
                                 source=f"Module:{mod_name} / module-prefab"))
        for obj in (_get(module, "objects") or []):
            refs.extend(extract_from_object(obj, mod_name))

    # Clip assets only from the named module, up through the named clip
    target_module = modules[idx]
    clip_idx = _find_clip_idx(target_module, clip_name)
    for clip in (_get(target_module, "clips") or [])[:clip_idx + 1]:
        refs.extend(extract_from_clip(clip, module_name))

    return dedup(refs)


def _find_module_idx(lab, module_name: str) -> int:
    modules = _get(lab, "modules") or []
    for i, m in enumerate(modules):
        if _get(m, "moduleName") == module_name:
            return i
    available = [_get(m, "moduleName", "") for m in modules]
    _die(f"Module '{module_name}' not found.\nAvailable modules:\n" +
         "\n".join(f"  - {n}" for n in available))


def _find_clip_idx(module, clip_name: str) -> int:
    mod_name = _get(module, "moduleName", "unknown")
    clips = _get(module, "clips") or []
    for i, c in enumerate(clips):
        if _get(c, "clipName") == clip_name:
            return i
    available = [_get(c, "clipName", "") for c in clips]
    _die(f"Clip '{clip_name}' not found in module '{mod_name}'.\n"
         f"Available clips:\n" + "\n".join(f"  - {n}" for n in available))


# ── Filesystem Audit ──────────────────────────────────────────────────────────

class AssetScanner:
    def __init__(self, asset_root: Path):
        self.root = asset_root
        self.textures:  set[str] = set()
        self.prefabs:   set[str] = set()
        self.models:    set[str] = set()
        self.fbx:       set[str] = set()
        self.scripts:   set[str] = set()
        self.audio:     set[str] = set()
        self.audio_dir_exists = False

        self._scan_dir("textures",      self.textures,  {".jpg", ".png"})
        self._scan_dir("prefabs",       self.prefabs,   {".prefab"})
        self._scan_dir("models",        self.models,    {".glb"})
        self._scan_dir("fbx",           self.fbx,       {".fbx"})
        self._scan_dir("object scripts",self.scripts,   {".cs"})

        # Audio: check common locations
        for audio_subdir in ("audio", "Audio", "Sounds", "sounds"):
            audio_path = asset_root / audio_subdir
            if audio_path.is_dir():
                self.audio_dir_exists = True
                for f in audio_path.iterdir():
                    if f.suffix.lower() in {".mp3", ".wav", ".ogg", ".aiff", ".aif"}:
                        self.audio.add(f.stem)

    def _scan_dir(self, subdir: str, target: set, extensions: set):
        d = self.root / subdir
        if not d.is_dir():
            print(f"WARNING: asset subfolder not found: {d}", file=sys.stderr)
            return
        for f in d.iterdir():
            if f.suffix.lower() in extensions:
                target.add(f.stem)

    def resolve(self, ref: AssetRef) -> AssetRef:
        ref.fallback = FALLBACK_TABLE.get(ref.asset_type)
        if ref.asset_type == "texture":
            if ref.name in self.textures:
                # Determine actual extension
                for ext in (".jpg", ".png"):
                    if (self.root / "textures" / (ref.name + ext)).exists():
                        ref.resolved_path = f"textures/{ref.name}{ext}"
                        break
                ref.found = True
            else:
                ref.found = False

        elif ref.asset_type == "prefab":
            has_prefab = ref.name in self.prefabs
            has_glb    = ref.name in self.models
            has_fbx    = ref.name in self.fbx
            if has_prefab or has_glb or has_fbx:
                ref.found = True
                parts = []
                if has_prefab: parts.append(f"prefabs/{ref.name}.prefab")
                if has_glb:    parts.append(f"models/{ref.name}.glb")
                if has_fbx:    parts.append(f"fbx/{ref.name}.fbx")
                ref.resolved_path = " + ".join(parts)
            else:
                ref.found = False

        elif ref.asset_type == "audio":
            if not self.audio_dir_exists:
                ref.found = False
                ref.resolved_path = None
            elif ref.name in self.audio:
                ref.found = True
                ref.resolved_path = f"audio/{ref.name}.*"
            else:
                ref.found = False

        elif ref.asset_type == "script":
            if ref.name in self.scripts:
                ref.found = True
                ref.resolved_path = f"object scripts/{ref.name}.cs"
            else:
                ref.found = False

        return ref


# ── Loading ───────────────────────────────────────────────────────────────────

def load_lab(json_path: Path, no_validate: bool = False) -> "Lab | dict":
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _die(f"Lab JSON not found: {json_path}")
    except json.JSONDecodeError as e:
        _die(f"Invalid JSON in {json_path}: {e}")

    if no_validate:
        # Return raw dict — extraction functions handle both typed and dict inputs
        return data

    if not _PYDANTIC_AVAILABLE:
        _die(f"Could not import Pydantic models: {_import_err_msg}\n"
             f"Make sure pydantic is installed and Code/Tools/pydantic_json_lab_claude.py exists.\n"
             f"Use --no-validate to run without schema validation.")

    try:
        return Lab.model_validate(data)
    except ValidationError as e:
        print(f"ERROR: Lab JSON failed schema validation:\n{e}", file=sys.stderr)
        print("Tip: use --no-validate to skip validation for hand-edited JSON.", file=sys.stderr)
        sys.exit(1)


def find_asset_root(override: Optional[str], script_path: Path) -> Path:
    candidates = []

    if override:
        p = Path(override)
        if not p.is_dir():
            _die(f"--asset-root path does not exist: {p}")
        return p

    # Relative to this script's location
    p = script_path.parent.parent / "viewer" / "arlem_preview_toolkit" / "arlem_preview_toolkit" / "assets"
    candidates.append(p)
    if p.is_dir():
        return p

    # Walk up from script looking for viewer/
    for parent in script_path.parents:
        p = parent / "viewer" / "arlem_preview_toolkit" / "arlem_preview_toolkit" / "assets"
        candidates.append(p)
        if p.is_dir():
            return p

    _die("Could not auto-detect asset root. Searched:\n" +
         "\n".join(f"  {c}" for c in candidates) +
         "\nUse --asset-root PATH to specify the location.")


# ── Output Formatting ─────────────────────────────────────────────────────────

def _colorize(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{_RESET}" if use_color else text


def format_screen_text(refs: list[AssetRef], mode: str, lab_id: str,
                       audio_category_missing: bool, use_color: bool) -> str:
    lines = []
    is_audit = mode in ("audit", "missing")

    # Header
    header = f"=== Asset Audit: {lab_id} ==="
    lines.append(_colorize(header, _BOLD, use_color))

    if is_audit:
        total   = len(refs)
        found   = sum(1 for r in refs if r.found is True)
        missing = sum(1 for r in refs if r.found is False)
        summary = f"Mode: {mode}  |  Total: {total}  |  Found: {found}  |  Missing: {missing}"
        lines.append(summary)
    else:
        lines.append(f"Mode: {mode}")
    lines.append("")

    # Group by type
    by_type: dict[str, list[AssetRef]] = {t: [] for t in ASSET_TYPE_ORDER}
    for ref in refs:
        by_type.setdefault(ref.asset_type, []).append(ref)

    for asset_type in ASSET_TYPE_ORDER:
        group = sorted(by_type.get(asset_type, []), key=lambda r: r.name.lower())
        if not group:
            continue

        # Skip entire group in --missing mode if all found
        if mode == "missing" and all(r.found is True for r in group):
            continue

        type_label = asset_type.upper() + "S"
        count_label = f"({len(group)} unique)"
        category_note = ""
        if asset_type == "audio" and audio_category_missing:
            category_note = "  " + _colorize("[CATEGORY MISSING - no audio folder found]", _YELLOW, use_color)

        lines.append(_colorize(f"{type_label} {count_label}", _BOLD, use_color) + category_note)

        fallback_shown = False
        for ref in group:
            if mode == "missing" and ref.found is True:
                continue

            if not is_audit:
                # dedup / module / clip: just list names with source
                lines.append(f"  {ref.name:<30}  ({ref.source})")
            else:
                if ref.found is True:
                    tag = _colorize("[OK]     ", _GREEN, use_color)
                    path_str = f" -> {ref.resolved_path}" if ref.resolved_path else ""
                    lines.append(f"  {tag} {ref.name:<30}{path_str}")
                else:
                    tag = _colorize("[MISSING]", _RED, use_color)
                    if asset_type == "audio" and audio_category_missing:
                        lines.append(f"  {tag} {ref.name}")
                    else:
                        lines.append(f"  {tag} {ref.name:<30} -> not found")
                    if ref.fallback and not fallback_shown:
                        lines.append(f"           Fallback: {ref.fallback}")
                        if asset_type == "audio" and audio_category_missing:
                            fallback_shown = True  # show once for audio category

        lines.append("")

    return "\n".join(lines)


def format_json_output(refs: list[AssetRef], mode: str, lab_id: str,
                       audio_category_missing: bool) -> str:
    is_audit = mode in ("audit", "missing")
    total   = len(refs)
    found   = sum(1 for r in refs if r.found is True)  if is_audit else None
    missing = sum(1 for r in refs if r.found is False) if is_audit else None

    summary: dict = {"total": total}
    if is_audit:
        summary["found"]   = found
        summary["missing"] = missing
        summary["audio_category_missing"] = audio_category_missing

    assets = []
    for ref in refs:
        entry: dict = {
            "name":          ref.name,
            "type":          ref.asset_type,
            "source":        ref.source,
            "found":         ref.found,
            "resolved_path": ref.resolved_path,
            "fallback":      ref.fallback if ref.found is False else None,
        }
        assets.append(entry)

    result = {
        "mode":    mode,
        "lab":     lab_id,
        "summary": summary,
        "assets":  assets,
    }
    return json.dumps(result, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset_audit",
        description="Audit assets referenced by an ARLEM AR lab JSON file (v2.0 schema).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python asset_audit/asset_audit.py lab.json --dedup
  python asset_audit/asset_audit.py lab.json --audit
  python asset_audit/asset_audit.py lab.json --missing --output json
  python asset_audit/asset_audit.py lab.json --module "Synchronous Rotation"
  python asset_audit/asset_audit.py lab.json --clip "Synchronous Rotation" clip3
        """,
    )
    parser.add_argument("lab_json", help="Path to the lab JSON file (v2.0 schema)")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dedup",  action="store_true",
                      help="List all unique assets needed by the entire lab")
    mode.add_argument("--module", metavar="MODULE_NAME",
                      help="Assets needed up through a named module (cumulative)")
    mode.add_argument("--clip",   nargs=2, metavar=("MODULE_NAME", "CLIP_NAME"),
                      help="Assets needed up through a named clip (cumulative)")
    mode.add_argument("--audit",  action="store_true",
                      help="Check all referenced assets against the asset folders")
    mode.add_argument("--missing", action="store_true",
                      help="Like --audit but only report missing assets")

    parser.add_argument("--output", choices=["screen", "text", "json"], default="screen",
                        help="Output format (default: screen)")
    parser.add_argument("--asset-root", metavar="PATH",
                        help="Override the default asset folder root path")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip Pydantic schema validation (lenient load)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Load lab
    json_path = Path(args.lab_json)
    lab = load_lab(json_path, no_validate=args.no_validate)

    # Determine mode and collect refs
    mode_name: str
    if args.dedup:
        mode_name = "dedup"
        refs = dedup(extract_from_lab(lab))
    elif args.module:
        mode_name = "module"
        refs = refs_for_module(lab, args.module)
    elif args.clip:
        mode_name = "clip"
        refs = refs_for_clip(lab, args.clip[0], args.clip[1])
    elif args.audit:
        mode_name = "audit"
        refs = dedup(extract_from_lab(lab))
    else:  # --missing
        mode_name = "missing"
        refs = dedup(extract_from_lab(lab))

    # Run filesystem audit if needed
    audio_category_missing = False
    if mode_name in ("audit", "missing"):
        script_path = Path(__file__).resolve()
        asset_root  = find_asset_root(args.asset_root, script_path)
        scanner     = AssetScanner(asset_root)
        audio_category_missing = not scanner.audio_dir_exists
        refs = [scanner.resolve(ref) for ref in refs]
        if mode_name == "missing":
            refs = [r for r in refs if r.found is False]

    # Format and emit output
    use_color = (args.output == "screen") and sys.stdout.isatty()
    lab_id = _get(lab, "labId", str(json_path.stem))

    if args.output == "json":
        print(format_json_output(refs, mode_name, lab_id, audio_category_missing))
    else:
        print(format_screen_text(refs, mode_name, lab_id, audio_category_missing, use_color))


if __name__ == "__main__":
    main()

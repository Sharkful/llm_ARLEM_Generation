"""Stage 4: STL/OBJ/FBX/GLB -> GLB, scale/pivot/orientation normalization.

Every asset entering the catalog -- whether from Stage 3 (OpenSCAD STL),
a future mesh-cleanup pass, or Stage 6 (external intake, OBJ/FBX/GLB) --
passes through this one function so the "every asset must have known
canonical scale" requirement (offline_ar_asset_pipeline_requirements.md
section 8) is enforced once, at build time, rather than left as a runtime
guess.

Built around a headless Blender Python script, following the same pattern
as viewer/arlem_preview_toolkit/arlem_preview_toolkit/prefab_to_glb.py's
BLENDER_SCRIPT (subprocess, --background --python, capture stdout/stderr,
success judged by output file existing). Blender 4.0+ moved STL/OBJ import
off the legacy bpy.ops.import_mesh.* namespace onto bpy.ops.wm.*_import;
the script below tries the modern operator first and falls back to the
legacy one so it works across the Blender versions this pipeline might
encounter.

Normalization here means: recenter the pivot to the chosen convention
(center, or base-center for "stands on a surface" objects), apply a
uniform scale so the exported GLB's bounds already equal canonical_bounds_m
(so target_size_m -> scale at scene-build time is a simple ratio, never a
mesh re-processing step), decimate only if triangle count exceeds
config.MAX_TRIANGLE_COUNT, and export GLB.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import config

SourceFormat = Literal["stl", "obj", "fbx", "glb"]
Pivot = Literal["center", "base_center"]

_IMPORT_BY_FORMAT: dict[str, str] = {
    "stl": "stl",
    "obj": "obj",
    "fbx": "fbx",
    "glb": "gltf",
}


class MeshNormalizationResult(BaseModel):
    asset_id: str
    success: bool
    source_path: str | None = None
    glb_path: str | None = None
    original_bounds_m: list[float] | None = None
    final_bounds_m: list[float] | None = None
    pivot: Pivot | None = None
    scale_applied: float | None = None
    triangle_count_before: int | None = None
    triangle_count_after: int | None = None
    decimated: bool = False
    error_message: str | None = None


class MeshProcessingError(RuntimeError):
    pass


_BLENDER_SCRIPT_TEMPLATE = """\
import bpy, sys, json, math

argv = sys.argv[sys.argv.index('--') + 1:]
source_path, source_format, glb_path, report_path, pivot_mode, target_size_m, max_triangles, decimate_ratio = argv
target_size_m = float(target_size_m)
max_triangles = int(max_triangles)
decimate_ratio = float(decimate_ratio)

bpy.ops.wm.read_factory_settings(use_empty=True)

# --- Import ---------------------------------------------------------------
imported = False
import_errors = []

def try_import(op_path, **kwargs):
    global imported
    if imported:
        return
    parts = op_path.split('.')
    try:
        ns = bpy.ops
        for p in parts[:-1]:
            ns = getattr(ns, p)
        fn = getattr(ns, parts[-1])
        fn(filepath=source_path, **kwargs)
        imported = True
    except Exception as e:
        import_errors.append(f"{op_path}: {e}")

if source_format == "stl":
    try_import("wm.stl_import")
    try_import("import_mesh.stl")
elif source_format == "obj":
    try_import("wm.obj_import")
    try_import("import_scene.obj")
elif source_format == "fbx":
    try_import("import_scene.fbx")
elif source_format == "gltf":
    try_import("import_scene.gltf")

if not imported:
    json.dump({"success": False, "error": "; ".join(import_errors) or "no matching import operator"}, open(report_path, "w"))
    sys.exit(1)

mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objects:
    json.dump({"success": False, "error": "no mesh objects found after import"}, open(report_path, "w"))
    sys.exit(1)

# Join multiple mesh objects (e.g. multi-part OBJ/FBX) into one so bounds/
# pivot/scale operate on the whole asset as a single unit.
bpy.ops.object.select_all(action='DESELECT')
for o in mesh_objects:
    o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objects[0]
if len(mesh_objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# --- Original bounds (object space, before any transform) ----------------
def world_bounds(o):
    coords = [o.matrix_world @ v.co for v in o.data.vertices]
    xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

(x0, x1), (y0, y1), (z0, z1) = world_bounds(obj)
original_bounds = [x1 - x0, y1 - y0, z1 - z0]

# --- Recenter pivot --------------------------------------------------------
if pivot_mode == "base_center":
    origin = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, z0)
else:
    origin = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)

cursor = bpy.context.scene.cursor
cursor.location = origin
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
obj.location = (0.0, 0.0, 0.0)

# --- Uniform scale to target_size_m (largest original dimension -> target) -
largest_original = max(original_bounds) if max(original_bounds) > 0 else 1.0
scale_factor = (target_size_m / largest_original) if target_size_m > 0 else 1.0
obj.scale = (obj.scale.x * scale_factor, obj.scale.y * scale_factor, obj.scale.z * scale_factor)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

(x0, x1), (y0, y1), (z0, z1) = world_bounds(obj)
final_bounds = [x1 - x0, y1 - y0, z1 - z0]

# --- Triangle count / decimate ---------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

tri_count_before = len(obj.data.polygons)
decimated = False
if tri_count_before > max_triangles:
    mod = obj.modifiers.new(name="AssetPipelineDecimate", type='DECIMATE')
    mod.ratio = decimate_ratio
    bpy.ops.object.modifier_apply(modifier=mod.name)
    decimated = True
tri_count_after = len(obj.data.polygons)

# --- Export GLB --------------------------------------------------------------
bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format='GLB',
    export_apply=True,
    export_animations=False,
)

json.dump({
    "success": True,
    "original_bounds_m": original_bounds,
    "final_bounds_m": final_bounds,
    "scale_applied": scale_factor,
    "triangle_count_before": tri_count_before,
    "triangle_count_after": tri_count_after,
    "decimated": decimated,
}, open(report_path, "w"))
print(f"Exported: {glb_path}")
"""


def normalize_mesh(
    asset_id: str,
    source_path: Path,
    source_format: SourceFormat,
    glb_path: Path,
    target_size_m: float,
    pivot: Pivot = "center",
) -> MeshNormalizationResult:
    """Import, recenter, scale, decimate-if-needed, export GLB.

    Raises MeshProcessingError if Blender isn't available -- callers should
    catch this the same way they catch missing-API-key errors from
    llm/client_factory.py: report cleanly, don't silently skip
    normalization for an asset that then enters the catalog with unknown
    scale.
    """
    if not config.BLENDER_BIN:
        raise MeshProcessingError(
            "Blender binary not found. Install Blender or set ASSET_PIPELINE_BLENDER_BIN."
        )
    if not source_path.exists():
        raise MeshProcessingError(f"Source mesh not found: {source_path}")

    glb_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "normalize.py"
        report_path = Path(tmp_dir) / "report.json"
        script_path.write_text(_BLENDER_SCRIPT_TEMPLATE, encoding="utf-8")

        blender_import_format = _IMPORT_BY_FORMAT[source_format]

        try:
            result = subprocess.run(
                [
                    config.BLENDER_BIN, "--background", "--python", str(script_path),
                    "--",
                    str(source_path), blender_import_format, str(glb_path), str(report_path),
                    pivot, str(target_size_m), str(config.MAX_TRIANGLE_COUNT), str(config.DECIMATE_RATIO),
                ],
                capture_output=True,
                text=True,
                timeout=config.BLENDER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return MeshNormalizationResult(
                asset_id=asset_id,
                success=False,
                source_path=str(source_path),
                error_message=f"Blender timed out after {config.BLENDER_TIMEOUT_SECONDS}s",
            )

        if not report_path.exists():
            tail = (result.stderr or result.stdout)[-1500:]
            return MeshNormalizationResult(
                asset_id=asset_id,
                success=False,
                source_path=str(source_path),
                error_message=f"Blender produced no report (crashed or script error). Output: {tail}",
            )

        report = json.loads(report_path.read_text(encoding="utf-8"))

    if not report.get("success"):
        return MeshNormalizationResult(
            asset_id=asset_id,
            success=False,
            source_path=str(source_path),
            error_message=report.get("error", "unknown Blender-side failure"),
        )

    return MeshNormalizationResult(
        asset_id=asset_id,
        success=True,
        source_path=str(source_path),
        glb_path=str(glb_path),
        original_bounds_m=[round(v, 6) for v in report["original_bounds_m"]],
        final_bounds_m=[round(v, 6) for v in report["final_bounds_m"]],
        pivot=pivot,
        scale_applied=round(report["scale_applied"], 6),
        triangle_count_before=report["triangle_count_before"],
        triangle_count_after=report["triangle_count_after"],
        decimated=report["decimated"],
    )


def save_mesh_meta(asset_id: str, result: MeshNormalizationResult) -> Path:
    """Write library/generated/<asset_id>/meta.json per the implementation plan."""
    meta_path = config.LIBRARY_DIR / "generated" / asset_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    return meta_path

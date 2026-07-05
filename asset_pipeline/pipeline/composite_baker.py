"""Bake a resolved CompositeFragment into one multi-material GLB (added
2026-07, "Frosty the Snowman is all gray" incident).

Composites previously lived only as a JSON parts-list with every part's
position/rotation/scale defaulted to the origin/unit-scale and no material
at all -- there was no rendering path for them anywhere in this repo, so a
"resolved" composite was invisible until this module existed. Baking turns
a composite into an ordinary cataloged asset (asset_class="composite",
a real library/composites/<id>/model.glb) by having Blender build/import
each part, position and color it, and JOIN everything into one mesh.
Blender's join preserves each source object's material slot, so the
result is one glTF with several materials -- exactly what "three white
snowballs, brown stick arms, an orange nose, a black hat" needs, and
exactly the representation every other pipeline stage already knows how
to preview/sync/validate, since it's just a GLB with an address like any
other asset. No composite-specific code needed anywhere else as a result.

Primitive parts (resolved_asset_id pointing at a seed catalog primitive,
e.g. sphere_basic) are built natively in Blender at bake time -- cheaper
and simpler than importing a GLB for a shape Blender can create with one
operator call. Non-primitive parts (an imported/generated/composite GLB
referenced as a sub-part) are imported as glTF and joined the same way.

Solid per-part colors need no UV data (a material with no texture image
ignores UVs entirely), so this does not depend on Stage 6c's texture/UV
work; a part that DOES carry a bound texture keeps its own UVs through the
join, for whenever a future part wants a real image rather than a flat
color.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

import config
from pipeline.composite_builder import CompositeFragment

_PRIMITIVE_ADDRESS_BUILDERS = {
    "sphere_basic": "sphere",
    "cube_basic": "cube",
    "cylinder_basic": "cylinder",
    "cone_basic": "cone",
    "torus_basic": "torus",
    "capsule_basic": "cylinder",  # Blender has no native capsule primitive op
    "plane_basic": "plane",
    "quad_basic": "plane",
}


class CompositeBakeResult(BaseModel):
    asset_id: str
    success: bool
    glb_path: str | None = None
    final_bounds_m: list[float] | None = None
    triangle_count: int | None = None
    uv_status: str = "none"
    error_message: str | None = None


class CompositeBakeError(RuntimeError):
    pass


_BLENDER_SCRIPT_TEMPLATE = """\
import bpy, sys, json, math, mathutils

argv = sys.argv[sys.argv.index('--') + 1:]
parts_json, glb_path, report_path, target_size_m = argv
target_size_m = float(target_size_m)
parts = json.loads(parts_json)

bpy.ops.wm.read_factory_settings(use_empty=True)

def make_primitive(kind):
    if kind == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, segments=32, ring_count=16)
    elif kind == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=1.0)
    elif kind == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, vertices=32)
    elif kind == 'cone':
        bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0, vertices=32)
    elif kind == 'torus':
        bpy.ops.mesh.primitive_torus_add(major_radius=0.35, minor_radius=0.15)
    elif kind == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=1.0)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0)  # recognizable fallback
    return bpy.context.active_object

def import_glb(path):
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    added = [o for o in bpy.context.scene.objects if o not in before and o.type == 'MESH']
    if len(added) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in added:
            o.select_set(True)
        bpy.context.view_layer.objects.active = added[0]
        bpy.ops.object.join()
        added = [bpy.context.active_object]
    return added[0] if added else None

def hex_to_rgb(hexstr):
    h = hexstr.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)) + (1.0,)

def to_blender_axes(x, y, z):
    # This pipeline (and glTF/three.js) is +Y-up, +Z-forward; Blender is
    # natively +Z-up. Blender's glTF exporter converts Z-up -> Y-up
    # automatically on export, so feeding it these Blender-native axes here
    # (swap Y/Z, negate the former Z into Blender's Y) makes that automatic
    # round-trip conversion restore the pipeline's own numbers instead of
    # quietly reinterpreting "up" as "forward".
    return (x, -z, y)

def apply_material(obj, base_color, metallic, smoothness, name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf is not None:
        bsdf.inputs['Base Color'].default_value = hex_to_rgb(base_color)
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = metallic
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = 1.0 - smoothness
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def finish_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=False)

built = []
atom_positions = {}  # label -> Blender-native Vector, for connector parts

# Pass 1: every non-connector part (atoms/regular pieces). Connectors need
# THESE parts' real final positions, so they must exist first.
for part in parts:
    if part.get('bond_between'):
        continue
    if part.get('glb_path'):
        obj = import_glb(part['glb_path'])
        if obj is None:
            continue
    else:
        obj = make_primitive(part['primitive_kind'])

    apply_material(obj, part['base_color'], part.get('metallic', 0.0),
                    part.get('smoothness', 0.5), part['part_id'] + '_mat')

    px, py, pz = part['position']
    sx, sy, sz = part['scale']
    rx, ry, rz = part.get('rotation', [0.0, 0.0, 0.0])
    obj.location = to_blender_axes(px * target_size_m, py * target_size_m, pz * target_size_m)
    obj.scale = to_blender_axes(sx * target_size_m, sy * target_size_m, sz * target_size_m)
    obj.rotation_euler = (math.radians(rx), math.radians(-rz), math.radians(ry))
    finish_object(obj)
    built.append(obj)
    if part.get('label'):
        atom_positions[part['label']] = mathutils.Vector(obj.location)

# Pass 2: connector parts (bond_between set) -- position, length, and
# rotation computed deterministically from the two named atoms' real
# positions above; an LLM is not asked to (and reliably cannot) get this
# 3D trigonometry right by itself.
for part in parts:
    bond = part.get('bond_between')
    if not bond:
        continue
    label_a, label_b = bond
    if label_a not in atom_positions or label_b not in atom_positions:
        continue  # referenced a label that didn't resolve/build; skip quietly
    pos_a, pos_b = atom_positions[label_a], atom_positions[label_b]
    direction = pos_b - pos_a
    length = direction.length
    if length < 1e-6:
        continue

    obj = make_primitive('cylinder')
    apply_material(obj, part['base_color'], part.get('metallic', 0.0),
                    part.get('smoothness', 0.5), part['part_id'] + '_mat')
    radius = part.get('bond_thickness', 0.06) * target_size_m
    obj.scale = (radius * 2.0, radius * 2.0, length)
    obj.location = (pos_a + pos_b) / 2.0
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    finish_object(obj)
    built.append(obj)

if not built:
    json.dump({"success": False, "error": "no parts could be built"}, open(report_path, "w"))
    sys.exit(1)

bpy.ops.object.select_all(action='DESELECT')
for o in built:
    o.select_set(True)
bpy.context.view_layer.objects.active = built[0]
if len(built) > 1:
    bpy.ops.object.join()
joined = bpy.context.active_object

# Recenter pivot to the assembly's own bounding-box center (base_center
# convention: whole assembly rests on Z=0, matching a figure standing on
# a surface, per the seed primitives' own pivot conventions).
coords = []
for v in joined.data.vertices:
    coords.append(joined.matrix_world @ v.co)
xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
cursor = bpy.context.scene.cursor
cursor.location = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, min(zs))
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
joined.location = (0.0, 0.0, 0.0)

final_bounds = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]

# UV unwrap any part with no layer (parts imported from a textured GLB
# already have one; native Blender primitives do not).
uv_status = "preserved" if joined.data.uv_layers else "none"
if not joined.data.uv_layers:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.02)
    except TypeError:
        bpy.ops.uv.smart_project(angle_limit=66.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    uv_status = "generated"

bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', export_apply=True, export_animations=False)

json.dump({
    "success": True,
    "final_bounds_m": final_bounds,
    "triangle_count": len(joined.data.polygons),
    "uv_status": uv_status,
}, open(report_path, "w"))
print(f"Baked: {glb_path}")
"""


def bake_composite(
    fragment: CompositeFragment,
    asset_id: str,
    target_size_m: float,
) -> CompositeBakeResult:
    """Assemble every resolved part (positioned/scaled/colored per its
    CompositePart fields) into one GLB. A part with no resolved_asset_id
    or material is skipped with a note in the result, not a hard failure --
    a partially-built snowman is more useful than none."""
    if not config.BLENDER_BIN:
        raise CompositeBakeError(
            "Blender binary not found. Install Blender or set ASSET_PIPELINE_BLENDER_BIN."
        )

    from pipeline.catalog_writer import load_catalog
    from models.catalog_models import MaterialDef

    catalog = {e.asset_id: e for e in load_catalog()}
    parts_payload = []
    skipped: list[str] = []
    for part in fragment.parts:
        material = MaterialDef(material_id="tmp", base_color="#aaaaaa")
        if part.material_id:
            mat_path = config.MATERIALS_DIR / f"{part.material_id}.json"
            if mat_path.is_file():
                material = MaterialDef.model_validate_json(mat_path.read_text(encoding="utf-8-sig"))

        # A connector (bond_between set) is always a plain cylinder built
        # from scratch -- its geometry comes from the two labeled atoms'
        # real positions (computed in Blender, pass 2), not from any
        # catalog lookup, so it has no resolved_asset_id requirement at all.
        if part.bond_between:
            parts_payload.append({
                "part_id": part.part_id,
                "label": part.label,
                "bond_between": part.bond_between,
                "bond_thickness": part.bond_thickness,
                "base_color": material.base_color,
                "metallic": material.metallic,
                "smoothness": material.smoothness,
            })
            continue

        if not part.resolved_asset_id:
            skipped.append(f"{part.part_id} (unresolved)")
            continue
        entry = catalog.get(part.resolved_asset_id)
        if entry is None:
            skipped.append(f"{part.part_id} (catalog entry {part.resolved_asset_id!r} missing)")
            continue

        payload = {
            "part_id": part.part_id,
            "label": part.label,
            "position": part.position,
            "rotation": part.rotation,
            "scale": part.scale,
            "base_color": material.base_color,
            "metallic": material.metallic,
            "smoothness": material.smoothness,
        }
        primitive_kind = _PRIMITIVE_ADDRESS_BUILDERS.get(part.resolved_asset_id)
        if primitive_kind:
            payload["primitive_kind"] = primitive_kind
            payload["glb_path"] = None
        elif entry.address.startswith("library/"):
            payload["primitive_kind"] = None
            payload["glb_path"] = str(config.LIBRARY_DIR / entry.address[len("library/"):])
        else:
            skipped.append(f"{part.part_id} (no bakeable geometry for {entry.address!r})")
            continue
        parts_payload.append(payload)

    if not parts_payload:
        return CompositeBakeResult(
            asset_id=asset_id, success=False,
            error_message="No parts could be baked: " + "; ".join(skipped) if skipped else "No parts to bake.",
        )

    glb_path = config.LIBRARY_DIR / "composites" / asset_id / "model.glb"
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "bake.py"
        report_path = Path(tmp_dir) / "report.json"
        script_path.write_text(_BLENDER_SCRIPT_TEMPLATE, encoding="utf-8")

        try:
            result = subprocess.run(
                [
                    config.BLENDER_BIN, "--background", "--python", str(script_path),
                    "--", json.dumps(parts_payload), str(glb_path), str(report_path),
                    str(target_size_m),
                ],
                capture_output=True, text=True, timeout=config.BLENDER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return CompositeBakeResult(
                asset_id=asset_id, success=False,
                error_message=f"Blender timed out after {config.BLENDER_TIMEOUT_SECONDS}s",
            )

        if not report_path.exists():
            tail = (result.stderr or result.stdout)[-1500:]
            return CompositeBakeResult(
                asset_id=asset_id, success=False,
                error_message=f"Blender produced no report (crashed or script error). Output: {tail}",
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))

    if not report.get("success"):
        return CompositeBakeResult(
            asset_id=asset_id, success=False,
            error_message=report.get("error", "unknown Blender-side failure"),
        )

    return CompositeBakeResult(
        asset_id=asset_id, success=True, glb_path=str(glb_path),
        final_bounds_m=[round(v, 6) for v in report["final_bounds_m"]],
        triangle_count=report["triangle_count"], uv_status=report.get("uv_status", "none"),
        error_message="; ".join(skipped) if skipped else None,
    )

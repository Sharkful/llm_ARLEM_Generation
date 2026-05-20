"""
prefab_to_glb.py
================
Convert Unity prefab assets to GLB files for the ARLEM preview viewer.

Two conversion paths:
  1. FBX files  -> Blender headless -> GLB
  2. Prefabs using Unity built-in primitives -> pygltflib procedural GLB

Usage
-----
    python prefab_to_glb.py [--prefabs DIR] [--fbx DIR] [--out DIR] [--blender PATH] [--force]

Defaults:
    --prefabs  assets/prefabs
    --fbx      assets/fbx   (optional; processed in addition to --prefabs)
    --out      assets/models
    --blender  auto-detected from common install locations

FBX name mapping
----------------
FBX files in --fbx are converted using their filename as the GLB name.
Override the output name with --fbx-map, e.g.:
    --fbx-map ArcadeButton=GenericBRB --fbx-map PushButton=BigRedButton

Examples
--------
    python prefab_to_glb.py
    python prefab_to_glb.py --force
    python prefab_to_glb.py --fbx assets/fbx --fbx-map ArcadeButton=GenericBRB --fbx-map PushButton=BigRedButton
    python prefab_to_glb.py --blender "C:/Program Files/Blender Foundation/Blender 4.4/blender.exe"
"""

import argparse
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Unity built-in mesh fileID -> shape tag
# https://docs.unity3d.com/Manual/BuiltInImporters.html
# ---------------------------------------------------------------------------
UNITY_BUILTIN_MESHES = {
    "10202": "cube",
    "10206": "quad",     # 1×1 flat plane facing +Z
    "10207": "sphere",
    "10208": "capsule",
    "10209": "plane",    # 10×10 subdivided ground plane (we emit a 1×1 unit quad)
    "10210": "cylinder",
}


# ---------------------------------------------------------------------------
# Minimal GLB / GLTF helpers using pygltflib
# ---------------------------------------------------------------------------
def _pack(fmt, *args):
    return struct.pack(fmt, *args)


def _sphere_mesh(radius=0.5, rings=16, sectors=32):
    """UV sphere vertices + triangle indices."""
    import array
    verts = []
    normals = []
    uvs = []
    for r in range(rings + 1):
        phi = math.pi * r / rings          # 0 -> π
        for s in range(sectors + 1):
            theta = 2 * math.pi * s / sectors  # 0 -> 2π
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            verts += [x, y, z]
            normals += [x / radius, y / radius, z / radius]
            uvs += [s / sectors, r / rings]
    indices = []
    for r in range(rings):
        for s in range(sectors):
            i0 = r * (sectors + 1) + s
            i1 = i0 + 1
            i2 = (r + 1) * (sectors + 1) + s
            i3 = i2 + 1
            indices += [i0, i2, i1, i1, i2, i3]
    return (
        array.array("f", verts).tobytes(),
        array.array("f", normals).tobytes(),
        array.array("f", uvs).tobytes(),
        array.array("H", indices).tobytes(),
        len(verts) // 3,
        len(indices),
    )


def _box_mesh(w=1.0, h=1.0, d=1.0):
    hw, hh, hd = w / 2, h / 2, d / 2
    # 6 faces × 4 verts
    faces = [
        # pos normal  u,v corners
        ([(-hw, -hh,  hd), ( hw, -hh,  hd), ( hw,  hh,  hd), (-hw,  hh,  hd)], [ 0,  0,  1]),
        ([( hw, -hh, -hd), (-hw, -hh, -hd), (-hw,  hh, -hd), ( hw,  hh, -hd)], [ 0,  0, -1]),
        ([(-hw, -hh, -hd), (-hw, -hh,  hd), (-hw,  hh,  hd), (-hw,  hh, -hd)], [-1,  0,  0]),
        ([( hw, -hh,  hd), ( hw, -hh, -hd), ( hw,  hh, -hd), ( hw,  hh,  hd)], [ 1,  0,  0]),
        ([(-hw,  hh,  hd), ( hw,  hh,  hd), ( hw,  hh, -hd), (-hw,  hh, -hd)], [ 0,  1,  0]),
        ([(-hw, -hh, -hd), ( hw, -hh, -hd), ( hw, -hh,  hd), (-hw, -hh,  hd)], [ 0, -1,  0]),
    ]
    import array
    verts, norms, uvcoords, indices = [], [], [], []
    base = 0
    for corners, n in faces:
        uvmap = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for (x, y, z), (u, v) in zip(corners, uvmap):
            verts += [x, y, z]
            norms += list(map(float, n))
            uvcoords += [u, v]
        indices += [base, base+1, base+2, base, base+2, base+3]
        base += 4
    nv = len(verts) // 3
    ni = len(indices)
    return (
        array.array("f", verts).tobytes(),
        array.array("f", norms).tobytes(),
        array.array("f", uvcoords).tobytes(),
        array.array("H", indices).tobytes(),
        nv,
        ni,
    )


def _quad_mesh(w=1.0, h=1.0):
    """Single flat quad in XY plane facing +Z (Unity Quad primitive)."""
    import array
    hw, hh = w / 2, h / 2
    verts = [-hw, -hh, 0,  hw, -hh, 0,  hw, hh, 0,  -hw, hh, 0]
    norms = [0, 0, 1] * 4
    uvs   = [0, 0,  1, 0,  1, 1,  0, 1]
    idxs  = [0, 1, 2,  0, 2, 3,  0, 2, 1,  0, 3, 2]  # double-sided
    return (
        array.array("f", verts).tobytes(),
        array.array("f", norms).tobytes(),
        array.array("f", uvs).tobytes(),
        array.array("H", idxs).tobytes(),
        4,
        len(idxs),
    )


def _cylinder_mesh(radius=0.5, height=1.0, segs=24):
    """Open cylinder without caps."""
    import array
    verts, norms, uvs, indices = [], [], [], []
    half = height / 2
    base = 0
    for i in range(segs + 1):
        theta = 2 * math.pi * i / segs
        x, z = math.cos(theta), math.sin(theta)
        for y in [-half, half]:
            verts += [radius * x, y, radius * z]
            norms += [x, 0.0, z]
            uvs   += [i / segs, (y + half) / height]
    for i in range(segs):
        b = i * 2
        indices += [b, b+1, b+2, b+1, b+3, b+2]
    nv = (segs + 1) * 2
    ni = segs * 6
    return (
        array.array("f", verts).tobytes(),
        array.array("f", norms).tobytes(),
        array.array("f", uvs).tobytes(),
        array.array("H", indices).tobytes(),
        nv,
        ni,
    )


def _capsule_mesh(radius=0.5, height=1.0, rings=8, sectors=16):
    """Capsule = cylinder body + two hemisphere caps."""
    import array
    # Combine sphere halves separated by height
    verts, norms, uvs, indices = [], [], [], []
    half = height / 2

    # Top hemisphere
    for r in range(rings + 1):
        phi = math.pi / 2 * r / rings      # 0 -> π/2
        for s in range(sectors + 1):
            theta = 2 * math.pi * s / sectors
            x = radius * math.cos(phi) * math.cos(theta)
            y = radius * math.sin(phi) + half
            z = radius * math.cos(phi) * math.sin(theta)
            nx, ny, nz = x / radius, (y - half) / radius, z / radius
            verts += [x, y, z];  norms += [nx, ny, nz];  uvs += [s / sectors, 0.5 + r / (2 * rings)]

    top_count = (rings + 1) * (sectors + 1)
    for r in range(rings):
        for s in range(sectors):
            i0 = r * (sectors + 1) + s
            indices += [i0, i0 + sectors + 2, i0 + 1, i0, i0 + sectors + 1, i0 + sectors + 2]

    # Bottom hemisphere
    base_idx = top_count
    for r in range(rings + 1):
        phi = math.pi / 2 + math.pi / 2 * r / rings  # π/2 -> π
        for s in range(sectors + 1):
            theta = 2 * math.pi * s / sectors
            x = radius * math.cos(phi) * math.cos(theta)
            y = radius * math.sin(phi) - half
            z = radius * math.cos(phi) * math.sin(theta)
            nx, ny, nz = x / radius, (y + half) / radius, z / radius
            verts += [x, y, z];  norms += [nx, ny, nz];  uvs += [s / sectors, 0.5 - r / (2 * rings)]

    for r in range(rings):
        for s in range(sectors):
            i0 = base_idx + r * (sectors + 1) + s
            indices += [i0, i0 + sectors + 2, i0 + 1, i0, i0 + sectors + 1, i0 + sectors + 2]

    nv = len(verts) // 3
    return (
        array.array("f", verts).tobytes(),
        array.array("f", norms).tobytes(),
        array.array("f", uvs).tobytes(),
        array.array("H", indices).tobytes(),
        nv,
        len(indices),
    )


def _build_glb(shape, color_rgba=(0.8, 0.8, 0.8, 1.0)):
    """Build a single-mesh GLB for `shape` using pygltflib."""
    import pygltflib

    mesh_fns = {
        "sphere":   lambda: _sphere_mesh(),
        "cube":     lambda: _box_mesh(),
        "quad":     lambda: _quad_mesh(),
        "plane":    lambda: _quad_mesh(10.0, 10.0),
        "cylinder": lambda: _cylinder_mesh(),
        "capsule":  lambda: _capsule_mesh(),
    }
    fn = mesh_fns.get(shape)
    if fn is None:
        raise ValueError(f"Unknown shape: {shape}")

    pos_bytes, norm_bytes, uv_bytes, idx_bytes, nv, ni = fn()

    # Pad each buffer section to 4-byte alignment
    def pad4(b):
        rem = len(b) % 4
        return b + b'\x00' * (4 - rem) if rem else b

    pos_bytes  = pad4(pos_bytes)
    norm_bytes = pad4(norm_bytes)
    uv_bytes   = pad4(uv_bytes)
    idx_bytes  = pad4(idx_bytes)

    # Byte offsets within the single blob
    off_idx  = 0
    off_pos  = off_idx  + len(idx_bytes)
    off_norm = off_pos  + len(pos_bytes)
    off_uv   = off_norm + len(norm_bytes)
    blob     = idx_bytes + pos_bytes + norm_bytes + uv_bytes

    # Compute bounding box for positions (accessor min/max)
    import struct as st
    nfloats = len(pos_bytes) // 4
    floats   = list(st.unpack(f"{nfloats}f", pos_bytes[:nv * 12]))
    xs = floats[0::3];  ys = floats[1::3];  zs = floats[2::3]
    pos_min = [min(xs), min(ys), min(zs)]
    pos_max = [max(xs), max(ys), max(zs)]

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[pygltflib.Mesh(primitives=[
            pygltflib.Primitive(
                attributes=pygltflib.Attributes(POSITION=1, NORMAL=2, TEXCOORD_0=3),
                indices=0,
                material=0,
            )
        ])],
        accessors=[
            pygltflib.Accessor(
                bufferView=0, componentType=pygltflib.UNSIGNED_SHORT,
                count=ni, type=pygltflib.SCALAR,
            ),
            pygltflib.Accessor(
                bufferView=1, componentType=pygltflib.FLOAT,
                count=nv, type=pygltflib.VEC3,
                min=pos_min, max=pos_max,
            ),
            pygltflib.Accessor(
                bufferView=2, componentType=pygltflib.FLOAT,
                count=nv, type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=3, componentType=pygltflib.FLOAT,
                count=nv, type=pygltflib.VEC2,
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(buffer=0, byteOffset=off_idx,  byteLength=len(idx_bytes),  target=pygltflib.ELEMENT_ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=off_pos,  byteLength=len(pos_bytes),  target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=off_norm, byteLength=len(norm_bytes), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=off_uv,   byteLength=len(uv_bytes),   target=pygltflib.ARRAY_BUFFER),
        ],
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
        materials=[pygltflib.Material(
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=list(color_rgba),
                metallicFactor=0.0,
                roughnessFactor=0.7,
            ),
            doubleSided=True,
        )],
    )
    gltf.set_binary_blob(blob)
    return gltf


# ---------------------------------------------------------------------------
# YAML-like prefab parser (Unity's dialect isn't standard YAML — parse manually)
# ---------------------------------------------------------------------------
_MESH_FILE_ID_RE = re.compile(r'm_Mesh:\s*\{fileID:\s*(\d+)')
_IS_PREFAB_INSTANCE_RE = re.compile(r'^PrefabInstance:', re.MULTILINE)
_NAME_RE = re.compile(r'm_Name:\s*(\S+)')
_GUID_RE = re.compile(r'guid:\s*([0-9a-f]+)')


def classify_prefab(path: Path):
    """
    Returns one of:
      ('builtin', shape_tag)       – uses a Unity built-in primitive
      ('fbx', fbx_path)            – references an FBX (resolved from sibling)
      ('prefab_instance', guid)    – references another prefab by GUID (not resolvable here)
      ('no_mesh', None)            – no mesh component found
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    if _IS_PREFAB_INSTANCE_RE.search(text):
        m = _GUID_RE.search(text)
        return ("prefab_instance", m.group(1) if m else "unknown")

    # Check for FBX references — any mesh fileID that points to a non-zero GUID
    # Unity built-ins always have guid: 0000000000000000e000000000000000
    # FBX meshes use asset-specific GUIDs
    mesh_blocks = re.findall(
        r'm_Mesh:\s*\{fileID:\s*(\d+),\s*guid:\s*([0-9a-f]+)',
        text
    )
    for file_id, guid in mesh_blocks:
        if guid == "0000000000000000e000000000000000":
            # Built-in mesh
            shape = UNITY_BUILTIN_MESHES.get(file_id)
            if shape:
                return ("builtin", shape)
        else:
            # External asset (FBX or other)
            # Try to resolve the FBX from sibling files
            meta_glob = list(path.parent.glob("*.fbx"))
            if meta_glob:
                return ("fbx", meta_glob[0])
            return ("fbx_missing", guid)

    # Legacy: plain fileID without guid (built-in, guid field absent)
    m = _MESH_FILE_ID_RE.search(text)
    if m:
        shape = UNITY_BUILTIN_MESHES.get(m.group(1))
        if shape:
            return ("builtin", shape)

    return ("no_mesh", None)


# ---------------------------------------------------------------------------
# Blender FBX->GLB converter
# ---------------------------------------------------------------------------
BLENDER_CANDIDATES = [
    # Traditional installer — newest first
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
    "blender",  # on PATH
]

# Glob pattern to find any MS Store Blender install regardless of version string
def _find_store_blender():
    import glob
    pattern = r"C:\Program Files\WindowsApps\BlenderFoundation.Blender_*\Blender\blender.exe"
    matches = glob.glob(pattern)
    return matches[0] if matches else None

BLENDER_SCRIPT = """\
import bpy, sys, os

fbx_path = sys.argv[sys.argv.index('--') + 1]
glb_path = sys.argv[sys.argv.index('--') + 2]

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX
bpy.ops.import_scene.fbx(filepath=fbx_path)

# Export GLB
bpy.ops.export_scene.gltf(
    filepath=glb_path,
    export_format='GLB',
    export_apply=False,
    export_animations=False,
)
print(f"Exported: {glb_path}")
"""


def find_blender(hint=None):
    store = _find_store_blender()
    candidates = ([hint] if hint else []) + ([store] if store else []) + BLENDER_CANDIDATES
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c))
        # Try which/where for bare names
        if c and not Path(c).suffix and not ('\\' in c or '/' in c):
            try:
                result = subprocess.run(
                    ["where", c] if sys.platform == "win32" else ["which", c],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip().splitlines()[0]
            except Exception:
                pass
    return None


def convert_fbx_to_glb(fbx_path: Path, glb_path: Path, blender_exe: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(BLENDER_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                blender_exe, "--background", "--python", script_path,
                "--", str(fbx_path), str(glb_path)
            ],
            capture_output=True, text=True, timeout=120
        )
        if glb_path.is_file():
            return True
        print(f"    Blender stdout: {result.stdout[-500:]}")
        print(f"    Blender stderr: {result.stderr[-500:]}")
        return False
    except subprocess.TimeoutExpired:
        print("    Blender timed out (120 s)")
        return False
    finally:
        Path(script_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------
SHAPE_COLORS = {
    "sphere":   (0.7, 0.7, 0.9, 1.0),
    "cube":     (0.8, 0.8, 0.8, 1.0),
    "quad":     (0.6, 0.6, 0.8, 1.0),
    "plane":    (0.5, 0.5, 0.5, 1.0),
    "cylinder": (0.8, 0.7, 0.6, 1.0),
    "capsule":  (0.7, 0.8, 0.7, 1.0),
}


def _convert_fbx_file(fbx: Path, out_path: Path, blender_exe, results, force):
    """Convert a single FBX to GLB, respecting --force. Updates results in place."""
    if out_path.exists() and not force:
        print(f"  [SKIP]  {out_path.name} already exists (use --force to overwrite)")
        results["skipped"].append(out_path.stem)
        return
    print(f"  {fbx.name}  ->  {out_path.name}", end="  ")
    if not blender_exe:
        print("SKIP (Blender not found)")
        results["failed"].append(out_path.stem)
        return
    ok = convert_fbx_to_glb(fbx, out_path, blender_exe)
    if ok:
        print("OK")
        results["ok"].append(out_path.stem)
    else:
        print("FAILED")
        results["failed"].append(out_path.stem)


def run(prefabs_dir: Path, out_dir: Path, fbx_dir: Path = None,
        fbx_map: dict = None, blender_hint=None, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    blender_exe = find_blender(blender_hint)
    fbx_map = fbx_map or {}

    prefab_files = sorted(prefabs_dir.glob("*.prefab"))
    prefab_fbx   = sorted(prefabs_dir.glob("*.fbx"))
    extra_fbx    = sorted(fbx_dir.glob("*.fbx")) if (fbx_dir and fbx_dir.is_dir()) else []

    print(f"\nPrefab directory : {prefabs_dir}")
    if fbx_dir:
        print(f"FBX directory    : {fbx_dir}  ({len(extra_fbx)} files)")
    print(f"Output directory : {out_dir}")
    print(f"Prefabs found    : {len(prefab_files)}")
    print(f"Blender          : {blender_exe or '(not found)'}")
    if fbx_map:
        print(f"Name mappings    : {fbx_map}")
    print()

    results = {"ok": [], "skipped": [], "failed": [], "no_mesh": []}

    # --- Process .prefab files ---
    for pf in prefab_files:
        stem = pf.stem
        out_path = out_dir / f"{stem}.glb"

        if out_path.exists() and not force:
            print(f"  [SKIP]  {stem}.glb already exists (use --force to overwrite)")
            results["skipped"].append(stem)
            continue

        kind, detail = classify_prefab(pf)
        print(f"  {stem}.prefab -> {kind}", end="")

        if kind == "builtin":
            shape = detail
            print(f" ({shape})", end="")
            try:
                glb = _build_glb(shape, SHAPE_COLORS.get(shape, (0.8, 0.8, 0.8, 1.0)))
                glb.save(str(out_path))
                print(f"  ->  {out_path.name}  OK")
                results["ok"].append(stem)
            except Exception as e:
                print(f"  ERROR: {e}")
                results["failed"].append(stem)

        elif kind == "fbx":
            fbx_path = detail
            print(f" (FBX: {fbx_path.name})", end="")
            if not blender_exe:
                print("  ->  SKIP (Blender not found)")
                results["failed"].append(stem)
            else:
                ok = convert_fbx_to_glb(fbx_path, out_path, blender_exe)
                if ok:
                    print(f"  ->  {out_path.name}  OK")
                    results["ok"].append(stem)
                else:
                    print(f"  ->  FAILED")
                    results["failed"].append(stem)

        elif kind == "fbx_missing":
            print(f" (FBX GUID {detail} not found in folder)  ->  SKIP")
            results["failed"].append(stem)

        elif kind == "prefab_instance":
            print(f" (PrefabInstance -> references external asset, cannot resolve here)  ->  SKIP")
            results["no_mesh"].append(stem)

        elif kind == "no_mesh":
            print("  (no mesh component)  ->  SKIP")
            results["no_mesh"].append(stem)

        else:
            print(f"  UNKNOWN kind {kind!r}")
            results["failed"].append(stem)

    # --- Process FBX files embedded in prefabs dir ---
    prefab_fbx_names = {pf.stem for pf in prefab_files}
    for fbx in prefab_fbx:
        if any(fbx.stem in name for name in prefab_fbx_names):
            continue  # already handled via the prefab
        out_name = fbx_map.get(fbx.stem, fbx.stem)
        _convert_fbx_file(fbx, out_dir / f"{out_name}.glb", blender_exe, results, force)

    # --- Process dedicated FBX directory ---
    if extra_fbx:
        print(f"\nFBX directory: {fbx_dir}")
        already_done = {f.stem for f in prefab_fbx}  # avoid re-converting same file
        for fbx in extra_fbx:
            if fbx.stem in already_done:
                continue
            out_name = fbx_map.get(fbx.stem, fbx.stem)
            _convert_fbx_file(fbx, out_dir / f"{out_name}.glb", blender_exe, results, force)

    # --- Summary ---
    print()
    print("=" * 50)
    print(f"Done.  OK={len(results['ok'])}  skipped={len(results['skipped'])}  "
          f"no_mesh={len(results['no_mesh'])}  failed={len(results['failed'])}")
    if results["ok"]:
        print(f"  Generated : {', '.join(results['ok'])}")
    if results["failed"]:
        print(f"  Failed    : {', '.join(results['failed'])}")
    if results["no_mesh"]:
        print(f"  No mesh   : {', '.join(results['no_mesh'])} (viewer placeholders will be used)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = Path(__file__).parent

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefabs", default=str(here / "assets" / "prefabs"),
                        help="Directory containing .prefab files (default: assets/prefabs)")
    parser.add_argument("--fbx",     default=str(here / "assets" / "fbx"),
                        help="Directory of standalone FBX files (default: assets/fbx)")
    parser.add_argument("--out",     default=str(here / "assets" / "models"),
                        help="Output directory for .glb files (default: assets/models)")
    parser.add_argument("--blender", default=None,
                        help="Path to blender.exe (auto-detected if omitted)")
    parser.add_argument("--fbx-map", action="append", default=[], metavar="SRC=DST",
                        help="Rename FBX stem to GLB name, e.g. --fbx-map ArcadeButton=GenericBRB")
    parser.add_argument("--force",   action="store_true",
                        help="Overwrite existing .glb files")
    args = parser.parse_args()

    fbx_map = {}
    for entry in args.fbx_map:
        if "=" in entry:
            src, dst = entry.split("=", 1)
            fbx_map[src.strip()] = dst.strip()

    fbx_dir = Path(args.fbx) if args.fbx else None

    run(Path(args.prefabs), Path(args.out),
        fbx_dir=fbx_dir, fbx_map=fbx_map,
        blender_hint=args.blender, force=args.force)

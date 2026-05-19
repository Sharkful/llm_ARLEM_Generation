# ARLEM / Unity JSON Preview Toolkit

This is a lightweight first-pass preview system for ARLEM-style JSON scenes used in Unity/Meta Quest 3 experiences.

## What it does

- Reads the original lab JSON directly in a browser.
- Displays each module as a manipulable Three.js scene.
- Represents spheres, suns, buttons, text labels, robot placeholders, and unknown prefabs with purpose-built placeholder geometry.
- Loads actual GLB models from `assets/models/<prefabName>.glb` when available, replacing placeholders automatically.
- Applies one selected clip at a time, producing static snapshots.
- Provides orbit/zoom/pan controls.
- Saves PNG screenshots from the current camera view.
- Looks for textures in `assets/textures/<texture>.png`, `.jpg`, or `.jpeg`.

## Quick start

From this folder, run a simple local web server:

```powershell
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/index.html
```

The included `Optomized_moon_lab_final.json` should auto-load. You can also choose another JSON file with the file picker.

## Suggested asset layout

```text
arlem_preview_toolkit/
  index.html
  arlem_to_preview_scene.py
  Optomized_moon_lab_final.json
  assets/
    textures/
      2k_earth_daymap.png
      2k_moon.png
      balldimpled.png
    models/
      robotIdle.glb
```

## Adding real 3D models (GLB workflow)

The viewer loads GLB models automatically — just drop the file in the right place.

**Step 1 — Convert FBX to GLB** using Blender or a CLI tool:

```powershell
# Blender CLI (adjust path for your Blender install)
& "C:\Program Files\Blender Foundation\Blender 4.x\blender.exe" --background --python-expr `
  "import bpy; bpy.ops.import_scene.fbx(filepath='robotIdle_modelimport.fbx'); bpy.ops.export_scene.gltf(filepath='robotIdle.glb', export_format='GLB')"
```

Or use the Blender GUI: File → Import → FBX, then File → Export → glTF 2.0, format = GLB.

**Step 2 — Name the output file** to match the Unity prefab name exactly (case-sensitive):

| Unity prefab | GLB filename |
| --- | --- |
| `robotIdle` | `robotIdle.glb` |
| `GenericBRB` | `GenericBRB.glb` |
| `BigRedButton` | `BigRedButton.glb` |

**Step 3 — Drop into `assets/models/`**:

```text
arlem_preview_toolkit/
  assets/
    models/
      robotIdle.glb
      GenericBRB.glb
```

**Step 4 — Reload the page.** The viewer checks `assets/models/<prefabName>.glb` for every object. If found, the placeholder geometry is replaced with the real model (static bind pose — no animation). Models that don't exist simply keep their placeholder.

## Prefab placeholder geometry

When no GLB is present, the viewer shows:

| Prefab pattern | Placeholder |
| --- | --- |
| `*sun*` | Yellow emissive sphere (r=0.5) + point light |
| `*sphere*`, `*basketball*`, `*clickable*`, `*moveable*` | Sphere r=0.5, textured if texture specified |
| `*tiny*` | Small red sphere r=0.08 |
| `*robot*` | Capsule body + sphere head (blue tint) |
| `*brb*`, `*button*` | Red cylinder r=0.07 h=0.15 + disc cap |
| `*cube*` | BoxGeometry 1×1×1 |
| `*plane*` | PlaneGeometry 1×1 (double-sided) |
| `*text*` | Canvas-rendered text panel |
| anything else | Grey box |

## Coordinate notes

The viewer defaults to a direct visual mapping of Unity-like coordinates:
`x -> x`, `y -> y`, `z -> z`.

There is also a "Flip Z" checkbox for a right-handed/export-style view:
`x -> x`, `y -> y`, `z -> -z`.

For rough screenshots and scale checks, the direct mapping is often easiest to interpret.

## Python utility

Inspect the JSON:

```powershell
python arlem_to_preview_scene.py Optomized_moon_lab_final.json
```

Export a normalized static scene for module 1, clip 5:

```powershell
python arlem_to_preview_scene.py Optomized_moon_lab_final.json --module 1 --clip 5 --out preview_scene.json
```

Use `--flip-z` if you want exported coordinates with the Z axis inverted.

## Good next enhancements

1. Add a `prefab_map.json` file that maps Unity prefab names to placeholder types or GLB files.
2. Add `GLTFLoader` support for actual exported Unity models.
3. Add batch screenshot generation using Playwright.
4. Add camera presets: front, top, side, student view, close-up object view.
5. Add script/component annotation panels so object interactions are visible to an LLM.

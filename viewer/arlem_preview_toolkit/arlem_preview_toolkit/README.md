# ARLEM / Unity JSON Preview Toolkit

A browser-based 3D preview system for ARLEM-style AR lab JSON scenes, with a Python CLI for headless screenshot capture.

---

## Installation

### 1. Python environment

```powershell
conda activate ar-experiments
```

Or with a plain venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Python dependencies

```powershell
pip install -r requirements.txt
```

`requirements.txt` is in the `arlem_preview_toolkit/` folder (one level up from this file).

### 3. Playwright + Chromium browser (required for screenshot CLI)

```powershell
python -m playwright install chromium
```

This downloads a self-contained Chromium binary used by `screenshot_cli.py`. It does **not** affect your system browser. Re-run this if Playwright upgrades.

---

## Files

| File | Purpose |
| --- | --- |
| `viewer2.html` | Main interactive viewer (use this) |
| `screenshot_cli.py` | Headless screenshot CLI (Playwright) |
| `moon_lab_with_buttons.json` | Default lab JSON — includes nav button objects |
| `Optomized_moon_lab_final.json` | Processed lab JSON — buttons stripped |
| `assets/models/` | GLB models keyed by Unity prefab name |
| `assets/textures/` | Textures referenced by JSON |
| `prefab_to_glb.py` | Converts FBX/Unity primitives to GLB |
| `arlem_to_preview_scene.py` | Inspects / exports normalized scene JSON |

---

## Interactive viewer

Run a local web server from this folder:

```powershell
python -m http.server 8000
```

Open `http://localhost:8000/viewer2.html`. The default JSON (`moon_lab_with_buttons.json`) auto-loads.

### Controls

| Control | Action |
| --- | --- |
| Left-drag | Orbit |
| Scroll wheel | Zoom |
| Right-drag | Pan |
| `‹` / `›` buttons | Step previous/next clip or module |
| Reset camera | Return to default viewpoint |
| Save screenshot | Download PNG of current view |
| Show room/grid | Toggle floor grid and reference human |
| Show object labels | Toggle name labels on objects |
| Flip Z | Mirror Z axis for right-handed/export view |

### JSON schema

The viewer reads the v2 processed schema produced by `convert_lab_json_with_buttons.py`:

- `position`, `eulerAngles`, `scale` as `[x, y, z]` arrays
- `prefab` (not `type`), `parent` (not `parentName`)
- `components` array with `componentType` discriminator
- Clip `changes` are sparse deltas — only changed fields per object

### Clip playback

Clips are **cumulative**: clip N represents the state after applying clips 0 through N in order. The viewer correctly accumulates state — selecting clip 5 automatically replays clips 0–4 first.

### Coordinate system

Unity coordinates map directly: `x→x`, `y→y`, `z→z`. Use the "Flip Z" checkbox for a right-handed/export view (`z→-z`).

### GLB model loading

The viewer attempts to load `assets/models/<prefabName>.glb` for every scene object. If found, the placeholder geometry is replaced with the real GLB model.

**Unity humanoid GLBs** (Mixamo/Armature rig exported from Unity) require special handling — no `.clone()`, root scale normalization via bounding-box measurement, and a +90° X rotation correction. This is handled automatically by detecting the `Armature` node.

**Prefab placeholder geometry** (when no GLB exists):

| Prefab pattern | Placeholder |
| --- | --- |
| `*sun*` | Yellow emissive sphere + point light |
| `*sphere*`, `*basketball*`, `*clickable*`, `*moveable*` | Sphere r=0.5, textured if JSON specifies |
| `*tiny*` | Small red sphere r=0.08 |
| `*robot*` | Capsule + sphere head (blue tint) |
| `*brb*`, `*button*` | Red cylinder + disc cap |
| `*text*` | Canvas-rendered text plane |
| `*cube*` | BoxGeometry 1×1×1 |
| `*plane*` | PlaneGeometry 1×1 (double-sided) |
| anything else | Grey box |

---

## Headless screenshot CLI

Captures screenshots programmatically using Playwright (headless Chromium). Designed for AI quality-assessment pipelines.

See the [Installation](#installation) section above for one-time setup.

### Usage

```text
python screenshot_cli.py [OPTIONS]

Options:
  --json PATH              Lab JSON file (default: moon_lab_with_buttons.json)
  --module INT|STR         Module index (0-based) or substring of module name
  --clip INT               Single clip index (-1 = base layout)
  --clips INT INT          Inclusive clip range: --clips 0 5 captures clips 0,1,2,3,4,5
  --camera PX PY PZ TX TY TZ   Camera position + look-at target (6 floats)
  --out DIR                Output directory (default: screenshots/ next to the JSON file)
  --prefix STR             Filename prefix
  --width INT              Viewport width (default: 1280)
  --height INT             Viewport height (default: 720)
  --cleanup                Delete existing images for this module before capturing
  --cleanup-all            Delete ALL PNG files in the screenshots directory, then exit
  --list                   Print module table and exit (no browser launched)
```

### Examples

```powershell
# List all modules and clip counts
python screenshot_cli.py --list

# Base layout of module 1
python screenshot_cli.py --module 1 --clip -1

# Full clip range, top-down view
python screenshot_cli.py --module 1 --clips 0 8 --camera 0 8 0 0 0 1.5

# Module by name substring
python screenshot_cli.py --module "Synchronous" --clip 3

# Custom JSON, cleanup old images first
python screenshot_cli.py --json path/to/my_lab.json --module 0 --clips 0 5 --cleanup

# High-res output
python screenshot_cli.py --module 1 --clip 5 --width 1920 --height 1080 --out ./eval

# Wipe the entire screenshots directory
python screenshot_cli.py --cleanup-all

# Wipe a custom output directory
python screenshot_cli.py --cleanup-all --out ./eval
```

### Output filenames

```text
m{idx:02d}_{ModuleName}_c{clip:03d}_{clipName}.png
```

Examples:

```text
m01_Synchronous_Rotation_c003_clip3.png
m07_Introduction_to_the_Earth_and_Moon_c005_clip5.png
m01_Synchronous_Rotation_base_base.png   <- clip -1 (base layout)
```

### Cleanup scope

| Flag | What is deleted |
| --- | --- |
| `--cleanup` | Only `*m{idx:02d}_{ModuleName}_*.png` for the current module — safe when multiple modules share one folder |
| `--cleanup-all` | Every `*.png` in the output directory — wipes the slate entirely, then exits without capturing |

### Clip state

The CLI uses the same cumulative playback as the interactive viewer. Capturing clip 5 automatically includes the state changes from clips 0–4. No manual stepped playback needed.

---

## Adding GLB models

**Step 1 — Convert FBX to GLB** (Blender CLI):

```powershell
& "C:\Program Files\Blender Foundation\Blender 4.x\blender.exe" --background --python-expr `
  "import bpy; bpy.ops.import_scene.fbx(filepath='robotIdle_modelimport.fbx'); bpy.ops.export_scene.gltf(filepath='robotIdle.glb', export_format='GLB')"
```

Or use `prefab_to_glb.py` which automates this for all prefabs.

**Step 2 — Name the file** to match the Unity prefab name exactly (case-sensitive) and place in `assets/models/`:

| Unity prefab | GLB filename |
| --- | --- |
| `robotIdle` | `robotIdle.glb` |
| `GenericBRB` | `GenericBRB.glb` |
| `demoButtonPrefab` | `demoButtonPrefab.glb` |

**Step 3 — Reload the viewer.** Placeholder geometry is replaced automatically on load.

---

## Data pipeline

```text
Raw JSON (Artifacts/Data/Raw/Full_Lab_Transmission.json)
  |  convert_lab_json.py              -> Optomized_moon_lab_final.json  (buttons stripped)
  |  convert_lab_json_with_buttons.py -> moon_lab_with_buttons.json     (buttons kept)
  |  viewer2.html / screenshot_cli.py -> 3D preview / PNG screenshots
```

Conversion scripts live in `../../Code/Data Processing/`.

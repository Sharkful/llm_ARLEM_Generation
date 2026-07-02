# Intake folder — manual external asset drop (Stage 6a)

To bring an externally sourced model into the library:

1. Verify the license **before** downloading. Allowed: `CC0`, `CC-BY`
   (attribution author must be recorded), `internal`, `proprietary-cleared`.
   Anything else does not enter the library — no exceptions without updating
   the allowlist in `pipeline/external_intake.py` after a human verifies the
   terms.
2. Download the file and place it in `intake/<asset_id>/` — exactly **one**
   model file per folder (`.stl`, `.obj`, `.fbx`, `.glb`, or `.gltf`).
3. Create `intake/<asset_id>/source.json` by hand:

```json
{
  "display_name": "Wooden lab stool",
  "license": "CC0",
  "source_site": "polyhaven.com",
  "source_url": "https://polyhaven.com/a/wooden_stool",
  "original_author": "Jane Modeler",
  "target_size_m": 0.45,
  "pivot": "base_center",
  "tags": ["furniture", "stool", "wood"],
  "semantic_type": "stool",
  "pedagogically_critical": false,
  "notes": ""
}
```

`display_name`, `license`, and `target_size_m` are required. `target_size_m`
is the real-world largest dimension in meters — nothing enters the catalog
with unknown scale. Use `"pivot": "base_center"` for objects that stand on
a surface.

4. Run:

```bash
python cli.py intake <asset_id>
```

This checks the license, normalizes the mesh through the Stage 4 Blender
pass (scale, pivot, triangulation, decimation if oversized), copies the
original + a LICENSE.txt + meta.json into `library/imported/<asset_id>/`,
and appends a catalog entry with full provenance (`license_status:
"approved"`, review level 3 — or 4 if `pedagogically_critical`). Any
rejection prints the specific reason; fix `source.json` and re-run.

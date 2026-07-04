# Asset Pipeline — User Guide

This guide is for *authors*: people building AR lab scenes who need 3D
assets. It covers the day-to-day workflows. (Setup, testing, and
architecture live in [README.md](README.md); the design rationale lives in
[the implementation plan](../3d_objects/asset_pipeline_implementation_plan.md).)

All commands run from the `asset_pipeline/` folder, inside its Python
environment (conda env `asset_pipeline` or the venv from README setup).

---

## 1. The mental model

The pipeline turns **a sentence** ("a small gray moon, 15cm across") into
**a catalogued, scaled, textured, Unity-ready asset**, choosing the
cheapest path that works:

1. **Catalog match** — an existing asset already fits. Nothing is built.
2. **Variant** — an existing asset fits with a different material/size.
3. **Composite** — the object is an arrangement of existing assets
   (a water molecule = three spheres).
4. **Parametric** — mechanical/geometric shapes (brackets, stands, dials)
   are generated as OpenSCAD scripts, compiled, and normalized.
5. **Imported** — organic/complex objects (furniture, realistic props) are
   flagged for you to source from the internet, license-gated.

Textures follow the same philosophy (archive first, then generate or
source), and **everything is remembered**: once the Moon map or a rust
texture is in the library, later requests reuse it — nothing is
re-downloaded or re-generated.

Two invariants the pipeline enforces for you:

- **Nothing enters the catalog with unknown scale.** Every asset records
  its real-world bounds in meters.
- **Nothing enters the library with unknown license.** Imported files need
  a recorded, allowlisted license before they're accepted.

## 2. First run on any machine

```bash
python cli.py doctor          # verifies OpenSCAD, Blender, and API keys
python cli.py catalog list    # what geometry exists
python cli.py textures list   # what textures exist
python cli.py validate        # library health check
```

If `doctor` reports a missing tool, install it (OpenSCAD from openscad.org,
Blender desktop from blender.org — the Microsoft Store Blender does *not*
work headless) or point `.env` at it. Missing API keys go in
`asset_pipeline/.env`.

## 3. Making a single asset (the app)

```bash
python cli.py review
```

This opens the local web app (nothing leaves your machine except LLM/API
calls). Three tabs:

- **Library** — browse every catalog asset with thumbnails, filter by
  class/tag, click to inspect in a live 3D viewport, and **Approve**
  (records who/when — required before an asset counts as human-reviewed).
- **Create** — type a description, hit *Generate*. What you get back
  depends on the route: an existing asset, a generated part with an
  **editable parameter grid** (edit numbers → *Recompile* — instant, no AI
  call), a **tweak box** ("make the base twice as wide" → *Regenerate*),
  and *Save to catalog* when you're happy. Textured surfaces (the Earth, a
  rusty panel) appear textured in the viewport.
- **Worklist** — paste a list of descriptions (one per line, or a JSON
  array), then step through them: *Generate next* → inspect → tweak →
  *Save* or *Skip* → next. You can also go **backwards**: click any item
  (or *◀ Back*) to select it, then *Regenerate current (alternative)* to
  produce a fresh take — even for items already saved. Progress survives
  restarts. This is the production workflow for building out a lab's full
  asset set.

While anything generates, the panel under the viewport streams each
pipeline step live ("Checking local library for possible matches…",
"Compiling with OpenSCAD…", "Normalizing mesh in Blender…"). Saved
parametric assets keep their OpenSCAD script and parameters permanently —
select one in the Library and click **Edit parameters** to reopen the
editor at any time. Saving a *binding* (an object that reuses existing
geometry, like a planet) records your approval (who/when) on the object
itself via the same Save button, relabeled **Approve binding**.

## 4. Making a whole scene (batch)

Put your scene's asset list in a JSON file — strings, spec objects, or a
mix (see [examples/scene_example.json](examples/scene_example.json)):

```json
[
  "a small gray moon with visible craters, about 15cm across",
  { "object_id": "stand", "description": "a four-legged stand", "desired_size_m": 0.4 }
]
```

Then:

```bash
python cli.py resolve-scene my_scene.json          # resolve everything
python cli.py resolve-scene my_scene.json --sync   # ...and copy into the viewer
```

Every item is resolved end-to-end; a **build report**
(`my_scene.build_report.json`) records each outcome:

| status | meaning | what you do |
|---|---|---|
| `resolved_existing` | matched a catalog asset | nothing |
| `composite` | decomposed into existing parts | nothing |
| `saved_parametric` | generated + auto-cataloged (review level 2) | eyeball it in the app when convenient |
| `needs_review` | pipeline stopped on purpose | see §5/§6 — source a file or inspect in the app |
| `failed` | hard error | read the message, fix, re-run |

Exit code 0 = everything clean; **2 = some items need you** (distinct from
1 = hard failure). The pipeline never quietly substitutes a wrong-but-
present asset — anything uncertain is flagged, per the no-silent-fallback
rule. Re-running the same scene is cheap: already-resolved items hit the
catalog.

### What resolve-scene actually writes (where to look)

| artifact | location | notes |
|---|---|---|
| build report | `<scene>.build_report.json` | one entry per object: status + what it resolved to + why |
| the scene object itself | `library/generated/<object_id>/draft.json` | which catalog asset it uses + its bound material |
| its material | `library/materials/<object_id>_mat.json` | shader values + texture reference |
| new geometry | `library/generated/<object_id>/model.glb` + a catalog entry | **only** for generated (parametric) objects |

Important: an object that resolves to existing geometry (the Moon =
`sphere_basic` + the `moon_surface` texture) does **not** create a new
catalog entry or preview thumbnail — the object is a *binding*, recorded in
its draft. To see it, open `cli.py review` → Library tab → **"Scene objects
& drafts"** section at the bottom: every scene object and work-in-progress
is listed there; click one to view it textured in the 3D viewport.

## 5. Bringing in external models (imported route)

When an item is classified `imported` (a couch, realistic lab equipment),
the pipeline does the searching for you:

1. **Confident match with a known size → downloaded automatically.**
   Two sources are searched: Poly Haven (CC0 — props, furniture, nature)
   and NASA 3D Resources (public domain — spacecraft and NASA hardware,
   via the official github.com/nasa/NASA-3D-Resources repository). A
   confident hit is fetched, license-gated, normalized, and cataloged with
   no action from you — e.g. "the Apollo lunar excursion module" resolves
   by itself to NASA's official LM model.
2. **Possible matches → candidates in the app.** The draft (Library tab →
   Scene objects & drafts) shows each candidate with a thumbnail, license,
   confidence, and a link to the source page. Enter the real-world size
   and click **Approve & download** — one click does fetch + intake +
   resolve.
3. **No match → specific links.** The draft carries direct search URLs
   for verified sources (Poly Haven, NASA 3D Resources, Smithsonian 3D).
   Download by hand into `intake/<asset_id>/` with a `source.json` and run
   `python cli.py intake <asset_id>` — see [intake/README.md](intake/README.md).

The CLI equivalents still exist (`search-external`, `fetch-external`,
`intake`). Licenses accepted: CC0, CC-BY (with author recorded),
public-domain, internal, proprietary-cleared — anything else is rejected
with the reason, and attribution-required licenses are never
auto-downloaded.

Every candidate is automatically checked against your description before
it's ever shown — first by name/context, then (for whatever survives) by
its actual picture — so a source that merely shares a word with your
request (a NASA asteroid feature nicknamed "Snowman Craters" is not a
snowman) gets filtered out instead of presented as a match. Candidate cards
show a **✓ reviewed** badge and the reviewer's one-line reasoning; a
candidate that failed review is never shown at all.

### If the result still isn't right

Composites (a snowman built from spheres/cylinders/a cone, a molecule)
cover a lot more than you'd expect — the pipeline builds up to 12 primitive
parts before giving up on that route. If a draft still lands somewhere
wrong (`needs_human` with bad or no candidates, or an odd parametric
attempt), an **override panel** appears on the draft with three options:
*Build with OpenSCAD instead*, *Build as composite instead*, and
*Search again* with your own keywords typed into the box. Each bypasses
whatever the pipeline decided the first time and tries that path directly
— you don't have to re-describe the object, and you're never stuck with
"the pipeline said no."

## 6. Bringing in real textures (planets, basketballs, wood…)

When a request names a real-world surface (Earth, the Moon) and no library
texture matches, the item reports **texture pending** with a suggested
search query — the pipeline will *not* fake a real surface. Fix it once,
benefit forever:

```bash
python cli.py search-external "wood planks" --type texture   # Poly Haven CC0
python cli.py fetch-external wood_planks_grey --type texture
python cli.py intake-texture wood_planks_grey
```

For planetary maps, download by hand from a verified source (NASA SVS,
USGS Astrogeology, Solar System Scope — list and `source.json` example in
[intake/README.md](intake/README.md)) into `intake/<texture_id>/`, then
`intake-texture`. Equirectangular maps (the standard 2:1 planet format)
work on the built-in sphere automatically — that's how "the planet earth"
renders with real continents. After intaking, regenerate the asset (app)
or re-run the scene, and the new texture is found by the archive matcher.

**UV maps are separate from textures on purpose**: an object's UV layout
is established once (built-ins ship with one; generated meshes get
unwrapped automatically, with a layout image saved next to the model as
`uv_layout.png`), so giving the same object a different surface is just a
different texture — no mesh work.

## 7. Checking and shipping

```bash
python cli.py validate        # schema, files, scale, licenses, textures; exit 0 = deployable
python cli.py preview --all   # thumbnails + library/previews/contact_sheet.html
python cli.py sync --all      # copy the whole resolved library into the viewer
python cli.py sync moon_demo equipment_stand --target path/to/unity/Assets  # a subset, elsewhere
```

`validate` must pass with zero errors before a bundle ships; warnings are
allowed but recorded. `sync` is copy-only — the library stays the source
of truth.

## 8. Where things live

```text
asset_pipeline/
  library/catalog.json        the asset registry (id, bounds, pivot, UVs, provenance, review level)
  library/textures/index.json the texture registry (mapping type, maps, provenance)
  library/generated/<id>/     OpenSCAD source, STL, model.glb, uv_layout.png, draft.json
  library/imported/<id>/      original file, model.glb, LICENSE.txt, meta.json
  library/materials/          material JSONs (+ procedural textures)
  library/previews/           thumbnails + contact_sheet.html
  intake/                     drop folder for human-sourced files
  examples/scene_example.json a 5-object scene exercising every route
```

## 9. Troubleshooting

- **"Blender/OpenSCAD binary not found"** — run `doctor`; install the
  desktop builds; MS Store Blender can't run headless.
- **"no API key"** — set `ANTHROPIC_API_KEY` (or another provider's) in
  `.env`; switch provider with `--provider` or `ASSET_PIPELINE_LLM_PROVIDER`.
- **A texture I intook isn't being picked up** — check
  `python cli.py textures list`; the matcher needs word overlap with your
  description *and* geometric fit (an equirect map only fits spheres). Add
  tags/semantic_type to the texture's index entry, or lower
  `MATCH_THRESHOLD` in `pipeline/texture_matcher.py`.
- **Generated part looks wrong** — open it in `review`, edit parameters
  (instant) or send a tweak instruction (one AI call), then Save.
- **Something claims to exist but validate fails** — the validator's
  message names the asset and field; fix the file or entry it names and
  re-run.

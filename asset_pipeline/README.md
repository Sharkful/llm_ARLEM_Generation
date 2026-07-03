# Asset Pipeline

Offline AR asset acquisition pipeline: turns a sentence-to-paragraph object
description into a scaled, catalogued, Unity-ready asset. See
[`3d_objects/offline_ar_asset_pipeline_requirements.md`](../3d_objects/offline_ar_asset_pipeline_requirements.md)
for the full requirements and
[`3d_objects/asset_pipeline_implementation_plan.md`](../3d_objects/asset_pipeline_implementation_plan.md)
for the staged build plan this module implements.

This is a **semi-independent sub-project**: it has its own dependencies,
environment file, and test suite, and can be set up without touching the
rest of the monorepo.

## Status

Stage 0 (scaffolding, models, catalog read/write, seed catalog, multi-provider
LLM client factory), Stage 1 (free-text description -> validated `AssetSpec`
via `cli.py spec`), Stage 2 (classifier/resolver routing an `AssetSpec` to
catalog_match / variant / composite / parametric / imported via
`cli.py resolve`), Stage 3 (OpenSCAD-based parametric generation with
compile-failure repair and bounds-divergence checking via
`cli.py generate-parametric`), Stage 4 (Blender-based mesh
normalization -- STL/OBJ/FBX/GLB -> pivoted, scaled, decimated-if-needed
GLB, via `cli.py normalize-mesh`), and Stage 5 (description -> `MaterialDef`
JSON plus locally synthesized procedural textures -- noise/craters/stripes/
grid/gradient/rust via numpy+Pillow -- via `cli.py generate-material`), and
Stage 6a (manual external intake: license-gated, provenance-recorded
normalization of a human-provided model file from `intake/<asset_id>/`
into `library/imported/` + the catalog, via `cli.py intake` -- see
`intake/README.md` for the drop-folder workflow), and Stage 6b (Poly Haven
search assist: `cli.py search-external` lists CC0 models read-only,
`cli.py fetch-external <id>` downloads only the one asset a human
explicitly names into the intake folder with a pre-filled source.json --
Poly Haven is the entire source allowlist by design), and Stage 7
(executable validation of the catalog + library per req. doc section 12:
schema, completeness, scale/geometry, licensing/provenance, and runtime
readiness via `cli.py validate`, with colorized screen output or `--json`,
exit 0 only when error-free), and Stage 8 (batch preview thumbnails:
`cli.py preview <id>` / `preview --all` renders each catalog asset --
GLBs and Unity-style primitives alike -- to `library/previews/<id>.png`
via Playwright + `webapp/asset_preview.html`, plus a contact_sheet.html
for fast human scanning), and Stage 8b (local Flask review/creation app
via `cli.py review`: a Library browser with live three.js inspection and
human approval stamps, a Create mode -- type a description, generate,
edit OpenSCAD parameters or send a free-text tweak, save to the catalog
-- and a Worklist mode that steps through a pasted list of descriptions
one by one for production authoring, with progress persisted across
restarts; all orchestration lives in `pipeline/asset_factory.py`, which
Stage 9's batch flow will reuse) are implemented. Stages 8/8b need
`playwright` + chromium (`pip install playwright && playwright install
chromium`), `Flask`, and network access to the three.js CDN (same one
`viewer2.html` uses). Catalog matching uses
tag/token/fuzzy string matching only (no embeddings yet -- see
`pipeline/catalog_matcher.py` docstring). Resolver output for `parametric`
and `imported` results still needs to be wired to Stage 3/4's generator
and Stage 6a's intake respectively -- both are currently always flagged
for human review by `pipeline/resolver.py`. Later stages (preview
rendering, the Flask review app, and sync) are not yet built -- see the
implementation plan for what's next.

## Setup

```bash
cd asset_pipeline
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env        # fill in whichever provider key(s) you use
```

No external tool binaries (OpenSCAD, Blender) are required for Stage 0-2.
Stage 3 (`generate-parametric`) needs OpenSCAD; Stage 4 (`normalize-mesh`)
needs Blender (4.0+; the mesh import step tries the modern
`bpy.ops.wm.*_import` operators first and falls back to the legacy
`import_mesh`/`import_scene` namespace). `config.py` auto-detects both from
common install locations if installed, and never fails at import time if
they're missing -- only the stages that actually shell out to them will
error, and only when invoked. Install OpenSCAD from openscad.org (or
`winget install OpenSCAD.OpenSCAD` / `brew install openscad`) and Blender
from blender.org (or `winget install BlenderFoundation.Blender` / `brew
install --cask blender`) if a command reports it can't find the binary;
override the detected paths with `ASSET_PIPELINE_OPENSCAD_BIN` /
`ASSET_PIPELINE_BLENDER_BIN` in `.env` if needed.

When moving to a new machine, run `python cli.py doctor` first: it reports
which tool binaries were detected (and actually runs each one, since a
binary can exist on disk but be unlaunchable -- e.g. the MS Store Blender's
`blender.exe` is blocked by WindowsApps ACLs and cannot be used headless;
install the desktop build instead) and which LLM API keys are set.

## Running tests

```bash
cd asset_pipeline
pytest tests/unit          # fast, no API keys or external tools required
pytest tests               # full suite; integration tests self-skip if a
                            # required tool/key isn't present on this machine
```

Every pipeline stage's CLI subcommand is independently runnable -- no stage
requires another stage's output to exist first. This is what keeps the unit
test tier fast and keeps development of one stage from blocking on another.

## CLI

```bash
python cli.py doctor         # check tool binaries + API keys on this machine
python cli.py catalog seed   # (re)write the seed catalog of core primitives
python cli.py catalog list   # list all current catalog entries
python cli.py spec "a small gray moon with visible crater texture, about 15cm across"
python cli.py spec "..." --out spec_moon.json --provider openai --model gpt-4o-mini
python cli.py resolve "a small gray moon with visible crater texture"
python cli.py resolve --spec-file spec_moon.json   # skips Stage 1, resolves a pre-parsed spec
python cli.py generate-parametric bracket_01 "an L-shaped mounting bracket, 5cm wide"
python cli.py normalize-mesh bracket_01 library/generated/bracket_01/source.stl --format stl --size 0.05
python cli.py normalize-mesh bracket_01 ... --format stl --size 0.05 --pivot base_center
python cli.py generate-material "rough red rust with visible texture"
python cli.py generate-material "dull gray rock" --id moon_surface --force
python cli.py intake wooden_stool   # normalize + catalog intake/wooden_stool/ (see intake/README.md)
python cli.py search-external "wooden table"                    # Poly Haven search (read-only)
python cli.py fetch-external small_wooden_table_01 --size 0.6   # download YOUR pick into intake/
python cli.py validate               # library health check; exit 0 only if error-free
python cli.py validate --json        # machine-readable report
python cli.py validate --records resolution_records.json   # also check for unresolved specs
python cli.py preview sphere_basic --force   # render one thumbnail
python cli.py preview --all                  # render all + library/previews/contact_sheet.html
python cli.py review                         # start the local review/creation web app
```

## LLM provider configuration

All LLM calls go through `llm/client_factory.py`, which wraps `anthropic`,
`openai`, or `google-genai` clients with `instructor` for structured output.
Provider and model are config-level choices (`ASSET_PIPELINE_LLM_PROVIDER`,
`ASSET_PIPELINE_LLM_MODEL` in `.env`, or passed explicitly per call) --
pipeline code never imports a provider SDK directly. Default is Anthropic
`claude-sonnet-4-6`, matching `narration_generator`'s default, but any call
site can be pointed at a different provider/model (e.g. a cheaper model for
high-volume classification calls) without code changes.

## Layout

```text
asset_pipeline/
  config.py          # env loading, paths, provider/model defaults, tool binary detection
  cli.py              # CLI entry point
  llm/                # multi-provider instructor client factory
  models/             # Pydantic models (catalog, authoring spec, generation, logging)
  pipeline/           # stage implementations
  library/            # the authoritative asset library (catalog.json + generated/imported/composites/materials/previews)
  intake/             # human-provided external asset drop folder (Stage 6)
  tests/
    unit/             # no network, no API keys, no external tools required
    integration/      # requires OpenSCAD/Blender/network/API keys; self-skips if unavailable
    fixtures/         # small static test fixtures
```

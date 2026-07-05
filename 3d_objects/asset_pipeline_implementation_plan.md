# Asset Acquisition Pipeline — Implementation Plan

## 0. Purpose and relationship to the requirements doc

[offline_ar_asset_pipeline_requirements.md](offline_ar_asset_pipeline_requirements.md) defines *what* the
system must do (authoring/runtime split, asset catalog, resolver, validation, review levels). This document
defines *how* to build it, in stages, as concrete Python modules a coding agent can implement one at a time.

It answers the hard problem the requirements doc leaves open (section 20.3): given a sentence-to-paragraph
description of an object, how does software actually produce a scaled, textured, Unity-ready asset?

**Decisions locked in for this plan** (do not revisit without updating this doc):

- Generation stack: **OpenSCAD primary**. It is a single small CLI binary, fully scriptable, deterministic,
  and LLMs are reliably good at writing its declarative CSG language. **Blender (`bpy`, headless)** is used
  only as a second-stage tool: FBX/OBJ→GLB conversion (already implemented in
  [prefab_to_glb.py](../viewer/arlem_preview_toolkit/arlem_preview_toolkit/prefab_to_glb.py)), mesh cleanup
  (decimate, recalculate normals, apply transforms), UV unwrap, and texture bake — not as a script-generation
  target for the LLM.
- External sourcing: **manual-first**, with a **semi-automated search stage added once manual works**. A
  human downloads from an approved source and drops the file into an intake folder; the pipeline normalizes,
  converts, and catalogs it. Stage 6 below adds an optional search assist against a small allowlist of APIs
  (Poly Haven confirmed CC0; others added only after license terms are verified), but a human still approves
  every download before it enters the intake folder.
- Asset storage: **new top-level asset library**, not the viewer's asset folder. `asset_pipeline/library/` is
  the source of truth. A sync step copies/builds the resolved subset into
  `viewer/arlem_preview_toolkit/arlem_preview_toolkit/assets/` (and later a Unity `Assets/` folder) — the
  same role `prefab_to_glb.py` already plays for prefab→GLB, just generalized.
- LLM default: **`claude-sonnet-4-6`**, matching `narration_generator`'s `DEFAULT_LLM_MODEL`
  ([config.py:19](../narration_generator/config.py#L19)), but **multi-provider from the start**, not
  Claude-only. Every LLM call goes through `instructor`, which this repo already uses across three providers
  (`instructor.from_openai()`, `instructor.from_anthropic()`, `instructor.from_google(use_async=False)` —
  see `CLAUDE.md`'s "LLM Integration" section and `requirements.txt`'s `instructor[google-genai]` /
  `instructor[anthropic]` extras). The asset pipeline reuses that same seam rather than hardcoding Anthropic:
  a `get_instructor_client(provider: str)` factory in `config.py` returns the right wrapped client, and
  every `pipeline/*_generator.py` function takes a `provider` + `model` pair (config-defaulted, overridable
  per call) instead of assuming Anthropic. This matters for this pipeline specifically because OpenSCAD/JSON
  generation is exactly the kind of structured, high-volume, low-ambiguity call where a cheaper OpenAI or
  Gemini model is worth being able to drop in without touching call sites. Retries use `tenacity`, same
  pattern as [brief_generator.py](../narration_generator/pipeline/brief_generator.py).

## 1. Where this lives in the repo

`asset_pipeline/` is a **semi-independent sub-project** inside this repo, not just a package. Treat it the
way `narration_generator/` is already treated: it has its own `requirements.txt`, its own `README.md`, its
own `.env`/`.env.example`, and its own `pytest` test suite that can run without the rest of the monorepo
being set up. Someone should be able to `cd asset_pipeline`, install just that folder's dependencies, and run
its tests or CLI without touching `Code/Tools` or `narration_generator` at all. It still lives inside the
larger GitHub project and eventually feeds JSON/assets to it, but it is developed and versioned as its own
unit.

```text
asset_pipeline/
  README.md                    # what this sub-project is, setup, CLI usage, how to run tests standalone
  requirements.txt             # this sub-project's own dependency list (superset it needs beyond the root repo's)
  .env.example                 # ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY placeholders + tool paths
  pytest.ini                   # standalone test config, mirrors narration_generator/pytest.ini
  config.py                    # API keys, paths, default provider/model, tool binary paths
  cli.py                       # entry point: `python -m asset_pipeline.cli resolve <scene.json>`
  llm/
    __init__.py
    client_factory.py          # get_instructor_client(provider) -> instructor-wrapped client (openai/anthropic/google)
  models/
    __init__.py
    catalog_models.py          # AssetCatalogEntry, MaterialDef, ProvenanceInfo (Pydantic, per req. doc §5.3, §9, §11)
    spec_models.py             # AssetSpec (authoring), ResolvedAsset, ResolutionRecord (per req. doc §5.2, §6)
    generation_models.py       # OpenSCADPlan, GenerationResult — structured LLM output for instructor
    log_models.py               # mirrors narration_generator's LLMCallEntry for cost tracking
  pipeline/
    classifier.py               # Stage 2: classify AssetSpec -> asset class (primitive/variant/composite/parametric/imported)
    catalog_matcher.py          # Stage 2: embedding/tag search against the catalog
    resolver.py                 # Stage 2 orchestrator: produces ResolutionRecord
    openscad_generator.py       # Stage 3: LLM -> OpenSCAD script -> STL
    mesh_processor.py           # Stage 4: STL/OBJ/FBX -> GLB, decimate, normalize scale/pivot/orientation (Blender headless)
    material_generator.py       # Stage 5: LLM -> material JSON (+ optional procedural texture)
    external_intake.py          # Stage 6: normalize a human-provided file (or semi-auto search hit) into the library
    catalog_writer.py           # writes/updates library/catalog.json, enforces schema + uniqueness
    validator.py                # Stage 7: schema, completeness, scale, license checks (per req. doc §12)
    preview_renderer.py         # Stage 8: reuse screenshot_cli.py / viewer2.html pattern for per-asset thumbnails
    sync.py                     # copies resolved library subset into a target consumer folder (viewer or Unity)
  library/
    catalog.json                # authoritative AssetCatalogEntry[] registry
    generated/                  # OpenSCAD-sourced assets: <asset_id>/{source.scad, source.stl, model.glb, meta.json}
    imported/                   # externally sourced assets: <asset_id>/{original.*, model.glb, meta.json, LICENSE.txt}
    composites/                 # composite prefab-fragment definitions (JSON, no new geometry)
    materials/                  # material_id.json definitions + procedural texture outputs
    previews/                   # per-asset thumbnail renders
  intake/
    README.md                   # instructions for manually dropping approved external files here
  tests/
    fixtures/                   # small static fixtures: sample .scad/.stl/.glb files, sample catalog.json
    unit/                       # no external tools/network required — see §1.1
    integration/                # requires OpenSCAD/Blender/network — see §1.1
```

`Code/Tools/arlem_full.py` and `pydantic_json_lab_claude.py` are **not modified**. This pipeline is a
producer of catalog entries and resolved `asset_id`s that those existing scene models already reference
(`prefab` fields, `componentsToAdd`, etc.). Integration is at the JSON boundary, not the Python import
boundary — keeps this module independently testable and avoids coupling asset generation to the ARLEM scene
schema's release cycle. `asset_pipeline/requirements.txt` can simply include everything the root
`requirements.txt` already has plus its own additions (§3) — duplication here is fine and preferred over a
shared file, since it's what lets the sub-project be set up in isolation.

### 1.1 Testing outside the full pipeline

A coding agent should be able to verify each stage's logic without OpenSCAD, Blender, or a live LLM installed
— those are the pieces most likely to be missing or flaky in a fresh environment (CI, a new machine, a
sandbox). Split tests into two tiers from Stage 0 onward:

- **`tests/unit/`** — no subprocess calls, no network, no API keys required. Covers: Pydantic model
  validation (catalog/spec/generation models), `catalog_matcher`'s tag/string matching path, `classifier`'s
  routing decision tree (feed it hand-built `AssetSpec`s and pre-computed match scores, assert the resulting
  `resolution_method` — no LLM call needed to test the *branching logic* itself), `catalog_writer`'s
  uniqueness enforcement, `validator`'s checks against a fixture `catalog.json` with deliberately broken
  entries. LLM-calling functions are tested here by mocking the `instructor` client (matches
  `narration_generator`'s existing use of `pytest-mock`, already a shared dependency) so the *shape* of the
  prompt-building and response-handling code is checked without spending money or needing a key.
- **`tests/integration/`** — allowed to shell out to OpenSCAD/Blender if present, and allowed to make a real
  (cheap, small) LLM call if an API key is present; each test skips itself (`pytest.mark.skipif`) rather than
  failing when the tool/key isn't available, so `pytest tests/unit` always works and `pytest` (full suite)
  degrades gracefully on a machine that only has some tools installed. Covers: actual OpenSCAD compilation of
  a known-good fixture `.scad` file, actual Blender headless GLB export bounds-checking, one real end-to-end
  `spec → resolve → generate` call per provider (guarded by `skipif` on missing key) to catch prompt/schema
  drift against the real API — this is the main place multi-provider support gets exercised, since it's cheap
  to parametrize one test over `["anthropic", "openai", "google"]` and skip whichever provider has no key set.
- CLI: `pytest asset_pipeline/tests/unit` should be the fast, always-green command a coding agent runs after
  every change; `pytest asset_pipeline/tests` (full) is the pre-merge check that exercises whatever tools
  happen to be installed locally.
- Also keep a couple of tiny **manual smoke-test scripts** outside pytest for the geometry-producing stages,
  since "does this look right" for a generated mesh is easier to eyeball than to assert: e.g.
  `python -m asset_pipeline.cli generate-parametric --dry-run "a small L-bracket"` that runs Stage 3 alone
  and opens the resulting STL/GLB in whatever viewer is available, without touching the catalog or requiring
  Stage 2's classifier to have run first. Each stage's CLI subcommand should be independently invocable like
  this — this is already implicit in the per-stage CLI commands listed below, but call it out explicitly:
  **no stage's CLI command should require the full `resolve-scene` pipeline to have been run first.**

## 2. Staged build order

Each stage is independently testable and produces a runnable CLI command before the next stage starts. Build
in this order; do not parallelize across stages until Stage 4 is done, since Stages 2–4 form the critical
path every asset class depends on.

### Stage 0 — Scaffolding and shared models (foundation)

**Goal:** empty pipeline that can load/save a catalog and validate an `AssetSpec`, with nothing that
generates geometry yet.

Deliverables:
- `asset_pipeline/config.py` — mirrors `narration_generator/config.py`: `.env` loading,
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (all optional individually — only the ones
  actually used need to be set), `DEFAULT_LLM_PROVIDER = "anthropic"`, `DEFAULT_LLM_MODEL = "claude-sonnet-4-6"`,
  `LIBRARY_DIR`, `INTAKE_DIR`, `OPENSCAD_BIN` (auto-detect like `find_blender()` in `prefab_to_glb.py`),
  `BLENDER_BIN` (reuse the exact `BLENDER_CANDIDATES` / `_find_store_blender()` logic from
  `prefab_to_glb.py:382-401` rather than reimplementing).
- `llm/client_factory.py` — `get_instructor_client(provider: str | None = None)` returning an
  `instructor`-wrapped client for `"anthropic"` (`instructor.from_anthropic(anthropic.Anthropic())`),
  `"openai"` (`instructor.from_openai(openai.OpenAI())`), or `"google"`
  (`instructor.from_google(genai.Client(), use_async=False)`) — the exact three patterns already documented
  in this repo's root `CLAUDE.md`. Every `pipeline/*_generator.py` function accepts `provider` and `model`
  parameters that default from `config.py` but can be overridden per call, so e.g. Stage 1's cheap
  classification call and Stage 3's OpenSCAD generation call can independently be pointed at different
  providers/models without editing pipeline code — only a config value or a CLI flag changes.
- `models/catalog_models.py` — `AssetCatalogEntry` Pydantic model matching req. doc §5.3 fields exactly
  (`asset_id`, `display_name`, `asset_class` as `Literal["primitive","variant","composite","parametric","imported"]`,
  `address`, `material_slots: list[str]`, `canonical_bounds_m: Annotated[list[float], Len(3,3)]`, `pivot`,
  `default_forward_axis`, `default_up_axis`, `tags: list[str]`, `capabilities`, `collider`, `provenance`,
  `review_level: Literal[0,1,2,3,4]`).
- `models/spec_models.py` — `AssetSpec` (authoring-side, per req. doc §5.1/§6.1) and `ResolutionRecord` (per
  §5.2).
- `pipeline/catalog_writer.py` — `load_catalog() -> list[AssetCatalogEntry]`, `append_entry(...)`,
  `save_catalog(...)`. Enforce `asset_id` uniqueness on write; raise, don't silently overwrite.
- Seed `library/catalog.json` with ~10 core primitives (sphere, cube, cylinder, cone, torus, plane, arrow,
  label placeholder) whose `address` points at Unity built-in primitives — copy the bounds/pivot/axis
  conventions already implicit in `prefab_to_glb.py`'s `UNITY_BUILTIN_MESHES` and `_sphere_mesh` /
  `_box_mesh` functions so scale stays consistent with what the viewer already renders.

Exit test: `python -m asset_pipeline.cli catalog list` prints the 10 seeded primitives; a malformed
`AssetCatalogEntry` fails Pydantic validation with a clear error.

### Stage 1 — Authoring spec intake and validation

**Goal:** take a raw sentence/paragraph description plus placement info, produce a validated `AssetSpec`
JSON — no resolution yet.

Deliverables:
- `models/spec_models.py` extended with a `parse_description()` helper that wraps an `instructor` call via
  `llm/client_factory.py`: LLM turns free text into a structured `AssetSpec` (`description`, `kind` guess,
  `semantic_type`, `desired_size_m` guess if stated, `visual_style`). This is a **thin, cheap LLM call** —
  classification, not generation — so it's the right place to default to a cheaper provider/model override
  (config-controlled, not hardcoded) even though the pipeline's global default is Sonnet.
- CLI: `python -m asset_pipeline.cli spec "a small gray moon with visible crater texture, about 15cm across"`
  → prints/saves `AssetSpec` JSON matching req. doc §5.1's example shape.

Exit test: five varied hand-written descriptions (a primitive-shaped one, a variant-shaped one, a composite,
a parametric-apparatus one, a "this is basically a lab table" one) each produce a plausible `AssetSpec`
without crashing, reviewed by hand — this is calibration, not unit testing.

### Stage 2 — Classifier and resolver (routing, no generation)

**Goal:** given an `AssetSpec`, decide which asset class it belongs to and whether an existing catalog entry
already satisfies it. This is the most leveraged stage — get routing right and the expensive generation
stages are only invoked when truly needed.

Deliverables:
- `pipeline/catalog_matcher.py` — cheap-first matching: (1) exact/fuzzy tag and `semantic_type` string match
  against `catalog.json`, (2) if no strong match, embed the spec description and catalog `display_name` +
  `tags` strings (use `anthropic`'s embeddings or a local sentence-transformers model — **prefer a local
  model** here to avoid a paid call on every single asset lookup, since this runs far more often than
  generation does) and rank by cosine similarity. Return top-k candidates with confidence scores.
- `pipeline/classifier.py` — decision logic per req. doc §7:
  1. If `catalog_matcher` returns confidence ≥ threshold (e.g. 0.85) → `resolution_method: "catalog_match"`.
  2. Else if the match is close but wrong color/scale/transparency → `"variant"` (Stage 5 handles material,
     Stage 4 handles scale — no new geometry).
  3. Else if `AssetSpec.kind` names 2+ known primitives assembled together (e.g. "water molecule") →
     `"composite"` (Stage 2.5 below).
  4. Else if description implies mechanical/geometric structure (bracket, stand, dial, cutaway box) →
     `"parametric"` → Stage 3 (OpenSCAD).
  5. Else if description implies organic/complex/realistic form (furniture, "realistic lab equipment",
     anything a CSG script can't reasonably approximate) → `"imported"` → Stage 6, flagged for human
     sourcing.
  6. Else → flag for human review with the LLM's reasoning attached (do not guess).
- `pipeline/resolver.py` — orchestrates the above, produces the `ResolutionRecord` JSON from req. doc §5.2,
  writes it next to the `AssetSpec`.
- Composite handling (`"composite"` case): a composite is **not new geometry** — it's a list of existing
  `asset_id`s with relative transforms, stored as a JSON fragment in `library/composites/<id>.json`. Build a
  minimal `pipeline/composite_builder.py` that takes an LLM-proposed decomposition ("water molecule = 1 large
  sphere_basic (oxygen, red) + 2 small sphere_basic (hydrogen, white) at 104.5° apart") validated against the
  catalog, and writes the fragment. This reuses Stage 2's classifier output recursively — each sub-part goes
  through resolution too.

Exit test: run all five Stage-1 spec examples through the resolver; each gets a `resolution_method` that
matches human judgment of what it should be. The moon example should resolve to `catalog_match` or `variant`
against `sphere_basic`, not trigger generation.

### Stage 3 — Parametric generation via OpenSCAD

**Goal:** for specs classified `"parametric"`, get a working OpenSCAD script and a compiled mesh out the
other end, with automatic repair on compile failure.

Deliverables:
- `models/generation_models.py` — `OpenSCADPlan` structured output: `parameters: dict[str, float]`,
  `scad_source: str`, `expected_bounds_m: list[float]`, `notes: str`. Using `instructor` with a Pydantic
  response model here (not raw text) means the LLM's own reasoning about expected bounds becomes a
  machine-checkable field, not just a guess.
- `pipeline/openscad_generator.py`:
  1. System prompt gives the LLM: OpenSCAD language reference summary, the target `canonical_bounds_m`
     convention (unit cube-ish, centered pivot, +Y up — matching Stage 0's primitive conventions), and 2-3
     worked examples (a bracket, a dial, a stand) as few-shot context.
  2. Call `instructor` → `OpenSCADPlan`.
  3. Write `scad_source` to `library/generated/<asset_id>/source.scad`, shell out to
     `openscad -o source.stl source.scad` (subprocess, same style as `convert_fbx_to_glb`'s subprocess call
     in `prefab_to_glb.py:445-467`, including a timeout).
  4. **On compile failure**: feed OpenSCAD's stderr back to the LLM in a repair prompt ("this script failed
     with error X, fix it") for up to 3 attempts (use `tenacity` for the retry loop, same as
     `brief_generator.py`). Log every attempt in the build report — never silently retry forever.
  5. On success, measure the actual STL bounding box (trivial with `numpy-stl` or manual STL parsing) and
     compare to `expected_bounds_m`; if they diverge by more than e.g. 20%, flag `requires_author_review` in
     the `ResolutionRecord` rather than silently trusting the LLM's stated expectation.
- CLI: `python -m asset_pipeline.cli generate-parametric <asset_id> --spec spec.json` — runs steps 1-5 and
  reports pass/fail with the repair attempt count.

Dependency: **OpenSCAD CLI** must be installed and on PATH or configured via `OPENSCAD_BIN`; add install
instructions to `setup_windows.ps1` / `setup_unix.sh` (winget/choco on Windows, apt/brew elsewhere) — do not
vendor the binary.

Exit test: generate 5 parametric examples (bracket, stand, dial, cutaway box, simple clamp) from text
descriptions alone; all compile to STL within the retry budget; bounding boxes are sane by manual inspection
of the preview render (Stage 8 needed to actually see them — acceptable to defer visual check until Stage 8
exists, but functional/compile success is checked here).

### Stage 4 — Mesh processing: STL/OBJ/FBX → GLB, scale/pivot/orientation normalization

**Goal:** every asset entering the catalog — whether from OpenSCAD, Blender, or an imported file — passes
through one common normalization step so the "every asset must have known canonical scale" requirement
(req. doc §8) is actually enforced, not just documented.

Deliverables:
- `pipeline/mesh_processor.py`, built around a headless Blender Python script (extend the existing
  `BLENDER_SCRIPT` pattern in `prefab_to_glb.py:402-422`, don't rewrite it from scratch):
  1. Import source mesh (STL from Stage 3, or OBJ/FBX/GLB from Stage 6 intake).
  2. Compute bounding box; recenter pivot to the convention chosen in Stage 0 (center, or base-center for
     "stands on a surface" objects — decide per `asset_class`/tag, record which convention was used in
     `canonical_bounds_m`/`pivot` catalog fields).
  3. Apply the author-specified or LLM-estimated `target_size_m` as a uniform scale so the **exported GLB's
     bounds already match `canonical_bounds_m`** — i.e., normalization happens once at build time, so the
     resolved runtime JSON's `target_size_m`-to-`scale` conversion (req. doc §8) is a simple ratio at scene
     time, never a mesh re-processing step.
  4. Recalculate normals, triangulate, decimate if triangle count exceeds a configurable threshold (flag,
     don't silently decimate below a quality floor — record the before/after triangle count in metadata).
  5. Export GLB via `bpy.ops.export_scene.gltf` (same call already used in `prefab_to_glb.py:415-420`).
  6. Write `library/generated/<asset_id>/meta.json` recording: original file, transform applied, final
     triangle count, bounds, decimation applied y/n.
- Triangle count threshold and decimation ratio go in `config.py`, not hardcoded in the processor function.

Exit test: feed it a Stage 3 STL and a hand-picked sample FBX; confirm the output GLB's bounding box (checked
via `pygltflib`, same accessor min/max fields `prefab_to_glb.py` already computes at
`prefab_to_glb.py:267-273`) matches the requested `target_size_m` within floating point tolerance.

**Addendum (2026-07, not yet implemented):** Stage 4 also owns UV unwrapping — Smart UV Project for
meshes that arrive without UVs (all OpenSCAD STLs), run after decimation, plus UV layout image export
and a UV hash recorded in meta.json. Specified in Stage 6c.2 below; build it as part of Stage 6c.

### Stage 5 — Materials and (optional) procedural textures

**Goal:** LLM proposes a material definition matching req. doc §9's schema; simple procedural textures
(noise, gradient, simple pattern) are generated locally rather than fetched, since bespoke photorealistic
textures are explicitly out of scope for v1.

Deliverables:
- `models/catalog_models.py` extended with `MaterialDef` matching req. doc §9's example exactly
  (`material_id`, `shader_family`, `base_color`, `alpha`, `metallic`, `smoothness`, `transparent`,
  `emissive`).
- `pipeline/material_generator.py`:
  1. `instructor` call: description → `MaterialDef`. This is a cheap, structured, low-risk call — good
     candidate for the cheaper-model override from Stage 1.
  2. If the description implies a patterned surface (craters, stripes, rust, grid lines) and no suitable
     existing texture exists, generate a simple procedural texture with **Pillow + numpy** (Perlin/simplex
     noise, cellular noise for craters, simple gradients) — this keeps the dependency footprint tiny compared
     to standing up a diffusion image model, and is explicitly enough for req. doc §9's initial scope
     ("normal maps, tiling... procedural textures" are listed as open questions, not requirements, in
     §20.2). Write to `library/materials/textures/<name>.png`.
  3. Write `library/materials/<material_id>.json`.
- Explicitly **out of scope for this stage**: AI image-generation-model textures. If photorealistic textures
  become necessary later, this is the seam where an image-gen API call would slot in without touching any
  other stage — note this in the module docstring so a future agent finds the extension point.

Exit test: five material descriptions ("dull gray rock", "glowing blue energy field", "brushed metal",
"transparent glass", "rough red rust with visible texture") produce valid `MaterialDef` JSON; the last one
also produces a texture file that visibly looks rust-like when opened.

**Addendum (2026-07):** procedural synthesis is no longer the whole texture story — it becomes step 3 of
the texture resolution ladder in Stage 6c.3 (archive match first, procedural for generic patterns,
human-sourced real maps for authentic surfaces like planets and basketballs). The synthesis code itself
is unchanged; its outputs get registered in the Stage 6c texture index so they are reused too.

### Stage 6 — External asset intake (manual first, semi-automated second)

**Goal:** normalize a human-provided model file into the library with correct provenance, and — once that
works — add an assisted search step that still requires human approval before download.

**6a — Manual intake (build first):**
- `intake/README.md` documents the process: human downloads an approved-license file, places it in
  `intake/<asset_id>/` with a `source.json` stub (URL, license, author) they fill in by hand.
- `pipeline/external_intake.py`:
  1. Read `intake/<asset_id>/source.json`; reject if `license` is missing or not in an allowlist
     (`CC0`, `CC-BY` + attribution captured, or explicitly `internal`/proprietary-cleared).
  2. Run the file through **Stage 4's mesh processor** unchanged — this is why Stage 4 is built as a
     standalone, format-agnostic function rather than something Stage 3 owns privately.
  3. Write `ProvenanceInfo` (req. doc §11 schema exactly) into the catalog entry.
  4. Move processed output into `library/imported/<asset_id>/`.
- CLI: `python -m asset_pipeline.cli intake <asset_id>` runs the above and reports pass/fail with the
  specific missing-field reason on failure (never silently skip provenance).

**6b — Semi-automated search assist (build after 6a is solid):**
- `pipeline/external_intake.py` gets a `search()` function hitting a **small, explicit allowlist** — start
  with the Poly Haven API only (confirmed CC0, has a public API, no auth needed for search). Do not add
  Sketchfab or other sources until their API terms and license-per-asset metadata are actually verified by a
  human — the requirements doc is explicit that unrestricted external retrieval is out of scope (§3.2), and
  scope creep here is exactly the failure mode that guardrail exists to prevent.
- Search results are presented (CLI table: name, thumbnail URL, license, poly count) — human picks one,
  confirms, *then* the file is downloaded into `intake/<asset_id>/` and 6a's pipeline runs normally. The
  agent never auto-downloads.

**Policy amendment (2026-07, user decision — supersedes "never auto-downloads" for one narrow case):**
the sourcing workflow was too manual — an `imported` classification just told the user to go search.
`pipeline/source_assist.py` now runs automatically for `imported` specs:

1. **Auto-download IS permitted** when all three hold: the source is on the verified allowlist with
   machine-verifiable unrestricted licensing (Poly Haven = all CC0), the top search hit matches the spec
   confidently (token coverage ≥ `AUTO_ADOPT_CONFIDENCE`), and the real-world size is known
   (`desired_size_m`) so intake can complete unattended. The download still flows through the normal 6a
   gate (license check, normalization, provenance).
2. Otherwise, **candidates** (name, thumbnail, license, source URL, confidence) are attached to the draft
   and the review app offers one-click *Approve & download* (`POST /api/adopt`).
3. With no usable hits, the draft carries **specific search URLs** for verified sources (Poly Haven,
   NASA 3D Resources, Smithsonian 3D) — the user clicks a link, never hunts blind.

Unchanged: only allowlisted sources are contacted; unverified sources still require human license
verification before being added; attribution-required licenses are never auto-adopted. Extending the same
assist to the texture `pending` flow (6c.3 step 4) is the natural follow-up.

Exit test (6a): manually source one CC0 GLB (e.g. a simple prop), run intake, confirm it appears in
`catalog.json` with correct provenance and normalized scale. Exit test (6b): search "wooden table" returns
real Poly Haven results with correct license field populated.

**Policy amendment (2026-07 — "Frosty the Snowman" incident):** a request for a snowman surfaced NASA's
"Vesta - Snowman Craters" (an asteroid crater formation nicknamed for its shape, not a snowman) as a
sourcing candidate, purely because keyword scoring cannot distinguish an object from something merely
*named after* it. Root-caused to three compounding gaps, all fixed:

1. **Search queries were positional, not semantic.** `build_query()` sliced the first 4 "non-generic"
   words out of the raw sentence, which for anything longer than a terse description picks up grammatical
   filler ("friendly", "made", "from") ahead of the actual identity words. Fixed by extending
   `GeometryClassification` with `search_keywords: list[str]` — the same LLM call that decides `imported`
   now also names the identity words to search for (zero extra LLM calls), carried onto
   `ResolutionRecord.search_keywords` and preferred by `build_query()` over mechanical extraction.
2. **No semantic relevance gate.** A single coincidental word match could clear the visibility floor and
   be shown as if it were a real option. New `pipeline/source_relevance.py`: every candidate is checked by
   an LLM against the object's own description ("is this actually the same kind of object, or just named
   similarly?") before being shown, and — for the survivors — checked again against its actual thumbnail
   image (vision, anthropic/openai only; degrades to text-only for google or on fetch failure). Auto-adopt
   additionally requires an explicit `matches=True` verdict, not just "not rejected".
3. **The composite path gave up too early.** The composite hard-rule capped parts at 2-6, so anything
   needing more (a snowman: 3 body spheres + 2 arms + a hat + facial features) fell through to `imported`
   — a dead end for whimsical/fictional subjects no real-world source will ever have. Raised to 2-12 and the
   classifier now explicitly prefers a crude composite approximation over `imported` when one is possible.
   Composite sub-parts also gained a `_guess_primitive_kind()` hint (`build_part_specs()`,
   `pipeline/composite_builder.py`) so an obvious part like "large white sphere" catalog-matches the seed
   primitive instantly instead of costing its own LLM classification call — a 12-part composite went from
   22 LLM calls to 2 once both fixes landed.

**Follow-up (2026-07 — "Frosty is all gray" incident):** the routing fix above was necessary but not
sufficient. Once Frosty actually classified as `composite`, it turned out composites had **never** produced
a visible or colored result at all: `CompositePart` had no material field, no code path ever generated one,
and the transform fields (`position`/`rotation`/`scale`) were never populated either -- always the
constructor defaults `[0,0,0]`/`[0,0,0]`/`[1,1,1]`. Every composite ever built by this pipeline was, in
effect, a pure inventory list with zero placement or appearance data; nothing anywhere (webapp, sync,
preview) had a rendering path for one. Fixed as a full loop-closer, by decision to **bake composites into
one ordinary GLB** rather than teach every consumer to assemble a live parts list:

1. `GeometryClassification.composite_parts` changed from `list[str]` to `list[CompositePartPlan]`
   (`models/classification_models.py`) -- each part now carries `color_hint` plus an approximate
   `relative_position`/`relative_scale` (fractions of the object's own bounding box, `+Y` up matching the
   rest of the pipeline's convention), authored by the same classification call at zero extra LLM cost.
   `composite_builder.decompose_into_primitives()` (the forced-redirect path) produces the same structured
   shape.
2. `asset_factory.py`'s composite branches (`create_from_spec` and `redirect_draft`) now call
   `_materialize_and_bake_composite()`: generate one material per part (reusing Stage 5's
   `generate_material()`, keyed off each part's own `color_hint`), then bake.
3. New `pipeline/composite_baker.py`: a headless-Blender script that builds/imports each part (seed
   primitives built natively with `bpy.ops.mesh.primitive_*_add`; non-primitive parts imported as glTF),
   positions/scales/colors it, and **joins everything into one mesh** -- Blender's join preserves each
   source object's material slot, so the result is one glTF with several materials. This is why baking (not
   a live parts-list renderer) was the right call: a baked composite is indistinguishable from any other
   cataloged asset as far as preview/sync/validate/save are concerned, so none of those needed
   composite-specific code once this landed.
4. Axis-convention pitfall worth flagging for future Blender-scripting work in this codebase: Blender is
   natively Z-up; this pipeline (and glTF/three.js) is Y-up. Feeding a part's Y-up position/scale straight
   into `obj.location`/`obj.scale` produces a Blender-native result that Blender's own Z-up→Y-up export
   conversion then re-interprets wrong (a vertical stack came out laid out sideways). The fix is a fixed
   swap at the point objects are placed: Blender-`(x, y, z)` = pipeline-`(x, -z, y)` for both position and
   scale, letting the exporter's automatic conversion round-trip the pipeline's own numbers correctly.
5. `AssetCatalogEntry.asset_class="composite"` (not `"parametric"`) and provenance wording were fixed in
   `save_draft()`, which had hardcoded the parametric case for any draft with a `glb_address`.

**Scope note:** solid per-part colors need no UV data (a material with no texture ignores UVs), so this
does not depend on or block Stage 6c's texture/UV work. A part that later wants a real image texture keeps
its own UVs through the join. Also out of scope for now, flagged for later: parametric (single-CSG-mesh)
assets still get only one flattened material each, since OpenSCAD's STL export has no concept of named
sub-regions -- unifying "composite" (existing-primitive parts) and "parametric" (freshly-generated parts)
into one general "assembly of independently-colorable pieces" concept, where a part's geometry can be
either a catalog reference OR a small freshly-generated OpenSCAD piece, is the natural next step if
multi-color parametric assets are needed.

**Manual override (2026-07, same incident):** a stuck draft previously had no way out except editing
OpenSCAD parameters that don't exist yet for non-parametric items. `asset_factory.redirect_draft()` lets a
human force any existing draft down a specific path — `parametric` (run Stage 3 directly), `composite`
(force a decomposition via `composite_builder.decompose_into_primitives()`, deliberately more permissive
than the normal classifier judgment since a human already overrode it), or `imported` (re-run the sourcing
assist, optionally with the human's own search words via `manual_query`) — bypassing the original
classify()/resolve() verdict for that asset_id rather than re-asking the same question. Exposed in the
review app as three buttons on a `needs_human` draft (`POST /api/redirect`).

**Follow-up (2026-07 — "methane sticks don't connect" incident):** a ball-and-stick methane model baked
with its four C–H bonds rendered as flattened discs pointing the same fixed direction regardless of where
their two atoms actually were — and the user had no way to fix it, since composites had no editor and no
OpenSCAD script to hand-edit. Root-caused and closed as three separate deliverables:

1. **Structural fix — connectors need real trigonometry, not an LLM guess.** Pointing a cylinder from one
   arbitrary 3D position to another requires computing a rotation from the direction vector between them;
   every attempt at asking the LLM for this left `rotation=[0,0,0]`, so every "bond" pointed the same fixed
   direction no matter where its two atoms were. Fixed by giving the LLM an easier, more reliable job: name
   the two endpoints instead of computing geometry. `CompositePartPlan`/`CompositePart` gained `label` (a
   short stable name like `"C"`, `"H1"`) and `bond_between: [labelA, labelB]` (set only on a connector part,
   which then ignores `relative_position`/`relative_scale`) plus `bond_thickness`. `composite_baker.py` now
   bakes in two passes: pass 1 places every labeled atom/regular part and records its real final Blender-space
   position; pass 2 builds each connector as a cylinder whose midpoint, length, and rotation are computed
   deterministically from its two named atoms' actual positions via `mathutils.Vector.to_track_quat('Z', 'Y')`.
   The classifier prompt (`pipeline/classifier.py`) explains this mechanism explicitly and adds CPK-style atom
   coloring guidance (carbon=dark gray/black, hydrogen=white, oxygen=red, nitrogen=blue, sulfur=yellow) for
   molecule requests. Verified live: a hand-built two-atom "dumbbell" test, then the user's exact original
   methane prompt rebuilt from scratch, both rendered with bonds correctly reaching their atoms.
2. **Composite editor — "I probably would be able to fix this if I could edit parameters."** Composites had
   no equivalent of the OpenSCAD parameter-edit loop. New `asset_factory.edit_composite()`: given
   `{part_id: {field: value}}` edits (position/scale/rotation/bond_between/bond_thickness/label/color_hex),
   mutates the stored `CompositeFragment` directly and rebakes — **no LLM call**, same cost profile as
   `regenerate()`'s numeric parameter substitution. A `color_hex` edit (`'#rrggbb'`) writes straight to that
   part's material file immediately; a free-text `color_hint` edit only updates the hint, since actually
   changing the rendered color from a hint requires an LLM material call — that's the separate, explicitly
   opt-in `regenerate_materials_for` list, so an edit action never silently does nothing while looking like it
   took effect. `get_composite_fragment()` + `GET /api/composite/<id>` (inlining each part's *current* resolved
   `base_color`, since the fragment JSON only stores a `material_id`) feed a part-by-part editor panel in the
   review app — every part's label/position/scale/bond endpoints/color swatch/color hint editable, with an
   "Apply edits & rebake" button (`POST /api/composite/<id>/edit`). Available from both the Create panel (a
   loaded composite draft) and the Library ("Edit composite parts" button, mirroring "Edit parameters" for
   parametric assets). Verified live end-to-end (real Blender rebake, real browser render, actual button
   click) moving/recoloring parts and confirming the bond geometry followed correctly.
3. **Automated visual review + auto-repair — "we need to have it review the images to see if they actually
   correspond to reality."** Nothing in the pipeline had ever noticed its own bad output; a broken bake looked
   exactly as "successful" as a correct one until a human happened to look. New `pipeline/visual_review.py`
   (same two-tier-review shape as `source_relevance.py`, but always an image call — there's no cheap
   text-only pass, since the point is judging what was actually *built*, not what was intended): renders the
   draft's current GLB (`preview_renderer.render_glb_to_png()`, a single-shot version of the Stage 8 preview
   pipeline for an arbitrary draft GLB that isn't yet a catalog entry) and asks a vision-capable LLM
   (anthropic/openai only) whether it matches the original description, returning `matches`, `reasoning`, and
   — when `matches=False` — a specific, actionable `suggested_fix`. `asset_factory.review_and_repair()`
   wires this into a repair pass when the verdict is a mismatch (opt-in via `auto_repair`, default on):
   for a `parametric` draft, the reviewer's complaint becomes a `regenerate()` tweak instruction (the existing
   LLM-revision path); for a `composite` draft, new `composite_builder.revise_composite_parts()` shows the LLM
   the description, the CURRENT part list exactly as built (`fragment_to_part_plans()`), and the complaint,
   and asks for a corrected FULL part list that fixes only what's wrong — explicitly instructed to leave
   parts the complaint doesn't implicate unchanged, rather than regenerating everything from scratch. The
   revised parts re-enter the normal composite pipeline via `resolve(..., forced_composite_parts=...)` and
   `_materialize_and_bake_composite()`. Exposed as "Review vs. intent" in the review app
   (`POST /api/review/<asset_id>`), with the verdict shown as a badge and the draft's `visual_review` field
   persisted either way. **Verified live end-to-end with a real vision LLM call**: seeded an intentionally
   broken two-atom composite (no connecting bond, both atoms defaulted to the same gray) matching the
   methane incident's actual defect shape; the review correctly identified "disconnected... both uniformly
   default gray" and produced a specific fix; auto-repair regenerated a corrected part list (bond +
   distinct red/blue atom colors) and rebaked; a second review of the repaired asset returned `matches=True`.
   Confirmed through the actual browser UI (Playwright-driven click of "Review vs. intent") as well as
   directly against pipeline functions.

### Stage 6c — Texture library: archive, intake, search, and separate UV maps (added 2026-07)

**Goal:** textures get the same resolution ladder geometry already has — *local archive first, then
generation or human-approved external sourcing* — instead of Stage 5's current "procedural or nothing".
Motivating cases: basketballs, Earth, the Moon. Real, authoritative surface maps for these exist online
(NASA/USGS planetary maps are public domain and already equirectangular; Poly Haven's texture section is
CC0 with physical tile dimensions in its API), and a synthesized crater-noise moon is categorically worse
than the real map. **Download once, reuse forever**: a texture that enters the library is never re-fetched
or re-generated for a later request that matches it.

**Design principle (user requirement, 2026-07): UV layout is a separate concern from texture image.**
A texture is only reusable if you know what UV layout it assumes; conversely, once an object's UV layout
is known and recorded, generating *N* different surfaces for it costs one texture each, with no
re-unwrapping. The pipeline therefore records UV information per geometry asset and mapping conventions
per texture, and checks compatibility at bind time.

**6c.1 — Data model**

- `library/textures/index.json` — the texture registry (same single-file + uniqueness pattern as
  `catalog.json`; payload files live in `library/textures/<texture_id>/`). `TextureAsset` fields:
  - `texture_id`, `display_name`, `semantic_type` (e.g. `"moon_surface"`, `"basketball_skin"`),
    `tags: list[str]` — the searchable surface for archive matching.
  - `mapping: Literal["equirectangular", "tileable", "atlas"]` — the reuse contract:
    *equirectangular* (planet maps, ball skins) fits any standard UV sphere; *tileable* (wood, rust,
    asphalt) fits anything with UVs, scaled by tiling factors; *atlas* was painted/baked against one
    specific mesh's UV islands and only fits that mesh (records `bound_asset_id` + `uv_hash`).
  - `maps: dict[str, str]` — map-kind → relative file path. Kinds: `albedo`, `normal_gl`, `normal_dx`,
    `roughness`, `ao`, `arm`, `displacement`. **v1 consumes `albedo` only** but *stores* whatever the
    source provides (Poly Haven ships full PBR sets; keeping them costs disk, re-downloading costs a
    human). Normal/roughness wiring into `MaterialDef` is a later, purely additive step — do not solve
    the Unity nor_gl-vs-nor_dx question now, just keep both files when offered.
  - `resolution: [w, h]`; `tile_size_m: [u, v] | null` — physical meters per tile for tileables
    (Poly Haven's `dimensions` field, mm → m). Enables scale-correct tiling: `uv_tiling =
    object_extent / tile_size_m` computed at bind time, not guessed.
  - `authentic: bool` — photographic/measured (NASA map, Poly Haven scan) vs synthesized (Stage 5
    procedural). Archive matching prefers authentic when the request names a real thing.
  - `provenance: ProvenanceInfo` (the existing §11 model, unchanged) + `LICENSE.txt` in the payload
    folder for imported textures — same rules as 6a.
- `MaterialDef` additions (backward-compatible): `texture_id: str | None` (the registry link;
  `texture` keeps holding the resolved runtime path), `uv_tiling: [float, float]` default `[1, 1]`.
  This answers req. doc §20.2's tiling open question.
- Catalog entry addition — `uv: UVInfo` on `AssetCatalogEntry`:
  `{status: "none" | "builtin" | "preserved" | "generated", convention: "equirect" | "generic" | "atlas",
  layout_image: str | None, uv_hash: str | None}`. Seed primitives get `builtin`/`equirect` for the
  sphere family and `builtin`/`generic` for the rest (Unity and three.js built-in primitives share
  those conventions — this is exactly why an equirect Earth map "just works" on `sphere_basic`).
  Imported GLBs get `preserved`; Stage 4-unwrapped meshes get `generated` + a layout image + a hash.

**6c.2 — UV unwrap and layout export (Stage 4 addendum)**

- `mesh_processor.py` gains an unwrap step: if the imported mesh has no UV layer (always true for
  OpenSCAD STLs), run Blender Smart UV Project **after** decimation (decimating after unwrapping would
  invalidate the layout), and report `uv_status` in the normalization result/meta.json.
- New `pipeline/uv_tools.py`: read the exported GLB's UV accessors + triangle indices with `pygltflib`
  (already a dependency) and rasterize the UV island wireframe to
  `library/.../<asset_id>/uv_layout.png` with Pillow — **no second Blender run needed**. This layout
  image is the human/LLM-facing template for authoring atlas textures later. `uv_hash` = md5 of the UV
  buffer, so an atlas texture can assert which layout it was made for and the validator can catch
  drift after a regenerate.
- Out of scope: multiple UV channels per mesh (lightmaps), in-browser UV editing.

**6c.3 — Resolution ladder (Stage 5 revision)**

Stage 5's LLM call is extended to classify the texture need, then route — mirroring the geometry
classifier's cheap-first philosophy:

1. `texture_need: "none" | "pattern" | "authentic"` + search terms + mapping preference come back from
   the (existing, cheap) material LLM call. "authentic" means the description names a real-world
   surface (Earth, Moon, basketball, oak) where a real map beats synthesis.
2. **Archive match first** (new `pipeline/texture_matcher.py`, same token/fuzzy scoring as
   `catalog_matcher` over `semantic_type`/`tags`/`display_name`, filtered by mapping compatibility
   with the target object's `uv` info). Hit → reuse, zero cost. This is the "download once" guarantee.
3. `pattern` + no match → **procedural synthesis** (existing Stage 5 code, unchanged) — and the output
   is *registered in the index* (`authentic: false`) so even procedural textures are reused next time.
4. `authentic` + no match → **flag for human sourcing** (6c.4/6c.5). Never synthesize a fake Earth —
   this extends req. doc §17's no-silent-fallback rule to textures. (A human can override to
   procedural explicitly.)

**6c.4 — Manual texture intake (build first, mirrors 6a)**

- `intake/<texture_id>/` with image file(s) (png/jpg/tif; exr deferred — extra dependency) + the same
  hand-filled `source.json`, extended with `mapping`, `tile_size_m` (tileables), `semantic_type`, and
  optional per-file map-kind labels (filename conventions `*_diff*`/`*_nor_gl*`/`*_rough*`/`*_ao*`
  auto-detected, explicit dict wins). Same license-gate code as 6a (refactor the allowlist check out of
  `external_intake.py` into a shared helper rather than duplicating); allowlist gains
  `public-domain` for NASA/USGS material — added only with the source URL recorded, same
  human-verifies-the-source rule as ever.
- Validation on intake: image opens; equirectangular textures must be ~2:1 aspect (warn otherwise);
  images larger than `TEXTURE_MAX_DIM` (config, default 4096) get a downscaled **runtime derivative**
  while the original is kept in the payload folder (planet maps run 8k–21k px; keep the original so
  higher-quality derivatives never require re-downloading).
- CLI: `python cli.py intake-texture <texture_id>`.
- `intake/README.md` gains a curated list of verified planetary-map sources (NASA SVS, USGS
  Astrogeology) — these are browse-and-download-by-hand sources, not APIs; the curation list is the
  search assist for them.

**6c.5 — Search assist (extends 6b, same human-approval contract)**

- `pipeline/polyhaven.py` already speaks this API: the textures endpoint is the same shape with
  `t=textures` (verified 2026-07: per-texture file trees expose `Diffuse`, `nor_gl`, `nor_dx`,
  `Rough`, `AO`, `arm`, `Displacement` exactly like model map sets, plus physical `dimensions`).
  Add `search-external --type texture` and `fetch-external --type texture <id>`; a fetch downloads
  the chosen-resolution map set into `intake/<texture_id>/` with a pre-filled `source.json`
  (`mapping: "tileable"`, `tile_size_m` from `dimensions`), then 6c.4's intake runs normally.
  The human's explicit fetch remains the approval step; nothing auto-downloads.

**6c.6 — Binding and preview**

- Binding = resolving a `(geometry asset, texture)` pair into a `MaterialDef`: check mapping
  compatibility against the object's `uv` info, compute `uv_tiling` for tileables from
  `canonical_bounds_m / tile_size_m`, set `texture_id` + runtime `texture` path.
- `webapp/asset_preview.html` learns a material override on `_previewLoad` (albedo texture URL +
  color/metallic/roughness/tiling applied to the loaded object's material) so the review app and the
  Stage 8 thumbnails both show textured results — an Earth request should *look like Earth* in the
  create loop, not like a gray sphere with a JSON attachment.
- Texture previews/contact sheet: thumbnails of each registry texture alongside the existing asset
  contact sheet.

**6c.7 — Validation (Stage 7 additions)**

Index schema validates; every `maps` file exists; imported textures have approved licenses; every
`MaterialDef.texture_id` exists in the index; equirect aspect sanity; atlas `uv_hash` matches the
bound asset's current hash (catches regenerate-invalidated atlases); orphan files under
`library/textures/` not referenced by the index.

**Migration:** the five existing Stage 5 procedural textures get backfilled into the index
(`authentic: false`, `mapping: "tileable"`) by a one-off script so the archive-match step sees them.

**Build order when implementing:** models + index IO → migration backfill → texture matcher + Stage 5
ladder rewiring → manual intake (6c.4) → Poly Haven textures (6c.5) → Stage 4 unwrap + uv_tools +
catalog `uv` field → preview binding (6c.6) → validator additions (6c.7).

**Exit tests:**
1. *Basketball*: "an orange basketball with black seams, 24cm" → sphere variant; texture ladder flags
   `authentic`; human intakes a real (or Poly Haven) ball texture; preview shows a textured ball.
2. *Download-once*: after a Moon map is intaken, a second "gray cratered moon" request resolves via
   archive match with zero LLM-texture/download activity.
3. *UV separation payoff*: Earth and Moon materials both bind to `sphere_basic` — one geometry, two
   materials, no mesh work.
4. *Generated-mesh path*: a Stage 3 bracket gets Smart-UV'd by Stage 4, a Poly Haven rust tileable
   binds to it with computed tiling, and the preview shows rusted metal.

### Stage 7 — Validation

**Goal:** implement req. doc §12 as an actual executable check, reusing `asset_audit.py`'s scanning approach
generalized to this library's layout.

Deliverables:
- `pipeline/validator.py` — checks per req. doc §12.1–§12.5 against `catalog.json` + `library/`:
  schema validation (Pydantic already gives this for free on load), every referenced `asset_id`/`material_id`
  exists, every asset has `canonical_bounds_m` and `pivot` set, triangle counts under threshold, every
  imported asset has `license_status: "approved"`, no unresolved `AssetSpec`s remain in a scene being
  finalized.
- Output shape mirrors `asset_audit.py`'s existing `format_json_output`/`format_screen_text` split — reuse
  that pattern (colorized screen output + machine-readable JSON mode) rather than inventing a new report
  format, since `asset_audit.py` already established the convention this codebase expects for these reports.
- CLI: `python -m asset_pipeline.cli validate` — exit code 0 only if zero errors (warnings allowed, per req.
  doc §17's fallback-must-be-recorded rule).

Exit test: intentionally break a catalog entry (missing `canonical_bounds_m`), confirm validator catches it
with a clear message and non-zero exit code; fix it, confirm clean pass.

### Stage 8 — Preview rendering (batch, static)

**Goal:** every catalog asset gets a thumbnail automatically, reusing the Playwright/three.js viewer
infrastructure that already exists rather than standing up a second renderer.

Deliverables:
- `pipeline/preview_renderer.py` — thin wrapper that either (a) extends `viewer2.html`/`screenshot_cli.py`
  with a single-object preview mode (place one GLB at origin, fixed camera, render), or (b) if that proves
  awkward for single assets, writes a minimal companion `asset_preview.html` using the same three.js loader
  code already proven in `viewer2.html`. Prefer (a) — it's less code and keeps one rendering path for the
  whole project instead of two.
- CLI: `python -m asset_pipeline.cli preview <asset_id>` and `preview --all` (batch, skip existing unless
  `--force`, matching `prefab_to_glb.py`'s `--force` convention for consistency).
- Output to `library/previews/<asset_id>.png`; also produce a contact-sheet HTML/image for fast human
  scanning across many assets at once (req. doc §13's "thumbnail contact sheet" requirement).

Exit test: `preview --all` on the Stage 0 seed catalog produces 10 recognizable primitive thumbnails.

### Stage 8b — Interactive review/edit app (Flask + three.js)

**Goal:** the static contact sheet from Stage 8 is enough for a quick scan, but human review (req. doc §15's
review levels, §18's "LLM should not be sole authority on final visual approval") is far more effective with
a live, orbit-able 3D view and the ability to nudge parameters without re-running the whole CLI by hand. This
stage builds a small local Flask app that turns catalog review into a fast iterate-in-browser loop.

**Two usage modes (added 2026-07, user requirement):** the app serves both *one-shot testing* and
*production asset creation*:

1. **Create-from-description (one-shot testing):** a text box on the main page where the user types a free
   description, hits generate, and gets the full pipeline result (spec → resolve → generate/variant/composite
   → normalize → material → preview) live in the viewport, with a regenerate button that accepts a tweak
   instruction ("make it flatter", "more metallic") and re-runs only the stages that need it. This is the
   fastest way to calibrate prompts and eyeball pipeline quality.
2. **Worklist mode (production creation):** load a user-provided list of asset descriptions (JSON/CSV of
   `AssetSpec`s or raw sentences — the same shape Stage 9's `resolve-scene` consumes), then step through it
   one asset at a time: generate → inspect in the viewport → modify (parameter edits, material tweaks, or a
   regenerate instruction) → iterate until acceptable → save/approve into the catalog → next. Progress is
   persisted so a long list survives an app restart; each saved asset records the normal provenance/review
   fields. This is the bulk-authoring workflow for building out a lab's full asset set.

Stage 8's `webapp/asset_preview.html` is deliberately built as the shared viewport for both modes
(`_previewLoad()` hot-swaps assets without a page reload), so 8b adds Flask endpoints + a worklist UI around
it rather than a new renderer.

This is deliberately **not** the runtime AR viewer and does not touch Unity — it's a development-time tool
sitting next to `viewer2.html`/`screenshot_cli.py`, reusing the same three.js GLB-loading code rather than a
new rendering stack.

Deliverables:
- `asset_pipeline/webapp/app.py` — a small Flask app (new dependency: `Flask`, already extremely lightweight
  and consistent with the "simple is better" stack goal — avoids pulling in a full frontend build toolchain
  for what is fundamentally a local review tool):
  - `GET /` — asset library browser: grid of thumbnails from `library/previews/`, filterable by
    `asset_class`, `tags`, `review_level`, and pending-review status (`requires_author_review: true` from any
    `ResolutionRecord`).
  - `GET /asset/<asset_id>` — detail/edit page: live three.js viewport (orbit controls, same GLTFLoader setup
    as `viewer2.html`) loading `library/.../<asset_id>/model.glb`, plus a side panel showing the catalog
    entry fields and, for `generated` (OpenSCAD) assets, the editable `parameters` dict from the stored
    `OpenSCADPlan`.
  - `POST /asset/<asset_id>/regenerate` — for OpenSCAD-sourced assets only: accepts edited parameter values
    (or an edited free-text tweak instruction, e.g. "make the base twice as wide"), re-runs Stage 3's
    generator with those overrides, re-runs Stage 4 normalization, and returns the new GLB — the browser
    reloads the viewport in place without a full page refresh. This is the "mods and edits" loop: a human
    can iterate on a parametric asset visually instead of hand-editing `.scad` files.
  - `POST /asset/<asset_id>/material` — same idea for `MaterialDef` fields (color, alpha, metallic,
    smoothness) with a live material update in the three.js viewport (cheap — no regeneration needed, just
    reassign the `MeshStandardMaterial` in-browser) before writing back to `library/materials/<id>.json`.
  - `POST /asset/<asset_id>/approve` — sets `review_level` acknowledgment / clears
    `requires_author_review`, writing an approval record (who/when — local use, so "who" can just default to
    OS username) into the catalog entry's provenance/audit trail. This is the human-approval checkpoint
    req. doc §18 requires; the web UI is what makes it fast enough to actually happen instead of being
    skipped under deadline pressure.
  - `GET /scene-preview/<scene_id>` — optional stretch: load a whole resolved scene (multiple assets placed
    per their runtime JSON transforms) in one viewport, for reviewing composites and layouts together rather
    than one asset at a time.
- Serve `library/` as a static folder (Flask's `send_from_directory`, scoped to that one directory — do not
  serve the whole repo) so the browser can fetch GLBs/textures directly, same pattern `screenshot_cli.py`
  already uses via its local `http.server` instance for `viewer2.html`.
- Keep all mutating endpoints local-only by default (`app.run(host="127.0.0.1")`) — this is a developer tool,
  not something exposed to a network, and regeneration endpoints shell out to OpenSCAD/Blender so they must
  not be reachable beyond localhost.
- CLI: `python -m asset_pipeline.cli review` starts the Flask dev server and opens the browser to `/`.

Exit test: start the app against the Stage 0 seed catalog plus at least one Stage-3 parametric asset; load
its detail page, edit one OpenSCAD parameter (e.g. bracket width), submit, confirm the viewport updates with
the regenerated geometry and the catalog/meta files on disk reflect the new bounds; click approve and confirm
`requires_author_review` clears.

**Scope guard:** this stage edits *generation parameters and materials*, not raw mesh geometry (no
in-browser sculpting/vertex editing). Full mesh editing in-browser is a large undertaking (essentially
building a Blender-in-the-browser) and isn't needed — the OpenSCAD parameter loop plus "reject and send back
to Stage 3/6 with new instructions" covers the actual iteration need. If freeform mesh edits turn out to be
frequently necessary, the better lever is a sharper regeneration prompt back to the LLM, not a browser mesh
editor.

### Stage 9 — Sync to consumer folders + CLI end-to-end wiring

**Goal:** tie every stage together behind one CLI entry point that takes a scene's list of `AssetSpec`s and
produces a fully resolved, validated, previewed bundle — this is the "give it a sentence, get an asset" loop
the user asked for, now assembled from the pieces above.

Deliverables:
- `pipeline/sync.py` — copies the resolved subset of `library/` referenced by a given scene into
  `viewer/arlem_preview_toolkit/arlem_preview_toolkit/assets/{models,materials,textures}` (matching
  `asset_audit.py`'s expected folder layout exactly, since that tool already scans those paths) — and,
  later, into a Unity project's `Assets/` folder once one exists. Copy-only, never a build dependency the
  other direction: `library/` stays the source of truth.
- `cli.py` top-level command: `python -m asset_pipeline.cli resolve-scene <scene_specs.json>` runs spec
  parsing → classify → resolve → generate/intake as needed → normalize → material → validate → preview →
  sync, printing a build report (JSON, matching req. doc §17's example shape) at the end.
- Human-in-the-loop checkpoint: any `ResolutionRecord` with `requires_author_review: true` **halts** the
  batch for that asset and reports it distinctly from hard failures — per req. doc §18, the LLM is never the
  sole authority on final approval.

Exit test: a small 5-object scene description (mixing a catalog-match primitive, a color variant, a
composite, a parametric bracket, and one flagged-for-human-import item) run end-to-end produces a build
report matching expectations for each of the 5 resolution paths.

## 3. Software stack summary

| Concern | Choice | Why |
|---|---|---|
| Structured LLM output | `instructor`, multi-provider (`anthropic`, `openai`, `google-genai` extras — all already dependencies per root `requirements.txt`) | Matches this repo's existing multi-provider pattern (`CLAUDE.md`'s "LLM Integration" section already documents all three); the asset pipeline should not hardcode Anthropic just because `narration_generator` currently defaults to it. |
| Default provider/model | `anthropic` / `claude-sonnet-4-6` | Matches `narration_generator/config.py` `DEFAULT_LLM_MODEL` for consistency, but routed through `llm/client_factory.py` so any call site can override provider+model via config or CLI flag — e.g. swap Stage 1/5's cheap classification calls to a lower-cost OpenAI or Gemini model without code changes. |
| Retry/backoff | `tenacity` (already a dependency) | Same as `brief_generator.py`. |
| Parametric geometry | **OpenSCAD** (external CLI, new dependency) | Deterministic CSG, small install, LLM-friendly declarative language, headless via subprocess. |
| Mesh conversion/cleanup | **Blender headless** (`bpy` via `blender --background --python`, already a dependency in practice via `prefab_to_glb.py`) | Already integrated and working in this repo; reuse rather than add a second mesh library (e.g. `trimesh`) unless a specific gap appears. |
| GLB assembly for pure primitives | `pygltflib` (already a dependency) | Already used for Unity-built-in-primitive → GLB in `prefab_to_glb.py`; reuse for Stage-0 seed catalog. |
| Procedural textures | `Pillow` + `numpy` (new, lightweight dependency) | Sufficient for noise/gradient/pattern textures per req. doc §20.2's stated initial scope; avoids standing up an image-gen model. |
| Embeddings for catalog matching | local `sentence-transformers` model (new dependency) **preferred** over an API embedding call | Catalog lookup happens on every asset, not just generated ones — a paid API call per lookup is the wrong cost tradeoff even with cheap models. |
| Validation | Pydantic v2 (already a dependency) | Matches every other model in this repo. |
| Preview rendering | Reuse existing Playwright + `viewer2.html` (already a dependency) | Avoids a second render pipeline; `screenshot_cli.py` already proves headless Chromium + three.js works in this environment. |
| STL bounds checking | manual STL parse or `numpy-stl` (new, tiny dependency) | Needed only to sanity-check OpenSCAD output before the Blender normalization pass. |

New dependencies to add to `requirements.txt`: `numpy`, `Pillow`, `sentence-transformers` (or a smaller
alternative if install size matters — flag this as a decision point when Stage 2 is actually implemented,
since `sentence-transformers` pulls in `torch`), `numpy-stl`. OpenSCAD and Blender are external binaries, not
pip packages — add detection/install-guidance to the setup scripts, matching how `prefab_to_glb.py` already
handles Blender auto-detection.

## 4. Cost and review-level mapping

Ties req. doc §15's review levels to which pipeline stage produced the asset, so the catalog's
`review_level` field is set automatically rather than guessed by hand:

| `resolution_method` | Stage | Default `review_level` |
|---|---|---|
| `catalog_match` | 2 | 0 |
| `variant` (material/scale only) | 2 + 5 | 1 |
| `composite` | 2.5 | 1 (each sub-part inherits its own level; composite record takes the max) |
| `parametric` (OpenSCAD) | 3 | 2 |
| `imported` | 6 | 3 (or 4 if tagged pedagogically critical in the `AssetSpec`) |

## 5. What a coding agent should NOT do

- Do not build a general-purpose image-to-3D or text-to-3D neural generation path in v1. It's the obvious
  "AI generates it directly" option the user mentioned, but it's the highest-cost, least-controllable option
  and req. doc §3.2 explicitly excludes it from the initial version. If OpenSCAD/composite/import coverage
  proves insufficient after Stage 9 ships, that's a Stage 10 proposal with its own cost/licensing analysis —
  not a default to reach for now.
- Do not let the resolver silently fall back to a wrong-but-present asset. Per req. doc §17, unresolved or
  low-confidence cases must be flagged, not papered over.
- Do not skip Stage 4's normalization for any asset class, including primitives seeded in Stage 0. The whole
  point of req. doc §8 is that *nothing* enters the catalog with unknown scale.
- Do not add Sketchfab or other external sources to Stage 6b until their license metadata is verified by a
  human for that specific source — Poly Haven only until then.
- Do not call `anthropic`/`openai`/`google-genai` SDKs directly from pipeline code. Every LLM call goes
  through `llm/client_factory.py` so provider/model stays a config-level choice, not something baked into
  each generator module.
- Do not make a stage's CLI command depend on another stage having been run first (see §1.1) — each stage
  must be independently testable and runnable, which is also what makes the standalone unit-test tier
  possible.
- Do not synthesize a procedural stand-in for a texture the description identifies as a specific
  real-world surface (Earth, the Moon, a basketball) — that's the texture-flavored version of the
  silent-fallback failure mode. Flag it for human sourcing per Stage 6c.3; a human may explicitly
  choose procedural, the pipeline may not.
- Do not auto-download textures any more than models: Stage 6c.5's fetch runs only on an explicit
  human pick, and new texture sources (beyond Poly Haven and the hand-curated NASA/USGS list) require
  the same human license verification as 6b.

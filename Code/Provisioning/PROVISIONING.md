# Model and texture provisioning: goals spec

Status: **agreed** (#104, milestone "3D Model and Texture Provisioning", parent
#103). It decides what the provisioning tool produces and how. Built so far: the
asset library (#107, `library.py`, `seed_assets.json`) and the plan (#106: the
audit, describe and route, `provision_lab.py --dry-run`), with `smoke_test.py`.
The issues that build the rest are listed at the end.

## Why

Generated labs ask for 3D models and textures that don't exist. In the 104 wave-2
json_lab outputs (`lab_metrics.analyze_assets`):

- A median lab invents **9 prefabs** (max 31) and **1 texture**.
- Across all labs there are **590 distinct prefab names** and **336 distinct texture
  names**, many of them near-duplicates (`dnaHelixPrefab`, `dnaDoubleHelixPrefab`,
  `dnaLadderPrefab`).
- The headset loads prefabs **by name** from what is built into the app, and loads
  textures (JPG) from the lab resources zip. Anything else is missing, so most
  generated labs can't show their own content.

The lab gives little context per asset: the prefab name, the object name, scale,
color, texture and whatever the narration says. Hierarchical runs also give each
object a `purpose`.

## Goals

1. **Provide every asset a finished lab references** that the headset doesn't
   already have: models and textures.
2. **Prefer, in order:** reuse what we have → build it exactly when its form allows
   → find it in a free, openly licensed library → assemble it from simple shapes →
   leave it for a reviewer. Each step is tried only if the previous one gave nothing
   usable.
3. **Work for STEM topics we haven't seen** (next section). Our five lab topics are
   the development set, not the design basis.
4. **Never ship an asset without a known license and size.** Every asset records
   where it came from, who made it, its license and what we changed.
5. **Leave room for a reviewer.** Automatic runs choose for themselves, but keep
   enough (ranked search candidates, reasons) for a person to change the choice
   later without searching again.
6. **Headset-ready output.** Files go into the lab resources zip (#92) next to the
   narration audio, and the bundled copy of the lab points at them.
7. **Grow a library.** Every accepted asset is kept and reused by later labs.
8. **Usable as paper data.** Each run records routes, decisions, costs and times
   per asset, in the spirit of the benchmark metrics files.

Non-goals for now: AI text-to-3D generation (deferred, #119), image-generated
textures (#118), a review UI (#116), changing lab generation (#117), animation and
rigging, ARLEM specs, and audio (narration covers that).

## Designing for topics we haven't seen

- **Routes are chosen by the asset's form, not its subject.** The question is "is
  this a surface on a simple shape, a diagram element, a UI element, a structure
  that exists in a public scientific database, or a recognizable object?", never
  "is this astronomy or biology?"
- **The fallback always ends somewhere.** Search and the shape assembler work for
  any subject. Specialised builders only improve results where they apply; nothing
  depends on them.
- **Builders are plug-ins that describe themselves** (§ Builders). Routing offers
  the LLM the current list by those descriptions, so adding a builder needs no
  routing change.
- **Builders are general by data source**: all of PubChem, all of the Protein Data
  Bank, all of the Crystallography Open Database. No builder is written for one lab
  topic ("VSEPR", "DNA replication").
- **Generality is measured**: a fixed, held-out set of asset requests from STEM
  fields outside our labs is run whenever routing or a builder changes (#115).

A rough keyword split of the 590 wave-2 prefab names into forms, as a sanity check
of the route list (a heuristic, not a measurement):

| Form | Distinct names | Examples |
|---|---|---|
| Lines, arrows, rings, grids, markers, planes | 175 | `arrowPrefab`, `orbitRingPrefab`, `groundPlanePrefab` |
| Molecular and macromolecular structures | 213 | `sf4Prefab`, `dnaHelixPrefab`, `helicasePrefab` |
| UI panels, boards, sliders | 71 | `summaryPanelPrefab`, `sliderPrefab` |
| Anatomy and other organic forms | 50 | `heartPrefab`, `redBloodCellPrefab` |
| Physical objects | 14 | `cannonPrefab`, `projectileBallPrefab` |
| Textured spheres (planets, stars) | 11 | `earthPrefab`, `jupiterPrefab` |
| Other | 54 | `gravityMeterPrefab`, `orderTogglePrefab` |

Textures split into real surfaces (`2k_jupiter`), generic materials
(`cannon_metal_texture`) and diagram textures with content
(`sf4_lewis_structure_texture`). The moon lab's `2k_earth_daymap`, `2k_moon` and
`2k_sun` are Solar System Scope file names, and models already invent `2k_jupiter`,
`2k_mars` and `2k_saturn` in the same pattern.

## Input

- **A complete json_lab `Lab` file:** a one-shot output (`*_json_lab_L3_*`,
  `*_L4_*`, either multi-module structure) or a hierarchical run's assembled lab. It
  must validate as `Code/Schemas/json_lab.Lab`.
- **Optional context:** the paired `*_metrics.json`. For a hierarchical run its
  `registry` gives each invented asset's `purpose`, which goes into the asset
  description.
- **Not supported:** L1 `LabOutline`, module-only `DemoModule` outputs and ARLEM
  specs. The tool exits with a clear message.
- **What counts as an asset:** every distinct `SceneObject.prefab` and
  `SceneObject.texture` value, compared ignoring case and surrounding spaces, in
  initial objects. `demoPrefab` (system) and the moon-lab built-ins are already
  available and are never provided.

## Routes

Each asset gets one **route** from the describe-and-route step (#106). If a route
gives nothing usable, the asset falls through to the next applicable one:

```
reuse ─► builder | kit | textured primitive | ui ─► search ─► composite ─► needs review
```

| Route | For | How | Issue |
|---|---|---|---|
| `reuse` | Anything already in the library | Name or alias hit, else shortlist + LLM judge | #108 |
| `kit` | Diagram elements: arrows, rings, lines, grids, planes, domes, markers, basic solids | A fixed set of shapes built in Python; color from the lab | #109 |
| `textured_primitive` | A surface on a simple shape: planets and moons, but also coins, dishes, maps, screens | Kit sphere / plane / cylinder + a texture from the library or a texture search. Real places are never faked | #109 |
| `ui` | Panels, boards, readouts, controls | Panel from the kit + `textPrefab` children (§ UI-like prefabs) | #109 |
| `builder:<name>` | A structure a registered builder can make from data | The builder (§ Builders) | #111, #114 |
| `search` | Recognizable objects: anatomy, instruments, animals, vehicles | Free libraries, ranked top 10, auto-pick (§ Search) | #110 |
| `composite` | Anything describable as a few simple parts | LLM part plan, code assembles, vision review (§ Composite) | #112 |
| `needs_review` | Nothing above worked | A placeholder ships; a person decides | #116 |

Search falls through to `composite`; `composite` falls through to `needs_review`.
A builder or `textured_primitive` that can't find its data falls through to
`search`.

**Needs review vs failure.** "Needs review" means the tool did its job but an asset
has no acceptable result: it ships as a **placeholder** (a neutral gray kit shape
named for the asset) so the lab still loads, and is listed in the manifest. Failure
means the run itself broke (bad input, an error that stops the run).

## Steps per lab

1. **Audit** (no LLM): list each asset with its context: object names using it,
   module, scale, color, texture pairing, parent, `TextMeshProComponent` text,
   narration lines mentioning the object, and `purpose` if known.
2. **Describe and route** (one LLM call per lab, #106): per asset, a plain
   description, what must be right ("four chambers visible", "standard atom
   colors"), search keywords, its form, its route, and merges of duplicates within
   the lab (`dnaHelixPrefab` and `dnaDoubleHelixPrefab` → one asset).
3. **Reuse** (#108).
4. **Provide** by the route, falling through as above.
5. **Check**: render a thumbnail. Built and assembled assets get a vision review
   against the description and one repair attempt (#112).
6. **Normalize** every model (§ Model convention).
7. **Accept** into the library (license and size required).
8. **Rewrite and bundle** (§ Output).

The manifest is rewritten after every asset, so a run can resume (§ CLI).

## Describe and route (#106, built)

`audit.py` lists the assets (step 1), with up to 8 uses each (all are counted), up
to 3 narration sentences that mention the object or asset by its name's words
(`dnaHelix` → "dna helix"), text shown on the object (also text added by clip
changes), and the hierarchical `purpose`. On all 104 wave-2 labs it finds exactly
the invented assets `lab_metrics.analyze_assets` counts.

Each asset is then looked up in the library by name or alias (`find`; a different
kind or a non-reusable entry such as `demoPrefab` doesn't count). Without a library
the seed list's built-ins are used, so the headset's own assets are never planned.
Only what the library doesn't have goes to the LLM.

`routing.py` makes the one call per lab (`LabAssetPlan`): for each asset a
description, `must_be_right` facts, 2 to 6 search keywords, a form, a route, a
one-sentence reason, and `same_as` for two names of one object. The prompt lists the
lab (id, course, objectives, modules), each asset's context, and the forms and
routes with their descriptions; registered builders (`builders.py`, empty until
#111) are offered as `builder:<name>` routes by their own descriptions.

| Kind | Forms | Routes |
|---|---|---|
| model | `diagram_element`, `ui`, `text`, `textured_primitive`, `structure`, `organism`, `object`, `other` | `kit`, `textured_primitive`, `ui`, `search`, `composite`, `builder:<name>` |
| texture | `real_surface`, `material`, `diagram`, `other` | `search`, `draw` (drawn by code), `generate` (image model, #118) |

The response is checked under `routing_scope` (a ContextVar, like
`asset_plan_validation`): every listed asset planned exactly once with its listed
kind, a form and route allowed for that kind, a non-empty description, keywords, and
`same_as` pointing at another listed asset of the same kind that isn't itself
merged. All problems come back in one error, so instructor reasks once. Names and
`same_as` are returned in the lab's own spelling.

The call uses `create_instructor_client`, `provider_create_kwargs` and
`_transient_retryer`, tracked by `InstructorTracker`; the default model is Claude
Haiku 5.5 (`--model`). `routing.check_environment` stops before any call if the SDK
or API key is missing.

## Model convention

Starting proposal, to be confirmed with the headset importer in #105:

- **GLB** (glTF 2.0 binary), textures embedded.
- +Y up, +Z forward, meters, pivot at the center of the bounding box.
- Scaled so the largest dimension is **1.0**, so the lab's `scale` behaves as it
  does on the built-in sphere (its size to be confirmed).
- Budgets for Quest 3: at most 50,000 triangles per model and 2048 px per texture
  (proposal; set in #105). Larger models are decimated; if that fails, the next
  candidate is used.
- Materials: glTF PBR metallic-roughness, one material per colored part.

Textures are written as **JPG** (the headset loads JPG today), 2048 px or less on
the long side, sRGB.

Built with `trimesh` and `pygltflib` in Python. Blender is optional, only as a
converter for formats those can't read (e.g. FBX); no route requires it.

## Asset library (#107, built)

One shared library, separate from runs, that grows with every accepted asset
(`Code/Provisioning/library.py`).

**Storage: a local folder for now.** Assets build up in that folder over time. Later
the library may move to an API or web server; everything else talks to it through
one small interface (`AssetStore`: identity, entries, get, find, put, add_alias,
path_of, validate), so a server-backed store can replace `LocalFolderStore` without
changing the rest of the tool.

**Location: outside git.** Chosen by `--library`, else `$ARLEM_ASSET_LIBRARY`, else
`Artifacts/Assets/Library/` (gitignored):

```
<library>/
  library.json              the index: library id, revision, one entry per asset
  <asset_id>/model.glb      or texture.jpg
  <asset_id>/thumbnail.png
```

**Results differ between environments.** Because asset files are not in git, each
machine's library holds whatever was seeded and provided there. The same lab can
therefore reuse an asset on one machine and search or build it on another. To keep
this visible:

- Every library has a random `library_id`, a `revision` that goes up on every write,
  and a `content_hash` over its entries (file checksums included, download dates
  left out). Every provisioning run records these in its manifest. Two runs used the
  same library contents exactly when `library_id` and `content_hash` match; two
  libraries seeded from the same list on different machines share a `content_hash`.
- `library.py info` prints them; every library command prints them in its header.
- Tests never use a real library: the smoke test builds a throwaway one from a test
  seed list served by a mock server.
- To reproduce a result elsewhere, start from the seed list (below) and compare the
  recorded identity.

**Entry:** `asset_id` (snake_case), kind (`model` / `texture`), `name` (the name labs
use and the file stem it ships under), `builtin`, `reusable` (false for system
objects such as `demoPrefab`), form, `mapping` (textures: equirectangular, tileable
or uv), description, tags, **aliases** (other lab names matched to it), files (role,
name, sha256, bytes), bounds and triangle count (models), pixel size (textures),
review status (`builtin` / `auto` / `reviewed`), and provenance: method, source
site, download URL, page URL, record id, author, title, license, license URL,
changes made, notes, date obtained.

**Acceptance rules** (`put` refuses an entry that breaks them, and leaves the
library as it was):

- a license in the policy, and for CC-BY an author, a title and a source URL;
- a known size: bounds (three positive numbers) and triangles (positive) for a
  model, pixels (two positive numbers) for a texture; malformed sizes can't even be
  set on an entry;
- files that match their checksums, textures that are JPEG, models that are GLB,
  each stored under its role's fixed name (`model.glb`, `texture.jpg`,
  `thumbnail.png`), so a new file of one role can never overwrite another's;
- no name or alias already used by another entry (compared ignoring case), and no
  blank alias.

**License policy (decided):** CC0, public domain and CC-BY. Never NC, ND,
share-alike, "all rights reserved" or unknown. `credit_line()` gives the credits-file
line for each CC-BY asset. Hits outside the policy are counted and reported, never
silently dropped.

**Built-ins:** the moon-lab prefabs `textPrefab`, `demoPrefab`, `clickableSphere`,
`SunPrefab`, `tinySphere`, `moveableSphere` and `robotIdle`, and textures
`2k_earth_daymap`, `2k_moon`, `2k_sun` and `balldimpled`, are `builtin` entries:
matched and reused, never shipped. Their descriptions are written from the moon lab
and should be checked against the Unity prefabs. `2k_earth_daymap`, `2k_moon` and
`2k_sun` are Solar System Scope maps, so the library also keeps their files, in case
the headset needs them in the zip (#105). A built-in is recorded even when its file
can't be downloaded (it is on the headset either way), and reported as `partial`: it
keeps its current file if that file is intact (a failed download never deletes a
good file) and drops only a missing or damaged one.

**Seed list (in git):** `Code/Provisioning/seed_assets.json` holds metadata only:
the built-ins, plus the assets we explicitly keep with their download URL, license,
author and sha256. Today that is the 15 real Solar System Scope 2k maps (CC-BY 4.0,
"based on NASA elevation and imagery data"): Sun, Mercury, Venus surface and
atmosphere, Earth day, night and clouds, Moon, Mars, Jupiter, Saturn, Uranus,
Neptune, stars, and stars with the Milky Way. Its `_fictional` dwarf-planet maps,
the PNG ring map and the TIF normal and specular maps are left out. The kit's build
parameters join it in #109. Seeding:

- downloads only what is missing or doesn't match its pin, and checks the file on
  disk, not just the index;
- refuses a download whose checksum changed at the source, and, before downloading
  it, an unpinned one unless `--pin` is given; `--pin` records a checksum only once
  the asset is accepted, saves it to the seed list straight away (so a killed run
  keeps the pins it got), and never overwrites an existing pin; if the seed list
  can't be written, the asset is still reported as added, with the pin problem, and
  the run exits 1;
- treats a download that stops short of its `Content-Length` as a dropped
  connection, never as a complete file;
- retries transient failures with the shared policy in
  `Code/Benchmark/transient_policy.py` (the attempts, back-off and HTTP statuses
  LLM calls and narration use): those HTTP statuses, timeouts, refused, reset or
  cut-off connections, a temporary DNS failure, a proxy answering 429/5xx. A host
  that doesn't exist, a proxy refusing the host (e.g. 403), a certificate that fails
  verification and other 4xx fail at once;
- never stops for one entry: any failure fails only that entry, with its reason. An
  entry refused because another run wrote to the library meanwhile is retried once
  against the current library.

**Aliases** make reuse cheap: each time a lab name is matched to an entry, the name
is added, so the next lab with that name hits it with no LLM call. Aliases are
learned from accepted matches (#108), not written by hand for our five lab topics.

**Writers:** every write holds an OS lock on `library.lock` (`flock`, or `msvcrt`
on Windows) from its check to its finish; another writer waits up to 30 s. The
operating system releases the lock if a run dies, so there is never a stale lock to
clear. Under the lock, a write is
refused (`ConcurrentWriteError`) if `library.json` changed since that store read it,
so two runs can never silently overwrite each other. The refused store then holds the
current index, so the caller can look again and retry. `library.json` is written
atomically and readable by everyone (0644).

```
python "Code/Provisioning/library.py" seed     [--library PATH] [--only ID ...] [--pin]
python "Code/Provisioning/library.py" validate [--library PATH] [--json]
python "Code/Provisioning/library.py" info     [--library PATH]
python "Code/Provisioning/library.py" list     [--library PATH] [--kind model|texture] [--json]
python "Code/Provisioning/library.py" find     NAME [--library PATH]
```

Exit codes: 0 ok; 1 validation errors, or seed entries that failed or are `partial`;
2 bad input (no library, unreadable seed list, unknown `--only` id). `validate` never
creates a library, and `seed` checks its arguments before creating one.

## Reuse (#108)

1. Name or alias hit, case-insensitive.
2. Shortlist: the whole library while it has fewer than about 200 entries; above
   that, the 10 to 20 nearest by embedding similarity of descriptions.
3. One LLM judge per lab (all its assets at once): for each asset, the entry to
   reuse or none, a reason, and whether the lab's `color` and `scale` can adapt the
   match (a gray sphere tinted red is fine; a heart is not a lung).

## Search (#110)

**Sources** (one adapter each, all returning the same candidate record):

| Source | Content | License | Access |
|---|---|---|---|
| Poly Haven | Props, furniture, nature; PBR textures | CC0 | Public API, no key |
| NASA 3D Resources | Spacecraft, planets, moons | Public domain | GitHub repository |
| Smithsonian Open Access | Scanned objects and specimens | CC0 per item | API (api.data.gov key) |
| Poly Pizza | Low-poly models | CC0 or CC-BY per model | API, free key |
| ambientCG | PBR materials | CC0 | API, no key |

To evaluate before adding: **Objaverse** (about 800,000 Sketchfab-derived objects
with captions; per-object license) and **Sketchfab** (needs a user login; owner
changed in August 2026). NIH 3D has no documented API and is a manual source.
Each source's terms are checked before its adapter is built.

**Candidate record:** source, record id, name, page URL, thumbnail URL, download
URL and format, license, author, triangle count and file size if known.

**Ranking:**

1. Up to 30 hits per asset across sources, from the asset's search keywords.
2. Drop hits outside the license policy, in formats we can't read, or over budget
   (counted).
3. A cheap text pass drops obvious mismatches by name and tags.
4. **One vision call ranks the remaining thumbnails** against the description:
   order, pass or fail, one-line reason each.
5. An automatic run takes rank 1 if it passed; otherwise the asset goes to
   `composite`.

The manifest keeps the **ranked top 10** with thumbnails, links, licenses and
reasons, and who chose (`auto` / `reviewer`). A reviewer's choice is applied with
`--resume` and no new search (#116).

## Builders (#111, #114)

A builder is a plug-in that turns data into a model. Its contract:

- `name` and a **plain-words description** of what it can make and what it needs,
  e.g. "any small molecule, given its name or formula". The route step sees these
  descriptions, not code.
- `build(request) → result`: a GLB plus provenance (data source, record id,
  license), or "can't build this" with a reason, which sends the asset to `search`.
- Deterministic for the same input and data.

First builders:

| Builder | Data | Output | Issue |
|---|---|---|---|
| `molecule` | PubChem 3D coordinates (public domain; PUG REST, no key); RDKit from SMILES as a fallback | Ball and stick, standard CPK colors and radii, bond orders; lone pairs as translucent lobes when asked | #111 |
| `macromolecule` | RCSB Protein Data Bank (CC0) | Backbone tube with chains colored apart, or ball and stick for small pieces; DNA as helix or ladder, bases colored by a recorded key; an idealized B-DNA option | #114 |
| `crystal` | Crystallography Open Database (CC0 / public domain) | Unit cell or small lattice block, atoms colored by element | #114 |

## Composite (#112)

The fallback for anything describable as a few parts (kept from JW's pipeline,
rewritten):

- One LLM call returns a **part plan**: each part is one kit primitive with its own
  color, relative position and relative size, up to 12 parts.
- **Connectors** (`bond_between`): a part that joins two others names them; code
  computes its position, length and rotation. The LLM is not asked to do the 3D math.
- Code assembles the parts into one GLB with a material per part.
- A **vision review** of the render against the description, and one repair from
  its complaint (a corrected part list). A failed review means `needs_review`, not a
  silent pass.

## UI-like prefabs (#109)

The headset has no panel prefab. A UI-like object (`summaryPanelPrefab`,
`historyPanelPrefab`, `sliderPrefab`) is replaced in the bundled lab by:

- a **panel** object from the kit (a flat rounded board), sized from the original
  `scale` and colored from its `color`, and
- one or more `textPrefab` children with `TextMeshProComponent` text, written by
  an LLM from the module's narration and the object's role.

Clip deltas that changed the original object are moved to the panel. Interactive
controls (sliders, toggles) keep a label but lose their behaviour unless the lab
already gives them a component that provides it; each is listed in the manifest.
This mapping is a first try, judged in headset testing.

## Output

### Working folder

`Artifacts/Data/Provisioning/[<run group>/]<run name>/`, never pooled with
`Benchmark/`:

```
provisioning_manifest.json     the run record and one record per asset
assets/<asset_id>/             files this run made or fetched, before acceptance
candidates/<asset_id>/         thumbnails of ranked search candidates
<Name>Lab/                     the bundle folder (Models/, Textures/, credits, lab JSON)
<Name>Lab.zip
```

Model files, textures and zips are gitignored, as narration's audio is. The
manifest is tracked like narration's.

### Bundled lab

A copy of the input lab. The original is never edited. In the copy:

- `prefab` values for provided models name their file in `Models/`; reused
  built-ins keep their name.
- `texture` values name their file in `Textures/`.
- UI-like objects are rewritten as above.
- The result must validate as a json_lab `Lab`.

How the headset tells a provided model from a built-in is settled in #105
(proposal: a `prefab` with a matching `Models/<prefab>.glb` is loaded from the
file; anything else is a built-in).

### Resources zip

The zip from #92 gains:

- `Models/<name>.glb`, one per provided model used by the lab
- `Textures/<name>.jpg`, one per provided texture
- a credits file at the bundle root listing every CC-BY asset (title, author,
  source URL, license, changes made); its name and format are set in #105

`bundle.py validate` checks that every non-built-in `prefab` and every `texture`
in the bundled lab has its file (#105). Narration and provisioning can run in
either order on the same bundle.

### `provisioning_manifest.json`

**Run record:**

| Group | Fields |
|---|---|
| Source | input path; `labId`; source model and level; `lab_name` (from the paired metrics file, if any) |
| Settings | LLM model(s) per step; library identity (`store`, `location`, `library_id`, `revision`, `content_hash`, `entries`) at the start and end of the run; license policy; sources used |
| Counts | assets by route; reused, built, found, assembled, placeholders; aliases added |
| Totals | LLM cost; wall time; transient retries; download size |
| Provenance | tool git commit; package versions; timestamps; `runs` list (one entry per run of the folder) |
| Status | `complete`, `needs_review` (placeholders shipped), `incomplete` (some assets failed; resume it), `failed` (stopped early) or `invalid` (the zip failed validation) |

**Per asset:** lab name(s) and merged duplicates; context; description; form;
route and every fallback taken, with reasons; the reuse shortlist and verdict;
ranked candidates (search); builder input and record id; vision review verdict and
repair; the accepted library entry; license and provenance; cost and time; status
(`reused`, `built`, `found`, `assembled`, `placeholder`, `failed`); and a **source
hash** of the request and settings, so `--resume` skips assets whose inputs are
unchanged.

### Paper metrics

From the manifest: share of assets per route, reuse rate (and how it grows as the
library grows), auto-pick acceptance in review, cost and time per lab and per asset,
and placeholder rate, reported per source model and level, and for the held-out set
per field (#115).

## Provisioning CLI (#106 built; #113)

```
python "Code/Provisioning/provision_lab.py" <lab_output.json> --dry-run [--no-llm]
    [--model KEY] [--library PATH] [--metrics PATH] [--run-group NAME]
```

Built so far (#106): the plan. `--dry-run` is required until #113.

- `--dry-run` audits the lab, looks every asset up in the library, makes the one
  describe-and-route call for the rest, prints the plan and saves
  `provisioning_manifest.json` in `Artifacts/Data/Provisioning/[<run group>/]<lab>_<timestamp>/`
  (a second run in the same second gets its own folder). It downloads and builds
  nothing.
- `--no-llm` stops after the audit and the library lookup: no API key needed.
- `--metrics` names the lab's metrics file; by default the paired `*_metrics.json` is
  used (beside the output or in the sibling `Metrics/`), for the source model, level
  and hierarchical purposes.
- Exit codes: **0** planned (or audited with `--no-llm`); **1** the call failed
  (recorded in the manifest); **2** bad input or environment (not a lab, bad run
  group, unknown model, missing SDK or API key).
- The plan manifest records: `status` (`planned`, `audited`, `failed`); the source
  (input, `labId`, metrics file, source model, level, lab name, whether
  hierarchical); the library identity (or a note that there is none); the routing
  record (model, model id, provider, reasoning effort, `tool_choice`, calls, seconds,
  transient retries, served models, cost, the tracker summary, any error); the routes
  offered; counts by status, route and form; and per asset its spellings, uses,
  narration, purpose, `status` (`available` with the library entry and whether it
  matched by name or alias, `planned`, `merged` into another, or `unplanned`) and the
  plan.

Still to come (#113): the full run.

```
python "Code/Provisioning/provision_lab.py" <lab_output.json>
    [--library PATH] [--model KEY] [--bundle FOLDER] [--run-group NAME] [--overwrite]
python "Code/Provisioning/provision_lab.py" --resume <run folder> [--overwrite]
```

- `--bundle` adds to an existing bundle folder (e.g. one narration made) instead of
  starting a new one.
- `--resume` reruns a folder with its saved settings, skipping finished assets and
  applying reviewer choices recorded in the manifest.
- Exit codes: **0** complete and valid; **1** incomplete, failed or invalid (resume
  it); **2** bad input or environment; **3** complete but `needs_review`
  (placeholders shipped).
- One asset failing is recorded and the run continues. HTTP 401/403/404 from an LLM
  provider stops the run. A source's errors only affect that source's candidates.

## Engineering rules (from the repo, not from JW's code)

- **LLM calls** go through `benchmark.create_instructor_client` plus
  `provider_create_kwargs` and are tracked with `InstructorTracker`, like every LLM
  entry point (CLAUDE.md). No retry loop around instructor.
- **Retries:** transient errors only (429 / 5xx / connection), as in
  `benchmark._transient_retryer()`, counted separately; this applies to source APIs
  and downloads too. Authentication and configuration errors fail at once.
- **Pricing** in `Code/Benchmark/tracking/pricing.py`; cost from reported usage,
  including reasks.
- **Environment check** before any call, in the style of
  `benchmark.require_environment()`: SDKs installed, API keys set for the chosen
  LLM and keyed sources.
- **Dependencies** in the root `requirements.txt` (`trimesh`, `pygltflib`, `Pillow`;
  `rdkit` optional), keys in the root `.env`. No separate environment.
- **Downloads** are untrusted: each goes into its own new folder, archive paths are
  checked so nothing writes outside it, file types are checked before parsing, and
  nothing downloaded is executed.
- **Offline smoke test** (`smoke_test.py`) with a fake LLM and mock source servers,
  in the style of `Code/Hierarchical/smoke_test.py` and `Code/Narration/smoke_test.py`.
- **Atomic writes** for the manifest and `library.json`.

## Decisions (owner, 2026-11-03)

1. Prefabs load by name from what is built in. Provided models need a new headset
   importer and a `Models/` folder in the zip. Textures load today; all are JPG
   celestial maps.
2. The moon-lab built-ins are the prefabs the moon lab uses (listed above); their
   source files exist in the Unity project.
3. CC-BY is acceptable, with a credits file.
4. Model and texture files stay outside git, except what we explicitly include
   (the seed list).
5. No hosted AI 3D generation for now; judge the other routes first.
6. UI-like prefabs map to panels plus text prefabs, then are judged in testing.

## Open questions (settled in the listed issue)

- How the headset tells a provided model from a built-in, the built-in sphere's
  size, budgets, and the credits file's name (#105).
- How the importer applies a lab's `color` to a model with several materials:
  tint all, tint none, or tint only uncolored parts (#105).
- Default LLM model per step (#106), chosen from the current roster and recorded.

## Review of JW's `asset_pipeline/`

Branch `jw-expermiments`, commit ff8f529. A separate sub-project (about 10,000
lines of Python plus a Flask app) that turns **a typed sentence** ("a small gray
moon, 15 cm across") into a scaled, cataloged GLB:

1. An LLM parses the sentence into an `AssetSpec`.
2. A word-overlap matcher checks the catalog; otherwise an LLM classifies the object
   as `composite`, `parametric`, `imported` or `unclear`.
3. Composite: the LLM lists 2 to 12 primitive parts; Blender bakes them into one GLB.
4. Parametric: the LLM writes OpenSCAD, compiled with a repair loop and a check of
   the size it claimed.
5. Imported: search Poly Haven, NASA 3D Resources and Thingiverse; an LLM checks
   relevance by name, then by thumbnail; only a confident Poly Haven hit downloads
   without a person.
6. Blender normalizes every mesh (pivot, scale, decimation, UV unwrap).
7. Materials and a texture registry; real surfaces are never faked.
8. A vision LLM reviews a render and repairs it.
9. Validation, thumbnails, a build report, and a copy into a three.js viewer.

286 of its 287 unit tests pass offline on Linux; the one failure is a path-escape
test written with Windows separators. Its library holds 22 assets and 11 planet
textures, mostly test items.

**It never reads a lab** (JW's separate `asset_audit.py` reads one, but isn't
connected), and **it has no path to the headset** (output goes to the three.js
viewer). Reviewed against the goals above:

| Piece | Verdict | Why |
|---|---|---|
| Ladder: catalog → build → search → flag | **Keep idea** | Matches goal 2. Ours routes by form first and ships a placeholder instead of stopping. |
| Typed-sentence input (`spec`, `resolve-scene`) | **Drop** | Goal 1: the input is a lab. |
| Catalog matcher (word overlap + `difflib`, stopword lists) | **Drop** | **Reproduced:** it splits words only on non-letters, so camelCase lab names are one token. Against an entry tagged `dna`, `double`, `helix`, `dnaHelixPrefab` scores 0.16 and `dnaDoubleHelixPrefab` 0.23, far below its 0.85 match threshold. `dna ladder` scores 0.49, enough to be called a "variant" of the double helix. Its stopword lists were patched after each false match. Replaced by aliases, a shortlist and an LLM judge. |
| LLM geometry classifier | **Keep idea, rewrite** | One call per asset, subject-agnostic categories. Ours is one call per lab, routes by form, and lists registered builders. |
| Composite part plan with `bond_between` | **Keep idea, rewrite** | The best idea in the branch: the LLM decides what connects, code does the 3D math. Ours builds with `trimesh`, not Blender. |
| OpenSCAD parametric route | **Drop as default** | CSG output carries no per-part color, few of our assets are mechanical, and it needs a binary. Could return as a builder. |
| Blender for normalizing and baking | **Drop as default** | A hard dependency for most routes. `trimesh` / `pygltflib` cover building, normalizing and GLB export; Blender stays an optional converter. |
| Poly Haven and NASA 3D Resources clients | **Keep idea, rewrite** | Right sources (CC0 and public domain, no key). Ours share one candidate record and keep 30 hits, not 5. |
| Thingiverse | **Drop** | STL only (no color), mixed licenses, needs a token, mostly 3D-printing parts. |
| License tiers and allowlist; hidden hits counted | **Keep idea** | Our policy is fixed: CC0, public domain, CC-BY. |
| Provenance record and "no license, no size, no entry" | **Keep** | Goal 4. Ours adds the bundle credits file. |
| Two-pass relevance review (name, then thumbnail) | **Keep idea, extend** | Ours ranks the whole shortlist in one vision call and keeps the top 10, instead of a yes/no per candidate. |
| Auto-adopt only for Poly Haven | **Rewrite** | Ours auto-picks rank 1 from any source within the license policy if the vision ranking passed it; the top 10 stay available for review. |
| Vision review of built assets with repair | **Keep idea** | Catches broken builds (sticks that miss atoms). |
| Texture registry, download once and reuse | **Keep idea** | Folded into the one library. |
| Procedural noise textures (numpy) | **Drop** | Real surfaces come from maps, generic materials from ambientCG / Poly Haven. |
| Planet maps in `intake/` (Solar System Scope, CC-BY 4.0) | **Reuse as data, re-downloaded** | Their names match what our models invent. Fetched again from the source with licenses checked, not copied from the branch. |
| `llm/client_factory.py` with `tenacity` around instructor | **Drop** | **Reproduced:** `classifier._call_with_usage` retries every exception, including a 401 bad-key error (5 attempts before failing), and each attempt allows instructor 3 reasks, so up to 15 requests. It bypasses `create_instructor_client` and the Anthropic `tool_choice` guard. |
| `LLMCallEntry` cost logging | **Drop** | **Found:** `cost_usd` is never set anywhere (always 0.0), and usage is read only from the final response, so reasks go uncounted. We use `tracking/`. |
| "Needs review" result separate from failure | **Keep idea** | Ours is a status plus exit code 3, with a placeholder so the lab still loads. |
| Build report | **Keep idea, merge** | Into `provisioning_manifest.json`, rewritten after every asset. |
| Flask review app (Library / Create / Worklist) | **Defer** | The review data is kept now; a review tool comes later (#116), using this app as a reference. |
| Previews via Playwright + three.js | **Keep idea, rewrite** | Thumbnails are needed for review and vision checks; ours renders offline where possible. |
| `sync` into the viewer's assets folder | **Drop** | Our output is the resources zip. |
| Separate `requirements.txt`, `.env` and test tree | **Drop** | Use the repo's own. |
| Library contents (snowman, stool, armchair...) | **Drop** | Test items. |
| `datetime.utcnow()` | **Drop** | Deprecated; use timezone-aware times. |

Nothing from `asset_pipeline/` is copied. The ideas marked "keep" are reimplemented
to this spec.

## Sources researched (2026-11)

Model and texture libraries: Poly Haven, NASA 3D Resources, Smithsonian Open
Access, Poly Pizza, Sketchfab (download API needs a user login), Objaverse /
Objaverse++ / Cap3D, NIH 3D, ambientCG, NASA/USGS planetary maps, Solar System
Scope (CC-BY 4.0). Structure data: PubChem, RCSB PDB, Crystallography Open Database.
Generation (deferred): Meshy, Tripo, Rodin, Hunyuan3D (hosted, roughly $0.20 to
$1.50 per model in 2026, terms vary by plan) and TRELLIS.2 (MIT, open weights).
Code-CAD (not planned): OpenSCAD, CadQuery, build123d, Blender `bpy`; AutoCAD is
commercial and Windows-only with no good headless path. Verify terms, prices and
APIs before building against any of them.

## Issues

| Issue | Scope |
|---|---|
| #104 | This spec |
| #105 | Headset contract: `Models/`, naming, convention, budgets, credits, `bundle.py` |
| #106 | Audit, describe and route; `--dry-run` |
| #107 | Asset library, seed list, built-ins, planet maps |
| #108 | Reuse matcher |
| #109 | Kit, textured primitives, UI mapping |
| #110 | Search, ranked top 10, auto-pick |
| #111 | Builder interface; `molecule` builder |
| #112 | Composite builder with vision review |
| #113 | Normalize, rewrite, bundle, resume, smoke test |
| #114 | `macromolecule` and `crystal` builders |
| #115 | Held-out generality check |
| #116 | Later: reviewer tool |
| #117 | Later: library-aware lab generation |
| #118 | Later: texture generation and diagram textures |
| #119 | Later: AI 3D generation |

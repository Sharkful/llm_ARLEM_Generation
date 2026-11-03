# 3D Model and Texture Provisioning: Research and Draft Plan

**Status: draft for discussion (2026-11-03).** Nothing here is built yet. Once the
plan is agreed, it becomes a set of issues under the milestone "3D Model and
Texture Provisioning." It follows the narration work (#90 to #93): review JW's
code for ideas, then build our own tool on our stack.

## 1. Goal

A command-line tool that takes a finished lab (json_lab v2.0) and provides every
model and texture it references that we don't already have:

```
python "Code/Provisioning/provision_lab.py" <lab>_output.json --dry-run   # what does this lab need?
python "Code/Provisioning/provision_lab.py" <lab>_output.json --run-group testing
python "Code/Provisioning/provision_lab.py" --resume "Artifacts/Data/Provisioning/testing/<run>"
```

For each asset the lab asks for, the tool tries, in order:

1. **Reuse** something we already have (moon-lab built-ins, or anything provisioned
   before), matched by meaning, not just by name.
2. **Build it from known parts** when that is cheap and accurate (a planet is a
   sphere plus a NASA map; a water molecule is three balls and two sticks).
3. **Search** free, openly licensed 3D libraries, rank the hits, and auto-pick the
   best one, keeping the top 5 to 10 so a reviewer can choose later.
4. **Generate** a model when nothing suitable exists, and flag it for review.

The result goes into the lab resources zip next to the narration audio, and the
bundled copy of the lab points at the provided files.

## 2. What our labs actually ask for

Measured on the 104 wave-2 json_lab outputs (`lab_metrics.analyze_assets`):

- A median lab invents **9 prefabs** (max 31) and **1 texture**.
- Across all labs: **590 distinct prefab names** and **336 distinct texture names**.
  Many are near-duplicates (`dnaHelixPrefab`, `dnaDoubleHelixPrefab`,
  `dnaLadderPrefab`), so matching by meaning matters a lot.
- The lab gives little context per asset: the prefab name, object name, scale,
  color, texture, and whatever the narration says. There is no description field.
  Hierarchical runs are better: the asset plan gives every object a `purpose`
  (kept in the metrics file's `registry`).

A rough keyword split of the 590 distinct prefab names (a heuristic, not a
measurement):

| Kind | Distinct names | Uses | Examples | Best route |
|---|---|---|---|---|
| Lines, arrows, rings, grids, markers, planes | 175 | 288 | `arrowPrefab`, `orbitRingPrefab`, `groundPlanePrefab` | A small kit of standard shapes |
| DNA and proteins | 107 | 195 | `dnaHelixPrefab`, `helicasePrefab`, `okazakiFragmentPrefab` | Build (DNA); search (proteins) |
| Molecules and atoms | 106 | 163 | `atomPrefab`, `lonePairCloudPrefab`, `sf4Prefab` | Build from real chemistry data |
| UI panels, boards, sliders | 71 | 101 | `summaryPanelPrefab`, `sliderPrefab` | Not a 3D model: map to `textPrefab` or a panel |
| Anatomy | 50 | 96 | `heartPrefab`, `ventricleCutawayPrefab`, `redBloodCellPrefab` | Search, then AI generation |
| Physical objects | 14 | 26 | `cannonPrefab`, `projectileBallPrefab` | Search, or build from shapes |
| Planets, stars | 11 | 64 | `earthPrefab`, `jupiterPrefab` | Sphere plus texture |
| Other | 54 | 56 | `gravityMeterPrefab`, `orderTogglePrefab` | Case by case |

The takeaway: **open-library search is the right tool for only about 15% of the
names** (anatomy, real objects, some proteins). About 40% are simple shapes or UI
that should never trigger a search, and about 35% (molecules, DNA) are better
built from data, because the result must be scientifically correct and colored by
convention.

Textures split the same way: real surfaces (`2k_jupiter`, `2k_mars`), generic
materials (`cannon_metal_texture`), and **diagram textures** with content
(`sf4_lewis_structure_texture`, `graph_paper_texture`). The moon lab's own
`2k_earth_daymap`, `2k_moon` and `2k_sun` are Solar System Scope's file names, and
models already invent `2k_jupiter`, `2k_mars`, `2k_saturn` in the same pattern.
A curated planet set covers those by exact name.

## 3. Review of JW's `asset_pipeline` (branch `jw-expermiments`)

### What it is

A separate sub-project (`asset_pipeline/`, about 10,000 lines of Python plus a
Flask app, 287 unit tests). It turns **a typed sentence** ("a small gray moon,
15 cm across") into a scaled, cataloged GLB through nine stages:

1. LLM parses the sentence into an `AssetSpec`.
2. A word-overlap matcher checks the catalog. If nothing matches, an LLM classifies
   the object as `composite`, `parametric`, `imported` or `unclear`.
3. **Composite:** the LLM lists 2 to 12 primitive parts with colors and positions.
   Blender bakes them into one multi-material GLB.
4. **Parametric:** the LLM writes OpenSCAD, which is compiled (with a repair loop on
   compile errors) and checked against the size it claimed.
5. **Imported:** search Poly Haven, NASA 3D Resources and Thingiverse. An LLM checks
   relevance by name, then by thumbnail. A download happens automatically only for a
   confident Poly Haven hit; otherwise candidates wait for a person in the web app.
6. Blender normalizes every mesh (pivot, scale, decimation, UV unwrap).
7. Materials: LLM picks shader values, and textures are reused from a texture
   registry. Real surfaces (planets) are never faked; they wait for a person.
8. A vision LLM reviews a render against the request and repairs it.
9. Validation, thumbnails, a build report, and a copy into the three.js viewer.

It works: 286 of 287 unit tests pass offline. The one failure is a path test that
only makes sense on Windows, where JW built it. The library holds 22 assets and 11
planet textures, mostly test items (a snowman, a stool, a methane molecule).

### What it doesn't do

- **It never reads a lab.** Input is hand-typed sentences or a list of them. JW's
  separate `asset_audit.py` reads a lab and lists its assets, but the two aren't
  connected.
- **It has no path to the headset.** Output is synced into a three.js preview
  viewer. The headset loads prefabs by name and the resources zip has no models
  folder.
- The catalog matcher is word overlap plus `difflib`, patched incident by incident
  with stopword lists ("spacecraft", "furniture"). It won't tell `dnaHelixPrefab`
  from `dnaLadderPrefab` well.
- Search returns at most 5 hits per source, ranked by word overlap, so the top 5 to
  10 a reviewer should see may never be fetched.
- It has its own LLM client with an outer retry around instructor's retries (up to 15
  attempts per call), and none of our #44 Gemini/Anthropic guards or cost tracking.
- It needs Blender and OpenSCAD installed for most routes.

### Verdicts

| JW piece | Verdict | Why |
|---|---|---|
| Overall ladder: reuse → build → search → flag | **Keep idea** | Matches the goal; ours adds AI generation at the end and routes by asset kind first. |
| Provenance record: source, URL, author, license, modified | **Keep idea, rewrite** | Required for every asset. Ours also writes a credits file into the bundle for CC-BY. |
| License tiers (CC0 / +CC-BY / +CC-BY-SA; never NC, ND or unknown) | **Keep idea** | A clear, configurable rule. Hidden hits are counted, not silently dropped. |
| "Nothing enters the library without a license and a known scale" | **Keep** | Same invariant. |
| Composite part plan with `bond_between` (code computes stick placement, not the LLM) | **Keep idea, rewrite** | The best idea in the branch: the LLM chooses what connects, code does the 3D math. Ours builds with `trimesh` in Python, not Blender. |
| Relevance review in two passes: name, then thumbnail | **Keep idea, extend** | Ours ranks the whole shortlist in one vision call and keeps the top 10, instead of a yes/no per candidate. |
| Vision review of a built asset, then repair | **Keep idea** | Catches broken builds (sticks that don't touch atoms). |
| NASA 3D Resources via the GitHub tree API, cached | **Keep idea, rewrite** | Public domain, no key, GLB plus thumbnail per model. |
| Poly Haven client | **Keep idea, rewrite** | CC0, no key. Also covers textures. |
| "Needs review" exit code, separate from failure | **Keep** | Same as narration's run status. |
| Texture registry, download once and reuse | **Keep idea** | Folded into the one asset library. |
| Planet textures (Solar System Scope, CC-BY 4.0) | **Reuse as data, after checking** | Names match what our models invent. Credit required. |
| Catalog matcher (word overlap, `difflib`, stopword lists) | **Drop** | Replaced by name and alias lookup, then a shortlist, then an LLM judge. |
| Free-text scene input | **Drop** | Our input is the lab. |
| `llm/client_factory.py` with tenacity around instructor | **Drop** | Use `benchmark.create_instructor_client`, `provider_create_kwargs` and the tracker. |
| OpenSCAD as the main generator | **Drop as default** | CSG output has no per-part color, and few of our assets are mechanical. Revisit if needed. |
| Blender as a hard dependency | **Drop as default** | `trimesh` and `pygltflib` cover building, normalizing and exporting GLB in Python. Blender stays optional for FBX conversion. |
| Thingiverse | **Drop** | STL only (no color), mixed licenses, needs a token, mostly 3D-printing parts. |
| Procedural noise textures (numpy) | **Drop** | Real surfaces come from maps; generic materials from ambientCG or Poly Haven. |
| Flask review app (Library, Create, Worklist) | **Defer** | We need the data for review now (ranked candidates in the manifest) and a review tool later. |
| Library contents (snowman, stool, armchair...) | **Drop** | Test items. |

## 4. Options research

### 4.1 Where to find existing models and textures

Checked 2026-11; verify each before building against it.

| Source | Content | License | Access | Fit |
|---|---|---|---|---|
| Poly Haven | Props, furniture, nature; PBR textures | CC0 | Public API, no key | Good for generic objects and materials; little science content |
| NASA 3D Resources | Spacecraft, planets, moons | Public domain | GitHub repo; one tree call lists everything | Space labs |
| Smithsonian Open Access | Scanned museum objects, specimens | CC0 (per item) | Open Access API (api.data.gov key); bulk S3 | Some science and natural history; details to verify |
| Poly Pizza | Low-poly models, GLB | CC0 or CC-BY per model | API with free key | Simple objects, good for headset budgets |
| Sketchfab | 1M+ downloadable models, glTF | CC per model | Download API needs a user login (OAuth). KitBash bought Sketchfab in Aug 2026 | Huge anatomy and biology coverage; API future uncertain |
| Objaverse (Sketchfab-derived dataset) | ~800K objects, with captions (Cap3D) and quality scores (Objaverse++) | Per object (many CC-BY) | Python package; download from Hugging Face | Captions can be searched offline with embeddings; strong coverage |
| NIH 3D | Biomedical: proteins (from PDB), anatomy | Per model | Web download; no documented API found | Proteins (helicase, polymerase) and anatomy |
| ambientCG | ~2,000 PBR materials | CC0 | Free API, no key | Generic material textures |
| NASA / USGS planetary maps | Planet and moon surface maps | Public domain | Manual download | Planets, curated once |
| Solar System Scope | Planet maps, 2k to 8k | CC-BY 4.0 | Manual download | Same names as the moon lab's textures |

### 4.2 Matching against what we already have

Our library will be small for a long time (tens to a few hundred assets). Proposal:

1. Exact or alias name hit (case-insensitive; aliases are recorded each time a match
   is accepted, so `dnaDoubleHelixPrefab → dna_double_helix` is instant next time).
2. Shortlist by embedding similarity of descriptions. While the library is under
   about 200 entries, skip this and send the whole list.
3. One cheap LLM call judges the shortlist: reuse which one, or none, and why. It
   also says whether the lab's color or scale can adapt the match (a gray sphere
   plus a red color is fine; a heart is not a lung).

The same request record drives every later step, so it needs a good description
first. One LLM call per lab writes it for all assets from the lab context (object
names, module, narration lines that mention the object, scale, color), plus the
`purpose` from hierarchical runs when there is one.

### 4.3 Ranking search hits (the reviewer hook)

1. Search every allowed source, keeping up to about 30 hits per asset.
2. Drop hits outside the license tier, in unusable formats, or over the triangle
   budget (counted and reported, as JW does).
3. A cheap text pass removes obvious mismatches by name and tags.
4. **One vision call ranks the remaining thumbnails** against the description:
   order, pass or fail, and a one-line reason each.
5. Automatic runs take rank 1 if it passed. The **ranked top 10 are saved in the
   manifest** with thumbnails, links, licenses and reasons.
6. Later, a review command (or HTML page) shows those 10. A reviewer picks one or
   rejects all, and `--resume` applies the choice. Nothing else needs to change.

### 4.4 Building and generating models

| Option | Good for | Accuracy | Cost and dependencies | Recommendation |
|---|---|---|---|---|
| **Standard kit** (arrow, ring, line, grid, plane, dome, marker), built once in Python (`trimesh`) | ~40% of names | Exact | None; pure Python | **First.** Most lines and arrows only need aliasing to the kit. |
| **Textured sphere** | Planets, stars, moons | Exact | None | **First.** |
| **Molecule builder**: PubChem 3D coordinates (public domain) or RDKit from SMILES → ball and stick with standard CPK colors; lone pairs as translucent lobes | VSEPR lab, any molecule | Real geometry | PubChem API (no key) or RDKit (pip) | **Early.** One of five labs, and correctness matters. |
| **DNA builder**: parametric B-DNA (10.5 base pairs per turn, 3.4 Å rise, 2 nm wide), ladder and helix forms, base colors by a fixed key | DNA lab | Textbook geometry | None | **Early.** |
| **LLM primitive composite** (JW's part plan + `bond_between`), built with `trimesh` | Simple objects: cannon, launcher, target | Recognizable, crude | One LLM call; vision review | **Middle.** |
| **LLM-written CAD code**: OpenSCAD, CadQuery or build123d (Python CAD), or Blender `bpy` scripts | Mechanical parts | Good shapes; color varies by tool | Binaries; running LLM-written code needs a sandbox | **Later, if needed.** AutoCAD is commercial and Windows-only with no good headless path, so not recommended. |
| **AI text/image-to-3D**: hosted (Meshy, Tripo, Rodin, Hunyuan3D; roughly $0.20 to $1.50 per model, license terms vary by plan) or open weights (TRELLIS.2, MIT license) on the cluster GPU | Organic shapes: organs, cells, proteins | Looks plausible; scientific detail unreliable | API cost, or a GPU job | **Last resort**, always flagged for review. Try hosted first, cluster later (as narration did with #95). |
| **Image generation for textures** (OpenAI or Gemini image models) | Stylized generic surfaces | Not for real places | Per-image cost | **Later.** Never for planets. Diagram textures with text should be drawn by code or become text objects. |

## 5. Proposed design

### 5.1 Steps per lab

1. **Audit** (no LLM): list every `prefab` and `texture` reference with context,
   grouped case-insensitively. Built-ins are marked as already available.
2. **Describe and route** (one LLM call per lab): for each asset, a description,
   what must be right ("four chambers visible", "CPK colors"), search keywords, a
   kind, a preferred route (`not_a_model`, `kit`, `textured_sphere`, `molecule`,
   `dna`, `search`, `composite`, `generate`), and merges of duplicates within the lab.
3. **Reuse** (§4.2).
4. **Provide** by the preferred route, falling back along
   build → search → composite → generate.
5. **Check**: render a thumbnail, vision review against the description, one repair
   attempt for built assets.
6. **Normalize**: GLB, +Y up, +Z forward, pivot at the center, scaled to fit a unit
   cube (so the lab's `scale` behaves as it does on the built-in sphere; to be
   confirmed), under the triangle and texture budget for Quest 3.
7. **Bundle**: `Models/` and `Textures/` in the resources zip; the bundled lab's
   `prefab` and `texture` names are rewritten to the provided files (the original lab
   is untouched, like narration's `audioClip` changes); a credits file lists every
   CC-BY item.

Same conventions as narration: `--dry-run`, `--run-group`, `--resume` with a
manifest rewritten after every asset, cost from `tracking/pricing.py`, a
needs-review status separate from failure, and an offline `smoke_test.py` with a
fake LLM and mock source servers.

### 5.2 The asset library

One shared library, separate from runs, that grows with every accepted asset:

```
Artifacts/Assets/Library/
  library.json                  id, kind, description, tags, aliases, bounds,
                                triangle count, provenance, license, review status
  <asset_id>/model.glb | texture.png
  <asset_id>/thumbnail.png
```

Seeded with the moon-lab built-ins (descriptions written by hand; files if we have
them) and the checked planet textures.

## 6. Blockers and open questions

1. **Headset loading (blocker).** The headset loads prefabs by name from the app's
   built-in resources. A new model needs runtime GLB loading (Unity's glTFast can
   load a GLB from a file path) and a `Models/` folder in the zip. Does the headset
   load textures from `Textures/` today, and in what format? Who makes the Unity
   change? `bundle.py validate` currently rejects any top-level folder other than
   `Audios/`, `Textures/` and `Videos/`.
2. **Moon-lab built-in files.** Do we have the source models for `SunPrefab`,
   `moveableSphere`, `robotIdle` and the others? JW's viewer folder has some FBX files.
3. **License tier default.** CC0 and public domain only, or also CC-BY with a credits
   file? The moon lab already ships CC-BY planet textures.
4. **Where binaries live.** GLBs and textures in git (repo growth), git LFS, or
   outside storage. Narration keeps audio out of git.
5. **AI generation budget.** Hosted API spend, or wait for the cluster?
6. **UI-like prefabs.** Map them to `textPrefab` (or a panel prefab), or treat them
   as a lab-generation problem and feed back into the schema prompt?
7. **Feedback into generation.** Later, the library could be listed in the lab
   prompts (`SceneObject.prefab` description, hierarchical registry) so new labs reuse
   provided assets instead of inventing names.

## 7. Draft issue breakdown

Under milestone "3D Model and Texture Provisioning":

1. Define goals and review JW's `asset_pipeline` against them (this document, turned
   into a `PROVISIONING.md` spec).
2. Headset contract: how the headset loads models and textures from the zip, the
   normalization convention, and the `bundle.py` update.
3. Audit, describe and route: `provision_lab.py --dry-run` with the per-lab LLM call.
4. Asset library: format, provenance and license gate, credits file, seed with
   built-ins and planet textures.
5. Reuse matcher: names and aliases, shortlist, LLM judge.
6. Standard kit and textured spheres.
7. Open-library search: source adapters, license tiers, ranked top 10 with vision
   ranking, auto-pick, saved for review.
8. Molecule and DNA builders.
9. LLM primitive composite builder with vision review and repair.
10. Later: AI 3D generation (hosted first, cluster second).
11. Later: reviewer tool for the saved top 10.
12. Later: library-aware lab generation.
13. Later: texture generation and diagram textures.

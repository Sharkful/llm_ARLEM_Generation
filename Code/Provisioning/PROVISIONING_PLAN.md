# 3D Model and Texture Provisioning: Research and Plan

**Status: plan, revision 2 (2026-11-03).** Revised after review: the design no
longer depends on our five lab topics, and the owner's answers are folded in
(§7). Nothing is built yet. Next step: turn §8 into issues under the milestone
"3D Model and Texture Provisioning." It follows the narration work (#90 to #93):
review JW's code for ideas, then build our own tool on our stack.

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
2. **Build it** when the asset's form makes that cheap and exact (a textured sphere,
   an arrow, a molecule from a chemistry database).
3. **Search** free, openly licensed 3D libraries, rank the hits, and auto-pick the
   best one, keeping the top 10 so a reviewer can choose later.
4. **Assemble** a recognizable model from simple shapes when nothing suitable
   exists, and flag it for review.

The result goes into the lab resources zip next to the narration audio, and the
bundled copy of the lab points at the provided files.

## 2. Designing for topics we haven't seen

The system has to work for any STEM topic, including ones we have no labs for
yet. So:

- **Routes are chosen by the asset's form, not its subject.** The question is
  "is this a surface on a simple shape, a line diagram, a structure that exists in a
  public scientific database, a recognizable real object, or a UI element?", never
  "is this astronomy or biology?"
- **The fallback chain always ends somewhere.** Search and the shape assembler work
  for any subject. Specialised builders only make results better when they apply;
  nothing depends on them.
- **Builders are plug-ins.** Each declares what it can make, in plain words. The
  routing step offers the LLM the current list, so adding a builder (crystals,
  circuits, terrain) needs no change to routing.
- **Builders are general by source.** A molecule builder covers every compound in
  PubChem, and a macromolecule builder every structure in the Protein Data Bank. They
  don't special-case "VSEPR" or "DNA".
- **Generality is measured.** Our five lab topics are the development set. A
  held-out set of asset requests from other STEM fields (geology, ecology, optics,
  electronics, engineering, earth science...) checks how far each route reaches, and
  is re-run whenever the router or a builder changes (§8, issue 12).

### What our current labs ask for (development set only)

Measured on the 104 wave-2 json_lab outputs (`lab_metrics.analyze_assets`). This
is a sanity check of the route list, not its basis:

- A median lab invents **9 prefabs** (max 31) and **1 texture**.
- Across all labs: **590 distinct prefab names** and **336 distinct texture names**,
  with many near-duplicates (`dnaHelixPrefab`, `dnaDoubleHelixPrefab`,
  `dnaLadderPrefab`), so matching by meaning matters.
- The lab gives little context per asset: prefab name, object name, scale, color,
  texture, and what the narration says. Hierarchical runs also give every object a
  `purpose` (kept in the metrics file's `registry`).

A rough keyword split of the 590 names into forms (a heuristic):

| Form | Distinct names | Examples |
|---|---|---|
| Lines, arrows, rings, grids, markers, planes | 175 | `arrowPrefab`, `orbitRingPrefab`, `groundPlanePrefab` |
| Molecular and macromolecular structures | 213 | `sf4Prefab`, `dnaHelixPrefab`, `helicasePrefab` |
| UI panels, boards, sliders | 71 | `summaryPanelPrefab`, `sliderPrefab` |
| Anatomy and other organic forms | 50 | `heartPrefab`, `redBloodCellPrefab` |
| Physical objects | 14 | `cannonPrefab`, `projectileBallPrefab` |
| Textured spheres (planets, stars) | 11 | `earthPrefab`, `jupiterPrefab` |
| Other | 54 | `gravityMeterPrefab`, `orderTogglePrefab` |

Textures fall into real surfaces (`2k_jupiter`), generic materials
(`cannon_metal_texture`), and diagram textures with content
(`sf4_lewis_structure_texture`, `graph_paper_texture`). The moon lab's
`2k_earth_daymap`, `2k_moon` and `2k_sun` are Solar System Scope's file names, and
models already invent `2k_jupiter`, `2k_mars`, `2k_saturn` in the same pattern.

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

- **It never reads a lab.** Input is hand-typed sentences. JW's separate
  `asset_audit.py` reads a lab and lists its assets, but the two aren't connected.
- **It has no path to the headset.** Output is synced into a three.js preview viewer.
- The catalog matcher is word overlap plus `difflib`, patched incident by incident
  with stopword lists ("spacecraft", "furniture").
- Search returns at most 5 hits per source, ranked by word overlap, so the top 10 a
  reviewer should see may never be fetched.
- It has its own LLM client with an outer retry around instructor's retries (up to 15
  attempts per call), and none of our #44 Gemini/Anthropic guards or cost tracking.
- It needs Blender and OpenSCAD installed for most routes.

### Verdicts

| JW piece | Verdict | Why |
|---|---|---|
| Overall ladder: reuse → build → search → flag | **Keep idea** | Matches the goal; ours routes by form first. |
| Composite route (LLM lists primitive parts) | **Keep idea, rewrite** | The general-purpose fallback that works for any subject. Ours builds with `trimesh` in Python, not Blender. |
| `bond_between` (code computes connector placement, not the LLM) | **Keep idea** | The best idea in the branch: the LLM decides what connects, code does the 3D math. |
| Provenance record: source, URL, author, license, modified | **Keep idea, rewrite** | Required for every asset. Ours also writes a credits file into the bundle. |
| License tiers (CC0 / +CC-BY / +CC-BY-SA; never NC, ND or unknown) | **Keep idea** | Default is CC0 + CC-BY (§7). Hidden hits are counted, not silently dropped. |
| "Nothing enters the library without a license and a known scale" | **Keep** | Same invariant. |
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
| OpenSCAD as the main generator | **Drop as default** | CSG output has no per-part color. May return as a builder plug-in. |
| Blender as a hard dependency | **Drop as default** | `trimesh` and `pygltflib` cover building, normalizing and exporting GLB. Blender stays optional for FBX conversion. |
| Thingiverse | **Drop** | STL only (no color), mixed licenses, needs a token, mostly 3D-printing parts. |
| Procedural noise textures (numpy) | **Drop** | Real surfaces come from maps; generic materials from ambientCG or Poly Haven. |
| Flask review app (Library, Create, Worklist) | **Defer** | We need the review data now (ranked candidates in the manifest) and a review tool later. |
| Library contents (snowman, stool, armchair...) | **Drop** | Test items. |

## 4. Options research

### 4.1 Where to find existing models and textures

Checked 2026-11; verify each before building against it. All are subject-agnostic.

| Source | Content | License | Access |
|---|---|---|---|
| Poly Haven | Props, furniture, nature; PBR textures | CC0 | Public API, no key |
| NASA 3D Resources | Spacecraft, planets, moons | Public domain | GitHub repo; one tree call lists everything |
| Smithsonian Open Access | Scanned museum objects, specimens | CC0 (per item) | Open Access API (api.data.gov key); bulk S3 |
| Poly Pizza | Low-poly models, GLB | CC0 or CC-BY per model | API with free key |
| Sketchfab | 1M+ downloadable models, glTF | CC per model | Download API needs a user login (OAuth). KitBash bought Sketchfab in Aug 2026 |
| Objaverse (Sketchfab-derived dataset) | ~800K objects, with captions (Cap3D) and quality scores (Objaverse++) | Per object (many CC-BY) | Python package; download from Hugging Face; captions searchable offline |
| NIH 3D | Biomedical: proteins, anatomy | Per model | Web download; no documented API found |
| ambientCG | ~2,000 PBR materials | CC0 | Free API, no key |
| NASA / USGS planetary maps | Planet and moon surface maps | Public domain | Manual download |
| Solar System Scope | Planet maps, 2k to 8k | CC-BY 4.0 | Manual download |

Scientific structure databases (inputs for builders, not models):

| Source | Content | License | Access |
|---|---|---|---|
| PubChem | 3D coordinates for 100M+ compounds | Public domain | PUG REST, no key |
| RCSB Protein Data Bank | Proteins, DNA, RNA, complexes | CC0 | REST, no key |
| Crystallography Open Database | Crystal structures | CC0 / public domain | REST, no key |

### 4.2 Matching against what we already have

The library will be small for a long time (tens to a few hundred assets):

1. Exact or alias name hit (case-insensitive; aliases are recorded each time a match
   is accepted, so `dnaDoubleHelixPrefab → dna_double_helix` is instant next time).
2. Shortlist by embedding similarity of descriptions. While the library is under
   about 200 entries, skip this and send the whole list.
3. One cheap LLM call judges the shortlist: reuse which one, or none, and why. It
   also says whether the lab's color or scale can adapt the match (a gray sphere
   plus a red color is fine; a heart is not a lung).

Every later step uses the same request record, so it needs a good description
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

### 4.4 Building models

| Option | Applies to | Accuracy | Dependencies | Plan |
|---|---|---|---|---|
| **Standard kit** (arrow, ring, line, grid, plane, dome, marker, panel), built once in Python (`trimesh`) | Diagram elements in any subject | Exact | Pure Python | **First** |
| **Textured primitive** (sphere, plane, cylinder + an image) | Planets and moons, but also coins, dishes, maps, screens | Exact shape; texture from a source | Pure Python | **First** |
| **UI mapping**: panel from the kit + `textPrefab` children, with text written from the lab context | Panels, boards, readouts, controls | Static only; controls lose their behaviour | One LLM call | **First**, then judged on the headset (§5.3) |
| **Structure builders** (plug-ins): PubChem → ball and stick (standard CPK colors, lone pairs as translucent lobes); PDB → backbone tube or ball and stick; COD → crystal lattice | Any compound, macromolecule or crystal | Real geometry | PubChem, RCSB and COD APIs; RDKit or Biopython optional | **Early** (molecules first) |
| **LLM primitive composite** (JW's part plan + `bond_between`), built with `trimesh` | Anything describable as parts | Recognizable, crude | One LLM call; vision review | **Early**: the universal fallback |
| **LLM-written CAD code**: OpenSCAD, CadQuery or build123d, or Blender `bpy` scripts | Mechanical parts | Good shapes; color varies by tool | Binaries; LLM-written code needs a sandbox | **Later, if needed.** AutoCAD is commercial and Windows-only with no headless path: not recommended |
| **AI text/image-to-3D**: hosted (Meshy, Tripo, Rodin, Hunyuan3D) or open weights (TRELLIS.2, MIT license) on the cluster | Organic shapes | Plausible; scientific detail unreliable | API cost or a GPU job | **Deferred** (§7): judge the other routes first |
| **Image generation for textures** | Stylized generic surfaces | Not for real places | Per-image cost | **Later.** Diagram textures with text are better drawn by code or turned into text objects |

## 5. Design

### 5.1 Steps per lab

1. **Audit** (no LLM): list every `prefab` and `texture` reference with context,
   grouped case-insensitively. Built-ins are marked as already available.
2. **Describe and route** (one LLM call per lab): for each asset, a description,
   what must be right ("four chambers visible", "standard atom colors"), search
   keywords, its form, a route, and merges of duplicates within the lab. The prompt
   lists the registered builders by their own descriptions (§2).
3. **Reuse** (§4.2).
4. **Provide** by the chosen route, falling back along
   builder → search → composite → needs review.
5. **Check**: render a thumbnail, vision review against the description, one repair
   attempt for built assets.
6. **Normalize**: GLB, +Y up, +Z forward, pivot at the center, scaled to fit a unit
   cube (so the lab's `scale` behaves as it does on the built-in sphere; to be
   confirmed on the headset), under the triangle and texture budget for Quest 3.
   Textures are written as **JPG**, which is what the headset loads today.
7. **Bundle**: `Models/` (new) and `Textures/` in the resources zip; the bundled
   lab's `prefab` and `texture` names are rewritten to the provided files, and UI
   objects are replaced by panel + text objects (the original lab is untouched, like
   narration's `audioClip` changes); a credits file lists every CC-BY item.

Same conventions as narration: `--dry-run`, `--run-group`, `--resume` with a
manifest rewritten after every asset, cost from `tracking/pricing.py`, a
needs-review status separate from failure, and an offline `smoke_test.py` with a
fake LLM and mock source servers.

### 5.2 The asset library (outside git)

One shared library, separate from runs, that grows with every accepted asset. Its
location is configurable (an environment variable or `--library`); the default
folder is gitignored:

```
<library>/
  library.json                  id, kind, description, tags, aliases, bounds,
                                triangle count, provenance, license, review status
  <asset_id>/model.glb | texture.jpg
  <asset_id>/thumbnail.png
```

What git does track is a small **seed list** (`Code/Provisioning/seed_assets.json`):
source URL, license, author and checksum for each asset we explicitly choose to
keep (the checked planet maps, the standard kit's build parameters). One command
rebuilds a library from it on a new machine. Nothing else enters git.

The seven moon-lab built-ins (`textPrefab`, `demoPrefab`, `clickableSphere`,
`SunPrefab`, `tinySphere`, `moveableSphere`, `robotIdle`) and its textures
(`2k_earth_daymap`, `2k_moon`, `balldimpled`) are library entries with
hand-written descriptions and no file: they are already on the headset, so they are
matched but never shipped.

### 5.3 UI-like prefabs

There is no panel prefab on the headset (the moon lab uses only `textPrefab` for
text). A UI-like request (`summaryPanelPrefab`, `historyPanelPrefab`,
`sliderPrefab`) becomes:

- one **panel** object from the standard kit (a flat rounded board, sized from the
  original object's scale, colored from its `color`), and
- one or more `textPrefab` children with `TextMeshProComponent` text, written by the
  LLM from the module's narration and the object's role.

Interactive controls (sliders, toggles) keep a label but lose behaviour unless the
lab already gives them a component that provides it; the manifest reports each one.
How this looks and works is judged in headset testing before it's refined.

## 6. Dependencies outside this repo

- **Headset model importer.** The headset loads prefabs by name from what is built
  in. Provisioned models need a new importer (Unity's glTFast can load a GLB from a
  file) reading the zip's new `Models/` folder, plus a way for the lab JSON to say
  "load this file" (proposed: the `prefab` value names the GLB in `Models/`). Until it
  exists, results are checked with thumbnails and a browser preview.
- **`bundle.py`**: accept a `Models/` folder and a credits file (validator rule
  change, #92 follow-up).

## 7. Decisions (owner, 2026-11-03)

1. Prefabs load by name from the built-in set. New models need a new importer and a
   `Models/` folder in the zip. Textures load today; all are JPG celestial maps.
2. The moon-lab built-in source files exist; the list is the prefabs the moon lab uses.
3. **CC-BY is acceptable**: the license tier is CC0 + public domain + CC-BY, with a
   credits file.
4. Model and texture files stay **outside git**, except what we explicitly include.
5. **No hosted AI 3D generation for now.** Judge the other routes first.
6. UI-likes: **map to panels plus text prefabs**, then judge in testing.

## 8. Issue breakdown

Under milestone "3D Model and Texture Provisioning":

1. **Spec**: turn this plan into `PROVISIONING.md` (goals, JW review verdicts,
   decisions), as #91 did for narration.
2. **Headset contract**: the model importer, the `Models/` folder, how the lab names
   a provided model, the normalization convention, JPG textures; `bundle.py` accepts
   `Models/` and the credits file.
3. **Audit, describe and route**: `provision_lab.py --dry-run` lists what a lab needs,
   with the per-lab LLM call (descriptions, form, route, duplicate merges).
4. **Asset library**: location outside git, `library.json` format, provenance and
   license gate, credits file, seed list in git, moon-lab built-ins as file-less
   entries, checked planet maps.
5. **Reuse matcher**: names and aliases, shortlist, LLM judge.
6. **Standard kit, textured primitives and UI mapping** (panel + text).
7. **Open-library search**: source adapters (Poly Haven, NASA 3D, Smithsonian, Poly
   Pizza first; Objaverse and Sketchfab evaluated), license tier, ranked top 10 with
   vision ranking, auto-pick, saved for review.
8. **Builder plug-in interface** and the first structure builder (PubChem molecules).
9. **LLM primitive composite builder** with vision review and repair.
10. **Normalize and bundle**: GLB normalization, budgets, lab rewrite, zip output,
    manifest, resume, offline smoke test.
11. **More structure builders**: PDB macromolecules, COD crystals.
12. **Generality check**: a held-out set of asset requests from STEM fields outside
    our five labs, with route coverage and pass rates per route, re-run on changes.
13. Later: reviewer tool for the saved top 10.
14. Later: library-aware lab generation (list provided assets in the lab prompts and
    the hierarchical registry, so new labs reuse them).
15. Later: texture generation and diagram textures.
16. Later: AI 3D generation (cluster TRELLIS.2 or hosted), once the other routes
    have been judged.

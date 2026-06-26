# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project generates structured JSON specifications for Augmented Reality Lab (ARLEM) educational experiences using LLMs (OpenAI, Google Gemini, Anthropic Claude). It uses Pydantic v2 models and the `instructor` library to produce validated, structured output from LLMs.

## Environment Setup

```bash
# Windows
./setup_windows.ps1

# macOS/Linux
chmod +x setup_unix.sh && ./setup_unix.sh
```

Manual setup:
```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

API keys go in `.env` (git-ignored):
```
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Running Scripts

```bash
# Data conversion (legacy JSON → v2.0 schema)
python "Code/Data Processing/convert_lab_json.py"

# Run Jupyter notebooks
jupyter notebook "Code/Examples/ARLEM_Test.ipynb"

# Generate a single JSON Lab for AR headset use (saves to Artifacts/Data/Generated Labs/)
python "Code/Testing/generate_lab.py" --model claude-sonnet-4.6 --topic "Photosynthesis"
python "Code/Testing/generate_lab.py" --model gpt-4o-mini --topic "DNA Replication" --name dna_lab
python "Code/Testing/generate_lab.py" --list-models

# Benchmark with a free-form topic (legacy path, single prompt template)
python "Code/Testing/benchmark.py" --model gpt-4o-mini
python "Code/Testing/benchmark.py" --model gpt-4o-mini claude-haiku-4.5 gemini-2.5-flash \
    --topic "Human Heart Anatomy"

# Benchmark with a YAML lab description at a specificity level (L1-L4)
python "Code/Testing/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L2
python "Code/Testing/benchmark.py" --model gpt-4o-mini --lab vsepr_molecular_geometry --level L4

# Switch output structure for L2-L4 (L1 ignores this)
#   single-module           - one DemoModule, many clips
#   multi-module (default)  - Lab with one DemoModule per scene
#   module-only             - bare DemoModule, no Lab wrapper
python "Code/Testing/benchmark.py" --model gpt-4o-mini --lab phases_of_the_moon \
    --level L3 --structure multi-module

# Benchmark ARLEM specs via the YAML path (L1-L4)
python "Code/Testing/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L2 --spec arlem

# Sweep multiple output formats together (--spec takes 1, 2, or all 3 formats)
python "Code/Testing/benchmark.py" --model gpt-4o-mini --lab phases_of_the_moon --level L2 \
    --spec json_lab arlem arlem_simple

# Run a pre-defined suite (quick = cheap models, full = all models)
python "Code/Testing/benchmark.py" --suite quick
python "Code/Testing/benchmark.py" --suite full

# List registered models / available lab YAMLs
python "Code/Testing/benchmark.py" --list-models
python "Code/Testing/benchmark.py" --list-labs
```

There is no test runner or linter configured. Validation is done via Pydantic model instantiation and Jupyter notebooks.

## Architecture

### Data Flow

```
Original moon lab reference data:
  Artifacts/Data/Original Moon Lab/Raw/       ← raw Unity transmission JSON
  Artifacts/Data/Original Moon Lab/Processed/ ← manually optimized versions

Lab generation:
  generate_lab.py → instructor → json_lab.py (Pydantic)
  → Artifacts/Data/Generated Labs/  ← clean output JSON for headset

Benchmark runs:
  Artifacts/Lab Descriptions/*.yaml   ← topic source-of-truth (six fields)
  → prompt_builder.py (loads YAML, builds L1-L4 prompt + picks response model)
  → benchmark_config.py (model registry + legacy prompt templates + BenchmarkRunConfig)
  → benchmark.py (runner + client factory)
  → instructor + tracking/ (token/retry tracking)
  → Lab / DemoModule / LabOutline generation
  → lab_metrics.py (structural analysis)
  → Artifacts/Data/Benchmark/ (output + metrics JSON)
  → Artifacts/Data/Benchmark/prompts/ (deduped prompt artifacts, one per
                                       lab × level × spec × structure tuple)
  → Artifacts/Data/Errors/   ← instructor error logs
```

### Key Models

**`Code/Schemas/json_lab.py`** — Lab JSON models (Claude/OpenAI):
- `Lab` → `DemoModule` → `Clip` → `SceneObject` → components
- Components use discriminated unions on `componentType` field
- `ObjectChange` uses sparse delta format (only changed fields per clip)
- `DemoModule` is also a valid top-level response model (used by the
  `--structure module-only` mode of the prompt builder)

**`Code/Schemas/lab_outline.py`** — Minimal response model for L1 outline prompts:
- `LabOutline` → `OutlineScene` (scene_name, brief_purpose, key_visuals, student_actions)
- No discriminated unions or prefab refs — works on every provider including Gemini
- Intentionally permissive; will be refined as L1 failure modes surface

**`Code/Schemas/arlem_full.py`** — Full ARLEM specification:
- `ARLEMScenario` contains a `Workplace` (static environment) + `Activity` (logic/workflow)
- Cross-validation: `ARLEMScenario.validate_activity_flows()` checks activity actions against workplace resources
- `Tangible` subtypes (Thing/Place/Person) use discriminated unions on `type` field

### Benchmark System (`Code/Testing/`)

**`benchmark.py`** — CLI runner. Creates instructor clients per provider, wraps them with `InstructorTracker`, generates a lab, analyzes output, saves results. Supports two prompt-construction paths:
- Legacy: `--topic "..."` uses the templates in `benchmark_config.py`
- YAML-driven: `--lab <topic_name> --level {L1,L2,L3,L4}` delegates to `prompt_builder.build_prompt()` and also picks the response model

**`benchmark_config.py`** — Model registry (`MODELS` dict), legacy prompt templates, `BenchmarkRunConfig` dataclass. Add new models here. `BenchmarkRunConfig` carries optional `lab_name`, `level`, and `structure` fields for the YAML-driven path; when both `lab_name` and `level` are set, the benchmark runner bypasses `get_prompt()` and calls `build_prompt()` instead. Defaults: `min_objects=4`, `min_clips=5`.

**`prompt_builder.py`** — Loads a `<topic>_lab.yaml` from `Artifacts/Lab Descriptions/` into a `LabDescription` dataclass (plain dataclass, not Pydantic — input layer doesn't cross a system boundary). Strips `*` authoring flags and `[REVIEW: ...]` markers silently at load time. `build_prompt(lab, level, spec_type, structure=..., min_objects=..., min_clips=..., min_things=..., min_places=..., min_actions=...)` returns `(prompt_string, response_model_class)`. Levels:
- **L1** — field/course/description → `LabOutline` (rough outline; not a full spec)
- **L2** — same input as L1 → full spec
- **L3** — L2 + `learning_objectives` → full spec
- **L4** — L3 + `detailed_script` → full spec

`--structure` (json_lab L2–L4 only) selects the output shape: `multi-module` (default, one DemoModule per scene), `single-module` (one DemoModule, many clips), or `module-only` (bare `DemoModule`, no `Lab` wrapper). For **ARLEM** (`arlem` / `arlem_simple`) at L2–L4, `build_prompt` returns an `ARLEMScenario` and ignores `--structure` (ARLEM has a single output shape). L1 returns the spec-agnostic `LabOutline` for every spec.

Prompts are deduped: one file per `(lab, level, spec, structure)` tuple under `Artifacts/Data/Benchmark/prompts/`. Every metrics record carries `lab_name`, `level`, `structure`, and `prompt_file` for traceability. Pass `--no-save-prompts` for large sweeps or `--overwrite-prompts` after a wrapper-template tweak.

**`lab_metrics.py`** — Post-generation structural analysis:
- `analyze_json_lab()` — counts objects, clips, components, text labels, object changes, prefab diversity. Shaped for the full `Lab` schema; reports zeros for `LabOutline` outputs (L1 runs)
- `analyze_assets()` — counts **novel assets** the model invented (prefabs/textures not in the moon-lab library, deduped per lab — what we'd have to author) plus audio volume. Holds the known-asset lists (`KNOWN_TEXTURES`/`KNOWN_PREFABS`), kept in sync with the `SceneObject.prefab`/`.texture` descriptions in `json_lab.py`; returns `novel_prefab_names`/`novel_texture_names` for qualitative review. Not stored in metrics files — computed at dataframe-load time from saved outputs. See `Code/Testing/STATISTICS_REPORT.md`
- `analyze_arlem()` — counts things, places, actions, activates/deactivates, triggers, POIs

**`tracking/`** — Token/retry tracking module (copied from `feature/instructor-tracking`):
- `InstructorTracker` / `TrackedClient` — wraps instructor clients, hooks into completion events
- `PricingCalculator` / `DEFAULT_PRICING` — cost calculation; update `tracking/pricing.py` when adding new models
- `BenchmarkExporter` — export to JSON, CSV, or Markdown

Benchmark outputs go to `Artifacts/Data/Benchmark/`:
- `{model}_{spec}[_{level}]_{timestamp}_output.json` — the generated Lab / DemoModule / LabOutline / ARLEM JSON (`{level}` is present on the YAML-driven L1-L4 path)
- `{model}_{spec}[_{level}]_{timestamp}_metrics.json` — full tracking record (includes `prompt_file` reference for YAML-driven runs)
- `suite_results_{timestamp}.json` — combined results across all suite runs
- `prompts/{lab}_{level}_{spec}[_{structure}].txt` — assembled user prompt, written once per unique tuple

### Data Processing Transformations (`Code/Data Processing/`)

The conversion scripts apply these transformations to raw lab JSON:
1. `ActivityModules`: `string[]` → `object[]` with `moduleType` discriminator
2. Unity structs (position, eulerAngles, scale, color) → compact arrays
3. TextMeshPro parsed into typed component objects
4. `componentsToAdd` strings → typed objects with `componentType`
5. Nav buttons (`brb*` prefixed objects) stripped entirely
6. `objectChanges` → sparse delta (only changed fields)

### LLM Integration

Uses the `instructor` library to enforce Pydantic schemas on LLM responses:
```python
import instructor
client = instructor.from_provider("anthropic/claude-opus-4-6")
result = client.create(
    response_model=Lab,
    messages=[...],
    max_tokens=8192,  # required for Anthropic
)
```

Use `instructor.from_provider("provider/model-id")` — this is the unified API. Anthropic always requires `max_tokens`.

**Gemini limitation (resolved for all specs)**: Gemini's function-calling schema validator rejects discriminated unions (`oneOf` + `discriminator`) and `const` tags. This is now resolved on the models themselves for **every** spec — `json_lab.py`, `arlem_full.py`, and `arlem_simplified.py` all inherit `ConstToEnumSchemaMixin` (`const`→`enum`, shared from `Code/Schemas/_schema_helpers.py`) and carry no field-level discriminated unions, so one provider-agnostic schema per spec runs on every provider via the free-decode `GENAI_TOOLS` path. All Gemini twins (`json_lab_gemini.py`, `arlem_full_gemini.py`, `arlem_simplified_gemini.py`) were retired (issues #26, #29), along with the constrained `GENAI_STRUCTURED_OUTPUTS` path and its response-schema-token estimation. `decode_mode_for()` in `benchmark.py` is the single source of truth for the per-run decode mode (recorded as `decode_mode` in metrics). Shared schema helpers (`ConstToEnumSchemaMixin`, `clamp_number`, `_coerce_number_list`) live in `_schema_helpers.py`.

## Code Patterns

- **Discriminated unions**: Use `Literal` + `Field(discriminator="fieldName")` for polymorphic models. New component/module/tangible types follow this pattern.
- **Sparse deltas**: `ObjectChange` only serializes fields with non-None values — use `model_dump(exclude_none=True)` when serializing.
- **Vec3 / Color4**: These are `Annotated` list types (not custom classes) — `list[float]` with length constraints.

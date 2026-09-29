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

# Benchmark with a free-form topic (legacy path, single prompt template)
python "Code/Benchmark/benchmark.py" --model gpt-4o-mini
python "Code/Benchmark/benchmark.py" --model gpt-4o-mini claude-haiku-4.5 gemini-2.5-flash \
    --topic "Human Heart Anatomy"

# Benchmark with a YAML lab description at a specificity level (L1, L3, L4; L2 retired)
python "Code/Benchmark/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L3
python "Code/Benchmark/benchmark.py" --model gpt-4o-mini --lab vsepr_molecular_geometry --level L4

# Switch output structure for L3-L4 (L1 ignores this)
#   single-module           - one DemoModule, many clips
#   multi-module (default)  - Lab with one DemoModule per scene
#   module-only             - bare DemoModule, no Lab wrapper
python "Code/Benchmark/benchmark.py" --model gpt-4o-mini --lab phases_of_the_moon \
    --level L3 --structure multi-module

# Benchmark ARLEM specs via the YAML path (L1, L3, L4)
python "Code/Benchmark/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L3 --spec arlem

# Sweep multiple output formats together (--spec takes 1, 2, or all 3 formats)
python "Code/Benchmark/benchmark.py" --model gpt-4o-mini --lab phases_of_the_moon --level L3 \
    --spec json_lab arlem arlem_simple

# Run a pre-defined suite (quick = cheap models, full = all models)
python "Code/Benchmark/benchmark.py" --suite quick
python "Code/Benchmark/benchmark.py" --suite full

# List registered models / available lab YAMLs
python "Code/Benchmark/benchmark.py" --list-models
python "Code/Benchmark/benchmark.py" --list-labs
```

Analysis and reporting (reads saved run artifacts, no LLM calls):

```bash
# Notebook/HTML reports -> Artifacts/Data/Benchmark/Reports/
python "Code/Analysis/reports/build_statistics_report.py" --since 00000000 --label all
python "Code/Analysis/reports/build_summary_report.py"    --since 00000000 --label all

# Publication figures -> Artifacts/Paper/figures/ (+ tables/)
python "Code/Analysis/figures/cost_figures.py"
python "Code/Analysis/figures/timing_figures.py"
python "Code/Analysis/figures/cost_by_model_figures.py"

# Re-cost a run table against current tracking/pricing.py rates
python "Code/Analysis/reprice_runs.py"     "Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2.csv" --dry-run

# Render any figure script against a repriced table, as _repriced siblings
CSV="Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv"
python "Code/Analysis/figures/cost_figures.py" --runs-csv "$CSV" --suffix _repriced
```

There is no test runner or linter configured. Validation is done via Pydantic model instantiation and Jupyter notebooks.

## Architecture

### Data Flow

```
Original moon lab reference data:
  Artifacts/Data/Original Moon Lab/Raw/       ← raw Unity transmission JSON
  Artifacts/Data/Original Moon Lab/Processed/ ← manually optimized versions

Benchmark runs:
  Artifacts/Lab Descriptions/*.yaml   ← topic source-of-truth (six fields)
  → prompt_builder.py (loads YAML, builds L1/L3/L4 prompt + picks response model)
  → benchmark_config.py (model registry + legacy prompt templates + BenchmarkRunConfig)
  → benchmark.py (runner + client factory)
  → instructor + tracking/ (token/retry tracking)
  → Lab / DemoModule / LabOutline generation
  → lab_metrics.py (structural analysis)
  → Artifacts/Data/Benchmark/ (output + metrics JSON)
  → Artifacts/Data/Benchmark/prompts/ (deduped prompt artifacts, one per
                                       lab × level × spec × structure tuple)
  → Artifacts/Data/Errors/   ← instructor error logs

Analysis (everything below reads run artifacts; nothing here calls an LLM):
  Artifacts/Data/Benchmark/Metrics/ + Outputs/
  → Code/Analysis/benchmark_dataframe.py  (load_runs → one tidy row per run)
  → Code/Analysis/reports/   → Artifacts/Data/Benchmark/Reports/ (ipynb, html, csv)
  → Code/Analysis/figures/   → Artifacts/Paper/figures/ (pdf, png)
                             → Artifacts/Paper/tables/  (tex)
```

**Reports/ vs Paper/**: `Artifacts/Data/Benchmark/Reports/` is the exploration
layer (notebooks, HTML, CSV). `Artifacts/Paper/` is the publication layer —
everything the LaTeX document `\includegraphics`es or `\input`s, figures and
tables alike, hand-authored and generated together.

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

### Benchmark System (`Code/Benchmark/`)

Generation only — this is the one layer that calls an LLM. Post-run analysis
lives in `Code/Analysis/` (below) and never imports the other way around.

**`benchmark.py`** — CLI runner. Creates instructor clients per provider, wraps them with `InstructorTracker`, generates a lab, analyzes output, saves results. Supports two prompt-construction paths:
- Legacy: `--topic "..."` uses the templates in `benchmark_config.py`
- YAML-driven: `--lab <topic_name> --level {L1,L3,L4}` delegates to `prompt_builder.build_prompt()` and also picks the response model

**`benchmark_config.py`** — Model registry (`MODELS` dict), legacy prompt templates, `BenchmarkRunConfig` dataclass. Add new models here. `BenchmarkRunConfig` carries optional `lab_name`, `level`, and `structure` fields for the YAML-driven path; when both `lab_name` and `level` are set, the benchmark runner bypasses `get_prompt()` and calls `build_prompt()` instead. Defaults: `min_objects=4`, `min_clips=5`.

**`prompt_builder.py`** — Loads a `<topic>_lab.yaml` from `Artifacts/Lab Descriptions/` into a `LabDescription` dataclass (plain dataclass, not Pydantic — input layer doesn't cross a system boundary). Strips `*` authoring flags and `[REVIEW: ...]` markers silently at load time. `build_prompt(lab, level, spec_type, structure=..., min_objects=..., min_clips=..., min_things=..., min_places=..., min_actions=...)` returns `(prompt_string, response_model_class)`. Levels:
- **L1** — field/course/description → `LabOutline` (rough outline; not a full spec)
- **L3** — L1 input + `learning_objectives` → full spec
- **L4** — L3 + `detailed_script` → full spec

**L2 retired (issue #31):** L2 was "full spec from the L1 input (no learning objectives)." Its structural metrics tracked L3 (same input + objectives) too closely to justify the run budget, so it was dropped. The gap at L2 is intentional — level numbers encode *input specificity*, not a contiguous ordinal (L4 = has the detailed script). Renumbering to a contiguous L1/L2/L3 is deferred to a possible later migration to avoid breaking comparability with the formative-run artifacts.

`--structure` (json_lab L3–L4 only) selects the output shape: `multi-module` (default, one DemoModule per scene), `single-module` (one DemoModule, many clips), or `module-only` (bare `DemoModule`, no `Lab` wrapper). For **ARLEM** (`arlem` / `arlem_simple`) at L3–L4, `build_prompt` returns an `ARLEMScenario` and ignores `--structure` (ARLEM has a single output shape). L1 returns the spec-agnostic `LabOutline` for every spec.

Prompts are deduped: one file per `(lab, level, spec, structure)` tuple under `Artifacts/Data/Benchmark/prompts/`. Every metrics record carries `lab_name`, `level`, `structure`, and `prompt_file` for traceability. Pass `--no-save-prompts` for large sweeps or `--overwrite-prompts` after a wrapper-template tweak.

**`lab_metrics.py`** — Post-generation structural analysis:
- `analyze_json_lab()` — counts objects, clips, components, text labels, object changes, prefab diversity. Shaped for the full `Lab` schema; reports zeros for `LabOutline` outputs (L1 runs)
- `analyze_assets()` — counts **novel assets** the model invented (prefabs/textures not in the moon-lab library, deduped per lab — what we'd have to author) plus audio volume. Holds the known-asset lists (`KNOWN_TEXTURES`/`KNOWN_PREFABS`), kept in sync with the `SceneObject.prefab`/`.texture` descriptions in `json_lab.py`; returns `novel_prefab_names`/`novel_texture_names` for qualitative review. Not stored in metrics files — computed at dataframe-load time from saved outputs. See `Code/Analysis/reports/STATISTICS_REPORT.md`
- `analyze_arlem()` — counts things, places, actions, activates/deactivates, triggers, POIs

**`tracking/`** — Token/retry tracking module (copied from `feature/instructor-tracking`):
- `InstructorTracker` / `TrackedClient` — wraps instructor clients, hooks into completion events
- `PricingCalculator` / `DEFAULT_PRICING` — cost calculation; update `tracking/pricing.py` when adding new models
- `BenchmarkExporter` — export to JSON, CSV, or Markdown

Benchmark outputs go to `Artifacts/Data/Benchmark/`:
- `{model}_{spec}[_{level}]_{timestamp}_output.json` — the generated Lab / DemoModule / LabOutline / ARLEM JSON (`{level}` is present on the YAML-driven L1/L3/L4 path)
- `{model}_{spec}[_{level}]_{timestamp}_metrics.json` — full tracking record (includes `prompt_file` reference for YAML-driven runs)
- `suite_results_{timestamp}.json` — combined results across all suite runs
- `prompts/{lab}_{level}_{spec}[_{structure}].txt` — assembled user prompt, written once per unique tuple

`probes/` holds one-shot provider/schema diagnostics (`probe_gemini_unions.py`,
`probe_unified_schema.py`, `probe_arlem_unified.py`). They are not part of the
pipeline — they exist to re-test a provider quirk on demand.

### Analysis System (`Code/Analysis/`)

Everything downstream of a run. Reads `Metrics/` + `Outputs/`; never calls an LLM.

**`benchmark_dataframe.py`** — the shared loader. `load_runs()` flattens every
`*_metrics.json` into one tidy DataFrame (one row per run) and re-reads each paired
`Outputs/*_output.json` for novel-asset counts. It reaches back into
`Code/Benchmark/` for `benchmark_config.MODELS` and `lab_metrics.analyze_assets`
via a `sys.path` insert — the only cross-package import in the codebase.

**`reprice_runs.py`** — re-costs a run CSV against current `tracking/pricing.py`
rates, written as a `_repriced` sibling so originals stay put.
**`quarantine_infra_failures.py`** — moves infra-caused failures out of the
live run set.

**`reports/`** — `build_statistics_report.py` (structural + cost/token/time
aggregates) and `build_summary_report.py` (which models/levels generated at all).
Both emit `.ipynb` + `.html` (+ `.csv`) into `Artifacts/Data/Benchmark/Reports/`.
See `STATISTICS_REPORT.md` / `SUMMARY_REPORT.md` alongside them.

**`figures/`** — publication assets, written to `Artifacts/Paper/`:
- `figure_style.py` — shared design tokens, `set_style()` rcParams, and the two
  custom marks (`rounded_bar`, `draw_provider_brackets`). Change the palette or
  `MODEL_ORDER` here and every figure moves together
- `cost_figures.py` — `wave2_tokens_cost_{mean,total}`: stacked tokens-over-cost panels
- `timing_figures.py` — the four `wave2_time_*` / `wave2_latency_*` figures, plus
  `generation_time_tables.tex`
- `cost_by_model_figures.py` — `cost_tokens_by_model{,_L1}`: per-model means over
  the L3/L4 base slice and the L1 slice

All three figure scripts take `--runs-csv` / `--suffix` so a repriced table can be
rendered as `_repriced` siblings without disturbing the originals.

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

Use `instructor.from_provider("provider/model-id")` — this is the unified API. Anthropic always requires `max_tokens`. For Anthropic runs the benchmark also forces `tool_choice={"type": "tool", "name": <response_model>, "disable_parallel_tool_use": True}`: a parallel tool call always fails parsing under instructor and its reask 400s (issue #44 — Haiku 4.5 retry-death), so we forbid it up front. This requires **instructor>=1.15** (earlier versions overwrite a caller-supplied `tool_choice`), and the `name` must equal `response_model.model_json_schema()["title"]` (currently `__name__`, since no response model sets a custom title).

**Gemini limitation (resolved for all specs)**: Gemini's function-calling schema validator rejects discriminated unions (`oneOf` + `discriminator`) and `const` tags. This is now resolved on the models themselves for **every** spec — `json_lab.py`, `arlem_full.py`, and `arlem_simplified.py` all inherit `ConstToEnumSchemaMixin` (`const`→`enum`, shared from `Code/Schemas/_schema_helpers.py`) and carry no field-level discriminated unions, so one provider-agnostic schema per spec runs on every provider via the free-decode `GENAI_TOOLS` path. All Gemini twins (`json_lab_gemini.py`, `arlem_full_gemini.py`, `arlem_simplified_gemini.py`) were retired (issues #26, #29), along with the constrained `GENAI_STRUCTURED_OUTPUTS` path and its response-schema-token estimation. `decode_mode_for()` in `benchmark.py` is the single source of truth for the per-run decode mode (recorded as `decode_mode` in metrics). Shared schema helpers (`ConstToEnumSchemaMixin`, `clamp_number`, `_coerce_number_list`) live in `_schema_helpers.py`.

**Gemini retryability caveat (issue #44)**: the schema story above is resolved, but Gemini *retry* fairness is not intrinsic — it depends on the `instructor==1.15.4` pin (`requirements.txt`) plus a monkeypatch. `benchmark.py::_patch_genai_parallel_call_retry()` (applied lazily inside `create_instructor_client`, guarded by `try/except ImportError` so it degrades on other instructor versions) re-raises instructor's `parse_genai_tools` `AssertionError` as a retryable `ResponseParsingError`, restoring retryability for the *text + single-functionCall* response shape only. True ≥2-functionCall responses still hard-fail on reask (upstream bug). 1.15.4 also no longer blind-retries transient API errors or `max_tokens` truncation; the benchmark restores a bounded, transient-only retry around `create()` (`_transient_retryer`, 429/5xx/connection, issue #54) counted separately in `transient_retries`, while truncation stays a hard fail (review F9). Any new entry point must build its client through `create_instructor_client` (which applies the patch) *and* the Anthropic `tool_choice` guard, or it reintroduces the Gemini/Anthropic retry-death — the standalone `generate_lab.py` entry point was removed for exactly this reason rather than re-guarded (issue #55).

## Code Patterns

- **Discriminated unions**: Use `Literal` + `Field(discriminator="fieldName")` for polymorphic models. New component/module/tangible types follow this pattern.
- **Sparse deltas**: `ObjectChange` only serializes fields with non-None values — use `model_dump(exclude_none=True)` when serializing.
- **Vec3 / Color4**: These are `Annotated` list types (not custom classes) — `list[float]` with length constraints.

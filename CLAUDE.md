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
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini claude-haiku-4.5 gemini-3.5-flash \
    --topic "Human Heart Anatomy"

# Benchmark with a YAML lab description at a specificity level (L1, L3, L4; L2 retired).
# --lab only discovers top-level YAMLs; Example/phases_of_the_moon_lab.yaml is not runnable.
python "Code/Benchmark/benchmark.py" --model claude-haiku-4.5 --lab apparent_retrograde_motion --level L3
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini --lab vsepr_molecular_geometry --level L4

# Switch output structure for L3-L4 (L1 ignores this)
#   single-module           - one DemoModule, many clips
#   multi-module (default)  - Lab with one DemoModule per scene
#   module-only             - bare DemoModule, no Lab wrapper
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini --lab apparent_retrograde_motion \
    --level L3 --structure multi-module

# Benchmark ARLEM specs via the YAML path (L3, L4; L1 returns LabOutline for every spec)
python "Code/Benchmark/benchmark.py" --model claude-haiku-4.5 --lab apparent_retrograde_motion --level L3 --spec arlem

# Sweep multiple output formats together (--spec takes 1, 2, or all 3 formats)
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini --lab apparent_retrograde_motion --level L3 \
    --spec json_lab arlem arlem_simple

# Run a pre-defined suite (quick = current small tier, full = current roster,
# wave2 = the paper's frozen roster)
python "Code/Benchmark/benchmark.py" --suite quick
python "Code/Benchmark/benchmark.py" --suite full
python "Code/Benchmark/benchmark.py" --suite wave2

# List registered models / available lab YAMLs
python "Code/Benchmark/benchmark.py" --list-models
python "Code/Benchmark/benchmark.py" --list-labs
```

Hierarchical generation (one module at a time: plan → asset plan → module; json_lab only):

```bash
# --run-group NAME -> Artifacts/Data/Hierarchical/NAME/ (keeps test and data-collection runs apart)
python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini --lab apparent_retrograde_motion --level L4 --run-group testing
python "Code/Hierarchical/generate_hierarchical.py" --resume "Artifacts/Data/Hierarchical/testing/Runs/<run name>"
python "Code/Hierarchical/smoke_test.py"   # offline check, no API keys (fake LLM + mock OpenAI/Anthropic)
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
  → Artifacts/Data/OneShot/[<run group>/] (output + metrics JSON, prompts/)
    (Artifacts/Data/Benchmark/ is the frozen wave-2 sweep and is never written)
  → Artifacts/Data/Errors/   ← instructor error logs (gitignored)

Hierarchical runs (Code/Hierarchical/, json_lab only):
  Lab YAML → plan (LabPlan) → per module: ModuleAssetPlan → PlannedDemoModule
  → Lab assembled in code → Artifacts/Data/Hierarchical/[<run group>/]{Runs,Outputs,Metrics}/

Analysis (everything below reads run artifacts; nothing here calls an LLM):
  Artifacts/Data/Benchmark/Metrics/ + Outputs/
  → Code/Analysis/benchmark_dataframe.py  (load_runs → one tidy row per run)
  → Code/Analysis/reports/   → Artifacts/Data/Benchmark/Reports/ (ipynb, html, csv)
  → Code/Analysis/figures/   → Artifacts/Paper/figures/ (pdf, png)
                             → Artifacts/Paper/tables/  (tex)
```

**Committed run data**: `Artifacts/Data/Benchmark/` holds only the wave 2 sweep
(2026-07-10, 385 runs, one provenance-stamped stack), which is the paper's dataset.
`Artifacts/Data/Benchmark_formative_202606-07/` is the frozen earlier archive;
never pool the two. `suite_results_*.json` is gitignored, so in a clone failed
runs survive only in the `Reports/` CSVs. `README.md` is the data guide. New
one-shot runs (any roster, `--suite wave2` included) go to
`Artifacts/Data/OneShot/[<run group>/]` (`benchmark.py --run-group NAME`), so they
are never pooled with the paper's data; `load_runs()` still reads `Benchmark/` (#79).

**Reports/ vs Paper/**: `Artifacts/Data/Benchmark/Reports/` is the exploration
layer (notebooks, HTML, CSV). `Artifacts/Paper/` is the publication layer —
everything the LaTeX document `\includegraphics`es or `\input`s, figures and
tables alike, hand-authored and generated together.

### Key Models

**`Code/Schemas/json_lab.py`** — Lab JSON models (Claude/OpenAI):
- `Lab` → `DemoModule` → `Clip` → `SceneObject` → components
- Components are a plain (smart) `Union` resolved on the LLM-facing `type` Literal (no `discriminator=`, which Gemini rejects); the headset-facing `componentType` is hidden from the schema and auto-filled
- `ObjectChange` uses sparse delta format (only changed fields per clip)
- `NewComponent.inputs` (optional, #74): structured inspector fields `{name, type, description, value}` (`ComponentInput` / `NewComponentInput`; `name` normalized to camelCase, `type` free text). It defaults to `None`, so outputs without it (wave 2 included) dump unchanged, but it **changes the one-shot tool schema**: future one-shot runs aren't comparable with wave 2 on this field
- `component_entry_checks()` (opt-in ContextVar scope): each component entry is validated against the one class its `type` tag names, so a bad entry gives one error line instead of every union member's failure. Only the hierarchical module step enters it; one-shot reask text is unchanged
- `DemoModule` is also a valid top-level response model (used by the
  `--structure module-only` mode of the prompt builder)

**`Code/Schemas/lab_outline.py`** — Minimal response model for L1 outline prompts:
- `LabOutline` → `OutlineScene` (scene_name, brief_purpose, key_visuals, student_actions)
- No discriminated unions or prefab refs — works on every provider including Gemini
- Intentionally permissive; will be refined as L1 failure modes surface

**`Code/Schemas/module_asset_plan.py`** — `ModuleAssetPlan`, the hierarchical pipeline's per-module plan:
- `design_notes` first, objects (with `purpose`), `new_components` (each new single-purpose component declared **once**: PascalCase `name`, `behavior_description`, `why_existing_insufficient`, required `inputs`; #77), components (attachments only: `object_name`, free-form `component_type`, `rationale`), interactions (`{object_name, action, effect}`), `clip_beats`
- One validator reports **all** problems in a single error (one reask): components/interactions must name a planned object, no duplicate object names or object/component pairs, no duplicate declarations or input names; under an `asset_plan_validation(catalog, allow_new=..., min_objects=..., min_clip_beats=...)` ContextVar scope, `component_type` must be in the catalog or in `new_components`, a declaration must not duplicate the catalog, `--no-new-components` means `new_components` is empty, and per-module minimums apply (instructor's `context=` is avoided because it Jinja-renders messages). A declared but unattached new component is a soft metric (`plan_adherence.unattached_new_components`), not a reask

**`Code/Schemas/arlem_full.py`** — Full ARLEM specification:
- `ARLEMScenario` contains a `Workplace` (static environment) + `Activity` (logic/workflow)
- Cross-validation: `ARLEMScenario.validate_activity_flows()` checks activity actions against workplace resources
- `things` / `places` are plain `List[Tangible]` (`Person` subclasses `Tangible`); no discriminated unions

**`Code/Schemas/arlem_simplified.py`** — Reduced ARLEM subset, same `ARLEMScenario` shape (`--spec arlem_simple`)

### Benchmark System (`Code/Benchmark/`)

Generation only — this is the one layer that calls an LLM. Post-run analysis
lives in `Code/Analysis/` (below) and never imports the other way around.

**`benchmark.py`** — CLI runner. Creates instructor clients per provider, wraps them with `InstructorTracker`, generates a lab, analyzes output, saves results. Supports two prompt-construction paths:
- Legacy: `--topic "..."` uses the templates in `benchmark_config.py`
- YAML-driven: `--lab <topic_name> --level {L1,L3,L4}` delegates to `prompt_builder.build_prompt()` and also picks the response model

**`benchmark_config.py`** — Model registry (`MODELS` dict), legacy prompt templates, `BenchmarkRunConfig` dataclass. Add new models here. The registry holds two rosters: `CURRENT_MODELS` (issue #72, 2026-10: GPT-6.1 Sol / GPT-6 Luna, Claude Opus / Sonnet / Haiku 5.5, Gemini 4 Argon / 3.8 Flash / 3.5 Flash-Lite), which the suites and tier flags use, and the frozen `WAVE2_MODELS` (the paper's 11 models, `--suite wave2`). Wave-2 entries stay registered with unchanged display names, which the figures key on. `BenchmarkRunConfig` carries optional `lab_name`, `level`, and `structure` fields for the YAML-driven path; when both `lab_name` and `level` are set, the benchmark runner bypasses `get_prompt()` and calls `build_prompt()` instead. Defaults: `min_objects=4`, `min_clips=5`.

**`prompt_builder.py`** — Loads a `<topic>_lab.yaml` from `Artifacts/Lab Descriptions/` into a `LabDescription` dataclass (plain dataclass, not Pydantic — input layer doesn't cross a system boundary). Strips `*` authoring flags and `[REVIEW: ...]` markers silently at load time. `build_prompt(lab, level, spec_type, structure=..., min_objects=..., min_clips=..., min_things=..., min_places=..., min_actions=...)` returns `(prompt_string, response_model_class)`. Levels:
- **L1** — field/course/description → `LabOutline` (rough outline; not a full spec)
- **L3** — L1 input + `learning_objectives` → full spec
- **L4** — L3 + `detailed_script` → full spec

**L2 retired (issue #31):** L2 was "full spec from the L1 input (no learning objectives)." Its structural metrics tracked L3 (same input + objectives) too closely to justify the run budget, so it was dropped. The gap at L2 is intentional — level numbers encode *input specificity*, not a contiguous ordinal (L4 = has the detailed script). Renumbering to a contiguous L1/L2/L3 is deferred to a possible later migration to avoid breaking comparability with the formative-run artifacts.

`--structure` (json_lab L3–L4 only) selects the output shape: `multi-module` (default, one DemoModule per scene), `single-module` (one DemoModule, many clips), or `module-only` (bare `DemoModule`, no `Lab` wrapper). For **ARLEM** (`arlem` / `arlem_simple`) at L3–L4, `build_prompt` returns an `ARLEMScenario` and ignores `--structure` (ARLEM has a single output shape). L1 returns the spec-agnostic `LabOutline` for every spec.

Prompts are deduped: one file per `(lab, level, spec, structure)` tuple under the output folder's `prompts/` (`Artifacts/Data/Benchmark/prompts/` for wave 2). Every metrics record carries `lab_name`, `level`, `structure`, and `prompt_file` for traceability. Pass `--no-save-prompts` for large sweeps or `--overwrite-prompts` after a wrapper-template tweak.

**`lab_metrics.py`** — Post-generation structural analysis:
- `analyze_json_lab()` — counts objects, clips, components, text labels, object changes, prefab diversity. Shaped for the full `Lab` schema; reports zeros for `LabOutline` outputs (L1 runs)
- `analyze_assets()` — counts **novel assets** the model invented (prefabs/textures not in the moon-lab library, deduped per lab — what we'd have to author) plus audio volume. Holds the known-asset lists (`KNOWN_TEXTURES`/`KNOWN_PREFABS`), kept in sync with the `SceneObject.prefab`/`.texture` descriptions in `json_lab.py`; returns `novel_prefab_names`/`novel_texture_names` for qualitative review. Not stored in metrics files — computed at dataframe-load time from saved outputs. See `Code/Analysis/reports/STATISTICS_REPORT.md`
- `analyze_arlem()` — counts things, places, actions, activates/deactivates, triggers, POIs

**`tracking/`** — Token/retry tracking module (copied from `feature/instructor-tracking`):
- `InstructorTracker` / `TrackedClient` — wraps instructor clients, hooks into completion events
- `PricingCalculator` / `DEFAULT_PRICING` — cost calculation; update `tracking/pricing.py` when adding new models. Entries can carry `cached_input` / `cache_write` rates (current roster only, issue #72), and cost bills uncached input, cache reads and cache writes each at its own rate. `TokenUsage.prompt_tokens` is all input on every provider (Anthropic's separately reported cache reads/writes are folded in); `cached_tokens` and `cache_write_tokens` are parts of it
- `BenchmarkExporter` — export to JSON, CSV, or Markdown
- **Served model**: each attempt records the model id the response reports (`served_model_of`: Gemini `model_version`, OpenAI/Anthropic `model`); `summary()` and both run records carry `served_models`, and both CLIs (and the gate probe) warn when it isn't the requested id or a dated snapshot of it. Google serves `gemini-3.5-flash` with `gemini-3.6-flash` since 2026-10-08; cost is still priced as the requested id

New benchmark outputs go to `Artifacts/Data/OneShot/[<run group>/]` (the wave-2 runs in `Artifacts/Data/Benchmark/` use the same layout):
- `{model}_{spec}[_{level}]_{timestamp}_output.json` — the generated Lab / DemoModule / LabOutline / ARLEM JSON (`{level}` is present on the YAML-driven L1/L3/L4 path)
- `{model}_{spec}[_{level}]_{timestamp}_metrics.json` — full tracking record (includes `prompt_file` reference for YAML-driven runs)
- `suite_results_{timestamp}.json` — combined results across all suite runs
- `prompts/{lab}_{level}_{spec}[_{structure}].txt` — assembled user prompt, written once per unique tuple

`probes/` holds one-shot provider/schema diagnostics (`probe_gemini_unions.py`,
`probe_unified_schema.py`, `probe_arlem_unified.py`) and the instructor-upgrade
gates (`probe_tool_schemas.py`, `probe_retry_gates.py`; see LLM Integration). They
are not part of the pipeline — they exist to re-test a provider quirk on demand.

### Hierarchical Generation (`Code/Hierarchical/`)

Second generation entry point (json_lab only). Builds a lab one module at a time
instead of in one `Lab` call; see `HIERARCHICAL.md`.

- **`pipeline.py`** — `HierarchicalPipeline`: plan (`LabPlan` = `LabOutline` + scene
  cap, default 8) → per scene a `ModuleAssetPlan` → a `PlannedDemoModule` (DemoModule
  + a check that every `NewComponent` was planned or is in the catalog and uses only
  its declared input names) → `Lab`
  assembled in code. Each step has its own retry budget, is regenerated alone on
  terminal failure (`step_attempts`; 401/403/404/auth/config errors fail at once),
  is saved to `Runs/<run>/`, and can be resumed; the metrics file is rewritten after
  every step. Accepted steps get new-component names/descriptions/input definitions
  normalized to one canonical form, keeping each instance's input values (logged as
  `normalizations`). Imports the client factory, provider
  kwargs, and transient retryer from `benchmark.py`, so the Gemini and Anthropic guards apply.
- **`registry.py`** — component catalog (built-ins with full field descriptions,
  derived from the json_lab classes, plus NewComponents from accepted modules),
  invented prefabs/textures with their planned purpose, and per-module summaries
  with a `covered:` line, carried forward into later prompts so they get reused.
- **`review.py`** — `Checkpoint` (with `feedback_history`, and `last_error` /
  `rejected_output` when an edit failed validation), `Approve` / `Revise(feedback)` /
  `Replace(output)`, `Reviewer` protocol, `AutoApproveReviewer` (default).
  Human-in-the-loop and LLM self-critique reviewers (#69) plug in here.
- **`prompts.py`** — templates for the plan, asset-plan and module steps. Every
  prompt opens with the same lab-description + approved-plan block so providers
  with prefix caching bill it at the cached rate. For Anthropic the pipeline marks the
  end of that opening (`prompts.shared_opening`) as a `cache_control` breakpoint on
  asset-plan and module requests (#73); the cache is shared per step type, because
  each step type sends its own tool and its own system tool line, and both come
  before the messages in the cached prefix.
- **`generate_hierarchical.py`** — CLI (`--run-group`, per-step retry budgets,
  `--min-objects`/`--min-clips` **per module**, `--max-scenes`, `--no-save`,
  `--resume`). **`smoke_test.py`** — offline test (fake LLM + mock OpenAI and
  Anthropic servers).

Outputs go to `Artifacts/Data/Hierarchical/[<run group>/]` and are never pooled with
`Benchmark/`. Metrics keep the benchmark record's keys (`load_runs()` reads them),
plus `status`, `steps[]`, `totals`, `normalizations`, `plan_adherence` (incl. soft
metrics), `review_log`, `registry`. `retries` is the true reask count (requests − 1
per instructor call; API errors end the call, transient re-sends are counted in
`transient_retries`) since the #71 fix; files recorded before it (wave 2
included) count each failed parse twice, so use `parse_errors` for those. Since the
same fix the error counters (`completion_errors` / `api_errors`) also count failures
instructor fires no hook for (max_tokens truncation, a non-retryable parse failure);
earlier records counted those as 0. Per-module
minimums make these labs deliberately bigger than one-shot runs (lab-wide minimums).

**Script names are PascalCase.** `json_lab.NewComponent.scriptName` normalizes to
PascalCase (`to_pascal_case` in `_schema_helpers.py`) instead of retrying, and
`ModuleAssetPlan` does the same for new component names; compare names with
`module_asset_plan.component_key()`. This also changes future one-shot runs' tool
schema text and output casing (wave 2 is unaffected).

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
- `structure_figures.py` — `wave2_json_lab_size`, `wave2_authoring_burden`,
  `wave2_arlem_validity`, `wave2_specificity_elasticity`, `wave2_arlem_floor`
  (what the generated labs contain), plus `structure_tables.tex`
- `cost_by_model_figures.py` — `cost_tokens_by_model{,_L1}`: per-model means over
  the L3/L4 base slice and the L1 slice

All four figure scripts take `--runs-csv` / `--suffix` so a repriced table can be
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

Use `instructor.from_provider("provider/model-id")` — this is the unified API. Anthropic always requires `max_tokens`. For Anthropic runs the benchmark sends `tool_choice={"type": "auto", "disable_parallel_tool_use": True}` plus a system line asking for exactly one call to the response model's tool (`provider_create_kwargs`). `disable_parallel_tool_use` is the #44 guard: a parallel tool call always fails parsing under instructor (Haiku 4.5 retry-death). The choice is `auto`, not forced, because Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 return 400 for a forced `tool_choice` (issue #70; Claude Haiku 5.5 accepts it, but forcing skips its thinking); a plain-text reply is reasked by instructor and costs one attempt. instructor only sets `tool_choice` when the caller hasn't, so ours always wins. Wave 2 ran with the forced choice on instructor 1.15.4; metrics records carry `tool_choice`, and Anthropic runs since #70 are never pooled with wave 2.

**Reasoning effort and reasks (issue #72).** Every current-roster model pins reasoning effort to `medium` (`ModelConfig.reasoning_effort`, recorded as `reasoning_effort`; wave-2 models keep the provider default, `None`): Anthropic `output_config.effort`, Gemini `thinking_config.thinking_level`, OpenAI `reasoning.effort`. GPT-6 (`responses_api=True`) runs on the **Responses API** (`decode_mode` `responses_tools`), because since GPT-5.4 Chat Completions rejects function tools whenever reasoning is on. Reasks keep the model's reasoning on every provider: instructor resends Claude's thinking blocks (with signatures) and Gemini's content (with thought signatures), and `benchmark._patch_openai_responses()` makes the OpenAI Responses reask do the same, stateless (`store=False`, `include=["reasoning.encrypted_content"]`): previous output items (encrypted reasoning + function call) plus a `function_call_output` carrying the validation error. The same patch drops instructor's strict `text.format` (redundant with the forced tool). It is applied lazily by `create_instructor_client` like the Gemini patch, so the same new-entry-point rule applies.

**Gemini limitation (resolved for all specs)**: Gemini's function-calling schema validator rejects discriminated unions (`oneOf` + `discriminator`) and `const` tags. This is now resolved on the models themselves for **every** spec — `json_lab.py`, `arlem_full.py`, and `arlem_simplified.py` all inherit `ConstToEnumSchemaMixin` (`const`→`enum`, shared from `Code/Schemas/_schema_helpers.py`) and carry no field-level discriminated unions, so one provider-agnostic schema per spec runs on every provider via the free-decode `GENAI_TOOLS` path. All Gemini twins (`json_lab_gemini.py`, `arlem_full_gemini.py`, `arlem_simplified_gemini.py`) were retired (issues #26, #29), along with the constrained `GENAI_STRUCTURED_OUTPUTS` path and its response-schema-token estimation. `decode_mode_for()` in `benchmark.py` is the single source of truth for the per-run decode mode (recorded as `decode_mode` in metrics). Shared schema helpers (`ConstToEnumSchemaMixin`, `clamp_number`, `_coerce_number_list`) live in `_schema_helpers.py`.

**Gemini retryability caveat (issue #44)**: the schema story above is resolved, but Gemini *retry* fairness is not intrinsic — it depends on the instructor pin (`requirements.txt`: 1.17.0 since issue #70; wave 2 ran on 1.15.4) plus a monkeypatch. `benchmark.py::_patch_genai_parallel_call_retry()` (applied lazily inside `create_instructor_client`, guarded by `try/except ImportError` so it degrades on other instructor versions) re-raises instructor's `parse_genai_tools` `AssertionError` as a retryable `ResponseParsingError`, restoring retryability for the *text + single-functionCall* response shape only. True ≥2-functionCall responses still hard-fail on reask (upstream bug). Since 1.15.4 instructor no longer blind-retries transient API errors or `max_tokens` truncation; the benchmark restores a bounded, transient-only retry around `create()` (`_transient_retryer`, 429/5xx/529/connection, issue #54) counted separately in `transient_retries` (the only transport retry: `create_instructor_client` builds the OpenAI and Anthropic SDK clients with `max_retries=0`), while truncation stays a hard fail (review F9). Its predicate (`_is_transient_api_error`) walks the exception's cause chain, because instructor 1.17 raises API errors wrapped in `InstructorRetryException`; each re-send is a separate tracked call with a fresh parse-retry budget (a 5xx on a reask restarts the call; hierarchical steps count those re-sends as `reask_restarts`, the field to filter on when comparing self-correction across providers). Any new entry point must build its client through `create_instructor_client` (which applies the patch) *and* the Anthropic `tool_choice` guard (`provider_create_kwargs`), or it reintroduces the Gemini/Anthropic retry-death — the standalone `generate_lab.py` entry point was removed for exactly this reason rather than re-guarded (issue #55). Moving the pin means re-clearing the #44 gates: `Code/Benchmark/probes/probe_tool_schemas.py` (offline; diffs the tool schemas every provider receives between two pins) and `probe_retry_gates.py` (live; a deliberately rejected attempt 1 must be reasked successfully on each model). Both generation CLIs (`benchmark.py` and `generate_hierarchical.py`), and both probes, call `benchmark.require_environment()` before any run: it exits with a fix-it message if a requested model's provider SDK is missing and warns if instructor isn't the `requirements.txt` pin (both usually mean the project venv isn't active).

## Code Patterns

- **Polymorphic models**: Give each member a single-value `Literal` tag and combine them in a *plain* `Union` (no `Field(discriminator=...)`, which Gemini rejects). Inherit `ConstToEnumSchemaMixin` so the tag emits as `enum`, not `const`. New component types define both the `type` and `componentType` literals and join the `Component` union.
- **Sparse deltas**: `ObjectChange` only serializes fields with non-None values — use `model_dump(exclude_none=True)` when serializing.
- **Vec3 / Color4**: These are `Annotated` list types (not custom classes) — `list[float]` with length constraints.

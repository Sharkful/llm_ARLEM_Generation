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
python "Code/Data Processing/convert_lab_json_claude.py"

# Run Jupyter notebooks
jupyter notebook "Code/Examples/ARLEM_Test.ipynb"

# Benchmark a single model (from project root)
python "Code/Testing/benchmark.py" --model gpt-4o-mini

# Benchmark multiple models on a custom topic
python "Code/Testing/benchmark.py" --model gpt-4o-mini claude-3-haiku gemini-2.0-flash \
    --topic "Human Heart Anatomy"

# Benchmark with ARLEM spec instead of JSON Lab
python "Code/Testing/benchmark.py" --model claude-sonnet-4 --spec arlem

# Run a pre-defined suite (quick = cheap models, full = all models)
python "Code/Testing/benchmark.py" --suite quick
python "Code/Testing/benchmark.py" --suite full

# List all registered models
python "Code/Testing/benchmark.py" --list-models
```

There is no test runner or linter configured. Validation is done via Pydantic model instantiation and Jupyter notebooks.

## Architecture

### Data Flow

```
Raw JSON (Artifacts/Data/Raw/)
  → Conversion scripts (Code/Data Processing/)
  → Pydantic validation (Code/Tools/)
  → LLM via instructor library
  → Optimized JSON (Artifacts/Data/Processed/)

Benchmark runs:
  benchmark_config.py (model registry + prompts)
  → benchmark.py (runner + client factory)
  → instructor + tracking/ (token/retry tracking)
  → Lab/ARLEM generation
  → lab_metrics.py (structural analysis)
  → Artifacts/Data/Benchmark/ (output + metrics JSON)
```

### Key Models

**`Code/Tools/json_lab.py`** — Lab JSON models (Claude/OpenAI):
- `Lab` → `DemoModule` → `Clip` → `SceneObject` → components
- Components use discriminated unions on `componentType` field
- `ObjectChange` uses sparse delta format (only changed fields per clip)

**`Code/Tools/arlem_full.py`** — Full ARLEM specification:
- `ARLEMScenario` contains a `Workplace` (static environment) + `Activity` (logic/workflow)
- Cross-validation: `ARLEMScenario.validate_activity_flows()` checks activity actions against workplace resources
- `Tangible` subtypes (Thing/Place/Person) use discriminated unions on `type` field

### Benchmark System (`Code/Testing/`)

**`benchmark.py`** — CLI runner. Creates instructor clients per provider, wraps them with `InstructorTracker`, generates a lab, analyzes output, saves results.

**`benchmark_config.py`** — Model registry (`MODELS` dict), prompt templates, `BenchmarkRunConfig` dataclass. Add new models here. JSON Lab prompt enforces exactly **1 DemoModule** with a clip sequence. ARLEM prompt targets a full Workplace + Activity.

**`lab_metrics.py`** — Post-generation structural analysis:
- `analyze_json_lab()` — counts objects, clips, components, text labels, object changes, prefab diversity
- `analyze_arlem()` — counts things, places, actions, activates/deactivates, triggers, POIs

**`tracking/`** — Token/retry tracking module (copied from `feature/instructor-tracking`):
- `InstructorTracker` / `TrackedClient` — wraps instructor clients, hooks into completion events
- `PricingCalculator` / `DEFAULT_PRICING` — cost calculation; update `tracking/pricing.py` when adding new models
- `BenchmarkExporter` — export to JSON, CSV, or Markdown

Benchmark outputs go to `Artifacts/Data/Benchmark/`:
- `{model}_{spec}_{timestamp}_output.json` — the generated Lab or ARLEM JSON
- `{model}_{spec}_{timestamp}_metrics.json` — full tracking record
- `suite_results_{timestamp}.json` — combined results across all suite runs

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
client = instructor.from_anthropic(anthropic.Anthropic())
result = client.chat.completions.create(
    model="claude-opus-4-6",
    response_model=Lab,
    messages=[...]
)
```

Provider-specific instructor patches: `instructor.from_openai()`, `instructor.from_anthropic()`, `instructor.from_gemini(use_async=False)`.

**Gemini limitation**: Gemini does not support Union types or discriminated unions. When benchmarking Gemini models, `benchmark.py` automatically switches to `json_lab_gemini.py` / `arlem_full_gemini.py` / `arlem_simplified_gemini.py`.

## Code Patterns

- **Discriminated unions**: Use `Literal` + `Field(discriminator="fieldName")` for polymorphic models. New component/module/tangible types follow this pattern.
- **Sparse deltas**: `ObjectChange` only serializes fields with non-None values — use `model_dump(exclude_none=True)` when serializing.
- **Vec3 / Color4**: These are `Annotated` list types (not custom classes) — `list[float]` with length constraints.

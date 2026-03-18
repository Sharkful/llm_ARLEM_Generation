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
```

### Key Models

**`Code/Tools/pydantic_json_lab_claude.py`** — Lab JSON models (Claude/OpenAI):
- `Lab` → `DemoModule` → `Clip` → `SceneObject` → components
- Components use discriminated unions on `componentType` field
- `ObjectChange` uses sparse delta format (only changed fields per clip)

**`Code/Tools/arlem_full.py`** — Full ARLEM specification:
- `ARLEMScenario` contains a `Workplace` (static environment) + `Activity` (logic/workflow)
- Cross-validation: `ARLEMScenario.validate_activity_flows()` checks activity actions against workplace resources
- `Tangible` subtypes (Thing/Place/Person) use discriminated unions on `type` field

**`Code/Tools/pydantic_json_lab_gemini.py`** / **`arlem_simplified.py`** — Provider-specific or simplified variants.

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

Provider-specific instructor patches: `instructor.from_openai()`, `instructor.from_anthropic()`, `instructor.from_google(use_async=False)`.

## Code Patterns

- **Discriminated unions**: Use `Literal` + `Field(discriminator="fieldName")` for polymorphic models. New component/module/tangible types follow this pattern.
- **Sparse deltas**: `ObjectChange` only serializes fields with non-None values — use `model_dump(exclude_none=True)` when serializing.
- **Vec3 / Color4**: These are `Annotated` list types (not custom classes) — `list[float]` with length constraints.

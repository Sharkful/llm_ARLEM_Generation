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
LLM client factory) is implemented. Later stages (spec parsing, classifier/
resolver, OpenSCAD generation, mesh normalization, materials, external
intake, validation, preview rendering, the Flask review app, and sync) are
not yet built -- see the implementation plan for what's next.

## Setup

```bash
cd asset_pipeline
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env        # fill in whichever provider key(s) you use
```

No external tool binaries (OpenSCAD, Blender) are required for Stage 0.
Later stages will need them; `config.py` auto-detects both if installed and
never fails at import time if they're missing -- only the stages that
actually shell out to them will error, and only when invoked.

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
python cli.py catalog seed   # (re)write the seed catalog of core primitives
python cli.py catalog list   # list all current catalog entries
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

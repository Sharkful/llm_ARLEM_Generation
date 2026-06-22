# `benchmark.py` — Command-Line Reference

`Code/Testing/benchmark.py` is the runner for benchmarking LLM providers on
structured AR lab generation. For each run it builds a prompt, calls a model
through the [`instructor`](https://github.com/instructor-ai/instructor) library
to enforce a Pydantic schema, tracks tokens / cost / retries, analyzes the
generated structure, and writes the output and metrics to disk.

Run it from the **project root** with the virtual environment activated:

```powershell
# Windows
.\.venv\Scripts\Activate.ps1
python "Code/Testing/benchmark.py" --list-models
```

```bash
# macOS / Linux
source .venv/bin/activate
python "Code/Testing/benchmark.py" --list-models
```

API keys are read from `.env` in the project root (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`). A model whose provider key is missing
will fail at call time, not at parse time.

---

## Two ways to build a prompt

`benchmark.py` has two mutually-exclusive prompt-construction paths. Which one
runs is decided by your flags.

| Path | Triggered by | Prompt source | Response model |
| --- | --- | --- | --- |
| **YAML-driven** (preferred) | `--lab` + `--level` (or their `--all-*` forms) | `prompt_builder.build_prompt()` reading a `*_lab.yaml` | Picked automatically per level/structure |
| **Legacy free-form** | `--topic "..."` (or the default topic) | Templates in `benchmark_config.py` | `get_response_model(spec_type)` |

The YAML path takes precedence: if both `--lab/--level` and `--topic` are
supplied, the YAML path wins and `--topic` is ignored.

---

## Flags

### Model selection (choose at least one, unless using `--suite`)

| Flag | Behavior |
| --- | --- |
| `--model`, `-m` `<id> [<id> ...]` | One or more explicit model IDs (space-separated). |
| `--small-models` | All `SMALL`-tier models. |
| `--medium-models` | All `MEDIUM`-tier models. |
| `--large-models` | All `LARGE`-tier models. |
| `--all-models` | Every registered model (all three tiers). |

Tier flags and `--model` combine (union, de-duplicated, registry order
preserved). For example `--small-models -m claude-opus-4.8` runs every small
model **plus** Opus.

### Lab / level (YAML-driven path)

| Flag | Behavior |
| --- | --- |
| `--lab <topic_name>` | A single lab YAML, identified by its `topic_name` (see `--list-labs`). |
| `--level {L1,L2,L3,L4}` | A single specificity level. |
| `--all-labs` | Iterate every discovered lab YAML (excludes the `Example/` subfolder). |
| `--all-levels` | Iterate all four levels `L1`–`L4`. |
| `--structure {single-module,multi-module,module-only}` | Output shape for L2–L4. Default `multi-module`. Ignored for L1. |

Coherence rules enforced at startup (non-suite path):

- `--lab` requires `--level` or `--all-levels`.
- `--level` requires `--lab` or `--all-labs`.
- `--all-labs` requires `--level` or `--all-levels`.
- `--all-levels` requires `--lab` or `--all-labs`.

> **There is no "list of specific labs" flag.** `--lab` takes exactly one
> `topic_name`. To sweep a chosen subset of labs (but not all of them), run one
> command per lab — see [Sweeping a subset of labs](#sweeping-a-subset-of-labs).

### Legacy free-form path

| Flag | Behavior |
| --- | --- |
| `--topic`, `-t "<string>"` | Free-form topic string. Defaults to the Solar System topic when omitted. |
| `--spec`, `-s {json_lab,arlem,arlem_simple}` | Which specification to generate. Default `json_lab`. ARLEM is only wired on the legacy path. |

### Suites

| Flag | Behavior |
| --- | --- |
| `--suite {quick,full}` | Run a predefined model set across `DEFAULT_TOPICS` (legacy path). |

- `quick` → `gpt-4o-mini`, `claude-haiku-4.5`, `gemini-2.5-flash-lite`
- `full` → every registered model

`--suite` **cannot** be combined with `--model`, the tier flags, `--all-labs`,
or `--all-levels`. (You may pair `--suite` with a single `--lab`/`--level` if
needed.)

### Generation / output controls

| Flag | Behavior |
| --- | --- |
| `--max-retries <int>` | Max `instructor` retries per call. Default `3`. |
| `--no-save` | Don't write any output, metrics, or combined-suite files. |
| `--no-save-prompts` | Skip writing the assembled-prompt artifact (use for large sweeps). |
| `--overwrite-prompts` | Rewrite the prompt artifact even if it already exists (use after editing wrapper templates). |

### Informational (print and exit)

| Flag | Behavior |
| --- | --- |
| `--list-models` | Print every registered model with tier, provider, and display name. |
| `--list-labs` | Print every discovered lab YAML and its path. |

---

## How a run expands

A "work item" is one `(lab, level)` pair (YAML path) or one `topic` (legacy
path). The suite runs **every selected model × every work item**:

```
total runs = (models) × (labs × levels)        # YAML path
total runs = (models) × (topics)               # legacy path
```

So `--all-models --all-levels --lab heart_anatomy_and_blood_flow` is
`(all models) × 1 lab × 4 levels`. Confirm your model count with
`--list-models` before launching a large paid sweep.

---

## Levels (L1–L4)

Levels control how much of the YAML lab description is fed to the model, and
therefore which response model is used. Higher levels add more authored detail.

| Level | Input fields used | Response model |
| --- | --- | --- |
| **L1** | field / course / description | `LabOutline` — a rough scene-by-scene outline, **not** a full spec |
| **L2** | same input as L1 | full spec (`Lab` / `DemoModule`) |
| **L3** | L2 + `learning_objectives` | full spec |
| **L4** | L3 + `detailed_script` | full spec |

L1 outputs `LabOutline`, so its structural metrics (object/clip counts) report
zeros — that's expected, not a failure.

## Structure modes (L2–L4 only)

| Value | Output shape |
| --- | --- |
| `multi-module` *(default)* | A `Lab` with one `DemoModule` per scene. |
| `single-module` | One `DemoModule` holding many clips. |
| `module-only` | A bare `DemoModule`, no `Lab` wrapper. |

L1 ignores `--structure` entirely.

## Spec types (legacy `--topic` path)

| Value | Model |
| --- | --- |
| `json_lab` *(default)* | `Lab` (`json_lab.py`) |
| `arlem` | `ARLEMScenario` (`arlem_full.py`) |
| `arlem_simple` | `ARLEMScenario` (`arlem_simplified.py`) |

ARLEM is currently only reachable on the legacy `--topic` path; the YAML path
raises `NotImplementedError` for ARLEM.

## Registered models

Snapshot of the registry (`MODELS` in `benchmark_config.py`). Run
`--list-models` for the live list — this can drift.

| Model ID | Tier | Provider |
| --- | --- | --- |
| `gpt-5.5` | large | OpenAI |
| `gpt-5.4-mini` | small | OpenAI |
| `gpt-5.4-nano` | small | OpenAI |
| `gpt-5-mini` | small | OpenAI |
| `gpt-5-nano` | small | OpenAI |
| `gpt-4o-mini` | small | OpenAI |
| `claude-opus-4.8` | large | Anthropic |
| `claude-sonnet-4.6` | medium | Anthropic |
| `claude-haiku-4.5` | small | Anthropic |
| `gemini-3.1-pro` | large | Google |
| `gemini-3.1-flash-lite` | small | Google |
| `gemini-3.5-flash` | medium | Google |
| `gemini-2.5-pro` | large | Google |
| `gemini-2.5-flash` | medium | Google |
| `gemini-2.5-flash-lite` | small | Google |

**Gemini handling**: Historically Gemini did not support Union / discriminated-union
types, so for any Google model the runner automatically swaps in the union-free
Pydantic variants (`json_lab_gemini.py`, etc.). ⚠️ **Under revalidation (2026-06):**
Google added `anyOf` / full-JSON-Schema support for Gemini 2.5+ (Nov 2025), so this
swap may no longer be required and the flat twins could eventually be retired — see
[`threats_to_validity.md`](../../Artifacts/Data/Benchmark/Reports/threats_to_validity.md)
§6. It also estimates Gemini's
response-schema input tokens, which Google bills but omits from reported usage,
and records them as `gemini_schema_tokens_*` / `cost_adjusted_usd` in the
metrics.

> **Methodology note:** Gemini generates under a different structured-output
> regime (`GENAI_STRUCTURED_OUTPUTS` constrained decoding on a flat schema) than
> the OpenAI/Anthropic tool-call-and-retry path, so cross-provider results carry a
> known confound. See
> [`threats_to_validity.md`](../../Artifacts/Data/Benchmark/Reports/threats_to_validity.md).

**Anthropic**: always called with an explicit `max_tokens` (required by the
API). The runner uses `32000` — full L2–L4 labs run ~18–22K completion tokens
and `8192` truncated them. To allow this above the SDK's ~21,333 non-streaming
threshold, the Anthropic client is built with an explicit timeout, which disables
the "streaming required" guard.

---

## Output files

Everything lands under `Artifacts/Data/Benchmark/` unless `--no-save` is set.
Base name is `{model}_{spec}[_{level}]_{timestamp}` (dots in the model ID become
dashes; `{level}` — L1-L4 — is included on the YAML-driven path so runs that differ
only by level are distinguishable by filename). The `Outputs/{base}_output.json`,
`Metrics/{base}_metrics.json`, and `../Errors/{base}_errors.txt` files all share this base.

| File | Contents |
| --- | --- |
| `Outputs/{base}_output.json` | The generated `Lab` / `DemoModule` / `LabOutline` / ARLEM JSON. |
| `Metrics/{base}_metrics.json` | Full record: tokens, cost, retries, wall time, `lab_metrics`, and references to the prompt / errors files. |
| `suite_results_{timestamp}.json` | Combined records for the whole invocation (at the Benchmark root). |
| `prompts/{lab}_{level}_{spec}[_{structure}].txt` | The assembled user prompt. |
| `Reports/` | Generated notebooks/HTML/CSV from the report builders. |
| `../Errors/{base}_errors.txt` | Per-attempt validation / API errors for that run (under `Artifacts/Data/Errors/`). |

**Prompt dedup**: prompts are keyed by the `(lab, level, spec, structure)`
tuple and written once; every run sharing that tuple references the same file
via `prompt_file` in its metrics. Use `--overwrite-prompts` after you tweak a
wrapper template, or `--no-save-prompts` for very large sweeps.

Every metrics record carries `lab_name`, `level`, `structure`, and
`prompt_file` for traceability back to the exact prompt.

---

## Examples

```powershell
# List what's available
python "Code/Testing/benchmark.py" --list-models
python "Code/Testing/benchmark.py" --list-labs

# Single model, one lab, one level
python "Code/Testing/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L2

# One lab, every level, with a chosen structure
python "Code/Testing/benchmark.py" --model gpt-4o-mini --lab vsepr_molecular_geometry --all-levels --structure single-module

# Every model × every lab × every level (the full matrix)
python "Code/Testing/benchmark.py" --all-models --all-labs --all-levels

# Tier subset: all small models on one lab at L3
python "Code/Testing/benchmark.py" --small-models --lab heart_anatomy_and_blood_flow --level L3

# Legacy free-form topic
python "Code/Testing/benchmark.py" --model gpt-4o-mini --topic "Volcanic Eruption Mechanics"

# Legacy ARLEM spec
python "Code/Testing/benchmark.py" --model claude-sonnet-4.6 --spec arlem --topic "Engine Maintenance"

# Predefined suites
python "Code/Testing/benchmark.py" --suite quick
python "Code/Testing/benchmark.py" --suite full
```

### Sweeping a subset of labs

Because `--lab` takes a single `topic_name`, sweeping a chosen subset (rather
than all labs via `--all-labs`) means one command per lab. To run every model
at every level on three specific labs:

```powershell
python "Code/Testing/benchmark.py" --all-models --all-levels --lab vsepr_molecular_geometry
python "Code/Testing/benchmark.py" --all-models --all-levels --lab heart_anatomy_and_blood_flow
python "Code/Testing/benchmark.py" --all-models --all-levels --lab apparent_retrograde_motion
```

Each command writes its own `suite_results_{timestamp}.json` at the
`Artifacts/Data/Benchmark/` root; the per-run `_output.json` / `_metrics.json`
files accumulate in the shared `Benchmark/Outputs/` and `Benchmark/Metrics/`
subfolders.

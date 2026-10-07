# Hierarchical JSON Lab generation

`Code/Hierarchical/` generates a JSON Lab **one module at a time**, along the lab's
own data structure, instead of in the single `Lab` call that `benchmark.py` makes.
Each step is a small instructor call with its own retry budget, and a checkpoint
after every step is where a reviewer (human or LLM) can later approve, critique,
or edit the output before generation moves on. For now every checkpoint is
auto-approved, so a run goes from start to finish unattended.

JSON Lab only (`multi-module` structure, L3 or L4 input).

## Steps

```
Lab YAML (L3/L4 context)
 ├─ 1. plan         LabOutline: one scene per future module            → checkpoint
 ├─ for each scene i:
 │   ├─ 2. asset plan   ModuleAssetPlan: objects, components, why      → checkpoint
 │   └─ 3. module       PlannedDemoModule implementing that plan       → checkpoint
 └─ 4. assemble     Lab wrapper filled in by code (no LLM call)
```

| Step | Response model | Sees |
|---|---|---|
| plan | `lab_outline.LabOutline` (the L1 outline model, reused) | lab description |
| asset plan | `module_asset_plan.ModuleAssetPlan` | lab description, approved plan, this scene, one-line summaries of earlier modules, asset library, component catalog |
| module | `pipeline.PlannedDemoModule` (a `DemoModule` with one extra check) | the same, plus the approved asset plan |

The Lab wrapper is filled in by code: `labId` = slug of the plan's `lab_title`,
`courseName` = the YAML's `course_context`, `objectives` = the plan's
`top_level_objectives`.

### Structured reasoning in the asset plan

`ModuleAssetPlan` asks the model to reason before it writes the module:

- `design_notes` comes **first**: what the student must see and do, and what that
  requires of the scene.
- Every object has a `purpose`, and every component has a `rationale`.
- `clip_beats` gives one line per clip. The module step writes at least one clip per beat.

Validators make instructor reask on bad plans:

- Every component's `object_name` must match a planned object, so each component
  has an object to attach to.
- `component_type` is a free string, **not** a fixed list. It must name a
  **catalog** component, or set `is_new=true` with a single-purpose
  `behavior_description` and `why_existing_insufficient`.
- A new component that re-invents a catalog entry is rejected with "reuse `<name>`".
- With `--no-new-components`, any `is_new` is rejected.

### Component catalog and asset registry (`registry.py`)

- The catalog starts with the built-in json_lab components. Their descriptions are
  generated from the schema classes, so they can't drift from the schema.
- When a module is accepted, every `NewComponent` it used is added to the catalog,
  so later modules see it and can reuse it by name instead of inventing a
  near-duplicate.
- Prefabs and textures that earlier modules invented are carried forward the same
  way.
- The prompts stress that a component can do almost anything, which is exactly why
  new ones should stay small: one behavior, clear inputs and outputs, and complex
  behavior split into reusable pieces.
- `PlannedDemoModule` only accepts a `NewComponent` whose `scriptName` is planned in
  this module's asset plan or already in the catalog. It serializes exactly like
  `DemoModule`.

The catalog state reaches the validators through ContextVar scopes
(`asset_plan_validation`, `module_validation`), **not** through instructor's
`context=`. Passing `context=` also makes instructor render every message as a
Jinja template, which would mangle prompts that embed JSON or text the model wrote.

## Reviewer hook (`review.py`)

After each step the pipeline calls `reviewer.review(Checkpoint(...))` and acts on
the decision:

| Decision | Effect |
|---|---|
| `Approve()` | accept and move on |
| `Revise(feedback)` | re-run the step with the previous output and the feedback appended (up to `--max-revisions`) |
| `Replace(output)` | accept an edited output (dict or model), re-validated against the step's model |

`AutoApproveReviewer` is the default. A human console reviewer or an LLM
self-critique reviewer only has to implement `review()`. Every decision is written
to `review_log.json`.

## Retries and failures

- `--plan-retries`, `--asset-plan-retries`, `--module-retries` set instructor's
  `max_retries` for each step type.
- `--step-attempts` sets how many times a step is regenerated from scratch after it
  fails terminally. Only that step is regenerated, never the whole lab.
- Transient API errors (429/5xx/connection) get the benchmark's bounded retry,
  counted in `transient_retries`.
- If a step still fails, the run is recorded as failed with `failed_step`. Every
  accepted step stays on disk, so `--resume <run dir>` continues from the first
  missing step without paying for the earlier ones again.

All LLM calls go through `benchmark.create_instructor_client`,
`benchmark.provider_create_kwargs` and `benchmark._transient_retryer`, the same
Gemini and Anthropic guards as the one-shot runner.

## Running

```bash
python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini \
    --lab apparent_retrograde_motion --level L4

python "Code/Hierarchical/generate_hierarchical.py" --model claude-haiku-4.5 gemini-3.5-flash \
    --all-labs --level L3 --module-retries 2 --no-new-components

python "Code/Hierarchical/generate_hierarchical.py" \
    --resume "Artifacts/Data/Hierarchical/Runs/<run name>"

# offline check (no API keys): fake LLM + local mock OpenAI server
python "Code/Hierarchical/smoke_test.py"
```

## Outputs (`Artifacts/Data/Hierarchical/`)

```
Runs/{model}_json_lab_{level}_{ts}/
    run_config.json                          # what --resume rebuilds from
    00_plan.json, 00_plan_prompt.txt
    01_asset_plan.json, 01_asset_plan_prompt.txt, 01_module.json, 01_module_prompt.txt, ...
    review_log.json, errors.txt (if any)
Outputs/{run}_output.json                    # assembled Lab (successful runs)
Metrics/{run}_metrics.json                   # written for failed runs too
```

The metrics record keeps the one-shot runner's top-level keys, so
`Code/Analysis/benchmark_dataframe.load_runs(Path("Artifacts/Data/Hierarchical"))`
loads it as is. Fields added by the hierarchical pipeline:

- `pipeline: "hierarchical"`, `granularity: "module"`, `reviewer`, `config`
- `steps[]`: per-step calls, attempts, `retries`, `parse_errors`, revisions,
  tokens, cost, wall time, `resumed`
- `totals`: the sum over steps, including steps resumed from an earlier session
  (`tracking` covers only this session)
- `plan_adherence[]`: planned objects and components actually present in each module
- `registry`: new components, invented prefabs, invented textures

`retries` is the tracker's `retry_count`, kept for comparability with one-shot
runs. The tracker counts each failed parse attempt twice, so `parse_errors` is the
actual number of reasks.

This folder is separate from `Artifacts/Data/Benchmark/` (the wave-2 paper
dataset). Don't pool the two.

## Next steps

- **Clip-level granularity**: after the asset plan, generate the module's objects,
  then one clip per call against the fixed object list.
- **LLM self-critique reviewer**: critique a plan against the objectives and asset
  rules, return `Revise`, capped at N rounds.
- **Human console reviewer**: approve, give feedback, or edit the JSON at each checkpoint.

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
 ├─ 1. plan         LabPlan: one scene per future module (max 8)       → checkpoint
 ├─ for each scene i:
 │   ├─ 2. asset plan   ModuleAssetPlan: objects, components, why      → checkpoint
 │   └─ 3. module       PlannedDemoModule implementing that plan       → checkpoint
 └─ 4. assemble     Lab wrapper filled in by code (no LLM call)
```

| Step | Response model | Sees |
|---|---|---|
| plan | `pipeline.LabPlan` (the L1 `LabOutline` plus a scene cap) | lab description |
| asset plan | `module_asset_plan.ModuleAssetPlan` | lab description, approved plan, asset library, component catalog, earlier-module summaries, this scene |
| module | `pipeline.PlannedDemoModule` (a `DemoModule` with one extra check) | the same, plus the approved asset plan |

The Lab wrapper is filled in by code: `labId` = slug of the plan's `lab_title`,
`courseName` = the YAML's `course_context`, `objectives` = the plan's
`top_level_objectives`.

### Structured reasoning in the asset plan

`ModuleAssetPlan` asks the model to reason before it writes the module:

- `design_notes` comes **first** (fields generate in order): what the student must
  see and do, and what that requires of the scene.
- Every object has a `purpose`, every component a `rationale`.
- Each interaction is tied to an object: `{object_name, action, effect}`.
- `clip_beats` gives one line per clip; the module writes at least one clip per beat.

Its validator reports **every problem in one error**, so a bad plan costs one reask
instead of one per problem. It checks:

- Unique object names, and components or interactions that name an unknown object.
- The same component planned twice on one object.
- **Per-module minimums**: at least `--min-objects` objects and `--min-clips` beats
  (defaults 4 and 5).
- `component_type` is free text, **not** a fixed list. It must name a **catalog**
  component, or set `is_new=true` with a single-purpose `behavior_description` and
  `why_existing_insufficient`.
- A new component that re-invents a catalog entry is rejected with "reuse `<name>`",
  and so is one declared twice with different descriptions.
- With `--no-new-components`, any `is_new` is rejected.

### Component names: PascalCase, one version per component

New component names are **PascalCase** (C# class convention): `orbit_trail` and
`orbitTrail` both become `OrbitTrail`.

- `json_lab.NewComponent.scriptName` normalizes this in a `mode="before"`
  validator, so the model is never re-asked over casing. The check then only
  rejects names under 5 characters or starting with a digit.
- `ModuleAssetPlan` normalizes new names the same way.
- Names are compared through `component_key()`, so `pulse_glow`, `pulseGlow` and
  `PulseGlow` count as one component.
- **After a step is accepted**, every reference is rewritten to its canonical
  form: the catalog's spelling, and for a reused component the catalog's
  description. Every rewrite is logged under `normalizations` in the metrics. On
  a real 9-module lab, this unified 21 divergent copies.

> **Comparability:** the `json_lab` change also affects the one-shot benchmark's
> tool schema and its output casing for future runs. The frozen wave-2 dataset is
> unaffected; all 104 of its Lab-shaped outputs still validate.

### Component catalog and asset registry (`registry.py`)

- The catalog starts with the built-in json_lab components. Their descriptions
  (the docstring plus every field's full description) are generated from the
  schema classes, so they can't drift.
- When a module is accepted, every `NewComponent` it used joins the catalog, so
  later modules can reuse it by name.
- Prefabs and textures that earlier modules invented are carried forward with the
  `purpose` their asset plan gave them.
- Module summaries (built in code, no LLM call) list each earlier module's name,
  description, objects and clip count, plus a `covered:` line made of its scene
  purpose and educational objectives.
- The prompts stress that a component can do almost anything, which is exactly why
  new ones should stay small: one behavior, clear inputs and outputs, and complex
  behavior split into reusable pieces.
- `PlannedDemoModule` only accepts a `NewComponent` that this module's asset plan
  introduces or that's already in the catalog. It serializes exactly like
  `DemoModule`.

The validators get their per-run state (catalog, new-component switch, minimums,
scene cap) through ContextVar scopes (`asset_plan_validation`, `module_validation`,
`plan_validation`), **not** instructor's `context=`. Passing `context=` also makes
instructor render every message as a Jinja template, which would mangle prompts
that embed JSON or model-written text.

### Prompt order (caching)

Every step prompt opens with the same lab-description block. The asset-plan and
module prompts continue with the same approved plan; the current module is named in
the task section, not marked inside the plan. About 80% of each prompt is
therefore a prefix shared with the previous call of the same step, which OpenAI and
Gemini bill at their cached rate automatically. Anthropic needs explicit cache
markers, tracked in #73.

## Reviewer hook (`review.py`)

After each step the pipeline calls `reviewer.review(Checkpoint(...))` and acts on
the decision:

| Decision | Effect |
|---|---|
| `Approve()` | accept and move on |
| `Revise(feedback)` | re-run the step with the previous output and **every** round of feedback so far (up to `--max-revisions`) |
| `Replace(output)` | accept an edited output (a dict or a model), re-validated against the step's model |

If an edit fails validation, the run doesn't crash. The edit comes back to the
reviewer with `Checkpoint.last_error` holding all of its validation errors and
`rejected_output` holding the edit, so it can be fixed rather than redone. It counts
as a revision round.

If a `Revise` round can't be generated (every attempt fails), the step keeps its
last valid output instead of failing the run; the step's `decision` is
`Approve (revision failed)`, the round isn't counted in `revisions`, and the review
log notes the error. An error no regeneration can fix (auth, configuration) still
fails the run.

The checkpoint also carries:
- `feedback_history`
- a `context` with the scene, the approved plan, the earlier-module summaries, the
  asset plan (for module steps) and `run_dir`

`AutoApproveReviewer` is the default. A human console reviewer or an LLM
self-critique reviewer (#69) only has to implement `review()`. Every decision is
written to `review_log.json`.

## Retries and failures

- `--plan-retries`, `--asset-plan-retries` and `--module-retries` set instructor's
  `max_retries` for each step type.
- `--step-attempts` regenerates a step from scratch after it fails terminally. Only
  that step is regenerated, never the whole lab.
- Errors a regeneration can't fix (401/403/404, authentication, permission,
  not-found or configuration errors) fail the step at once. A provider 400 still
  regenerates, because Gemini's parallel-call reask 400 (#44) is often survived by
  a fresh attempt.
- Transient API errors (429/5xx/529/connection) get the benchmark's bounded retry,
  counted in `transient_retries`. It is the only transport retry: the OpenAI and
  Anthropic SDK clients are built with `max_retries=0`, so their own re-sends don't
  stack with it (or with `step_attempts`) uncounted.
- The metrics file is rewritten after every step, so an interrupted run keeps its
  cost records. Ctrl-C saves a record with status `interrupted` naming the step it
  stopped in; any other unexpected error (a broken reviewer, say) ends the run as
  `failed` at its step, with its traceback printed, stored as `traceback` in the
  metrics and appended to `errors.txt`.
- The console shows one line per failed attempt (for an exhausted instructor retry
  loop: the attempt count and the last error's first problem). `errors.txt` has one
  entry per instructor attempt plus one per failed step attempt: a one-line summary
  for an exhausted instructor retry loop (its attempts are already listed), the full
  text for any other error. Instructor's own ERROR log is silenced by the CLI.
- `--resume <run dir>` continues from the first missing step without paying for the
  earlier ones again. Retry budgets may be changed on resume; level, minimums, scene
  cap and the new-component switch stay as the run started.
  - A step that failed or was interrupted in an earlier session keeps that session's
    attempts, tokens, cost and errors when it is regenerated.
  - The run continues where its directory is now. A copied or moved run writes its
    outputs and metrics next to its new location (the run group is re-derived when
    it sits under `Artifacts/Data/Hierarchical/`).

> **Resume: known limits.** `--resume` is a convenience for long runs, not a core
> workflow: a run that stops during testing can simply be rerun from the start.
> Resume and the reviewer checkpoints don't fully fit together yet. That work waits
> for the lab-making UI, where the checkpoints are what matter. Known gaps:
> - A step regenerated on resume carries over its earlier `revisions` count while
>   its feedback history starts empty, so its review starts partway through the
>   `--max-revisions` budget.
> - If only `Runs/<run>` is moved, the earlier metrics aren't found: resumed steps
>   show no tokens or cost, and the run's `review_log.json` is rewritten with the new
>   entries only.
> - A run moved outside `Artifacts/Data/Hierarchical/` keeps the `run_group` it
>   started with, and a group taken from a folder name isn't checked against the
>   `--run-group` naming rule.
> - A metrics file whose step records have keys this version doesn't know (e.g.
>   written by a later version) fails to resume with a `TypeError`.

All LLM calls go through `benchmark.create_instructor_client`,
`benchmark.provider_create_kwargs` and `benchmark._transient_retryer`, the same
Gemini and Anthropic guards as the one-shot runner.

> Anthropic requests use `tool_choice: auto` with `disable_parallel_tool_use`, so
> Claude Sonnet 5.5, Opus 5.5 and Fable 5.1 (which reject a forced choice) can run
> too (#70).

## Running

```bash
python "Code/Hierarchical/generate_hierarchical.py" --model gpt-5.4-mini \
    --lab apparent_retrograde_motion --level L4 --run-group testing

python "Code/Hierarchical/generate_hierarchical.py" --model claude-haiku-4.5 gemini-3.5-flash \
    --all-labs --level L3 --run-group pilot_oct --module-retries 2 --no-new-components

python "Code/Hierarchical/generate_hierarchical.py" \
    --resume "Artifacts/Data/Hierarchical/pilot_oct/Runs/<run name>" --module-retries 5

# offline check (no API keys): fake LLM + mock OpenAI and Anthropic servers
python "Code/Hierarchical/smoke_test.py"
```

Other flags:

| Flag | What it does |
|---|---|
| `--max-scenes` | the scene cap, 3–12 (default 8); the plan prompt asks for 3–N scenes |
| `--min-objects` / `--min-clips` | the per-module minimums (defaults 4 / 5) |
| `--no-save` | write nothing; the run can't be resumed |

A sweep exits 1 if any run failed.

Before any run, the CLI checks the Python environment (`benchmark.require_environment`).
It stops with a fix-it message if a requested model's provider SDK isn't installed,
and warns if instructor doesn't match the `requirements.txt` pin. Both usually mean
you're outside the project venv.

## Outputs (`Artifacts/Data/Hierarchical/[<run group>/]`)

`--run-group NAME` (a single folder name) keeps test runs and data-collection runs
apart. Without it, runs go to `Hierarchical/` itself.

```
Runs/{model}_json_lab_{level}_{ts}/
    run_config.json                          # what --resume rebuilds from
    00_plan.json, 00_plan_prompt.txt
    01_asset_plan.json, 01_asset_plan_prompt.txt, 01_module.json, 01_module_prompt.txt, ...
    review_log.json, errors.txt (if any)
Outputs/{run}_output.json                    # assembled Lab (successful runs)
Metrics/{run}_metrics.json                   # rewritten after every step
```

The metrics record keeps the one-shot runner's top-level keys, so
`Code/Analysis/benchmark_dataframe.load_runs(Path("Artifacts/Data/Hierarchical/<group>"))`
loads it as is. Fields added by the hierarchical pipeline:

- `pipeline: "hierarchical"`, `granularity: "module"`, `run_group`, `reviewer`, `config`
- `status`: `in_progress` while running, then `complete`, `failed` or `interrupted`
- `served_models`: the model ids the responses reported (this session). The run
  summary warns when one isn't the requested model or a dated snapshot of it
  (e.g. Google serving `gemini-3.5-flash` with `gemini-3.6-flash`)
- `steps[]`: per-step calls, attempts, `retries`, `parse_errors`, revisions,
  `transient_retries`, `reask_restarts` (see below), tokens (including
  `cached_tokens`, the part of the input read from the provider's prompt cache,
  and `cache_write_tokens`, the part written to it), cost, wall time, `resumed`,
  `non_retryable`
- `totals`: the sum over steps, including steps resumed from an earlier session
  (`tracking` covers only this session)
- `normalizations`: every name or description rewrite (see above)
- `plan_adherence[]`, per module:
  - planned objects and components actually present
  - clip and object counts against the minimums
  - interaction objects with no `PointerReceiverComponent`
  - `is_new_asset` claims that disagree with the registry
  - `new_components_planned` / `new_components_added` (new components the plan
    declared, and those the module actually added to the catalog)
- `new_components_total`: new components invented across the lab
- `registry`: new components, invented prefabs and textures (with their purposes)

Cost prices each part of the input at its own rate: uncached input at the input
rate, cache reads at `cached_input` and cache writes at `cache_write`
(`tracking/pricing.py`; current-roster models only, issue #72). OpenAI caches
automatically, and GPT-5.6 and later bill cache writes at 1.25x input. Gemini's
implicit caching has no write charge. Claude reads from cache only once the
prompts carry `cache_control` markers (#73). Anthropic reports cache reads and
writes outside `input_tokens`; the tracker folds them into `prompt_tokens`, so
`prompt_tokens` is all input on every provider. Models without cache rates (the
wave-2 roster) bill all input at the input rate, as they always did.

`retries` is the tracker's `retry_count`: the number of reasks, i.e. requests
minus one per instructor call. A failed parse is reasked, so `retries` equals
`parse_errors`, except that a call which ran out of attempts has one more parse
error than reasks (its last failure isn't reasked). An API error ends the call
rather than being retried inside it; the transient retryer's re-sends (429, 5xx,
connection) are separate calls, counted per step and in `totals` as
`transient_retries` (each failed call's error is listed in the step's errors). A
re-send restarts the instructor call with a fresh parse-retry budget, so a step
whose reask hit a 5xx can get more reasks than its retry budget allows one call.
`reask_restarts` counts exactly those re-sends (a 5xx on a call's first request
gives no extra reasks and isn't counted); filter on `reask_restarts > 0` when
comparing self-correction across providers.
Runs recorded before the #71 fix (wave 2, and earlier hierarchical runs, including
the 2026-10-13 `testing` runs) counted each failed parse twice (`retries` = 2 ×
`parse_errors`, or one less when the call ran out of attempts); use `parse_errors`
for those.

Each step's `errors` and the error counters also include failures instructor fires
no hook for: a max_tokens truncation, or a parse failure it won't retry (recorded
as the underlying exception, not instructor's wrapper). Records from before the
#71 fix counted those as 0.

Per-module minimums make hierarchical labs deliberately bigger than one-shot runs,
whose minimums are lab-wide. Keep that in mind when comparing the two. This folder
is separate from `Artifacts/Data/Benchmark/` (the wave-2 paper dataset); don't pool
the two.

## Next steps

- #68 clip-level granularity (generate objects, then one clip per call)
- #69 LLM self-critique reviewer and a human console reviewer
- #70 instructor upgrade
- #72 model roster review
- #73 Anthropic prompt caching
- #74 structured inputs for new components

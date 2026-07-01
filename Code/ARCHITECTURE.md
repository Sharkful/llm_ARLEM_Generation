# Code Map / Architecture Notes

Orientation doc for working in this repo. CLAUDE.md has the project overview and
run commands; [Testing/BENCHMARK.md](Testing/BENCHMARK.md) is the exhaustive
`benchmark.py` CLI reference. **This file is the code map** — module
relationships, the Pydantic patterns, the benchmark/tracking/report pipeline, and
the cross-cutting gotchas that aren't obvious from any single file.

Read this first when picking up benchmark/model work.

---

## 1. The one-paragraph mental model

The project asks LLMs to emit a **structured AR lab spec** (JSON validated by
Pydantic v2 via the `instructor` library), then measures how well each
model/prompt does it. A *lab description* (YAML, six fields) is expanded into a
*prompt* at one of four specificity *levels* (L1–L4), sent to a model wrapped in
a *tracker* (tokens/cost/retries), and the returned object is *analyzed*
(structural metrics). Everything is written to `Artifacts/Data/Benchmark/`, then
two *report builders* roll the runs up into notebooks/HTML/CSV.

```
YAML lab desc ──► prompt_builder ──► (prompt, response_model)
                                          │
                              instructor client (+ InstructorTracker)
                                          │
                                   model_dump(json)
                                          │
                        lab_metrics.analyze_* ──► metrics record (JSON)
                                          │
                  benchmark_dataframe.load_runs ──► build_*_report ──► notebooks
```

---

## 2. Directory layout (code only)

| Path | Role |
| --- | --- |
| `Code/Schemas/` | **Pydantic schemas only** — the response models. No I/O, no LLM calls. |
| `Code/Testing/` | The benchmark runner, prompt builder, metrics, report builders. |
| `Code/Data Processing/` | Legacy Unity-JSON → v2.0 converters (`convert_lab_json.py`, `prototype_json_patcher.py`). Separate concern from benchmarking. |
| `tracking/` | **At repo root, not under Code/.** Token/retry/cost tracking, copied from a feature branch. |
| `Artifacts/Lab Descriptions/` | `*_lab.yaml` topic source-of-truth (six fields). `Example/` subfolder is excluded by `--all-labs`. |
| `Artifacts/Data/Benchmark/` | Run artifacts: `Outputs/` (generated JSON), `Metrics/` (`*_metrics.json`), `prompts/`, `suite_results_*.json`, and `Reports/`. |
| `Artifacts/Data/Errors/` | Per-run `_errors.txt` (per-attempt validation/API errors). |
| `Artifacts/Data/Benchmark/Reports/` | Generated notebooks/HTML/CSV from the report builders. |

---

## 3. Import / dependency layout (read before touching imports)

`benchmark.py` does **path surgery at startup** so the rest of the code uses flat
imports:

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # repo root
sys.path.insert(0, str(PROJECT_ROOT))                # makes `tracking` importable
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Schemas"))  # makes `json_lab` etc. importable
```

Consequences:
- `from json_lab import Lab`, `from tracking import InstructorTracker` are flat
  imports that only resolve because of those inserts. They will look "unresolved"
  to a static tool but work at runtime when entered through `benchmark.py` /
  `generate_lab.py` (both do the same surgery).
- Response models are imported lazily inside functions (`get_response_model`,
  `prompt_builder.build_prompt`) so a schema module loads only when used. There are
  no longer per-provider variants — one schema serves every provider.

**Module dependency arrows (within `Code/Testing/`):**

```
prompt_builder.py   ← pure, imports nothing project-local (avoids a cycle)
        ▲
benchmark_config.py ── imports Level/Structure from prompt_builder
        ▲
benchmark.py ── imports both, plus lab_metrics, plus tracking
```

`prompt_builder` is deliberately **dependency-free of `benchmark_config`** to
break the circular import — it takes `spec_type` as a plain `str`, which works
because `SpecType` inherits from `str`.

---

## 4. The response models (`Code/Schemas/`)

### 4.1 Spec families (one provider-agnostic schema each)

| Spec | Model | Selected by |
| --- | --- | --- |
| JSON Lab | `json_lab.py` → `Lab` | `SpecType.JSON_LAB` (default) |
| ARLEM full | `arlem_full.py` → `ARLEMScenario` | `SpecType.ARLEM` |
| ARLEM simplified | `arlem_simplified.py` → `ARLEMScenario` | `SpecType.ARLEM_SIMPLIFIED` |
| L1 outline | `lab_outline.py` → `LabOutline` | Level L1 only (any spec) |

**No more Gemini twins.** Gemini's function-calling schema validator used to reject
`Union` / discriminated-union types and the `const` tags single-value `Literal`s emit,
which once required flat `*_gemini.py` twins. Every schema is now provider-agnostic:
single-value Literals emit as `enum` (`const`→`enum`) via the shared
`ConstToEnumSchemaMixin` in `_schema_helpers.py`, and the ARLEM models never used
field-level unions to begin with (the `Union` import appears only in validator
type-hints). So one schema per spec runs on every provider on the free-decode
`GENAI_TOOLS` path — the `json_lab_gemini.py` and `arlem_*_gemini.py` twins were all
retired (issues #26, #29). Shared coercion helpers (`ConstToEnumSchemaMixin`,
`clamp_number`, `_coerce_number_list`) live in `Code/Schemas/_schema_helpers.py`.

### 4.2 JSON Lab hierarchy

```
Lab
└── modules: list[DemoModule]          (Module = DemoModule; union collapsed to one type)
    └── DemoModule
        ├── objects: list[SceneObject]
        │   └── components: list[Component]   (discriminated union)
        └── clips: list[Clip]
            └── changes: list[ObjectChange]   (sparse delta)
```

`DemoModule` is **also a valid top-level response model** — the
`--structure module-only` mode returns a bare `DemoModule` with no `Lab` wrapper.
`lab_metrics.analyze_json_lab` handles all three shapes (Lab-with-modules, bare
module, L1 outline) transparently. `lab_metrics.analyze_assets` reuses the same
shape-normalization to count **novel assets** — prefabs/textures the model
invented (not in the moon-lab library) and that we'd have to author — plus audio
volume; it's the source of truth for the known-asset lists.

### 4.3 Two patterns to know cold

**Dual discriminator on components** (`json_lab.py`). Each component carries two
literal fields kept in lockstep:
- `type` — **LLM-facing**, value = the Python class name (e.g.
  `"TextMeshProComponent"`). The discriminated union resolves on this. Marked
  `exclude=True`, so it never reaches serialized output.
- `componentType` — **headset-facing**, value = the Unity identifier (e.g.
  `"textMeshPro"`). `SkipJsonSchema` + a default, so the LLM never sees or sets
  it; it's auto-filled and is the only one that survives serialization.

Net: the model reasons in class names; the headset receives Unity names. **When
adding a component type, define both literals and add it to the `Component`
union.**

**Sparse deltas** (`ObjectChange`). A clip only specifies the fields that change.
Always serialize labs with `model_dump(mode="json", exclude_none=True)` (the
runner does). `target` accepts `target` or `name` via `AliasChoices`.

`DemoModule.validate_object_references` enforces unique `SceneObject` names and
that every `ObjectChange.target` references a defined object — a common LLM
failure surfaces here as a validation error (→ instructor retry).

### 4.4 ARLEM (`arlem_full.py`)

`ARLEMScenario` = `Workplace` (static environment: things/places/persons,
detectables, primitives, predicates) + `Activity` (ordered `Action` steps with
enter/exit `ActionFlow`s and `Trigger`s). `validate_activity_flows()`
cross-checks activity actions against workplace resources. `Tangible` subtypes
use a `type` discriminator. ARLEM is only reachable on the **legacy `--topic`
path** today; the YAML path raises `NotImplementedError` for it.

---

## 5. The two prompt-construction paths

This is the single most important branch in `benchmark.py` (`run_single_benchmark`).

| | YAML-driven (preferred) | Legacy free-form |
| --- | --- | --- |
| Trigger | `run_config.lab_name` **and** `run_config.level` set | neither set |
| Prompt from | `prompt_builder.build_prompt()` | templates in `benchmark_config.py` |
| Response model | picked by `build_prompt` per level/structure | `get_response_model(spec_type)` |
| Levels | L1, L3, L4 (L2 retired) | n/a (single template) |
| ARLEM | not wired (`NotImplementedError`) | supported |

**Levels** (how much of the YAML is fed in):
- **L1** field/course/description → `LabOutline` (rough outline, *not* a full
  spec; structural metrics are legitimately all-zero).
- **L3** L1 input + `learning_objectives` → full spec.
- **L4** L3 + `detailed_script` → full spec.

**L2 retired (issue #31):** was "full spec from the L1 input, no learning
objectives." Its metrics tracked L3 too closely to be worth the run budget. The
gap is intentional — levels encode input specificity, not a contiguous ordinal
(L4 = has the script); renumbering is deferred to keep formative-run artifacts
comparable.

**Structure** (L3–L4 only; L1 ignores it): `multi-module` (default, one
`DemoModule` per scene) / `single-module` (one module, many clips) /
`module-only` (bare `DemoModule`).

Adding ARLEM to the YAML path is "one row in the dispatch table" in
`build_prompt` — the function is already parameterized on `spec_type`.

---

## 6. Provider-specific handling (gotchas live here)

**Anthropic** (`create_instructor_client`): builds the `anthropic.Anthropic`
client *manually* with an explicit `httpx.Timeout`. This is deliberate — a
non-default timeout disables the SDK's "streaming required" guard that otherwise
trips when `max_tokens > ~21,333`. The runner then requests `max_tokens=32000`
(full L3–L4 labs run ~18–22K completion tokens; the old 8192 truncated them with
`IncompleteOutputException`). Mode is `ANTHROPIC_TOOLS`.

**Gemini / Google:**
- `.env` uses `GEMINI_API_KEY`, but `from_provider` expects `GOOGLE_API_KEY` —
  the runner reads either and passes it explicitly.
- Mode is `GENAI_TOOLS` (free-decode: generate + validate/retry) for **every** spec,
  on the unified provider-agnostic schema — the same footing as OpenAI/Anthropic.
- The old constrained `GENAI_STRUCTURED_OUTPUTS` path and its response-schema-token
  estimation (`estimate_gemini_schema_tokens`, the `gemini_schema_*` /
  `*_adjusted` / `effective_*` fields) were **retired** with the ARLEM unification:
  free-decode sends the schema as a function declaration, billed normally, so there
  is nothing to recover (issue #29).

**OpenAI:** plain `from_provider("openai/<id>")`, no special handling.

---

## 7. Tracking pipeline (`tracking/`)

`InstructorTracker(model, provider, pricing_config).wrap(client)` returns a
`TrackedClient` that hooks instructor's completion events to capture per-attempt
tokens/errors/timing. Key call sites in the runner: `tracker.summary()` (the
nested dict embedded in every metrics record), `tracker.get_last_call_metrics()`
(per-attempt `errors` for the `_errors.txt` file), `tracker.get_total_cost()`.

`tracker.summary()` shape (this is the `tracking` block in each metrics record):

```
{ session_id, model, provider,
  total_calls, successful_calls, failed_calls, success_rate,
  tokens:  { prompt, completion, total },
  retries: { total, calls_with_retries, retry_rate, avg_per_call, max_single_call },
  errors:  { parse_errors, api_errors, completion_errors },
  timing:  { total_duration_ms, avg_duration_ms },
  cost:    { total_usd, formatted } }
```

Pricing lives in `tracking/pricing.py` (`DEFAULT_PRICING`); a model can override
it via `ModelConfig.input_price/output_price`. **Update `pricing.py` when adding
a model** or cost columns read zero.

---

## 8. Output files & metrics record

Per successful run, base name `{model}_{spec}[_{level}]_{timestamp}` (dots → dashes):
- `Outputs/{base}_output.json` — the generated `Lab`/`DemoModule`/`LabOutline`/ARLEM JSON.
- `Metrics/{base}_metrics.json` — the full record: identity, `tracking` block (§7),
  `lab_metrics` block (`lab_metrics.analyze_*`), Gemini adjustment fields,
  `prompt_file`/`errors_file` references.
- `prompts/{lab}_{level}_{spec}[_{structure}].txt` — deduped; one file per
  `(lab, level, spec, structure)` tuple, shared across models.
- `../Errors/{base}_errors.txt` — per-attempt error log.
- `suite_results_{timestamp}.json` — array of all records for the invocation.

**Failure visibility:** standalone `_metrics.json` files are written **only for
successful runs**. Failed runs survive only inside `suite_results_*.json` arrays.
`benchmark_dataframe.load_runs(include_failures=True)` unions both and dedupes.

---

## 9. Reporting layer

`benchmark_dataframe.load_runs()` flattens every `_metrics.json` into one tidy
pandas row (nested `tracking`/`lab_metrics` → flat scalars + derived ratios like
`clips_per_module`, `tokens_per_sec`). For each run it also re-reads the paired
`Outputs/_output.json` through `lab_metrics.analyze_assets` to add **novel-asset**
columns (`novel_prefabs`/`novel_textures` + `*_refs` + `audio_per_clip`; the
metrics files don't store these), and joins the model's `size` tier from the
`MODELS` registry. It prefers the Gemini-adjusted token/cost figures via the
`effective_*` columns. `fail_mode()` classifies terminal errors
(truncation / schema-validation / 404 / other) — kept byte-identical to the copy
in `build_summary_report` so both reports agree.

Two report builders (both emit executed notebook + HTML into `Artifacts/Data/Benchmark/Reports/`):
- `build_summary_report.py` — *which* providers/levels generated at all;
  per-lab-topic sections; Run 1 vs Run 2 fix comparison.
- `build_statistics_report.py` — *what* they produced quantitatively
  (tokens/cost/time/structure), sliced by model/provider/level; also writes
  `benchmark_runs.csv`.

---

## 10. Known issues / live caveats

See also memory notes (`project_benchmark_known_issues`,
`project_gemini_schema_token_billing`).

- Gemini reported tokens are understated by the schema tokens (handled via
  estimation, but it's an *estimate*).
- L1 runs report zero structural metrics by design (outline, not full spec) —
  not a failure.
- OpenAI models occasionally fail discriminated-union resolution
  (`union_tag_not_found`) → surfaces as schema-validation retries.
- ARLEM is not yet wired on the YAML/L1–L4 path.
- There is no test runner or linter. "Validation" = Pydantic instantiation +
  the notebooks.

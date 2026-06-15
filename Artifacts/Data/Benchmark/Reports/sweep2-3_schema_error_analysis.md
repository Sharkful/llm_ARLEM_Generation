# Schema Validation Errors — Sweeps 2 & 3 (post-fix)

Analysis date: 2026-06-15. Source data: benchmark runs of 2026-06-08.

## Scope & question

After the **first** topic sweep (`vsepr_molecular_geometry`), the two dominant
failure modes — **truncation over `max_tokens`** and **hard schema-validation
crashes** — were addressed. This report covers **only the 2nd and 3rd topic
sweeps** that ran afterward:

- **Sweep 2** — `heart_anatomy_and_blood_flow` (2026-06-08, ~17:00–18:30)
- **Sweep 3** — `apparent_retrograde_motion` (2026-06-08, ~20:00–21:45)

Question: *which parts of the schema still failed to validate, and on which
models?*

## Headline

The sweep-1 headliners are gone. In sweeps 2 & 3:

- **Only 7 runs failed outright** (terminal `success=False`, "schema validation"),
  all OpenAI small models: `gpt-5-mini`, `gpt-5-nano`, `gpt-4o-mini`.
- Every other run **recovered on instructor retry**. The errors below are the
  per-attempt validation misses that triggered those retries (and the 7 terminal
  failures).

## Failing schema fields

Ranked by **distinct runs affected** (raw occurrence counts are inflated by
retries — the same error repeats across a run's attempts).

| Schema field (normalized path) | Error type(s) | Runs | What's happening |
| --- | --- | ---: | --- |
| `modules[].clips[].changes[].target` | `missing` (740×), `string_type` | 15 | Required `target` omitted on sparse `ObjectChange` deltas. **Most frequent error.** |
| `…components[].TextMeshProComponent.fontSize` | `greater_than_equal`, `int_type` | 21 | fontSize given as a float, or below the allowed minimum. |
| `…components[].SimpleOrbitComponent.orbitalPeriod` | `missing` | 21 | Required orbit field omitted. |
| `…components[].SimpleOrbitComponent.initialPosition` | `missing`, `too_short`, `list_type` | 21 | Omitted, or wrong-length vector. |
| `…components[].SimpleRotationComponent.rotationTime` | `missing` | 23 | Required field omitted. |
| `…components[]` (discriminated union) | `union_tag_not_found`, `union_tag_invalid` | 5 | Invented a `componentType` not in the union, or missing discriminator. |
| `…components[].NewComponent.scriptName` / `.sciptDescription` | `missing` | 6 | See schema-bug note below. |
| `…components[].TextMeshProComponent.text` | `missing` | 4 | Required field omitted. |
| `…components[].TextMeshProComponent.color` | `list_type` | 1 | Wrong Color4 shape. |
| `modules[].objects[].position[]` | `int_type` / `float_type` | 2 | Vec3 scalar wrong type. |
| `…components[].CheckAngleComponent.azmTarget` / `.altTarget` | `missing` | 3 | Required fields omitted. |
| `modules` / `modules[]` / `modules[].{moduleName,description,clips,objects,educationalObjectives}` | `list_type`, `missing`, `value_error` | 1–5 | Top-level shape errors (rare). |

Error-type totals across both sweeps: `missing` ≫ `greater_than_equal` >
`int_type` > `union_tag_not_found` > `list_type`/`too_short` > others.

## By model

Runs that hit ≥1 schema error (of the standard 6 levels per topic):

| Model | Runs with errors | Notes |
| --- | ---: | --- |
| `gpt-5.5` | 5 | High raw volume; `target`-missing + `fontSize`. |
| `gpt-5.4-nano` | 6 | Worst `target`-missing offender. |
| `gpt-5.4-mini` | 6 | `target`-missing + `fontSize`. |
| `gpt-5-mini` | 6 | 2 terminal failures. |
| `gpt-5-nano` | 6 | terminal failures. |
| `gpt-4o-mini` | 3 | 1 terminal failure (L4 retrograde); Orbit fields. |
| `claude-haiku-4-5` | 6 | Almost entirely `fontSize`; all recovered. |
| `claude-opus-4-8` | 3 | `fontSize`; all recovered. |
| `gemini-3.1-flash-lite-preview` | 1 | One top-level shape error. |

Summary: error volume is concentrated in the **OpenAI gpt-5.x small models**.
Claude errors were almost all the `fontSize` type/min issue and recovered.
Gemini barely appears.

## Two fixes worth making

1. **Schema typo — `sciptDescription`.** The field is misspelled in the schema at
   `Code/Tools/json_lab.py:208` ("scipt", missing the `r`). Models naturally emit
   the correct `scriptDescription`, which Pydantic then rejects as a *missing*
   field. This is a schema bug inflating the error count, not a model failure.
2. **`ObjectChange.target` is the #1 retry driver.** Required with no default at
   `Code/Tools/json_lab.py:302` (it accepts a `name` alias via `AliasChoices`).
   Worth reinforcing in the prompt that every change entry must name its target.

## How to reproduce

The CSV (`benchmark_runs.csv`) is **not sufficient** on its own — it records the
coarse `fail_mode` and counts (`parse_errors`, `retries`) but not which fields
failed. Field-level detail lives only in `Artifacts/Data/Errors/*.txt`
(per-attempt Pydantic `ValidationError` dumps). Terminal failures write *only*
an error log (no `_metrics.json` / `_output.json`), so drive any field analysis
off the `Errors/` logs, bucketing to a topic by timestamp window (the three
sweeps are cleanly time-separated). Normalize list indices in the field paths
(`.0` → `[]`) before aggregating.

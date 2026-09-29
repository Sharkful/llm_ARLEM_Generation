# `build_statistics_report.py` — Benchmark Statistics

Where [`build_summary_report.py`](SUMMARY_REPORT.md) answers *which* providers and
levels generated at all, this report aggregates *what the successful runs
produced*: token counts, generation cost, generation time, and the structural
shape of each generated spec — for **json_lab** (modules, clips, objects,
components) and **ARLEM** (things, places, actions, triggers, POIs) — sliced by
model, provider, `spec_type`, and prompt-specificity level (L1–L4).

Two pieces:

- **`Code/Analysis/benchmark_dataframe.py`** — a reusable loader. `load_runs()`
  flattens every `*_metrics.json` in `Artifacts/Data/Benchmark/Metrics/` into one tidy
  pandas DataFrame (one row per run) with derived columns. It also re-reads each
  run's paired `Outputs/*_output.json` to add the [novel-asset](#novel-assets)
  counts (the metrics files themselves don't store them).
- **`Code/Analysis/reports/build_statistics_report.py`** — builds the report from that
  DataFrame.

## Usage

Run from the project root with the virtual environment active:

```bash
# Default: today's runs only, labelled with today's date
python "Code/Analysis/reports/build_statistics_report.py"

# A specific sweep window (e.g. the wave 2 matrix)
python "Code/Analysis/reports/build_statistics_report.py" --since 20260710 --until 20260710 --label wave2

# From a run CSV instead of Metrics/ (e.g. the repriced wave 2 table)
python "Code/Analysis/reports/build_statistics_report.py" \
    --runs-csv "Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv" --label wave2_repriced
```

`--since` / `--until` take `YYYYMMDD` or `YYYYMMDD_HHMMSS`, and `00000000`
includes all history. `--label` sets the filename suffix so successive reports
never overwrite each other. `--runs-csv` ignores the date window, because the
CSV defines it. The script produces, under `Artifacts/Data/Benchmark/Reports/`:

- **`benchmark_statistics_<label>.ipynb`** is a fully *executed*, self-contained
  notebook. The per-run data is embedded inline, so it re-runs without the
  source JSON.
- **`benchmark_statistics_<label>.html`** is a standalone HTML export. For a PDF,
  open it and choose **Print → Save as PDF**.
- **`benchmark_runs_<label>.csv`** is the full flat table, for slicing in any
  other tool.

The committed wave 2 outputs are `benchmark_statistics_wave2{,_repriced}.*` and
`benchmark_runs_wave2{,_repriced}.csv`. `benchmark_runs.csv` (June 2026) and
`benchmark_runs_final.csv` (the July formative sweep) are from earlier eras.
Keep them apart from wave 2.

## Slicing the DataFrame yourself

The loader is useful on its own:

```python
import sys; sys.path.insert(0, "Code/Analysis")
from benchmark_dataframe import load_runs

df = load_runs()                         # successful runs in Benchmark/ (wave 2)
df = load_runs(include_failures=True)    # also pull failed runs from suite_results (gitignored)
df = load_runs(levels_only=False)        # also include legacy --topic runs

df[df.provider == "anthropic"].wall_s.mean()    # avg generation time, one provider
df[df.spec_type == "json_lab"].groupby("level").num_clips.mean()  # avg clips/level (json_lab)
df[df.spec_type == "arlem"].groupby("level").num_actions.mean()   # avg actions/level (ARLEM)
df.groupby("display_name").cost_usd.mean()      # avg cost per run, per model (all specs)
```

`python "Code/Analysis/benchmark_dataframe.py"` prints a quick shape/columns dump.

## Columns

Each row is one run. Beyond identity columns (`model`, `display_name`, `provider`,
`spec_type`, `level`, `structure`, `lab_name`, `timestamp`, `success`, `fail_mode`):

- **Cost / tokens / time** — `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `cost_usd`, `duration_ms`, `wall_s`, `retries`. Every spec runs free-decode, so
  these raw provider-reported figures are the real ones — there is no Gemini
  schema-token adjustment (the old `effective_*` columns were retired with the
  ARLEM unification, issue #29).
- **Status** — `success`, `fail_mode`
  (`truncation (max_tokens)` / `schema validation` / `404 model-not-found` /
  `other`), and `had_usage` (`total_tokens > 0`) — separates *billable* failures
  that burned real tokens/time from *no-op* failures (bad model id / config
  error) that never reached the model and carry no meaningful stats.
- **json_lab structural** (0 for ARLEM rows) — `num_modules`, `num_objects`,
  `num_clips`, `num_components`, `num_object_changes`, `num_text_labels`,
  `num_unique_prefabs`, `num_unique_objects`, `num_objectives`,
  `num_educational_objectives`.
- **ARLEM structural** (0 for json_lab rows) — workplace resources `num_things`,
  `num_places`, `num_persons`, `num_sensors`, `num_devices`, `num_apps`,
  `num_detectables`, `num_primitives`, `num_predicates`, `num_warnings`; activity
  flow `num_actions`, `total_activates`, `total_deactivates`, `total_messages`,
  `total_triggers`, `total_pois`; and trigger-mode scalars `trigger_click`,
  `trigger_voice`, `trigger_detect`, `trigger_sensor`. A row holds both the
  json_lab and ARLEM column families; slice by `spec_type` before aggregating
  structural columns.
- **Novel assets** — `novel_prefabs`, `novel_textures` (distinct assets the model
  *invented* — not in the moon-lab library — and that we'd have to author, deduped
  across the whole lab), `novel_prefab_refs` / `novel_texture_refs` (how many
  object-instances point at a novel asset), `prefab_refs` / `texture_refs` (total
  references, the denominators), and `audio_refs` / `audio_unique`. See
  [Novel assets](#novel-assets) below.
- **Model size** — `size` (`small` / `medium` / `large`), joined from the
  `MODELS` registry in `benchmark_config.py` — the "rough model size" axis.
- **Derived ratios** — `clips_per_module`, `objects_per_module`,
  `components_per_object`, `changes_per_clip`, `tokens_per_sec` (completion
  throughput), plus novel-asset ratios `novel_prefabs_per_module`,
  `novel_textures_per_module`, `novel_prefabs_per_object`,
  `novel_textures_per_object`, `audio_per_clip`. Ratios are `NaN` when the
  denominator is zero.

## Novel assets

Today the headset only has the **moon-lab** asset library:

- Textures: `2k_earth_daymap`, `2k_moon`, `2k_sun`, `balldimpled`
- Prefabs: `SunPrefab`, `moveableSphere`, `clickableSphere`, `tinySphere`,
  `textPrefab`, `robotIdle` (plus `demoPrefab`, the auto-filled module prefab)

To build a lab on any other topic, a model must **invent** asset names — prefabs
and textures we don't have and would have to author. `lab_metrics.analyze_assets()`
re-reads each saved `Outputs/*_output.json` at load time (no LLM calls, no
re-running of the sweep) and diffs every asset reference against that known set.
The source of truth for the known lists lives in `analyze_assets` (kept in sync
with the `SceneObject.prefab` / `.texture` field descriptions in
`Code/Schemas/json_lab.py`); membership is case-insensitive.

Asset reference points are bounded: prefab/texture come only from `SceneObject`
(an `ObjectChange` delta has no prefab/texture field, so it can't introduce a new
visual asset); audio comes from `Clip.audioClip` plus any
`CheckAngleComponent.audioClipSuccess`.

**Audio is reported as volume, not novelty.** There is no existing audio library
at all, so every `audioClip` is novel by definition — calling it "novel" would
always be 100%. Instead the report shows `audio_per_clip` (what share of clips
carry narration audio).

### Finding the invented asset *names*

The CSV carries only scalar **counts** (it's a flat table). To see the actual
invented names for a given lab — e.g. to spot-check or build an authoring backlog
— call `analyze_assets` on that run's output JSON directly:

```python
import json, sys
sys.path.insert(0, "Code/Benchmark")
from lab_metrics import analyze_assets

# Claude Opus 4.8, heart_anatomy_and_blood_flow, L4 (wave 2)
out = "Artifacts/Data/Benchmark/Outputs/claude-opus-4-8_json_lab_L4_20260710_183321_output.json"
assets = analyze_assets(json.loads(open(out, encoding="utf-8").read()))
print(assets["novel_prefab_names"])   # ['bloodStreamPrefab', 'heartCutawayPrefab', 'heartIntactPrefab', ...]
print(assets["novel_texture_names"])  # ['heart_exterior_texture', 'heart_interior_texture', ...]
```

`analyze_assets` returns the counts (`novel_prefabs`, `novel_textures`,
`*_refs`, `audio_refs`, `audio_unique`) plus the `novel_prefab_names` /
`novel_texture_names` lists. The name lists are intentionally **not** flattened
into the CSV — they're per-lab, variable-length, and belong to qualitative review,
not the one-row-per-run table.

## What's in the report

- **Cost & tokens** — grouped by provider, **spec_type**, level, and model.
- **Generation speed** — wall-clock time and completion throughput by provider,
  level, and model.
- **json_lab structural output (L3–L4)** — modules, clips, clips-per-module,
  objects, unique objects, and components (over `FULL_JSON`), by level / provider /
  model, plus a level × provider clips pivot.
- **ARLEM structural output (L3–L4)** — things, places, predicates, actions,
  activates, triggers, and POIs (over `FULL_ARLEM`), by spec / level / provider /
  model, plus a trigger-mode mix (full vs simplified).
- **Novel assets (json_lab, L3–L4)** — distinct invented prefabs/textures per lab
  and their per-module / per-object density, by level / provider / **model size** /
  model, plus a novel-prefabs level × provider pivot. Audio shown as
  `audio_per_clip`. (ARLEM has no asset library, so this section is json_lab-only.)
- **Failed runs** — failure-mode breakdown (the "common errors"), billable vs
  no-op split, what the billable failures cost (by model and level), the no-op
  list, and total spend wasted on failures.
- **Charts** — cost per run by model, clips by level, throughput by provider,
  objects-vs-clips by level, and spend burned per failure mode.

### Caveats

- **L1 is outline-only.** L1 prompts produce a `LabOutline` (the same model for
  every spec), not a full spec, so all structural counts are 0 at L1 by design.
  Structural sections restrict to L3–L4.
- **Slice structural columns by `spec_type`.** Each row carries both the json_lab
  and ARLEM structural column families, with 0 where a metric doesn't apply. The
  report scopes json_lab shape to `FULL_JSON` and ARLEM shape to `FULL_ARLEM`; do
  the same in ad-hoc analysis. `structure` is `None` for ARLEM rows (ARLEM has one
  ARLEMScenario shape; `--structure` only applies to json_lab).
- **Where failures come from.** A failed run does not write an individual metrics
  file — failures survive only in the `suite_results_*.json` sweeps, which the
  report loads via `include_failures=True`. The suite files are a clean superset
  (every success in a suite also has a standalone metrics file), so the union is
  deduped on `(model, spec_type, level, structure, lab_name, timestamp)`. The
  cost/token and structural sections aggregate successes; the *Failed runs* section
  handles the rest. For the success/failure-by-provider picture, see
  `build_summary_report.py`.
- **Failures in a fresh clone.** `Metrics/` and `Outputs/` are committed, but
  `suite_results_*.json` is gitignored. A clone therefore has no failed-run
  records except those in the exported run CSVs, so build wave 2 reports with
  `--runs-csv` to include the 34 failures.

## Dependencies

`nbformat`, `nbconvert`, `matplotlib`, `pandas`, and a `python3` Jupyter kernel
(`ipykernel`) — all already in the project venv. The notebook is executed via
`nbconvert`'s `ExecutePreprocessor`, so generation fails loudly if a cell errors
rather than shipping a broken report.

## Regenerating after a new sweep

Re-run `build_statistics_report.py` with a `--since` / `--until` window that
covers the sweep and a new `--label`. It reads the matching `*_metrics.json`
records from `Artifacts/Data/Benchmark/Metrics/` plus any local
`suite_results_*.json`. Outputs with the same label are overwritten in place.

## Repricing after a provider rate change

`cost_usd` is **baked into each `*_metrics.json` at generation time** from the
`DEFAULT_PRICING` entry in effect then — it is stored, never derived. When a
provider changes a published rate after the fact, every downstream artifact keeps
quoting the stale number and nothing flags it.

`reprice_runs.py` re-derives cost from the token counts a run already recorded,
writing a *new* CSV beside the original. Nothing is overwritten: the source CSV and
the `*_metrics.json` behind it are read-only, and the output is a byte-exact
superset — the recomputed figure lands in `cost_usd`, the original is preserved
verbatim in `cost_usd_recorded`.

```bash
# 1. Update the rate in tracking/pricing.py, then reprice the run table.
python "Code/Analysis/reprice_runs.py" \
    "Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2.csv"

# 2. Rebuild the figures and reports as _repriced siblings.
CSV=Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv
python "Code/Analysis/figures/cost_figures.py"      --runs-csv "$CSV" --suffix _repriced
python "Code/Analysis/figures/cost_by_model_figures.py"  --runs-csv "$CSV" --suffix _repriced
python "Code/Analysis/reports/build_statistics_report.py" --runs-csv "$CSV" --label wave2_repriced
python "Code/Analysis/reports/build_summary_report.py"    --runs-csv "$CSV" --label wave2_repriced
```

It works off the run CSV rather than `load_runs()` on purpose. The CSV is the
only committed record that includes failed runs, because `suite_results_*.json`
is gitignored. `--runs-csv` on the two
report builders exists for the same reason, and it ignores `--since`/`--until`
because the CSV defines the window.

**Integrity check.** Models whose rate did *not* change must reproduce their
recorded cost to the cent; if any doesn't, the stored cost and stored tokens
disagree for some unrelated reason and the script exits non-zero rather than
publishing a number it can't account for. That check is what makes the models that
*did* move trustworthy — pass `--dry-run` to see the delta table without writing.

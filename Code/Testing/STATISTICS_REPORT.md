# `build_statistics_report.py` — Benchmark Statistics

Where [`build_summary_report.py`](SUMMARY_REPORT.md) answers *which* providers and
levels generated at all, this report aggregates *what the successful runs
produced*: token counts, generation cost, generation time, and the structural
shape of each generated lab (modules, clips, objects, unique objects, components)
— sliced by model, provider, and prompt-specificity level (L1–L4).

Two pieces:

- **`Code/Testing/benchmark_dataframe.py`** — a reusable loader. `load_runs()`
  flattens every `*_metrics.json` in `Artifacts/Data/Benchmark/Metrics/` into one tidy
  pandas DataFrame (one row per run) with derived columns. It also re-reads each
  run's paired `Outputs/*_output.json` to add the [novel-asset](#novel-assets)
  counts (the metrics files themselves don't store them).
- **`Code/Testing/build_statistics_report.py`** — builds the report from that
  DataFrame.

## Usage

Run from the project root with the virtual environment active:

```bash
python "Code/Testing/build_statistics_report.py"
```

No arguments — it loads every run under `Artifacts/Data/Benchmark/Metrics/`. It
produces, under `Artifacts/Data/Benchmark/Reports/`:

- **`benchmark_statistics.ipynb`** — a fully *executed*, self-contained notebook
  (per-run data embedded inline; re-runs without the source JSON).
- **`benchmark_statistics.html`** — standalone HTML export. For a PDF, open it and
  **Print → Save as PDF**.
- **`benchmark_runs.csv`** — the full flat table, for slicing in any other tool.

## Slicing the DataFrame yourself

The loader is useful on its own:

```python
import sys; sys.path.insert(0, "Code/Testing")
from benchmark_dataframe import load_runs

df = load_runs()                         # successful L1–L4 runs (the formative sweep)
df = load_runs(include_failures=True)    # also pull failed runs from suite_results
df = load_runs(levels_only=False)        # also include legacy --topic runs

df[df.provider == "anthropic"].wall_s.mean()    # avg generation time, one provider
df.groupby("level").num_clips.mean()            # avg clips per specificity level
df.groupby("display_name").effective_cost_usd.mean()  # avg cost per run, per model
```

`python "Code/Testing/benchmark_dataframe.py"` prints a quick shape/columns dump.

## Columns

Each row is one run. Beyond identity columns (`model`, `display_name`, `provider`,
`level`, `structure`, `lab_name`, `timestamp`, `success`, `fail_mode`):

- **Cost / tokens / time** — `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `cost_usd`, `duration_ms`, `wall_s`, `retries`, plus the Gemini-adjusted
  `effective_prompt_tokens` / `effective_total_tokens` / `effective_cost_usd`
  (fold in the response-schema tokens Google bills as input but omits from
  reported usage; equal to the raw figures for other providers).
- **Status** — `success`, `fail_mode`
  (`truncation (max_tokens)` / `schema validation` / `404 model-not-found` /
  `other`), and `had_usage` (`total_tokens > 0`) — separates *billable* failures
  that burned real tokens/time from *no-op* failures (bad model id / config
  error) that never reached the model and carry no meaningful stats.
- **Structural** — `num_modules`, `num_objects`, `num_clips`, `num_components`,
  `num_object_changes`, `num_text_labels`, `num_unique_prefabs`,
  `num_unique_objects`, `num_objectives`, `num_educational_objectives`.
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
`Code/Tools/json_lab.py`); membership is case-insensitive.

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
sys.path.insert(0, "Code/Testing")
from lab_metrics import analyze_assets

out = "Artifacts/Data/Benchmark/Outputs/claude-opus-4-8_json_lab_L2_20260608_171853_output.json"
assets = analyze_assets(json.loads(open(out, encoding="utf-8").read()))
print(assets["novel_prefab_names"])   # ['heartCutawayPrefab', 'heartPrefab', 'valveDiscPrefab', 'vesselTubePrefab']
print(assets["novel_texture_names"])  # ['heart_muscle_texture']
```

`analyze_assets` returns the counts (`novel_prefabs`, `novel_textures`,
`*_refs`, `audio_refs`, `audio_unique`) plus the `novel_prefab_names` /
`novel_texture_names` lists. The name lists are intentionally **not** flattened
into the CSV — they're per-lab, variable-length, and belong to qualitative review,
not the one-row-per-run table.

## What's in the report

- **Cost & tokens** — grouped by provider, level, and model.
- **Generation speed** — wall-clock time and completion throughput by provider,
  level, and model.
- **Structural output (L2–L4)** — modules, clips, clips-per-module, objects,
  unique objects, and components, by level / provider / model, plus a
  level × provider clips pivot.
- **Novel assets (L2–L4)** — distinct invented prefabs/textures per lab and their
  per-module / per-object density, by level / provider / **model size** / model,
  plus a novel-prefabs level × provider pivot. Audio shown as `audio_per_clip`.
- **Failed runs** — failure-mode breakdown (the "common errors"), billable vs
  no-op split, what the billable failures cost (by model and level), the no-op
  list, and total spend wasted on failures.
- **Charts** — cost per run by model, clips by level, throughput by provider,
  objects-vs-clips by level, and spend burned per failure mode.

### Caveats

- **L1 is outline-only.** L1 prompts produce a `LabOutline`, not a full `Lab`, so
  modules/clips/objects/components are 0 at L1 by design. Structural sections
  restrict to L2–L4.
- **Where failures come from.** A failed run does not write an individual metrics
  file — failures survive only in the `suite_results_*.json` sweeps, which the
  report loads via `include_failures=True`. The suite files are a clean superset
  (every success in a suite also has a standalone metrics file), so the union is
  deduped on `(model, level, structure, lab_name, timestamp)`. The cost/token and
  structural sections aggregate successes; the *Failed runs* section handles the
  rest. For the success/failure-by-provider picture, see
  `build_summary_report.py`.

## Dependencies

`nbformat`, `nbconvert`, `matplotlib`, `pandas`, and a `python3` Jupyter kernel
(`ipykernel`) — all already in the project venv. The notebook is executed via
`nbconvert`'s `ExecutePreprocessor`, so generation fails loudly if a cell errors
rather than shipping a broken report.

## Regenerating after a new sweep

Just re-run `build_statistics_report.py` — it picks up every `*_metrics.json`
currently in `Artifacts/Data/Benchmark/Metrics/`, so new runs are included automatically.
The `.ipynb`, `.html`, and `.csv` are overwritten in place.

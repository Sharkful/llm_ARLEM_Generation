# `build_statistics_report.py` — Benchmark Statistics

Where [`build_summary_report.py`](SUMMARY_REPORT.md) answers *which* providers and
levels generated at all, this report aggregates *what the successful runs
produced*: token counts, generation cost, generation time, and the structural
shape of each generated lab (modules, clips, objects, unique objects, components)
— sliced by model, provider, and prompt-specificity level (L1–L4).

Two pieces:

- **`Code/Testing/benchmark_dataframe.py`** — a reusable loader. `load_runs()`
  flattens every `*_metrics.json` in `Artifacts/Data/Benchmark/` into one tidy
  pandas DataFrame (one row per run) with derived columns.
- **`Code/Testing/build_statistics_report.py`** — builds the report from that
  DataFrame.

## Usage

Run from the project root with the virtual environment active:

```bash
python "Code/Testing/build_statistics_report.py"
```

No arguments — it loads every run under `Artifacts/Data/Benchmark/`. It produces,
under `Artifacts/Reports/`:

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
- **Derived ratios** — `clips_per_module`, `objects_per_module`,
  `components_per_object`, `changes_per_clip`, `tokens_per_sec` (completion
  throughput). Ratios are `NaN` when the denominator is zero.

## What's in the report

- **Cost & tokens** — grouped by provider, level, and model.
- **Generation speed** — wall-clock time and completion throughput by provider,
  level, and model.
- **Structural output (L2–L4)** — modules, clips, clips-per-module, objects,
  unique objects, and components, by level / provider / model, plus a
  level × provider clips pivot.
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
currently in `Artifacts/Data/Benchmark/`, so new runs are included automatically.
The `.ipynb`, `.html`, and `.csv` are overwritten in place.

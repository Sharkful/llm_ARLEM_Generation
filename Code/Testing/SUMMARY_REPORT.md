# `build_summary_report.py` — Shareable Benchmark Summary

`Code/Testing/build_summary_report.py` collates one or more benchmark sweeps
(`suite_results_*.json`) into a single, shareable report — one section per lab
topic, plus a before/after comparison of the fixes applied between runs.

It produces, under `Artifacts/Data/Benchmark/Reports/`:

- **`benchmark_summary.ipynb`** — a fully *executed*, self-contained notebook. The
  per-run data is embedded inline, so it opens and re-runs without the source
  JSON files.
- **`benchmark_summary.html`** — a standalone HTML export that opens in any
  browser. For a PDF, open it and **Print → Save as PDF** (direct PDF export needs
  LaTeX/pandoc, which is not assumed to be installed).

## Usage

Run from the project root with the virtual environment active:

```bash
python "Code/Testing/build_summary_report.py"
```

No arguments — the sweeps to include are declared in the `SWEEPS` list at the top
of the script.

## What it reads

Each entry in `SWEEPS` is a tuple:

```python
(topic_key, "Pretty Title", "Run N — phase label", "suite_results_<timestamp>.json")
```

The filename is resolved under `Artifacts/Data/Benchmark/`. To add a topic or a new
run, append a row — nothing else needs to change. Ordering in the list is the
ordering in the report.

## What's in the report

- **Fixes between Run 1 and Run 2** — a narrative section plus a bar chart of
  success rate by provider, pre-fix vs post-fix.
- **One section per topic**, each with:
  - topic stats (success rate, wall time, cost, avg objects/clips at L3–L4),
  - success by provider × level,
  - a per-model OK/✗ grid with the dominant failure mode,
  - a failure-mode breakdown.
- **Takeaways**.

Failure modes are classified from each run's error string into
`truncation (max_tokens)`, `404 model-not-found`, `schema validation`, or `other`
(see `fail_mode()`).

## Dependencies

`nbformat`, `nbconvert`, `matplotlib`, `pandas`, and a `python3` Jupyter kernel
(`ipykernel`) — all already in `requirements.txt` / the project venv. The script
executes the notebook via `nbconvert`'s `ExecutePreprocessor`, so generation will
fail loudly if a cell errors rather than shipping a broken report.

## Regenerating after a new sweep

1. Run the new sweep (see [BENCHMARK.md](BENCHMARK.md)).
2. Add its `suite_results_*.json` as a row in `SWEEPS`.
3. Re-run `build_summary_report.py`. The `.ipynb` and `.html` are overwritten in
   place.

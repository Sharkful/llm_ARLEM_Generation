# Benchmark data

Run records from the LLM lab-generation benchmark. Everything the analysis
code in `Code/Analysis/` reads is here; `load_runs()` in
`Code/Analysis/benchmark_dataframe.py` joins it into one row per run.

| Path | Contents |
|---|---|
| `Benchmark/` | **Wave 2** (clean re-run from 2026-07-10). Single software stack; each metrics record carries a `provenance` block. Use this for the paper's results. |
| `Benchmark_formative_202606-07/` | Formative runs (2026-06 → 2026-07) spanning several instructor/validator eras. Kept for the record; do not mix with wave 2. See its `README.md`. |
| `Original Moon Lab/` | Reference Unity lab JSON (`Raw/`) and hand-optimised versions (`Processed/`). |
| `Generated Labs/` | Two hand-checked example generations. |

Inside each benchmark directory:

| Folder | Contents |
|---|---|
| `Metrics/` | One `{model}_{spec}[_{level}]_{timestamp}_metrics.json` per run: model, provider, spec type, lab, level, structure, decode mode, success/error, retries, wall time, token and cost tracking, structural `lab_metrics`, and a `prompt_file` reference. |
| `Outputs/` | The paired `*_output.json`: the generated Lab / DemoModule / LabOutline / ARLEM JSON. Failed runs have a metrics record but no output. |
| `prompts/` | Deduplicated prompt text, one file per `(lab, level, spec, structure)` tuple. |
| `Reports/` | Live dir only: run CSVs, executed notebooks and HTML. The `*_wave2*` files are wave 2. `benchmark_runs.csv` (June 2026) and `benchmark_runs_final.csv` (July formative sweep) are earlier eras. |
| `logs/` | Sweep console logs. |

Not included: instructor error logs (`Errors/`), quarantined infrastructure
failures (`quarantine/`), and `suite_results_*.json`. The suite files are the
only per-run record of *failed* runs, so in a clone the failures survive only in
the run CSVs. For wave 2 that is
`Reports/benchmark_runs_wave2_repriced.csv`, which has all 385 runs including
the 34 failures. API keys are never stored in run records.

Costs are computed at the rates in `tracking/pricing.py` when a run is made.
`Code/Analysis/reprice_runs.py` re-costs a table against current rates. For wave
2, use the `_repriced` files, which correct Claude Sonnet 5 to its billed $2/$10
rate.

See the top-level [README](../../README.md) for how to find a given lab's output,
the full list of metrics, and parsing examples.

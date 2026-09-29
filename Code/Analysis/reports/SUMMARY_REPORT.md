# `build_summary_report.py` — Benchmark Success/Failure Summary

`Code/Analysis/reports/build_summary_report.py` answers *which* model × level ×
spec × lab cells generated at all, and classifies the failures. It is the
companion to [`build_statistics_report.py`](STATISTICS_REPORT.md), which covers
*what* the successful runs produced. Both read the same deduplicated DataFrame
from `benchmark_dataframe.load_runs(include_failures=True)`.

For each run window it produces, under `Artifacts/Data/Benchmark/Reports/`:

- **`benchmark_summary_<label>.ipynb`** is a fully *executed*, self-contained
  notebook. The per-run data is embedded inline, so it opens and re-runs without
  the source JSON files.
- **`benchmark_summary_<label>.html`** is a standalone HTML export that opens in
  any browser. For a PDF, open it and choose **Print → Save as PDF** (direct PDF
  export needs LaTeX/pandoc, which the project does not assume).

The committed wave 2 reports are `benchmark_summary_wave2.*` and
`benchmark_summary_wave2_repriced.*`. They are identical except for Claude
Sonnet 5's cost; see
[STATISTICS_REPORT.md § Repricing](STATISTICS_REPORT.md#repricing-after-a-provider-rate-change).

## Usage

Run from the project root with the virtual environment active:

```bash
# Default: today's runs only, labelled with today's date
python "Code/Analysis/reports/build_summary_report.py"

# A specific sweep window (e.g. the wave 2 matrix)
python "Code/Analysis/reports/build_summary_report.py" --since 20260710 --until 20260710 --label wave2

# From a run CSV instead of Metrics/ (e.g. the repriced wave 2 table)
python "Code/Analysis/reports/build_summary_report.py" \
    --runs-csv "Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv" --label wave2_repriced
```

| Flag | Behavior |
| --- | --- |
| `--since YYYYMMDD[_HHMMSS]` | Inclusive lower bound. Default is today at 00:00; `00000000` includes all history. |
| `--until YYYYMMDD[_HHMMSS]` | Optional inclusive upper bound. |
| `--label <str>` | Suffix for output filenames (default: today's date), so successive reports never overwrite each other. |
| `--runs-csv PATH` | Build from an existing run CSV rather than re-reading `*_metrics.json`. `--since`/`--until` are ignored because the CSV defines the window. |

**Failures need the suite files or a CSV.** Failed runs write no standalone
metrics file. They survive only in the gitignored `suite_results_*.json` files
and in the exported run CSVs. In a fresh clone, therefore, build the wave 2
summary with `--runs-csv`. Reading `Metrics/` alone would report 100% success.

## What's in the report

- **Overall:** headline stats for the window, success by provider × level and
  by spec × level, the failure-mode table, and a success-rate-by-provider chart.
- **Per-lab breakdown:** one section per lab topic present in the window, with
  topic stats, success by provider × level, a per-model grid of
  successes/attempts at each level (L3/L4 fan out to three specs, so `2/3` means
  one spec failed), and that topic's failure modes.
- **Takeaways:** computed from the data rather than written by hand, so they
  stay true when the report is re-run.

Failure modes come from `benchmark_dataframe.fail_mode()`, which classifies
each run's terminal error as `schema validation`, `truncation (max_tokens)`,
`404 model-not-found`, `gemini parallel-call reask 400`, or `other`.

## Dependencies

`nbformat`, `nbconvert`, `matplotlib`, `pandas`, and a `python3` Jupyter kernel
(`ipykernel`) are all in `requirements.txt` and the project venv. The script
executes the notebook with `nbconvert`'s `ExecutePreprocessor`, so generation
fails loudly if a cell errors rather than shipping a broken report.

## History

Before issue #61, this script hard-coded the three June 2026 formative sweeps
in a `SWEEPS` list and wrote a VSEPR pre-fix/post-fix narrative. It was
generalized to take any run window so that it could describe the wave 2
matrix.

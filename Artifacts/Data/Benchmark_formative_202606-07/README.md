# Formative benchmark archive (2026-06 → 2026-07)

Frozen copy of every benchmark run record produced **before the wave-2 clean
re-run** (2026-07-10). Nothing in here should be mixed with wave-2 data in
reports: this dataset spans three ARLEM-validator eras and two instructor
retry regimes, distinguishable only by run timestamp, and served as the
shakedown that surfaced the issues fixed in PRs #46–#59. The live
`Artifacts/Data/Benchmark/` directory was emptied of run data at the same
time, so everything written there after 2026-07-10 is single-era wave-2 data
(and carries an explicit `provenance` block in each metrics record — records
in this archive predate that field).

## Era boundaries inside this archive

| Records from | Stack |
|---|---|
| 2026-06-08 | June json_lab runs — instructor 1.14.4, Gemini on constrained `GENAI_STRUCTURED_OUTPUTS` + flat twin schemas (see `Reports/threats_to_validity.md` §1–6) |
| 2026-07-05 → 07-08 | Formative sweep + recovery — instructor 1.14.4 (blind unfiltered retry), unified free-decode schemas, pre-#50 ARLEM validators (None crashes, contradictory Activate/Deactivate rules) |
| 2026-07-09 → 07-10 | Gate-verification probes for PR #50/#51 only — not comparison data |

Wave-2 stack (not represented here): instructor 1.15.4 pin + parse-retry
patch (#51), post-#57 ARLEM validators, bounded transient-error retry (#58),
parallel-call fail-mode detection (#59), provenance stamping.

## Contents

| Path | What | Moved from |
|---|---|---|
| `Metrics/` | per-run `*_metrics.json` records | `Benchmark/Metrics/` |
| `Outputs/` | generated Lab/ARLEM JSON | `Benchmark/Outputs/` |
| `suite_results_*.json` | combined per-chunk suite records | `Benchmark/` |
| `quarantine/` | 156 infra-failure records + their error logs (issue #45) | `Benchmark/quarantine/` |
| `logs/` | per-chunk sweep console logs | `Benchmark/logs/` |
| `sweep_progress.log` | final checkpoint state of the July sweep | `Benchmark/` |
| `Errors/` | instructor retry/error logs | `Artifacts/Data/Errors/` |
| `prompts/` | prompt artifacts (COPY — the live `Benchmark/prompts/` kept them) | `Benchmark/prompts/` |

Path-fidelity note: `prompt_file`, `errors_file`, and `output_path` fields
inside the records still hold the original live-directory paths; resolve them
against this archive root instead.

## Loading

```python
from benchmark_dataframe import load_runs
df = load_runs(benchmark_dir=Path("Artifacts/Data/Benchmark_formative_202606-07"))
```

Git-tracked (same policy as the live dir): `Metrics/`, `Outputs/`, `prompts/`,
`logs/`, `sweep_progress.log`, and this README. Not tracked: `suite_results_*.json`,
`quarantine/`, and `Errors/`, which exist only on the machine that ran the
sweeps. Failed formative runs are therefore not in the repository, except as
rows in the formative-era CSVs under `Benchmark/Reports/`
(`benchmark_runs.csv` for June, `benchmark_runs_final.csv` for the July sweep).

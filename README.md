# llm_ARLEM_Generation

LLM generation of structured Augmented Reality lab specifications, with the
benchmark data behind the paper.

We ask LLMs from OpenAI, Anthropic, and Google to write a complete AR lab
experience as structured JSON, validated against Pydantic v2 schemas through the
[`instructor`](https://github.com/567-labs/instructor) library. Each run records
what the model produced and what that cost: tokens, dollars, wall time, retries,
and the structural shape of the generated lab. **All run data from the final
benchmark (wave 2) is committed to this repository**, so you can inspect every
generated lab and reproduce every figure without making an API call.

**Contents**

- [The wave 2 benchmark at a glance](#the-wave-2-benchmark-at-a-glance)
- [Where the data lives](#where-the-data-lives)
- [Looking at a generated lab](#looking-at-a-generated-lab)
- [Metrics we collected](#metrics-we-collected)
- [Parsing the data](#parsing-the-data)
- [Reports, figures, and tables](#reports-figures-and-tables)
- [Formative runs (archive)](#formative-runs-archive)
- [Running it yourself](#running-it-yourself)
- [Further documentation](#further-documentation)
- [Citing](#citing)

---

## The wave 2 benchmark at a glance

Wave 2 is a single clean sweep run on **2026-07-10**, and it is the dataset
the paper reports on. Every run shares one software stack: the same git commit,
`instructor==1.15.4`, and no uncommitted changes. Each metrics record carries a
`provenance` block that says so.

| | |
|---|---|
| Matrix | 11 models × 5 lab topics × 7 prompt/format cells = **385 runs** |
| Outcome | **351 succeeded**, 34 failed (28 schema-validation, 6 other) |
| Tokens | ~10.5 M total |
| Cost | ~$49.47, at the rates in [`tracking/pricing.py`](tracking/pricing.py) ([why "repriced"](#a-note-on-cost)) |
| Generation time | ~6.8 h of summed wall time |

The seven cells per model and topic are **Outline (L1)** plus
**Objectives (L3)** and **Script (L4)**, each in three output formats:

| Level | Paper name | Prompt input (from the lab YAML) | Output |
|---|---|---|---|
| L1 | Outline | field, course, topic description | `LabOutline`, a rough scene list (not a full spec) |
| L3 | Objectives | L1 + learning objectives | full spec, in each of the three formats below |
| L4 | Script | L3 + a detailed scene-by-scene script | full spec, in each of the three formats below |

L2 was retired, so the level numbers skip it on purpose: they encode how
specific the input is. See [BENCHMARK.md § Levels](Code/Benchmark/BENCHMARK.md#levels-l1l4).

| Format (`spec_type`) | Schema | What it is |
|---|---|---|
| `json_lab` | [`Code/Schemas/json_lab.py`](Code/Schemas/json_lab.py) | Our headset format: `Lab` → `DemoModule` (one per scene) → `SceneObject`s + `Clip`s with sparse per-clip deltas |
| `arlem` | [`Code/Schemas/arlem_full.py`](Code/Schemas/arlem_full.py) | Full IEEE ARLEM: `Workplace` (things, places, detectables, predicates) + `Activity` (actions, triggers) |
| `arlem_simple` | [`Code/Schemas/arlem_simplified.py`](Code/Schemas/arlem_simplified.py) | A reduced ARLEM subset |

**Models** (successes out of 35 runs each):

| Provider | Model (`display_name`) | Model id in file names | Size tier | OK |
|---|---|---|---|---|
| Anthropic | Claude Opus 4.8 | `claude-opus-4-8` | large | 35 |
| Anthropic | Claude Sonnet 5 | `claude-sonnet-5` | medium | 35 |
| Anthropic | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | small | 32 |
| OpenAI | GPT-5.5 | `gpt-5-5` | large | 35 |
| OpenAI | GPT-5.4 | `gpt-5-4` | medium | 35 |
| OpenAI | GPT-5.4 Mini | `gpt-5-4-mini` | small | 35 |
| OpenAI | GPT-5.4 Nano | `gpt-5-4-nano` | small | 29 |
| Google | Gemini 3.1 Pro | `gemini-3-1-pro-preview` | large | 34 |
| Google | Gemini 3.5 Flash | `gemini-3-5-flash` | medium | 35 |
| Google | Gemini 3.1 Flash Lite | `gemini-3-1-flash-lite` | small | 34 |
| Google | Gemini 2.5 Flash Lite | `gemini-2-5-flash-lite` | small | 12 |

**Lab topics**, one YAML each in [`Artifacts/Lab Descriptions/`](Artifacts/Lab%20Descriptions/):

- `apparent_retrograde_motion` (astronomy)
- `dna_structure_and_replication` (biology)
- `heart_anatomy_and_blood_flow` (anatomy)
- `vectors_and_projectile_motion` (physics)
- `vsepr_molecular_geometry` (chemistry)

Each YAML holds the six fields that prompts are built from: `topic_name`,
`field`, `course_context`, `topic_description`, `learning_objectives`, and
`detailed_script`. `Example/phases_of_the_moon_lab.yaml` is a worked example of the format only.
It was not part of the sweep, and `--lab` does not look in `Example/`.

---

## Where the data lives

```
Artifacts/
├── Lab Descriptions/*.yaml            ← the 5 topic inputs
├── Data/
│   ├── Benchmark/                     ← WAVE 2 (use this)
│   │   ├── Outputs/                   ← 351 generated labs, one *_output.json per successful run
│   │   ├── Metrics/                   ← 351 *_metrics.json run records (paired 1:1 with Outputs/)
│   │   ├── prompts/                   ← 35 exact prompts sent, one per (lab, level, spec, structure)
│   │   ├── Reports/                   ← flat CSVs + executed notebooks/HTML reports
│   │   └── logs/                      ← console logs, one per model × level sweep chunk
│   ├── Benchmark_formative_202606-07/ ← earlier shakedown runs (do not mix with wave 2)
│   ├── Original Moon Lab/             ← the human-authored reference lab (Raw/ + Processed/)
│   └── Generated Labs/                ← two early hand-checked generations (not wave 2)
└── Paper/
    ├── figures/                       ← publication figures (.pdf + .png)
    └── tables/                        ← LaTeX tables (\input by the paper)
```

The files you will most often want:

| You want… | Open |
|---|---|
| Every wave 2 run as one table, **including failures** | [`Reports/benchmark_runs_wave2_repriced.csv`](Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv) |
| A browsable statistics report | [`Reports/benchmark_statistics_wave2_repriced.html`](Artifacts/Data/Benchmark/Reports/benchmark_statistics_wave2_repriced.html) |
| Success/failure by model, level, and topic | [`Reports/benchmark_summary_wave2_repriced.html`](Artifacts/Data/Benchmark/Reports/benchmark_summary_wave2_repriced.html) |
| A lab a model generated | `Outputs/*_output.json` (see [below](#looking-at-a-generated-lab)) |
| The prompt that produced it | `prompts/*.txt` |
| Known confounds and caveats | [`Reports/threats_to_validity.md`](Artifacts/Data/Benchmark/Reports/threats_to_validity.md) |

All paths in this table are relative to `Artifacts/Data/Benchmark/`.

Other CSVs in `Reports/` belong to earlier runs. `benchmark_runs.csv` covers
the June 2026 runs and `benchmark_runs_final.csv` covers the July formative
sweep. Neither is wave 2.

**Not in the repository:** per-attempt instructor error logs (`Artifacts/Data/Errors/`),
the per-invocation `suite_results_*.json` files, and quarantined infrastructure
failures. Because failed runs write no standalone metrics file, **the wave 2 CSVs
are the committed record of the 34 failures**, including their `fail_mode`,
tokens, and cost.

---

## Looking at a generated lab

Every successful run writes two files that share a base name:

```
{model}_{spec}_{level}_{YYYYMMDD_HHMMSS}_output.json    ← Outputs/
{model}_{spec}_{level}_{YYYYMMDD_HHMMSS}_metrics.json   ← Metrics/
```

Dots in the model id become dashes. For example:

```
Outputs/claude-haiku-4-5-20251001_json_lab_L4_20260710_193454_output.json
```

**The lab topic is not in the file name.** It is stored as `lab_name` inside the
metrics record. To find every generation for one topic, level, and format:

```python
import json
from pathlib import Path

bench = Path("Artifacts/Data/Benchmark")
for p in sorted((bench / "Metrics").glob("*_metrics.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    if (r["lab_name"], r["level"], r["spec_type"]) == ("heart_anatomy_and_blood_flow", "L4", "json_lab"):
        print(r["display_name"], "→", bench / "Outputs" / p.name.replace("_metrics", "_output"))
```

What you will find inside an output:

- **`json_lab`** has top-level `labId`, `courseName`, `objectives`, and `modules[]`.
  Each module holds `objects[]` (prefab, texture, transform, components) and
  `clips[]` (narration, `audioClip`, and `changes[]`, which are sparse deltas
  that list only the fields that change in that clip).
- **`arlem` / `arlem_simple`** has a `workplace` (things, places, persons,
  detectables, primitives, predicates) and an `activity` (ordered actions with
  enter/exit activations and triggers).
- **L1 (`LabOutline`)** has one entry per scene with `scene_name`, `brief_purpose`,
  `key_visuals`, and `student_actions`. It is the same shape for every format.

To compare against a lab a human wrote, the original Moon Phases lab is in
[`Artifacts/Data/Original Moon Lab/`](Artifacts/Data/Original%20Moon%20Lab/). `Raw/`
is the Unity transmission JSON and `Processed/Optomized_moon_lab_final.json` is
the same lab in the v2.0 `json_lab` shape the models were asked to produce.

The metrics record's `prompt_file` field names the exact prompt that was sent.
Prompts are deduplicated, so every model that ran the same
`(lab, level, spec, structure)` cell points at the same file.

---

## Metrics we collected

Every run record (`Metrics/*_metrics.json`) contains the following blocks, and
the CSV flattens them into one row per run.

| Group | Fields | Notes |
|---|---|---|
| **Identity** | `model`, `display_name`, `provider`, `spec_type`, `lab_name`, `level`, `structure`, `timestamp`, `decode_mode`, `prompt_file` | `structure` is `multi-module` for json_lab and empty for ARLEM |
| **Provenance** | `provenance.git_sha`, `git_dirty`, `python`, `packages` (instructor, anthropic, openai, google-genai, pydantic versions) | Identical across all of wave 2 |
| **Outcome** | `success`, `error`; the CSV adds `fail_mode` and `had_usage` | `fail_mode` ∈ `schema validation` / `truncation (max_tokens)` / `404 model-not-found` / `gemini parallel-call reask 400` / `other` |
| **Tokens and cost** | `tracking.tokens.{prompt, completion, total}`, `tracking.cost.total_usd` | Summed over every attempt in the run, retries included |
| **Time** | `wall_time_seconds` (CSV `wall_s`), `tracking.timing.total_duration_ms`; the CSV adds `tokens_per_sec` | Wall time includes retries and client overhead |
| **Reliability** | `tracking.retries.*`, `tracking.errors.{parse_errors, api_errors}`, `transient_retries` | Parse retries are schema-validation reasks. `transient_retries` counts 429/5xx/connection retries and is 0 across wave 2 |
| **json_lab structure** | `num_modules`, `num_objects`, `num_unique_objects`, `num_clips`, `num_components`, `num_object_changes`, `num_text_labels`, `num_unique_prefabs`, `num_objectives`; the CSV adds `clips_per_module`, `objects_per_module`, `components_per_object`, `changes_per_clip` | From [`lab_metrics.analyze_json_lab`](Code/Benchmark/lab_metrics.py) |
| **ARLEM structure** | `num_things`, `num_places`, `num_persons`, `num_devices`, `num_sensors`, `num_apps`, `num_detectables`, `num_primitives`, `num_predicates`, `num_warnings`, `num_actions`, `total_activates`, `total_deactivates`, `total_messages`, `total_triggers`, `total_pois`, and trigger modes (`click` / `voice` / `detect` / `sensor`) | From `lab_metrics.analyze_arlem` |
| **Novel assets** (json_lab, **CSV only**) | `novel_prefabs`, `novel_textures`, their `*_refs`, `prefab_refs`, `texture_refs`, `audio_refs`, `audio_unique`, and per-module/per-object ratios | Assets the model invented that are not in the moon-lab library, i.e. the art that would have to be authored before the lab can run. These are computed at load time from `Outputs/` and not stored in the metrics JSON |
| **Size tier** (CSV only) | `size` (`small` / `medium` / `large`) | Joined from the model registry |

Two caveats apply before you aggregate:

- **L1 structural columns are always 0.** An outline has no modules or clips.
  Restrict structural analysis to L3 and L4.
- **Each row carries both the json_lab and the ARLEM column families**, with 0
  wherever a metric does not apply. Filter on `spec_type` before averaging.

Full column definitions, including how novel assets are counted, are in
[STATISTICS_REPORT.md § Columns](Code/Analysis/reports/STATISTICS_REPORT.md#columns).

### A note on cost

`cost_usd` is written into each metrics record when the run is made. Claude
Sonnet 5 was recorded at $3/$15 per M tokens, but it was actually billed (and
is now permanently priced) at $2/$10. The **`_repriced`** CSV and reports carry
the corrected figure in `cost_usd` and keep the original in `cost_usd_recorded`.
Every other model is unchanged. **Use the `_repriced` files for any cost
numbers.** See [STATISTICS_REPORT.md § Repricing](Code/Analysis/reports/STATISTICS_REPORT.md#repricing-after-a-provider-rate-change).

---

## Parsing the data

Pick whichever of the three routes fits what you need.

**1. The flat CSV.** Needs only pandas, has no project imports, and includes
failures.

```python
import pandas as pd

df = pd.read_csv("Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv")

df.groupby("display_name").success.mean()                          # success rate per model
df.groupby(["provider", "level"]).cost_usd.mean()                  # mean $ per run
full = df[df.success & df.level.isin(["L3", "L4"])]                # structural slice
full[full.spec_type == "json_lab"].groupby("level").num_clips.median()
full[full.spec_type == "arlem"].groupby("display_name").num_actions.mean()
df[~df.success][["display_name", "spec_type", "level", "lab_name", "fail_mode"]]
```

**2. `load_runs()`.** Rebuilds the same table straight from `Metrics/` and
`Outputs/`. Use it if you change a metric definition, for example the known
asset list in `lab_metrics.py`. It needs the project venv (see below).

```python
import sys; sys.path.insert(0, "Code/Analysis")
from benchmark_dataframe import load_runs

df = load_runs()          # the 351 wave 2 successes (Benchmark/ holds only wave 2)
```

Failed runs live only in the gitignored `suite_results_*.json`, so in a fresh
clone `load_runs(include_failures=True)` returns the same 351 rows. Use the CSV
for failures. Costs come from the metrics files as recorded, so they are *not*
repriced.

**3. Raw JSON.** Read `Metrics/*_metrics.json` and `Outputs/*_output.json`
directly with any JSON tool. Each metrics record is self-describing (see the
table above), and `lab_metrics.analyze_assets(output_json)` returns the actual
invented prefab and texture *names* for a lab, which the CSV reduces to counts.

---

## Reports, figures, and tables

**Exploration reports** live in [`Artifacts/Data/Benchmark/Reports/`](Artifacts/Data/Benchmark/Reports/).
Each is an executed notebook plus a standalone HTML copy:

- `benchmark_statistics_wave2_repriced.{ipynb,html}` covers tokens, cost, time,
  structural output, novel assets, and failure costs, sliced by model, provider,
  size, level, and format.
  See [STATISTICS_REPORT.md](Code/Analysis/reports/STATISTICS_REPORT.md).
- `benchmark_summary_wave2_repriced.{ipynb,html}` shows which models, levels,
  and topics generated at all, with a per-model OK/✗ grid and the dominant
  failure modes. See [SUMMARY_REPORT.md](Code/Analysis/reports/SUMMARY_REPORT.md).

The non-`_repriced` `wave2` siblings are identical except for Sonnet 5's cost.

**Publication assets** live in [`Artifacts/Paper/`](Artifacts/Paper/), with
`.pdf` and `.png` for each figure:

| Figure(s) | Shows | Script |
|---|---|---|
| `wave2_tokens_cost_{mean,total}` | tokens stacked over cost, per model | [`cost_figures.py`](Code/Analysis/figures/cost_figures.py) |
| `cost_tokens_by_model{,_L1}` | per-model means, L3/L4 and L1 slices | [`cost_by_model_figures.py`](Code/Analysis/figures/cost_by_model_figures.py) |
| `wave2_time_*`, `wave2_latency_throughput` | generation time by level/format, decomposition, time vs tokens, throughput | [`timing_figures.py`](Code/Analysis/figures/timing_figures.py) |
| `wave2_json_lab_size`, `wave2_authoring_burden`, `wave2_arlem_validity`, `wave2_specificity_elasticity`, `wave2_arlem_floor` | what the generated labs contain: size against the human moon lab, novel-asset burden, ARLEM validity, what the script adds, minimum-count floor effects | [`structure_figures.py`](Code/Analysis/figures/structure_figures.py) |

The LaTeX tables in `Artifacts/Paper/tables/` (model roster, cost, time,
structure, prompt levels) are written by the same scripts or authored by hand.
To regenerate everything from the committed data (no API calls):

```bash
CSV="Artifacts/Data/Benchmark/Reports/benchmark_runs_wave2_repriced.csv"
python "Code/Analysis/figures/cost_figures.py"          --runs-csv "$CSV" --suffix _repriced
python "Code/Analysis/figures/cost_by_model_figures.py" --runs-csv "$CSV" --suffix _repriced
python "Code/Analysis/figures/timing_figures.py"
python "Code/Analysis/figures/structure_figures.py"
python "Code/Analysis/reports/build_statistics_report.py" --runs-csv "$CSV" --label wave2_repriced
python "Code/Analysis/reports/build_summary_report.py"    --runs-csv "$CSV" --label wave2_repriced
```

---

## Formative runs (archive)

[`Artifacts/Data/Benchmark_formative_202606-07/`](Artifacts/Data/Benchmark_formative_202606-07/)
holds 440 run records from June to early July 2026. That was the shakedown
period that surfaced the retry, validator, and provider bugs fixed before wave 2.
These records span several `instructor` versions and ARLEM-validator eras, so
**do not pool them with wave 2**. They are kept for the record, and the
[archive README](Artifacts/Data/Benchmark_formative_202606-07/README.md)
explains the era boundaries. Load them with
`load_runs(benchmark_dir=Path("Artifacts/Data/Benchmark_formative_202606-07"))`.

---

## Running it yourself

### 1. Environment Setup

To set up the Python environment and install dependencies, run the setup script
for your operating system:

Windows (PowerShell):
```PowerShell
./setup_windows.ps1
```
Note: If you get a permission error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.

macOS / Linux:
```Bash
chmod +x setup_unix.sh
./setup_unix.sh
```

### 2. Manual Installation

If you prefer to do it manually:

| OS | Create Venv | Activate | Install |
| --- | --- | --- | --- |
| Windows | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` | `pip install -r requirements.txt` |
| macOS | `python3 -m venv .venv` | `source .venv/bin/activate` | `pip install -r requirements.txt` |

`requirements.txt` pins `instructor==1.15.4`. The benchmark's retry handling
depends on that exact version, so do not upgrade it casually.

### 3. Configuration

The `.env` file is ignored by git for security. Create a file named `.env` in the
root directory and add your keys. You only need keys for the providers you
intend to call. Analyzing the committed data needs none.

```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### 4. Benchmarking LLM Lab Generation

The main entry point is the benchmark runner,
[Code/Benchmark/benchmark.py](Code/Benchmark/benchmark.py). It builds a prompt
for a lab topic, calls a model to generate a validated AR lab spec, and records
tokens, cost, retries, and structural metrics.

```powershell
# From the project root, with the venv active:
python "Code/Benchmark/benchmark.py" --list-models
python "Code/Benchmark/benchmark.py" --list-labs
python "Code/Benchmark/benchmark.py" --model claude-haiku-4.5 --lab heart_anatomy_and_blood_flow --level L3
python "Code/Benchmark/benchmark.py" --model gpt-5.4-mini --lab vsepr_molecular_geometry --level L4 --spec json_lab arlem arlem_simple
```

New runs are written to `Artifacts/Data/Benchmark/` next to the wave 2 data.
Use `load_runs(since=...)` or the report builders' `--since` flag to keep them
separate. The full wave 2 matrix was driven by
[`Code/Benchmark/run_sweep.ps1`](Code/Benchmark/run_sweep.ps1), a resumable
driver that runs 33 chunks (one per model × level group) and checkpoints its
progress.

For the full command-line reference, covering every flag, the levels, structure
and spec modes, the model registry, and output naming, see
[Code/Benchmark/BENCHMARK.md](Code/Benchmark/BENCHMARK.md).

---

## Further documentation

| Document | Covers |
|---|---|
| [Code/Benchmark/BENCHMARK.md](Code/Benchmark/BENCHMARK.md) | `benchmark.py` CLI reference: flags, levels, structures, specs, model registry, output files |
| [Code/ARCHITECTURE.md](Code/ARCHITECTURE.md) | Code map: module layout, schema patterns, tracking pipeline, provider-specific gotchas |
| [Code/Analysis/reports/STATISTICS_REPORT.md](Code/Analysis/reports/STATISTICS_REPORT.md) | Every CSV column, novel-asset methodology, `load_runs()` usage, repricing |
| [Code/Analysis/reports/SUMMARY_REPORT.md](Code/Analysis/reports/SUMMARY_REPORT.md) | The success/failure summary report |
| [Artifacts/Data/README.md](Artifacts/Data/README.md) | What is in each data directory and what is not committed |
| [Artifacts/Data/Benchmark/Reports/threats_to_validity.md](Artifacts/Data/Benchmark/Reports/threats_to_validity.md) | Known confounds in the cross-provider comparison, and which ones wave 2 resolves |
| [Artifacts/Data/Benchmark_formative_202606-07/README.md](Artifacts/Data/Benchmark_formative_202606-07/README.md) | The formative archive and its era boundaries |
| [CLAUDE.md](CLAUDE.md) | Condensed project overview and command cheat sheet |

---

## Citing

If you use this software or dataset, please cite it using
[`CITATION.cff`](CITATION.cff) (GitHub's **"Cite this repository"** button reads
it). Released under the BSD 2-Clause license; see [LICENSE](LICENSE).

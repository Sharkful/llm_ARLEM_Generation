"""
Build a quantitative statistics report from every benchmark run.

Where ``build_summary_report.py`` answers *which* providers/levels generated at
all, this report aggregates *what they produced*: token counts, cost, generation
time, modules/clips/objects, unique objects, and components made — sliced by
model, provider, and prompt-specificity level (L1-L4).

Data is loaded via ``benchmark_dataframe.load_runs()`` and embedded inline in the
notebook so the report stays self-contained / re-runnable. A flat
``benchmark_runs.csv`` is also written for slicing in any other tool.

Usage:
    python "Code/Testing/build_statistics_report.py"
Outputs:
    Artifacts/Reports/benchmark_statistics.ipynb   (executed)
    Artifacts/Reports/benchmark_statistics.html
    Artifacts/Reports/benchmark_runs.csv
"""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dataframe import load_runs  # noqa: E402


def build_notebook(records_json: str) -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        "# AR Lab Generation — Benchmark Statistics\n\n"
        "Quantitative analysis of the formative lab-generation sweep: every "
        "registered model × prompt-specificity level (L1–L4) across three lab "
        "topics. This is the companion to the success/failure summary "
        "(`benchmark_summary`) — here we look at **what the successful runs "
        "actually produced**: tokens, cost, generation time, the structural "
        "shape of each generated lab, and how many **novel assets** (prefabs, "
        "textures, audio we don't yet have) each lab would require us to author.\n\n"
        "The table covers **both successes and failures**. Cost/token/speed and "
        "structural sections aggregate the **successful** runs; a dedicated "
        "*Failed runs* section breaks down what the failures cost (some burned "
        "real tokens; others — bad model ids / config errors — never reached the "
        "model and carry no stats).\n\n"
        "**Caveat on L1.** L1 prompts produce a rough `LabOutline`, not a full "
        "`Lab` — so modules/clips/objects/components are 0 for L1 by design. "
        "Structural sections below therefore restrict to **L2–L4**."
    ))

    # ── Data + helpers (records embedded inline → self-contained notebook) ──
    cells.append(new_code_cell(
        "%matplotlib inline\n"
        "import json\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n\n"
        "pd.set_option('display.max_rows', 200)\n"
        "pd.set_option('display.width', 200)\n\n"
        "# Full per-run table embedded inline — notebook is self-contained.\n"
        f"RECORDS = json.loads(r'''{records_json}''')\n"
        "df = pd.DataFrame(RECORDS)            # ALL runs (successes + failures)\n"
        "OK = df[df.success].copy()           # successful runs only\n"
        "FAIL = df[~df.success].copy()        # failed runs\n\n"
        "LEVELS = ['L1', 'L2', 'L3', 'L4']\n"
        "PROVIDERS = ['openai', 'anthropic', 'google']\n"
        "FULL = OK[OK.level != 'L1']  # L2-L4 successes (any spec)\n"
        "# Structural columns differ by spec, so slice before aggregating shape.\n"
        "FULL_JSON = FULL[FULL.spec_type == 'json_lab']                       # json_lab shape\n"
        "FULL_ARLEM = FULL[FULL.spec_type.isin(['arlem', 'arlem_simple'])]    # ARLEM shape\n\n"
        "def agg(frame, by):\n"
        "    \"\"\"Standard cost/token/time aggregation grouped by a column.\"\"\"\n"
        "    g = frame.groupby(by)\n"
        "    out = pd.DataFrame({\n"
        "        'runs': g.size(),\n"
        "        'avg_cost_usd': g.cost_usd.mean().round(4),\n"
        "        'total_cost_usd': g.cost_usd.sum().round(2),\n"
        "        'avg_total_tokens': g.total_tokens.mean().round(0),\n"
        "        'avg_completion_tokens': g.completion_tokens.mean().round(0),\n"
        "        'avg_wall_s': g.wall_s.mean().round(1),\n"
        "        'avg_tokens_per_sec': g.tokens_per_sec.mean().round(1),\n"
        "    })\n"
        "    return out\n\n"
        "def struct_agg(frame, by):\n"
        "    \"\"\"json_lab structural-shape aggregation (use on FULL_JSON, L2-L4).\"\"\"\n"
        "    g = frame.groupby(by)\n"
        "    out = pd.DataFrame({\n"
        "        'runs': g.size(),\n"
        "        'avg_modules': g.num_modules.mean().round(1),\n"
        "        'avg_clips': g.num_clips.mean().round(1),\n"
        "        'avg_clips_per_module': g.clips_per_module.mean().round(1),\n"
        "        'avg_objects': g.num_objects.mean().round(1),\n"
        "        'avg_unique_objects': g.num_unique_objects.mean().round(1),\n"
        "        'avg_components': g.num_components.mean().round(1),\n"
        "        'avg_obj_changes': g.num_object_changes.mean().round(1),\n"
        "    })\n"
        "    return out\n\n"
        "def arlem_agg(frame, by):\n"
        "    \"\"\"ARLEM structural-shape aggregation (use on FULL_ARLEM, L2-L4).\"\"\"\n"
        "    g = frame.groupby(by)\n"
        "    out = pd.DataFrame({\n"
        "        'runs': g.size(),\n"
        "        'avg_things': g.num_things.mean().round(1),\n"
        "        'avg_places': g.num_places.mean().round(1),\n"
        "        'avg_predicates': g.num_predicates.mean().round(1),\n"
        "        'avg_actions': g.num_actions.mean().round(1),\n"
        "        'avg_activates': g.total_activates.mean().round(1),\n"
        "        'avg_triggers': g.total_triggers.mean().round(1),\n"
        "        'avg_pois': g.total_pois.mean().round(1),\n"
        "    })\n"
        "    return out\n\n"
        "print(f'{len(df)} runs ({int(df.success.sum())} ok / "
        "{int((~df.success).sum())} failed) | {df.model.nunique()} models | "
        "levels {sorted(df.level.unique())} | "
        "specs {sorted(df.spec_type.dropna().unique())} | "
        "labs {df.lab_name.nunique()}')"
    ))

    cells.append(new_markdown_cell(
        "## Slice it yourself\n\n"
        "The full flat table lives in `Artifacts/Reports/benchmark_runs.csv`, or "
        "load it live from the source metrics files:\n\n"
        "```python\n"
        "import sys; sys.path.insert(0, 'Code/Testing')\n"
        "from benchmark_dataframe import load_runs\n"
        "df = load_runs()\n"
        "df[df.provider == 'anthropic'].wall_s.mean()       # avg speed, one provider\n"
        "df.groupby('level').num_clips.mean()               # avg clips per level\n"
        "```\n\n"
        "Every column available:"
    ))
    cells.append(new_code_cell("list(df.columns)"))

    # ── Cost & tokens ──────────────────────────────────────────────
    cells.append(new_markdown_cell(
        "## Cost & tokens\n\n"
        "Every spec runs free-decode (generate + validate/retry), so the "
        "provider-reported `cost_usd` / `total_tokens` are the real figures — there "
        "is no Gemini schema-token adjustment to fold in. Aggregations here cover all "
        "specs; the **By spec** cut compares json_lab vs ARLEM full vs simplified."
    ))
    cells.append(new_code_cell(
        "# Successful runs only (failed-run spend is covered in its own section).\n"
        "print('By provider:'); display(agg(OK, 'provider'))\n"
        "print('By spec:'); display(agg(OK, 'spec_type'))\n"
        "print('By level:'); display(agg(OK, 'level').reindex(LEVELS))\n"
        "print('By model:'); display(agg(OK, 'display_name').sort_values('avg_cost_usd'))"
    ))

    # ── Generation speed ───────────────────────────────────────────
    cells.append(new_markdown_cell(
        "## Generation speed\n\n"
        "`avg_wall_s` is end-to-end wall-clock time per run; "
        "`avg_tokens_per_sec` is completion throughput "
        "(completion tokens ÷ API duration)."
    ))
    cells.append(new_code_cell(
        "speed_cols = ['runs', 'avg_wall_s', 'avg_tokens_per_sec']\n"
        "print('By provider:'); display(agg(OK, 'provider')[speed_cols])\n"
        "print('By level:'); display(agg(OK, 'level').reindex(LEVELS)[speed_cols])\n"
        "print('By model:'); display(agg(OK, 'display_name')[speed_cols].sort_values('avg_wall_s'))"
    ))

    # ── Structural output ──────────────────────────────────────────
    cells.append(new_markdown_cell(
        "## json_lab structural output (L2–L4)\n\n"
        "Shape of the generated **json_lab** labs: modules per lab, clips per module, "
        "objects (and how many were distinct), and components created. Restricted to "
        "`FULL_JSON` (json_lab, L2–L4); L1 outlines and ARLEM runs are excluded."
    ))
    cells.append(new_code_cell(
        "print('By level:'); display(struct_agg(FULL_JSON, 'level').reindex(['L2','L3','L4']))\n"
        "print('By provider:'); display(struct_agg(FULL_JSON, 'provider'))\n"
        "print('By model:'); display(struct_agg(FULL_JSON, 'display_name').sort_values('avg_clips', ascending=False))"
    ))

    cells.append(new_markdown_cell(
        "### Clips by specificity level × provider\n\n"
        "Does more prompt detail yield more clips, and does that hold across "
        "providers? (json_lab runs only.)"
    ))
    cells.append(new_code_cell(
        "if not FULL_JSON.empty:\n"
        "    ct = FULL_JSON.pivot_table(index='provider', columns='level', "
        "values='num_clips', aggfunc='mean').round(1)\n"
        "    display(ct)\n"
        "else:\n"
        "    print('No json_lab L2-L4 runs in this sweep.')"
    ))

    # ── ARLEM structural output ────────────────────────────────────
    cells.append(new_markdown_cell(
        "## ARLEM structural output (L2–L4)\n\n"
        "Shape of the generated **ARLEM** scenarios (`arlem` full + `arlem_simple`): "
        "workplace things/places/predicates and activity actions / activates / "
        "triggers / POIs. Restricted to `FULL_ARLEM` (L2–L4). json_lab's "
        "modules/clips/objects don't apply here; ARLEM has a single ARLEMScenario "
        "shape (no `--structure`). The **By spec** cut contrasts full vs simplified."
    ))
    cells.append(new_code_cell(
        "if not FULL_ARLEM.empty:\n"
        "    print('By spec:'); display(arlem_agg(FULL_ARLEM, 'spec_type'))\n"
        "    print('By level:'); display(arlem_agg(FULL_ARLEM, 'level').reindex(['L2','L3','L4']))\n"
        "    print('By provider:'); display(arlem_agg(FULL_ARLEM, 'provider'))\n"
        "    print('By model:'); display(arlem_agg(FULL_ARLEM, 'display_name').sort_values('avg_actions', ascending=False))\n"
        "else:\n"
        "    print('No ARLEM L2-L4 runs in this sweep.')"
    ))
    cells.append(new_markdown_cell(
        "### Trigger-mode mix (ARLEM)\n\n"
        "How the generated activities advance between steps. Full ARLEM allows "
        "click/voice/detect/sensor; simplified is limited to click/detect."
    ))
    cells.append(new_code_cell(
        "trig_cols = ['trigger_click', 'trigger_voice', 'trigger_detect', 'trigger_sensor']\n"
        "if not FULL_ARLEM.empty:\n"
        "    display(FULL_ARLEM.groupby('spec_type')[trig_cols].mean().round(1))\n"
        "else:\n"
        "    print('No ARLEM L2-L4 runs in this sweep.')"
    ))

    # ── Novel assets ───────────────────────────────────────────────
    cells.append(new_markdown_cell(
        "## Novel assets (json_lab, L2–L4)\n\n"
        "Novel assets are a **json_lab** concept (invented prefab/texture names); "
        "ARLEM has no comparable asset library, so this section restricts to "
        "`FULL_JSON`.\n\n"
        "Today the headset only has the **moon-lab** asset library "
        "(textures `2k_earth_daymap`/`2k_moon`/`2k_sun`/`balldimpled`; prefabs "
        "`SunPrefab`/`moveableSphere`/`clickableSphere`/`tinySphere`/`textPrefab`/"
        "`robotIdle`). To build a lab on any other topic, a model must **invent** "
        "asset names — prefabs/models and textures we don't have and would have to "
        "author. `novel_textures` / `novel_prefabs` count the **distinct** invented "
        "assets per lab (deduped across modules — a prefab reused three times is "
        "one asset to build); the `*_refs` columns count how many object-instances "
        "point at one.\n\n"
        "**Audio is different:** there is no existing audio library at all, so "
        "every `audioClip` is novel by definition. Reporting it as 'novel' would "
        "always be 100%, so audio is shown as **volume/density** (`audio_per_clip`) "
        "instead — effectively, what share of clips carry narration audio."
    ))
    cells.append(new_code_cell(
        "def asset_agg(frame, by):\n"
        "    \"\"\"Novel-asset aggregation (use on L2-L4 only).\"\"\"\n"
        "    g = frame.groupby(by)\n"
        "    out = pd.DataFrame({\n"
        "        'runs': g.size(),\n"
        "        'avg_novel_prefabs': g.novel_prefabs.mean().round(1),\n"
        "        'avg_novel_textures': g.novel_textures.mean().round(1),\n"
        "        'novel_prefabs_per_module': g.novel_prefabs_per_module.mean().round(2),\n"
        "        'novel_prefabs_per_object': g.novel_prefabs_per_object.mean().round(2),\n"
        "        'novel_textures_per_module': g.novel_textures_per_module.mean().round(2),\n"
        "        'audio_per_clip': g.audio_per_clip.mean().round(2),\n"
        "    })\n"
        "    return out\n\n"
        "print('By level:'); display(asset_agg(FULL_JSON, 'level').reindex(['L2','L3','L4']))\n"
        "print('By provider:'); display(asset_agg(FULL_JSON, 'provider'))\n"
        "print('By model size:'); display(asset_agg(FULL_JSON, 'size').reindex(['small','medium','large']))\n"
        "print('By model:'); display(asset_agg(FULL_JSON, 'display_name').sort_values('avg_novel_prefabs', ascending=False))"
    ))
    cells.append(new_markdown_cell(
        "### Novel prefabs by specificity level × provider\n\n"
        "Does more prompt detail change how many assets the model invents, and "
        "does that hold across providers? (A flat row would mean asset count is "
        "driven by lab size, not prompt specificity.)"
    ))
    cells.append(new_code_cell(
        "if not FULL_JSON.empty:\n"
        "    ct = FULL_JSON.pivot_table(index='provider', columns='level', "
        "values='novel_prefabs', aggfunc='mean').round(1)\n"
        "    display(ct)\n"
        "    print('Per-module (size-normalized):')\n"
        "    display(FULL_JSON.pivot_table(index='provider', columns='level', "
        "values='novel_prefabs_per_module', aggfunc='mean').round(2))\n"
        "else:\n"
        "    print('No json_lab L2-L4 runs in this sweep.')"
    ))
    cells.append(new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Novel prefabs & textures per module by model size (json_lab only)\n"
        "order = ['small', 'medium', 'large']\n"
        "pm = FULL_JSON.groupby('size').novel_prefabs_per_module.mean().reindex(order)\n"
        "tm = FULL_JSON.groupby('size').novel_textures_per_module.mean().reindex(order)\n"
        "x = np.arange(len(order)); w = 0.38\n"
        "axes[0].bar(x - w/2, pm.values, w, label='prefabs', color='#c84')\n"
        "axes[0].bar(x + w/2, tm.values, w, label='textures', color='#48c')\n"
        "axes[0].set_xticks(x); axes[0].set_xticklabels(order)\n"
        "axes[0].set_title('Novel assets per module by model size')\n"
        "axes[0].set_ylabel('avg per module'); axes[0].legend()\n\n"
        "# Novel prefabs per lab by specificity level (json_lab only)\n"
        "lv = ['L2', 'L3', 'L4']\n"
        "k = FULL_JSON.groupby('level').novel_prefabs.mean().reindex(lv)\n"
        "axes[1].bar(lv, k.values, color='#4a7')\n"
        "axes[1].set_title('Novel prefabs per lab by specificity level')\n"
        "axes[1].set_ylabel('distinct invented prefabs')\n"
        "for i, v in enumerate(k.values):\n"
        "    axes[1].annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ── Failed runs ────────────────────────────────────────────────
    cells.append(new_markdown_cell(
        "## Failed runs\n\n"
        "Failures live only in the `suite_results_*.json` sweeps (a successful run "
        "also writes a standalone metrics file; a failed one does not). They split "
        "cleanly into two kinds, keyed on whether the run actually reached the "
        "model and burned tokens (`had_usage`):\n\n"
        "- **Billable failures** — `truncation (max_tokens)` and `schema "
        "validation` retries generated real output before failing, so they carry "
        "meaningful token / cost / time stats (and are often *more* expensive than "
        "a success, because instructor retried).\n"
        "- **No-op failures** — `404 model-not-found` and config errors never "
        "reached the model: 0 tokens, $0, sub-second. They have no relevant "
        "statistics and are excluded from the cost/token aggregates below."
    ))
    cells.append(new_code_cell(
        "# Overview: how common is each failure mode, and is it billable?\n"
        "ov = FAIL.groupby('fail_mode').agg(\n"
        "    failed_runs=('model', 'size'),\n"
        "    billable=('had_usage', 'sum'),\n"
        "    total_tokens=('total_tokens', 'sum'),\n"
        "    total_cost_usd=('cost_usd', 'sum'),\n"
        "    avg_wall_s=('wall_s', 'mean'),\n"
        ").round({'total_cost_usd': 2, 'avg_wall_s': 1}).sort_values('failed_runs', ascending=False)\n"
        "print('Failure modes (the \"common errors\" breakdown):')\n"
        "display(ov)\n"
        "wasted = FAIL.cost_usd.sum(); spent = OK.cost_usd.sum()\n"
        "print(f'Spend on failed runs: ${wasted:.2f}  "
        "({wasted / (wasted + spent):.1%} of total ${wasted + spent:.2f}) — "
        "all from billable failures.')"
    ))
    cells.append(new_markdown_cell(
        "### Billable failures — what they cost\n\n"
        "Token/cost/time for the failures that actually did work, by model. These "
        "are the runs worth attention: real money spent on output that didn't "
        "validate or got truncated."
    ))
    cells.append(new_code_cell(
        "BILL = FAIL[FAIL.had_usage]\n"
        "by_model = BILL.groupby('display_name').agg(\n"
        "    failed_runs=('model', 'size'),\n"
        "    total_tokens=('total_tokens', 'sum'),\n"
        "    total_cost_usd=('cost_usd', 'sum'),\n"
        "    avg_wall_s=('wall_s', 'mean'),\n"
        "    avg_retries=('retries', 'mean'),\n"
        ").round({'total_cost_usd': 3, 'avg_wall_s': 1, 'avg_retries': 1})\n"
        "display(by_model.sort_values('total_cost_usd', ascending=False))\n"
        "print('By level:')\n"
        "display(BILL.groupby('level').agg(\n"
        "    failed_runs=('model', 'size'),\n"
        "    avg_tokens=('total_tokens', 'mean'),\n"
        "    avg_cost_usd=('cost_usd', 'mean'),\n"
        "    avg_wall_s=('wall_s', 'mean'),\n"
        ").round(2))"
    ))
    cells.append(new_markdown_cell(
        "### No-op failures — no relevant statistics\n\n"
        "Listed for completeness; these never billed. (`404` = deprecated/bad "
        "model id; `other` = a missing-argument config error.)"
    ))
    cells.append(new_code_cell(
        "NOOP = FAIL[~FAIL.had_usage]\n"
        "display(NOOP[['display_name', 'level', 'lab_name', 'fail_mode', "
        "'total_tokens', 'cost_usd', 'wall_s']]"
        ".reset_index(drop=True))"
    ))
    cells.append(new_code_cell(
        "# Cost burned per failure mode (billable only).\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "fc = BILL.groupby('fail_mode').cost_usd.sum().sort_values()\n"
        "ax.barh(fc.index, fc.values, color='#c55')\n"
        "ax.set_title('Spend burned on failed runs by failure mode (USD)')\n"
        "ax.set_xlabel('USD')\n"
        "for i, v in enumerate(fc.values):\n"
        "    ax.annotate(f'${v:.2f}', (v, i), va='center', fontsize=9)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # ── Charts ─────────────────────────────────────────────────────
    cells.append(new_markdown_cell("## Charts"))
    cells.append(new_code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n\n"
        "# 1. Avg cost per run by model\n"
        "c = df.groupby('display_name').cost_usd.mean().sort_values()\n"
        "axes[0, 0].barh(c.index, c.values, color='#4a7')\n"
        "axes[0, 0].set_title('Avg cost per run by model (USD)')\n"
        "axes[0, 0].set_xlabel('USD')\n\n"
        "# 2. Avg clips by level (json_lab, L2-L4)\n"
        "k = FULL_JSON.groupby('level').num_clips.mean().reindex(['L2','L3','L4'])\n"
        "axes[0, 1].bar(k.index, k.values, color='#47a')\n"
        "axes[0, 1].set_title('Avg clips per json_lab by specificity level')\n"
        "axes[0, 1].set_ylabel('clips')\n"
        "for i, v in enumerate(k.values):\n"
        "    axes[0, 1].annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n\n"
        "# 3. Avg throughput by provider\n"
        "s = df.groupby('provider').tokens_per_sec.mean().reindex(PROVIDERS)\n"
        "axes[1, 0].bar([p.capitalize() for p in s.index], s.values, color='#a47')\n"
        "axes[1, 0].set_title('Avg completion throughput by provider')\n"
        "axes[1, 0].set_ylabel('tokens / sec')\n"
        "for i, v in enumerate(s.values):\n"
        "    axes[1, 0].annotate(f'{v:.0f}', (i, v), ha='center', va='bottom')\n\n"
        "# 4. Avg objects vs clips by level (json_lab)\n"
        "lv = ['L2', 'L3', 'L4']\n"
        "objs = FULL_JSON.groupby('level').num_objects.mean().reindex(lv)\n"
        "clps = FULL_JSON.groupby('level').num_clips.mean().reindex(lv)\n"
        "x = np.arange(len(lv)); w = 0.38\n"
        "axes[1, 1].bar(x - w/2, objs.values, w, label='objects', color='#c84')\n"
        "axes[1, 1].bar(x + w/2, clps.values, w, label='clips', color='#48c')\n"
        "axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(lv)\n"
        "axes[1, 1].set_title('Avg objects & clips by specificity level')\n"
        "axes[1, 1].legend()\n\n"
        "plt.tight_layout(); plt.show()"
    ))

    cells.append(new_markdown_cell(
        "## Notes\n\n"
        "- Cost/token figures are the raw provider-reported `cost_usd` / "
        "`total_tokens`. Every spec runs free-decode, so there is no Gemini "
        "schema-token adjustment (the old `effective_*` columns were retired).\n"
        "- Structural sections split by **spec_type**: json_lab shape "
        "(modules/clips/objects) uses `FULL_JSON`; ARLEM shape "
        "(things/places/actions/triggers/POIs) uses `FULL_ARLEM`. A row carries 0 "
        "for the columns that don't apply to its spec.\n"
        "- Structural counts are over **successful** L2–L4 runs only; a model "
        "that failed a level contributes no row there.\n"
        "- Failed-run stats use `had_usage` (`total_tokens > 0`) to separate "
        "billable failures from no-op ones; only billable failures enter the "
        "cost/token aggregates.\n"
        "- `num_components` is the count of components created per lab "
        "(\"new components made\"); the per-type breakdown lives in the source "
        "metrics' `component_types` if a finer cut is needed.\n"
        "- Novel-asset counts are computed by re-reading each saved `*_output.json` "
        "at load time (no re-running of the sweep); membership in the known moon-lab "
        "library is case-insensitive. The invented asset *names* aren't in the CSV "
        "but are available per-lab via `lab_metrics.analyze_assets(output_json)` "
        "(`novel_prefab_names` / `novel_texture_names`) for qualitative review."
    ))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runs(include_failures=True)

    csv_path = OUT_DIR / "benchmark_runs.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path.relative_to(ROOT)} ({len(df)} rows)")

    # to_json handles numpy types and emits null for NaN — valid for json.loads.
    nb = build_notebook(df.to_json(orient="records"))

    ipynb_path = OUT_DIR / "benchmark_statistics.ipynb"
    from nbconvert.preprocessors import ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(OUT_DIR)}})
    nbformat.write(nb, ipynb_path)
    print(f"Wrote {ipynb_path.relative_to(ROOT)}")

    from nbconvert import HTMLExporter
    html, _ = HTMLExporter(exclude_input=False).from_notebook_node(nb)
    html_path = OUT_DIR / "benchmark_statistics.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

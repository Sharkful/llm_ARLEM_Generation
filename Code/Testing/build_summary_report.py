"""
Build a shareable benchmark success/failure summary notebook (+ HTML).

This is the qualitative companion to ``build_statistics_report.py``: where that
report aggregates *what* the successful runs produced (tokens, cost, structure),
this one answers *which* model × level × spec × lab cells generated at all, and
classifies the failures. Both source the same deduped dataframe via
``benchmark_dataframe.load_runs`` and embed it inline so the notebook is
self-contained / re-runnable without the source JSON.

By default it reports **only today's** runs (so a fresh sweep is isolated from
prior history) and stamps every output filename with the ``--label`` (default:
today's date) so successive reports never clobber each other. Point it at a
specific sweep with ``--since``/``--until``; e.g. the wave-2 matrix:

    python "Code/Testing/build_summary_report.py" \
        --since 20260710 --until 20260710 --label wave2

Outputs (all under Artifacts/Data/Benchmark/Reports/):
    benchmark_summary_<label>.ipynb   (executed)
    benchmark_summary_<label>.html

History: this script previously hard-coded the three June-2026 formative sweeps
and a VSEPR pre-fix/post-fix narrative. It was generalized to any run window
(issue #61) so it can describe the wave-2 single-era matrix.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "Artifacts" / "Data" / "Benchmark" / "Reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_dataframe import load_runs  # noqa: E402


# ── Notebook assembly ────────────────────────────────────────────────

def build_notebook(records_json: str) -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        "# AR Lab Generation — Benchmark Success/Failure Summary\n\n"
        "LLM-generated ARLEM lab specifications, benchmarked across providers, "
        "models, prompt-specificity levels (L1/L3/L4), output specs (`json_lab`, "
        "`arlem`, `arlem_simple`), and lab topics.\n\n"
        "This is the success/failure companion to the statistics report "
        "(`benchmark_statistics`, which covers **what** the successful runs "
        "produced). Here we look at **which** cells generated at all and **why** "
        "the failures failed. Cost/token figures are summed from each cell's "
        "tracking record. The failure taxonomy comes from "
        "`benchmark_dataframe.fail_mode` (`schema validation` = the model's output "
        "failed Pydantic/ARLEM cross-validation — real model signal; `other` = "
        "everything else, including the upstream Gemini multi-function-call "
        "retry-death, issue #53).\n\n"
        "The tables count **both successes and failures**; every 'X of Y' is "
        "successes over attempts for that slice."
    ))

    # Data + helpers cell (embeds records inline for a self-contained notebook).
    cells.append(new_code_cell(
        "%matplotlib inline\n"
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "pd.set_option('display.max_rows', 300)\n\n"
        "# Deduped per-run dataframe embedded inline — notebook is self-contained.\n"
        f"RECORDS = json.loads(r'''{records_json}''')\n"
        "df = pd.DataFrame(RECORDS)\n\n"
        "# Order axes by what's actually present, not a fixed assumption.\n"
        "LEVELS = [lv for lv in ['L1', 'L2', 'L3', 'L4'] if lv in set(df.level)]\n"
        "PROVIDERS = [p for p in ['openai', 'anthropic', 'google'] if p in set(df.provider)]\n"
        "SPECS = [s for s in ['json_lab', 'arlem', 'arlem_simple'] if s in set(df.spec_type)]\n\n"
        "def _frac(d):\n"
        "    return f\"{int(d.success.sum())}/{len(d)}\" if len(d) else '-'\n\n"
        "def success_by_provider_level(sub):\n"
        "    rows = []\n"
        "    for p in PROVIDERS:\n"
        "        d = sub[sub.provider == p]\n"
        "        row = {'provider': p}\n"
        "        for lv in LEVELS:\n"
        "            row[lv] = _frac(d[d.level == lv])\n"
        "        row['total'] = _frac(d)\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('provider')\n\n"
        "def success_by_spec_level(sub):\n"
        "    rows = []\n"
        "    for s in SPECS:\n"
        "        d = sub[sub.spec_type == s]\n"
        "        row = {'spec': s}\n"
        "        for lv in LEVELS:\n"
        "            row[lv] = _frac(d[d.level == lv])\n"
        "        row['total'] = _frac(d)\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('spec')\n\n"
        "def model_grid(sub):\n"
        "    # One row per model; per-level cell is successes/attempts across all\n"
        "    # specs and labs in `sub` (L3/L4 fan out to 3 specs, so not 0/1).\n"
        "    rows = []\n"
        "    order = sub.sort_values(['provider', 'model']).display_name.unique()\n"
        "    for m in order:\n"
        "        d = sub[sub.display_name == m]\n"
        "        row = {'model': m}\n"
        "        for lv in LEVELS:\n"
        "            row[lv] = _frac(d[d.level == lv])\n"
        "        row['ok'] = _frac(d)\n"
        "        fm = [x for x in d[~d.success].fail_mode.tolist() if x]\n"
        "        row['top fail mode'] = max(set(fm), key=fm.count) if fm else ''\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('model')\n\n"
        "def slice_stats(sub):\n"
        "    ok = int(sub.success.sum()); n = len(sub)\n"
        "    succ = sub[sub.success & (sub.level != 'L1')]\n"
        "    return {\n"
        "        'success': f\"{ok}/{n} ({ok/n:.0%})\" if n else '-',\n"
        "        'wall_min': round(float(sub.wall_s.sum()) / 60, 1),\n"
        "        'cost_usd': round(float(sub.cost_usd.sum()), 2),\n"
        "        'avg_objects_L3-4': round(float(succ.num_objects.mean()), 1) if len(succ) else 0,\n"
        "        'avg_clips_L3-4': round(float(succ.num_clips.mean()), 1) if len(succ) else 0,\n"
        "    }\n\n"
        "def fail_table(sub):\n"
        "    fm = sub[~sub.success].fail_mode\n"
        "    if not len(fm):\n"
        "        return None\n"
        "    return fm.value_counts().rename_axis('mode').to_frame('failed runs')\n"
    ))

    # Overall provenance + headline.
    cells.append(new_markdown_cell("## Overall\n\nThe whole run window at a glance."))
    cells.append(new_code_cell(
        "print('Overall stats:', slice_stats(df))\n"
        "print()\n"
        "print('Success by provider x level:')\n"
        "display(success_by_provider_level(df))\n"
        "print('Success by spec x level:')\n"
        "display(success_by_spec_level(df))\n"
        "ft = fail_table(df)\n"
        "if ft is not None:\n"
        "    print('Failure modes (whole window):')\n"
        "    display(ft)\n"
        "else:\n"
        "    print('No failures in window.')"
    ))

    # Overall success-by-provider chart.
    cells.append(new_code_cell(
        "import numpy as np\n"
        "rate = lambda d, p: (d[d.provider == p].success.mean() * 100) if len(d[d.provider == p]) else 0\n"
        "x = np.arange(len(PROVIDERS)); w = 0.6\n"
        "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
        "bars = ax.bar(x, [rate(df, p) for p in PROVIDERS], w, color='#4a7')\n"
        "ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in PROVIDERS])\n"
        "ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 105)\n"
        "ax.set_title('Generation success by provider (whole window)')\n"
        "for b in bars:\n"
        "    ax.annotate(f'{b.get_height():.0f}%', (b.get_x()+b.get_width()/2, b.get_height()+1), ha='center', fontsize=9)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # Per-lab sections (data-driven: one section per lab topic present).
    cells.append(new_markdown_cell(
        "## Per-lab breakdown\n\n"
        "Each lab topic swept across every model × level × spec. The per-model "
        "grid shows successes/attempts at each level (L3/L4 fan out to three "
        "specs, so a cell of `2/3` means one spec failed)."
    ))
    cells.append(new_code_cell(
        "for lab in sorted(df.lab_name.dropna().unique()):\n"
        "    sub = df[df.lab_name == lab]\n"
        "    print('=' * 70)\n"
        "    print(lab)\n"
        "    print('=' * 70)\n"
        "    print('Stats:', slice_stats(sub))\n"
        "    print()\n"
        "    print('Success by provider x level:')\n"
        "    display(success_by_provider_level(sub))\n"
        "    print('Per-model grid:')\n"
        "    display(model_grid(sub))\n"
        "    ft = fail_table(sub)\n"
        "    if ft is not None:\n"
        "        print('Failure modes:')\n"
        "        display(ft)\n"
        "    else:\n"
        "        print('No failures.')\n"
        "    print()"
    ))

    # Data-driven takeaways.
    cells.append(new_markdown_cell(
        "## Takeaways\n\n"
        "Computed from the window below rather than narrated, so this stays true "
        "on re-run."
    ))
    cells.append(new_code_cell(
        "n = len(df); ok = int(df.success.sum())\n"
        "print(f'Overall: {ok}/{n} cells generated ({ok/n:.1%}).')\n"
        "fails = df[~df.success]\n"
        "if len(fails):\n"
        "    print(f'{len(fails)} failures, by mode:')\n"
        "    for mode, c in fails.fail_mode.value_counts().items():\n"
        "        print(f'  - {mode or \"(unclassified)\"}: {c}')\n"
        "    print('Failures by provider:',\n"
        "          fails.provider.value_counts().to_dict())\n"
        "    print('Failures by spec:',\n"
        "          fails.spec_type.value_counts().to_dict())\n"
        "    worst = (df.groupby('display_name').success.mean().sort_values().head(3))\n"
        "    print('Lowest success-rate models:',\n"
        "          {k: f'{v:.0%}' for k, v in worst.items()})\n"
        "else:\n"
        "    print('No failures in the window.')"
    ))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def parse_args(argv=None) -> argparse.Namespace:
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description=(
            "Build a dated, non-colliding benchmark success/failure summary for a "
            "run window. Defaults to today's runs so a fresh sweep is isolated "
            "from prior history and its outputs don't clobber earlier reports."
        )
    )
    parser.add_argument(
        "--since",
        default=today,
        metavar="YYYYMMDD[_HHMMSS]",
        help="Inclusive lower bound (default: today's 00:00). Accepts a date "
        "(YYYYMMDD) or an exact second (YYYYMMDD_HHMMSS). Use 00000000 to "
        "include all history.",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="YYYYMMDD[_HHMMSS]",
        help="Optional inclusive upper bound, YYYYMMDD or YYYYMMDD_HHMMSS "
        "(default: no upper bound).",
    )
    parser.add_argument(
        "--label",
        default=today,
        help="Suffix appended to output filenames (default: today). "
        "e.g. --label wave2 -> benchmark_summary_wave2.html.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runs(
        include_failures=True,
        since=args.since,
        until=args.until,
        dedupe_latest=True,
    )

    if df.empty:
        window = f"since {args.since}"
        if args.until:
            window += f" until {args.until}"
        print(
            f"No runs found in the window ({window}). Nothing to report; "
            "no files written."
        )
        return

    nb = build_notebook(df.to_json(orient="records"))

    ipynb_path = OUT_DIR / f"benchmark_summary_{args.label}.ipynb"
    from nbconvert.preprocessors import ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(OUT_DIR)}})
    nbformat.write(nb, ipynb_path)
    print(f"Wrote {ipynb_path.relative_to(ROOT)} ({len(df)} rows)")

    from nbconvert import HTMLExporter
    html, _ = HTMLExporter(exclude_input=False).from_notebook_node(nb)
    html_path = OUT_DIR / f"benchmark_summary_{args.label}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

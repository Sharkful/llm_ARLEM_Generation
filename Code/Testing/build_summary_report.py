"""
Build a shareable benchmark summary notebook (+ HTML) from suite_results files.

Collates the three formative sweeps into one report, one section per lab topic,
with a "fixes between Run 1 and Run 2" comparison. Data is embedded inline so the
notebook is self-contained (re-runnable without the source JSON).

Usage:
    python "Code/Testing/build_summary_report.py"
Outputs:
    Artifacts/Reports/benchmark_summary.ipynb   (executed)
    Artifacts/Reports/benchmark_summary.html
"""

import json
import os
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = ROOT / "Artifacts" / "Data" / "Benchmark"
OUT_DIR = BENCH / "Reports"

# (topic_key, pretty title, phase label, suite_results filename)
SWEEPS = [
    ("vsepr_molecular_geometry", "VSEPR Molecular Geometry",
     "Run 1 — pre-fix baseline", "suite_results_20260608_130019.json"),
    ("heart_anatomy_and_blood_flow", "Heart Anatomy & Blood Flow",
     "Run 2 — post-fix", "suite_results_20260608_182801.json"),
    ("apparent_retrograde_motion", "Apparent Retrograde Motion",
     "Run 3 — post-fix", "suite_results_20260608_214138.json"),
]


def fail_mode(err: str) -> str:
    if not err:
        return ""
    e = err.lower()
    if "incomplete" in e or "max_tokens length" in e:
        return "truncation (max_tokens)"
    # Check schema validation before the 404 signature: discriminated-union
    # failures contain "union_tag_not_found", whose "not_found" must NOT be read
    # as a model-not-found 404. Use the specific 404 signature, not a substring.
    if "validation error" in e:
        return "schema validation"
    if "not_found_error" in e or "error code: 404" in e:
        return "404 model-not-found"
    return "other"


def compact(rec: dict, topic: str, phase: str) -> dict:
    t = rec.get("tracking") or {}
    toks = (t.get("tokens") or {})
    lm = rec.get("lab_metrics") or {}
    return {
        "topic": topic,
        "phase": phase,
        "model": rec.get("model"),
        "display_name": rec.get("display_name", rec.get("model")),
        "provider": rec.get("provider"),
        "level": rec.get("level"),
        "success": bool(rec.get("success")),
        "total_tokens": toks.get("total", 0),
        "completion_tokens": toks.get("completion", 0),
        "cost_usd": (t.get("cost") or {}).get("total_usd", 0.0),
        "wall_s": rec.get("wall_time_seconds", 0.0),
        "retries": (t.get("retries") or {}).get("total", 0),
        "num_objects": lm.get("num_objects", 0),
        "num_clips": lm.get("num_clips", 0),
        "fail_mode": fail_mode(rec.get("error")),
    }


def collect_records() -> list[dict]:
    records = []
    for topic, _title, phase, fname in SWEEPS:
        data = json.loads((BENCH / fname).read_text(encoding="utf-8"))
        for rec in data:
            records.append(compact(rec, topic, phase))
    return records


# ── Notebook assembly ────────────────────────────────────────────────

def build_notebook(records: list[dict]) -> nbformat.NotebookNode:
    nb = new_notebook()
    cells = []

    cells.append(new_markdown_cell(
        "# AR Lab Generation — Formative Benchmark Summary\n\n"
        "LLM-generated ARLEM lab specifications, benchmarked across providers, "
        "models, and prompt-specificity levels (L1–L4) on three lab topics.\n\n"
        "Each topic was swept across **all registered models × L1–L4** "
        "(`--all-models --all-levels`), generating the full `Lab` JSON schema "
        "(`multi-module` structure). The first topic (VSEPR) was run **before** a "
        "round of schema/runtime fixes; the latter two were run **after**. "
        "This report keeps each topic separate and shows the effect of those fixes."
    ))

    cells.append(new_markdown_cell(
        "## Fixes applied between Run 1 (VSEPR) and Run 2 (Heart)\n\n"
        "Run 1 surfaced three systematic failure modes. The fixes:\n\n"
        "**1. Component discriminator — dual-field scheme** "
        "(`Code/Schemas/json_lab.py`). Models emitted `{\"type\": "
        "\"TextMeshProComponent\"}` while the schema discriminated on "
        "`componentType: \"textMeshPro\"` — a field-name *and* value mismatch that "
        "caused thousands of `union_tag_not_found` errors. Components now carry two "
        "fields: an LLM-facing `type` (literal = class name, used as the "
        "discriminator, `exclude=True` so it is stripped on serialization) and a "
        "hidden `componentType` (`SkipJsonSchema`, the canonical Unity identifier "
        "the headset expects). The model reads/writes the naming it defaults to; the "
        "saved JSON keeps the headset contract.\n\n"
        "**2. `ObjectChange.target` alias.** Models emitted `name` instead of the "
        "required `target`; `target` now accepts `name` as a validation alias.\n\n"
        "**3. Anthropic completion truncation.** A hardcoded `max_tokens=8192` "
        "truncated every full L2–L4 lab (`IncompleteOutputException`). The Anthropic "
        "client is now built with an explicit timeout (disabling the SDK's "
        "non-streaming guard) and `max_tokens` raised to 32000 — no streaming "
        "rework, so token/cost tracking is unchanged.\n\n"
        "**4. Removed deprecated `claude-3-haiku`** (returned 404 on every run).\n\n"
        "The chart below shows success rate by provider, Run 1 vs Runs 2–3."
    ))

    # Data + helpers cell (embeds records inline for a self-contained notebook)
    data_literal = json.dumps(records)
    cells.append(new_code_cell(
        "%matplotlib inline\n"
        "import json\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "pd.set_option('display.max_rows', 200)\n\n"
        "# Compact per-run records embedded inline — notebook is self-contained.\n"
        f"RECORDS = json.loads(r'''{data_literal}''')\n"
        "df = pd.DataFrame(RECORDS)\n\n"
        "LEVELS = ['L1', 'L2', 'L3', 'L4']\n"
        "PROVIDERS = ['openai', 'anthropic', 'google']\n\n"
        "def success_by_provider_level(sub):\n"
        "    rows = []\n"
        "    for p in PROVIDERS:\n"
        "        d = sub[sub.provider == p]\n"
        "        row = {'provider': p}\n"
        "        for lv in LEVELS:\n"
        "            dl = d[d.level == lv]\n"
        "            row[lv] = f\"{int(dl.success.sum())}/{len(dl)}\" if len(dl) else '-'\n"
        "        row['total'] = f\"{int(d.success.sum())}/{len(d)}\"\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('provider')\n\n"
        "def model_grid(sub):\n"
        "    rows = []\n"
        "    for m in sub.display_name.unique():\n"
        "        d = sub[sub.display_name == m]\n"
        "        row = {'model': m}\n"
        "        for lv in LEVELS:\n"
        "            dl = d[d.level == lv]\n"
        "            row[lv] = ('OK' if bool(dl.success.iloc[0]) else 'X') if len(dl) else '-'\n"
        "        row['ok'] = f\"{int(d.success.sum())}/{len(d)}\"\n"
        "        fm = [x for x in d[~d.success].fail_mode.tolist() if x]\n"
        "        row['fail mode'] = max(set(fm), key=fm.count) if fm else ''\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows).set_index('model')\n\n"
        "def topic_stats(sub):\n"
        "    ok = int(sub.success.sum()); n = len(sub)\n"
        "    succ = sub[sub.success & (sub.level != 'L1')]\n"
        "    return {\n"
        "        'success': f\"{ok}/{n} ({ok/n:.0%})\",\n"
        "        'wall_min': round(float(sub.wall_s.sum())/60, 1),\n"
        "        'cost_usd': round(float(sub.cost_usd.sum()), 2),\n"
        "        'avg_objects_L2-4': round(float(succ.num_objects.mean()), 1) if len(succ) else 0,\n"
        "        'avg_clips_L2-4': round(float(succ.num_clips.mean()), 1) if len(succ) else 0,\n"
        "    }\n"
    ))

    # Overall before/after chart
    cells.append(new_code_cell(
        "before = df[df.phase.str.startswith('Run 1')]\n"
        "after = df[~df.phase.str.startswith('Run 1')]\n"
        "rate = lambda d, p: (d[d.provider == p].success.mean() * 100) if len(d[d.provider == p]) else 0\n"
        "import numpy as np\n"
        "x = np.arange(len(PROVIDERS)); w = 0.38\n"
        "fig, ax = plt.subplots(figsize=(8, 4.2))\n"
        "b1 = ax.bar(x - w/2, [rate(before, p) for p in PROVIDERS], w, label='Run 1 (VSEPR, pre-fix)', color='#c44')\n"
        "b2 = ax.bar(x + w/2, [rate(after, p) for p in PROVIDERS], w, label='Runs 2-3 (post-fix)', color='#4a7')\n"
        "ax.set_xticks(x); ax.set_xticklabels([p.capitalize() for p in PROVIDERS])\n"
        "ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 105)\n"
        "ax.set_title('Generation success by provider — before vs after fixes')\n"
        "ax.legend()\n"
        "for b in list(b1) + list(b2):\n"
        "    ax.annotate(f'{b.get_height():.0f}%', (b.get_x()+b.get_width()/2, b.get_height()+1), ha='center', fontsize=9)\n"
        "plt.tight_layout(); plt.show()"
    ))

    # Per-topic sections
    for topic, title, phase, _fname in SWEEPS:
        cells.append(new_markdown_cell(f"## {title}\n\n*{phase}*"))
        cells.append(new_code_cell(
            f"sub = df[df.topic == {topic!r}]\n"
            "print('Topic stats:', topic_stats(sub))\n"
            "print()\n"
            "print('Success by provider x level:')\n"
            "display(success_by_provider_level(sub))\n"
            "print('Per-model grid:')\n"
            "display(model_grid(sub))\n"
            "fm = sub[~sub.success].fail_mode\n"
            "if len(fm):\n"
            "    print('Failure modes:')\n"
            "    display(fm.value_counts().rename_axis('mode').to_frame('failed runs'))\n"
            "else:\n"
            "    print('No failures.')"
        ))

    cells.append(new_markdown_cell(
        "## Takeaways\n\n"
        "- The discriminator dual-field scheme moved OpenAI from a near-total L2–L4 "
        "failure to near-complete success without changing the saved JSON's headset "
        "contract.\n"
        "- The Anthropic timeout/`max_tokens` fix eliminated the truncation failures; "
        "full L4 labs now generate.\n"
        "- Gemini was already union-free and unaffected (perfect throughout).\n"
        "- Residual failures concentrate in the weakest small model (gpt-5-mini), "
        "which hallucinates field names / component types — a model-capability floor, "
        "not a schema issue."
    ))

    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    nb = build_notebook(records)

    ipynb_path = OUT_DIR / "benchmark_summary.ipynb"

    # Execute so outputs (tables, chart) are embedded for sharing.
    from nbconvert.preprocessors import ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(OUT_DIR)}})
    nbformat.write(nb, ipynb_path)
    print(f"Wrote {ipynb_path.relative_to(ROOT)}")

    # HTML export (reliable; print-to-PDF from a browser if a PDF is needed).
    from nbconvert import HTMLExporter
    html, _ = HTMLExporter(exclude_input=False).from_notebook_node(nb)
    html_path = OUT_DIR / "benchmark_summary.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

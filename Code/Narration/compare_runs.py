"""
Compare narration runs of the same labs across spoken-form modes (#94).

    python "Code/Narration/compare_runs.py" <run folders, or a run-group folder> [--out FILE.md]

Reads each run's ``narration_manifest.json`` (text-only runs included) and writes:

- ``spoken_form_comparison.md``: per mode, the added cost and time of the spoken-form
  pass, clips changed and flagged, and audio cost where there is audio; then, per lab,
  every clip whose spoken text differs between modes, with each mode's text and what
  it changed in bold. That list is what to listen to.
- ``spoken_form_comparison.csv``: one row per run, for the write-up.

Both go next to the runs (the folders' common parent) unless ``--out`` says otherwise.
No requests are sent.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

MANIFEST = "narration_manifest.json"


def find_manifests(paths: list[Path]) -> list[Path]:
    found = []
    for p in paths:
        if (p / MANIFEST).is_file():
            found.append(p / MANIFEST)
        elif p.is_dir():
            found += sorted(p.rglob(MANIFEST))
    return sorted(set(found))


def mode_label(m: dict) -> str:
    sf = m.get("spoken_form") or {}
    return f"llm {sf.get('model')}" if sf.get("mode") == "llm" else (sf.get("mode") or "none")


def _sum(rows, key):
    return sum(r.get(key) or 0 for r in rows)


def run_row(path: Path, m: dict) -> dict:
    clips = [c for c in m.get("clips", []) if c.get("spoken_text")]
    calls = m.get("calls") or []
    sf_calls = [c for c in calls if c.get("step") == "spoken_form"]
    tts_calls = [c for c in calls if c.get("step") == "tts"]
    runs = m.get("runs") or []
    # Time the pass took where it did its work: the LLM requests, or the longest run of
    # the rules (a later run reuses cached text and takes ~0 s).
    sf_seconds = (_sum(sf_calls, "seconds") if sf_calls
                  else max((r.get("spoken_form_seconds") or 0 for r in runs), default=0))
    totals = m.get("totals") or {}
    source = m.get("source") or {}
    return {
        "run": path.parent.name,
        "lab": source.get("topic") or source.get("labId") or "?",
        "input": source.get("input_path"),
        "mode": mode_label(m),
        "status": m.get("status"),
        "clips": len(clips),
        "changed": sum(1 for c in clips if c["spoken_text"] != (c.get("display_text") or "").strip()),
        "flagged": sum(1 for c in clips if (c.get("spoken_form") or {}).get("flags")),
        "spoken_form_requests": len(sf_calls),
        "spoken_form_failed": sum(1 for c in sf_calls if c.get("status") != "ok"),
        "spoken_form_cost_usd": _sum(sf_calls, "cost_usd"),
        "spoken_form_seconds": round(sf_seconds, 3),
        "glossary_version": (m.get("spoken_form") or {}).get("glossary_version"),
        "audio_seconds": totals.get("audio_seconds") or 0,
        "tts_requests": len(tts_calls),
        "tts_cost_usd": _sum(tts_calls, "cost_usd"),
        "total_cost_usd": totals.get("total_cost_usd") or 0,
    }


def bold_changes(written: str, spoken: str) -> str:
    """The spoken text with the parts that differ from the written text in bold."""
    a, b = written.split(), spoken.split()
    out = []
    for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        words = " ".join(b[j1:j2])
        if words:
            out.append(words if op == "equal" else f"**{words}**")
    return " ".join(out)


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build(manifests: list[Path]) -> tuple[list[dict], str]:
    rows, by_lab = [], defaultdict(dict)   # input -> mode -> manifest
    for path in manifests:
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row = run_row(path, m)
        rows.append(row)
        by_lab[row["input"]][row["mode"]] = (row, m)

    modes = sorted({r["mode"] for r in rows}, key=lambda x: (x != "none", x != "rules", x))
    out = ["# Spoken-form comparison", "",
           f"{len(rows)} runs, {len(by_lab)} labs. Spoken-form cost and time are what the pass "
           "adds to narrating a lab; audio columns fill in once a run has audio.", "",
           "## By mode", "",
           "| Mode | Labs | Clips changed | Flagged | Spoken-form $ / lab | Spoken-form s / lab "
           "| Failed requests | Audio $ / lab |",
           "|---|---|---|---|---|---|---|---|"]
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        n = len(mr)
        audio = [r for r in mr if r["tts_requests"]]
        audio_cost = f"${_sum(audio, 'tts_cost_usd') / len(audio):.4f}" if audio else "–"
        out.append(
            f"| {mode} | {n} | {_sum(mr, 'changed')} of {_sum(mr, 'clips')} | {_sum(mr, 'flagged')} "
            f"| ${_sum(mr, 'spoken_form_cost_usd') / n:.4f} | {_sum(mr, 'spoken_form_seconds') / n:.1f} "
            f"| {_sum(mr, 'spoken_form_failed')} | {audio_cost} |")
    out += ["", "## By lab", "", "| Lab | Mode | Changed | Flagged | Spoken-form $ | Spoken-form s | Run |",
            "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["lab"], modes.index(r["mode"]))):
        out.append(f"| {r['lab']} | {r['mode']} | {r['changed']} of {r['clips']} | {r['flagged']} "
                   f"| ${r['spoken_form_cost_usd']:.4f} | {r['spoken_form_seconds']:.1f} | `{r['run']}` |")

    out += ["", "## Clips whose spoken text differs between modes", "",
            "Bold marks what each mode changed from the written narration. These are the clips "
            "to listen to.", ""]
    for inp, per_mode in sorted(by_lab.items(), key=lambda kv: kv[1][next(iter(kv[1]))][0]["lab"]):
        lab_modes = [md for md in modes if md in per_mode]
        texts = {md: {c["audio_name"]: c for c in per_mode[md][1].get("clips", []) if c.get("spoken_text")}
                 for md in lab_modes}
        names = []
        for md in lab_modes:
            names += [n for n in texts[md] if n not in names]
        differing = [n for n in names if len({texts[md].get(n, {}).get("spoken_text") for md in lab_modes}) > 1]
        first = per_mode[lab_modes[0]][0]
        out += [f"### {first['lab']}", "", f"`{inp}`: {len(differing)} of {len(names)} clips differ.", ""]
        for n in differing:
            written = next((texts[md][n].get("display_text") or "" for md in lab_modes if n in texts[md]), "").strip()
            out += [f"**`{n}`**", "", f"- *Written:* {_md(written)}"]
            for md in lab_modes:
                c = texts[md].get(n)
                if not c:
                    continue
                flags = (c.get("spoken_form") or {}).get("flags") or []
                audio = " 🔊" if c.get("status") in ("generated", "skipped") else ""
                out.append(f"- *{md}*{audio}: {_md(bold_changes(written, c['spoken_text']))}"
                           + (f"  \n  ⚑ {_md('; '.join(flags))}" if flags else ""))
            out.append("")
    return rows, "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare narration runs across spoken-form modes (#94).")
    ap.add_argument("paths", nargs="+", type=Path, help="run folders, or folders that contain them")
    ap.add_argument("--out", type=Path, help="the Markdown report (the CSV goes beside it)")
    args = ap.parse_args(argv)
    manifests = find_manifests(args.paths)
    if not manifests:
        print(f"error: no {MANIFEST} under {', '.join(map(str, args.paths))}", file=sys.stderr)
        return 2
    rows, report = build(manifests)
    out = args.out or Path(os.path.commonpath([str(p.parent.parent) for p in manifests])) / "spoken_form_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    print(report.split("## By lab")[0].strip())
    print(f"\nWrote {out} and {out.with_suffix('.csv').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

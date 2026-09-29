"""
Quarantine benchmark failure records caused by infrastructure, not model behavior.

Failed runs survive only inside ``Artifacts/Data/Benchmark/suite_results_*.json``
arrays (mixed with the successes from the same chunk). When a failure was caused by
something outside what the benchmark evaluates — a billing-quota 429, a DNS drop, a
provider 503 burst, a misregistered model id 404 — the record must not count against
the model in failure-rate analyses, and (for the 404s) can never be superseded by
``load_runs(dedupe_latest=True)`` because the corrected model id changes the dedup
key (issues #42, #45).

This tool moves those records out of the loaders' sight while preserving them as
evidence (the suite files are untracked and ``Artifacts/Data/Errors/`` is
git-ignored, so deletion would be unrecoverable):

    Artifacts/Data/Benchmark/quarantine/<suite file name>   the extracted records
    Artifacts/Data/Benchmark/quarantine/errors/             their _errors.txt files

Source suite files are rewritten without the quarantined records (and dropped
entirely if emptied); successes and genuine failures are untouched. A record is
quarantined when ANY retry attempt died on an infra signature — one 503 mid-run
already contaminates the attempt budget, so the cell should be re-run either way.
Retry-death 400s (instructor replaying a malformed turn, issue #44) and Pydantic
validation failures are deliberately NOT matched: those are model/plumbing signal.

The default is a dry run that lists every record that would move; nothing is
written without ``--apply``. The run is idempotent — once moved, records are gone
from the sources and a re-run finds nothing.

Usage (from the repo root):
    python "Code/Analysis/quarantine_infra_failures.py" --since 20260705_182200
    python "Code/Analysis/quarantine_infra_failures.py" --since 20260705_182200 --apply

``--since`` / ``--until`` take the same YYYYMMDD[_HHMMSS] bounds as the report CLI.
Unbounded invocations scan all history — check the dry run carefully before
applying, since older sweeps' reports were built with their failures in place.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from benchmark_dataframe import _normalize_bound, _normalize_ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Per-attempt exception bodies inside an InstructorRetryException dump.
EXCEPTION_BLOCK = re.compile(r"<exception>\s*(.*?)\s*</exception>", re.DOTALL)


def infra_kind(exc_text: str) -> str | None:
    """Classify one attempt's exception text as an infra failure, or None.

    Signatures are deliberately narrow. In particular there is no bare
    "not found" / "429" match: retry-death 400s (#44) and schema-validation
    errors (whose text can mention arbitrary lab content) must stay unmatched.
    """
    e = " ".join(exc_text.split()).lower()
    if "exceeded your current quota" in e:
        return "quota-429"
    if "connection error" in e:
        return "connection"
    if "is not found for api version" in e:
        return "bad-model-404"
    if "503" in e and ("unavailable" in e or "high demand" in e):
        return "overloaded-503"
    if "overloaded_error" in e:
        return "overloaded-529"
    if "resource_exhausted" in e or ("429" in e and "rate limit" in e):
        return "rate-limit-429"
    return None


def record_infra_kinds(record: dict) -> list[str]:
    """Distinct infra kinds across a failed record's attempts (empty = genuine).

    Falls back to classifying the whole error string when there are no
    ``<exception>`` blocks (e.g. a FATAL ERROR record from the suite runner).
    """
    err = str(record.get("error") or "")
    blocks = EXCEPTION_BLOCK.findall(err) or [err]
    kinds: list[str] = []
    for block in blocks:
        kind = infra_kind(block)
        if kind and kind not in kinds:
            kinds.append(kind)
    return kinds


def partition_suite(
    records: list[dict], since_key: str | None, until_key: str | None
) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """Split one suite array into (records to keep, records to quarantine)."""
    keep: list[dict] = []
    quarantined: list[tuple[dict, list[str]]] = []
    for rec in records:
        if rec.get("success"):
            keep.append(rec)
            continue
        # Same window semantics as load_runs: a record with no timestamp can't
        # be placed, so a --since bound leaves it alone (conservative: keep).
        ts = _normalize_ts(rec.get("timestamp"))
        if since_key is not None and ts < since_key:
            keep.append(rec)
            continue
        if until_key is not None and ts > until_key:
            keep.append(rec)
            continue
        kinds = record_infra_kinds(rec)
        if kinds:
            quarantined.append((rec, kinds))
        else:
            keep.append(rec)
    return keep, quarantined


def _dump(records: list[dict]) -> str:
    # Match benchmark.py's suite serialization byte-for-byte-ish (indent=2).
    return json.dumps(records, indent=2, default=str)


def quarantine(
    root: Path, since: str | None, until: str | None, apply: bool
) -> int:
    since_key = _normalize_bound(since, is_upper=False) if since else None
    until_key = _normalize_bound(until, is_upper=True) if until else None

    bench_dir = root / "Artifacts" / "Data" / "Benchmark"
    q_dir = bench_dir / "quarantine"
    q_err_dir = q_dir / "errors"

    suite_paths = sorted(bench_dir.glob("suite_results_*.json"))
    if not suite_paths:
        print(f"No suite_results_*.json under {bench_dir} - nothing to do.")
        return 0

    by_kind: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    total_moved = 0
    errors_moved = 0
    errors_missing = 0

    for path in suite_paths:
        keep, quarantined = partition_suite(
            json.loads(path.read_text(encoding="utf-8")), since_key, until_key
        )
        if not quarantined:
            continue

        print(f"\n{path.name}  ({len(quarantined)} of {len(keep) + len(quarantined)} records):")
        for rec, kinds in quarantined:
            print(
                f"  {rec.get('model', '?'):26s} {str(rec.get('level')):4s} "
                f"{str(rec.get('spec_type')):13s} {str(rec.get('lab_name')):32s} "
                f"[{', '.join(kinds)}]"
            )
            for kind in kinds:
                by_kind[kind] += 1
            by_model[str(rec.get("model", "?"))] += 1
        total_moved += len(quarantined)

        if not apply:
            continue

        # Extracted records accumulate in a same-named file under quarantine/
        # (merged if a previous invocation already created it).
        q_dir.mkdir(parents=True, exist_ok=True)
        q_path = q_dir / path.name
        held = json.loads(q_path.read_text(encoding="utf-8")) if q_path.exists() else []
        held.extend(rec for rec, _ in quarantined)
        q_path.write_text(_dump(held), encoding="utf-8")

        if keep:
            path.write_text(_dump(keep), encoding="utf-8")
        else:
            path.unlink()  # fully-infra chunk: everything now lives in quarantine/

        for rec, _ in quarantined:
            ef = rec.get("errors_file")
            if not ef:
                continue
            src = root / Path(str(ef).replace("\\", "/"))
            if src.exists():
                q_err_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(q_err_dir / src.name))
                errors_moved += 1
            else:
                errors_missing += 1

    print(f"\n{'=' * 70}")
    if total_moved == 0:
        print("No infra-failure records found in the window.")
        return 0

    window = f"[{since or 'beginning'} .. {until or 'now'}]"
    print(f"{total_moved} infra-failure record(s) in window {window}")
    print("  by signature: " + ", ".join(f"{k}={n}" for k, n in by_kind.most_common()))
    print("  by model:     " + ", ".join(f"{m}={n}" for m, n in by_model.most_common()))
    if apply:
        print(f"Moved to {q_dir}")
        print(f"  errors files moved: {errors_moved}"
              + (f" (missing: {errors_missing})" if errors_missing else ""))
    else:
        print("DRY RUN - nothing was moved. Re-run with --apply to execute.")
    return total_moved


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move infra-caused failure records out of suite_results_*.json "
        "into Artifacts/Data/Benchmark/quarantine/ (dry run by default)."
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYYMMDD[_HHMMSS]",
        help="Inclusive lower bound on record timestamps (default: no bound). "
        "For the 2026-07 formative sweep use 20260705_182200.",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="YYYYMMDD[_HHMMSS]",
        help="Inclusive upper bound (default: no bound).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move records/files. Without this flag, only report.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        metavar="DIR",
        help="Repo root containing Artifacts/ (default: resolved from this file; "
        "override to rehearse against a copy).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    quarantine(args.root, args.since, args.until, args.apply)

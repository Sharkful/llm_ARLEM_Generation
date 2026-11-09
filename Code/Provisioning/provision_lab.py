"""
Provision a lab's models and textures (spec: PROVISIONING.md). So far: the plan
(#106). ``--dry-run`` audits the lab, looks every asset up in the library by name,
and makes one LLM call that describes and routes the rest; it downloads and builds
nothing. The full run (#113) comes later.

    python "Code/Provisioning/provision_lab.py" <lab>_output.json --dry-run
        [--model KEY] [--library PATH] [--metrics PATH] [--run-group NAME]
    python "Code/Provisioning/provision_lab.py" <lab>_output.json --dry-run --no-llm

``--no-llm`` stops after the audit and the library lookup (no API key needed).
The plan is saved as ``provisioning_manifest.json`` in
``Artifacts/Data/Provisioning/[<run group>/]<lab>_<timestamp>/``.

Exit codes: 0 planned (or audited, with --no-llm); 1 the describe-and-route call
failed; 2 bad input or environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "Artifacts" / "Data" / "Provisioning"
_RUN_GROUP = re.compile(r"^[A-Za-z0-9_-]+$")

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit  # noqa: E402
import library  # noqa: E402
import routing  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    packages = {}
    for pkg in ("instructor", "pydantic", "anthropic", "openai", "google-genai"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(status) if status is not None else None,
            "packages": packages, "python": sys.version.split()[0]}


class Known:
    """What's already available: the library, or, without one, the seed list's
    built-ins (so a machine with no library still knows what the headset has)."""

    def __init__(self, library_path: Optional[str]):
        path = library.resolve_library_path(library_path)
        try:
            self.store: Optional[library.LocalFolderStore] = library.LocalFolderStore(path)
            self.identity = self.store.identity()
            self.note = None
            self._entries = None
        except library.LibraryError:
            self.store = None
            self.identity = {"store": None, "location": str(path)}
            self.note = (f"no asset library at {path}: only the seed list's built-ins are known "
                         f"(build one with library.py seed)")
            seed = library.load_seed()
            self._entries = [library.LibraryEntry(
                asset_id=e.asset_id, kind=e.kind, name=e.name, builtin=True,
                reusable=e.reusable, description=e.description) for e in seed.entries if e.builtin]

    def find(self, name: str, kind: str) -> Optional[library.LibraryEntry]:
        if self.store is not None:
            e = self.store.find(name)
        else:
            k = library.name_key(name)
            e = next((x for x in self._entries if library.name_key(x.name) == k), None)
        return e if e is not None and e.kind == kind and e.reusable else None


def _asset_record(r: audit.AssetRequest) -> dict:
    rec = r.to_json()
    return {k: rec[k] for k in ("name", "kind", "spellings", "use_count", "modules", "uses",
                                "narration", "purpose")}


def build_manifest(input_path: Path, lab: dict, metrics: Optional[dict], known: Known,
                   requests: list[audit.AssetRequest], plan, route_record: Optional[dict],
                   *, no_llm: bool) -> dict:
    planned = {library.name_key(p.name): p for p in (plan.assets if plan else [])}
    assets = []
    for r in requests:
        rec = _asset_record(r)
        hit = known.find(r.name, r.kind)
        if hit is not None:
            rec["status"] = "available"
            rec["library"] = {"asset_id": hit.asset_id, "builtin": hit.builtin,
                              "matched": "name" if library.name_key(hit.name) == r.key else "alias"}
        elif r.key in planned:
            p = planned[r.key]
            rec["plan"] = p.model_dump(exclude={"name", "kind"}, exclude_none=True)
            rec["status"] = "merged" if p.same_as else "planned"
        else:
            rec["status"] = "unplanned"
        assets.append(rec)
    statuses = Counter(a["status"] for a in assets)
    routes = Counter(a["plan"]["route"] for a in assets if a["status"] == "planned")
    forms = Counter(a["plan"]["form"] for a in assets if "plan" in a)
    if no_llm:
        status = "audited"
    elif route_record and route_record.get("error"):
        status = "failed"
    else:
        status = "planned"
    return {
        "tool": "provision_lab",
        "stage": "plan (#106)",
        "status": status,
        "source": {
            "input": _rel(input_path),
            "labId": lab.get("labId"),
            "metrics": _rel(Path(metrics["_path"])) if metrics else None,
            "source_model": (metrics or {}).get("model"),
            "level": (metrics or {}).get("level"),
            "lab_name": (metrics or {}).get("lab_name"),
            "hierarchical": bool((metrics or {}).get("registry")),
        },
        "library": known.identity,
        "library_note": known.note,
        "routing": route_record,
        "routes_offered": {k: list(routing.routes_for(k)) for k in routing.KINDS},
        "counts": {
            "assets": len(assets),
            "models": sum(a["kind"] == "model" for a in assets),
            "textures": sum(a["kind"] == "texture" for a in assets),
            "by_status": dict(sorted(statuses.items())),
            "by_route": dict(sorted(routes.items())),
            "by_form": dict(sorted(forms.items())),
        },
        "assets": assets,
        "provenance": provenance(),
        "created": _now(),
    }


def _print_plan(manifest: dict) -> None:
    for a in manifest["assets"]:
        if a["status"] == "available":
            what = f"available ({a['library']['asset_id']}" + \
                   (", built-in)" if a["library"]["builtin"] else ")")
        elif a["status"] in ("planned", "merged"):
            p = a["plan"]
            what = (f"same as {p['same_as']}" if a["status"] == "merged"
                    else f"{p['route']:18} {p['form']:18} {p['description'][:70]}")
        else:
            what = a["status"]
        print(f"  {a['kind']:7} {a['name'][:32]:32} {what}")
    c = manifest["counts"]
    print(f"{c['assets']} assets ({c['models']} models, {c['textures']} textures): "
          + ", ".join(f"{n} {s}" for s, n in c["by_status"].items()))
    if c["by_route"]:
        print("routes: " + ", ".join(f"{n} {r}" for r, n in c["by_route"].items()))
    r = manifest.get("routing") or {}
    if r.get("calls"):
        cost = f"${r['cost_usd']:.4f}, " if "cost_usd" in r else ""
        print(f"describe and route: {r['model']}, {cost}{r['seconds']} s, {r['calls']} call(s)"
              + (f", {r['transient_retries']} transient retries" if r.get("transient_retries") else ""))


def run_dir_for(input_path: Path, run_group: Optional[str], stamp: str) -> Path:
    stem = input_path.name
    for suffix in ("_output.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "lab"
    base = OUTPUT_ROOT / run_group if run_group else OUTPUT_ROOT
    return base / f"{stem}_{stamp}"


def main(argv: Optional[list[str]] = None, *, call_fn: Optional[routing.CallFn] = None,
         output_root: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description="Provision a lab's models and textures (so far: "
                                             "the plan, with --dry-run).")
    ap.add_argument("lab", help="a json_lab *_output.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: audit, library lookup, describe and route (required for now)")
    ap.add_argument("--no-llm", action="store_true",
                    help="stop after the audit and library lookup (no API key needed)")
    ap.add_argument("--model", default=routing.DEFAULT_MODEL,
                    help=f"benchmark_config.MODELS key for describe and route "
                         f"(default {routing.DEFAULT_MODEL})")
    ap.add_argument("--library", help=f"library folder (default: ${library.LIBRARY_ENV}, else "
                                      f"{_rel(library.DEFAULT_LIBRARY)})")
    ap.add_argument("--metrics", help="the lab's metrics file (default: the paired *_metrics.json)")
    ap.add_argument("--run-group", help="save under Artifacts/Data/Provisioning/<NAME>/")
    args = ap.parse_args(argv)

    def fail(msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr)
        return 2

    if not args.dry_run:
        return fail("only --dry-run is available so far; the full run comes with #113")
    if args.run_group and not _RUN_GROUP.match(args.run_group):
        return fail("--run-group may use only letters, digits, '_' and '-'")
    input_path = Path(args.lab)
    try:
        lab = audit.load_lab(input_path)
        if args.metrics:
            metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
            metrics["_path"] = str(args.metrics)
        else:
            metrics = audit.paired_metrics(input_path)
        known = Known(args.library)
    except (audit.InputError, library.LibraryError, OSError, ValueError) as exc:
        return fail(str(exc))
    if known.note:
        print(f"note: {known.note}")

    requests = audit.audit_lab(lab, audit.purposes_from_metrics(metrics))
    to_plan = [r for r in requests if known.find(r.name, r.kind) is None]
    plan, record = None, None
    if not args.no_llm:
        if call_fn is None:
            try:
                routing.check_environment(args.model)
            except KeyError as exc:
                return fail(exc.args[0])
            except SystemExit as exc:  # benchmark.require_environment prints its own message
                return fail(f"environment check failed ({exc.code})")
        router = routing.Router(args.model, call_fn=call_fn)
        print(f"planning {len(to_plan)} of {len(requests)} assets with {args.model}...")
        plan, record = router.plan(lab, to_plan)
        if record.get("served_models") and router.model_config is not None:
            routing._benchmark().warn_served_model(router.model_config.model_id,
                                                   record["served_models"])

    manifest = build_manifest(input_path, lab, metrics, known, requests, plan, record,
                              no_llm=args.no_llm)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_dir_for(input_path, args.run_group, stamp)
    if output_root is not None:
        run_dir = Path(output_root) / run_dir.relative_to(OUTPUT_ROOT)
    base, n = run_dir, 1
    while run_dir.exists():  # two runs in the same second never share a folder
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "provisioning_manifest.json"
    library.write_json_atomic(manifest_path, manifest)

    _print_plan(manifest)
    print(f"manifest: {_rel(manifest_path)}")
    if manifest["status"] == "failed":
        print(f"error: describe and route failed: {record['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

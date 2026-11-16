"""
Provision a lab's models and textures (spec: PROVISIONING.md). So far: the plan
(#106, #108). ``--dry-run`` audits the lab, looks every asset up in the library by
name or alias, asks the reuse judge (one LLM call) which of the rest the library
already has by meaning, and makes one more call that describes and routes what is
left. It downloads and builds nothing. The full run (#113) comes later.

    python "Code/Provisioning/provision_lab.py" <lab>_output.json --dry-run
        [--model KEY] [--library PATH] [--metrics PATH] [--run-group NAME]
        [--no-reuse] [--save-aliases]
    python "Code/Provisioning/provision_lab.py" <lab>_output.json --dry-run --no-llm

``--no-llm`` stops after the audit and the library lookup (no API key needed).
``--no-reuse`` skips the reuse judge. ``--save-aliases`` saves the judge's
single-entry matches (not built-ins) as library aliases, so the next run finds them by name.
The plan is saved as ``provisioning_manifest.json`` in
``Artifacts/Data/Provisioning/[<run group>/]<lab>_<timestamp>/``.

Exit codes: 0 planned (or audited, with --no-llm); 1 the reuse judge or the
describe-and-route call failed; 2 bad input or environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
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
import llm  # noqa: E402
import reuse  # noqa: E402
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
        """Opens the library. A library that exists but can't be read, or a path chosen
        by ``--library`` or ``$ARLEM_ASSET_LIBRARY`` with no library, raises
        ``LibraryError``: only the default location falls back to the seed list when
        empty."""
        path = library.resolve_library_path(library_path)
        env = os.environ.get(library.LIBRARY_ENV)
        chosen = (f"--library {library_path}" if library_path
                  else f"${library.LIBRARY_ENV}={env}" if env else None)
        if chosen and not (path / library.INDEX_NAME).is_file():
            raise library.LibraryError(f"{chosen}: no asset library there "
                                       f"(build one with library.py seed --library {path})")
        if (path / library.INDEX_NAME).is_file():
            self.store: Optional[library.LocalFolderStore] = library.LocalFolderStore(path)
            self.identity = self.store.identity()
            self.note = None
            self._entries = None
        else:
            self.store = None
            self.identity = {"store": None, "location": str(path)}
            self.note = (f"no asset library at {path}: only the seed list's built-ins are known "
                         f"(build one with library.py seed)")
            seed = library.load_seed()
            self._entries = [library.LibraryEntry(
                asset_id=e.asset_id, kind=e.kind, name=e.name, builtin=True,
                reusable=e.reusable, description=e.description) for e in seed.entries if e.builtin]

    def entries(self) -> list[library.LibraryEntry]:
        """Every known entry (the reuse judge shortlists from these)."""
        return self.store.entries() if self.store is not None else list(self._entries)

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
                                "components", "narration", "purpose")}


def _reuse_record(d) -> dict:
    """A reuse decision for the manifest, without empty fields."""
    return {k: v for k, v in d.model_dump(exclude={"name", "kind"}).items() if v}


def _matched(d) -> str:
    """What a reuse decision matched the asset to, e.g. "sphere + 2k_mars"."""
    base = d.asset_id or d.shape
    return base + (f" + {d.texture_id}" if d.texture_id else "")


def alias_proposals(reuse_plan, entries: list[library.LibraryEntry]) -> list[dict]:
    """The aliases ``--save-aliases`` would save: accepted reuses of a single entry (no
    texture) that is not built-in. A built-in is a generic moon-lab prefab (a sphere, a
    text label) that stands in for many different things, so a lab's name for its stand-in
    is not a name of the entry: aliasing ``marsPrefab`` to ``tiny_sphere`` would keep every
    later Mars a plain sphere, with no call, even once a Mars map is in the library."""
    builtin = {e.asset_id for e in entries if e.builtin}
    return [{"name": d.name, "asset_id": d.asset_id}
            for d in (reuse_plan.decisions if reuse_plan else [])
            if d.decision == "reuse" and not d.texture_id and d.asset_id not in builtin]


def save_aliases(store: library.LocalFolderStore, proposals: list[dict]) -> list[dict]:
    """Add each proposed alias (``alias_proposals``) to its entry, so the next lab with
    the name hits by name with no call. A write refused because another run wrote
    meanwhile is retried once (the store has reloaded)."""
    out = []
    for p in proposals:
        rec = {**p, "saved": False}
        for attempt in (1, 2):
            already = store.find(p["name"])
            if already is not None and already.asset_id == p["asset_id"]:
                # There already (e.g. another run saved it meanwhile): nothing written.
                rec.update(saved=True, already=True)
                break
            try:
                store.add_alias(p["asset_id"], p["name"])
                rec["saved"] = True
                break
            except library.ConcurrentWriteError as exc:
                if attempt == 2:
                    rec["error"] = str(exc)
            except library.LibraryError as exc:
                rec["error"] = str(exc)
                break
        out.append(rec)
    return out


def build_manifest(input_path: Path, lab: dict, metrics: Optional[dict], known: Known,
                   requests: list[audit.AssetRequest], hits: dict, reuse_plan,
                   reuse_record: Optional[dict], plan, route_record: Optional[dict], *,
                   no_llm: bool, aliases: Optional[dict] = None, llm_summary: Optional[dict] = None,
                   metrics_problems: Optional[list[str]] = None) -> dict:
    """``hits``: (kind, name key) -> the library entry found by name, from ``main``."""
    decided = {(d.kind, library.name_key(d.name)): d for d in (reuse_plan.decisions if reuse_plan else [])}
    reused = {key: d for key, d in decided.items() if d.decision != "none"}
    planned = {(p.kind, library.name_key(p.name)): p for p in (plan.assets if plan else [])}

    def p_reused(p):
        """The reuse decision a planned asset merged into, if its same_as is reused."""
        return reused.get((p.kind, library.name_key(p.same_as))) if p.same_as else None

    assets = []
    for r in requests:
        rec = _asset_record(r)
        hit = hits.get((r.kind, r.key))
        if hit is not None:
            rec["status"] = "available"
            rec["library"] = {"asset_id": hit.asset_id, "builtin": hit.builtin,
                              "matched": "name" if library.name_key(hit.name) == r.key else "alias"}
        elif (r.kind, r.key) in reused:
            rec["status"] = "reused"
            rec["reuse"] = _reuse_record(reused[(r.kind, r.key)])
        elif (r.kind, r.key) in planned and p_reused(planned[(r.kind, r.key)]) is not None \
                and planned[(r.kind, r.key)].route not in routing.REVIEW_ROUTES:
            # Merged into an asset the judge reused: it follows that library match, and
            # has no plan of its own to provide. (Behaviour or impossible is still
            # flagged, below.)
            p = planned[(r.kind, r.key)]
            rec["status"] = "reused"
            rec["reuse"] = {**_reuse_record(p_reused(p)), "same_as": p.same_as,
                            "routing_reason": p.reason}
            own = decided.get((r.kind, r.key))
            if own is not None:
                rec["reuse"]["judge"] = _reuse_record(own)   # what the judge said of it alone
        elif (r.kind, r.key) in planned:
            p = planned[(r.kind, r.key)]
            rec["plan"] = p.model_dump(exclude={"name", "kind"}, exclude_none=True)
            target = planned.get((r.kind, library.name_key(p.same_as))) if p.same_as else None
            # A merged asset follows its target: if either is behaviour or impossible,
            # its objects need a reviewer too.
            flagged_by = next((x for x in (p, target)
                               if x is not None and x.route in routing.REVIEW_ROUTES), None)
            if flagged_by is not None:
                rec["status"] = "flagged"  # a reviewer decides (#116)
                # The route and reason that flagged it: its own, or its same_as target's.
                rec["review_route"] = flagged_by.route
                rec["review_reason"] = flagged_by.reason
            elif p.same_as:
                rec["status"] = "merged"
            else:
                rec["status"] = "planned"
        else:
            rec["status"] = "unplanned"
        d = decided.get((r.kind, r.key))
        if d is not None and d.decision == "none" and "reuse" not in rec:
            rec["reuse"] = _reuse_record(d)   # the judge's reason, for a reviewer
        assets.append(rec)
    statuses = Counter(a["status"] for a in assets)
    review_items = []
    for a in assets:
        if a["status"] != "flagged":
            continue
        item = {"name": a["name"], "kind": a["kind"], "route": a["review_route"],
                "reason": a["review_reason"], "description": a["plan"]["description"],
                "objects": sorted({u["object"] for u in a["uses"]})}
        if a["plan"].get("same_as"):
            item["same_as"] = a["plan"]["same_as"]
        review_items.append(item)
    # Assets that would each be provided (or reviewed) once: merged ones follow their target.
    routes = Counter(a["plan"]["route"] for a in assets
                     if a["status"] in ("planned", "flagged") and not a["plan"].get("same_as"))
    forms = Counter(a["plan"]["form"] for a in assets if "plan" in a)
    # Each asset provided (or reused) once: those merged into a reused one follow it.
    decisions = Counter(a["reuse"]["decision"] for a in assets
                        if a["status"] == "reused" and not a["reuse"].get("same_as"))
    if no_llm:
        status = "audited"
    elif any(rec and rec.get("error") for rec in (reuse_record, route_record)):
        status = "failed"
    else:
        status = "planned"
    return {
        "tool": "provision_lab",
        "stage": "plan (#106, #108)",
        "status": status,
        "source": {
            "input": _rel(input_path),
            "labId": lab.get("labId"),
            "metrics": _rel(Path(metrics["_path"])) if metrics else None,
            "metrics_problems": metrics_problems or [],
            "source_model": (metrics or {}).get("model"),
            "level": (metrics or {}).get("level"),
            "lab_name": (metrics or {}).get("lab_name"),
            "hierarchical": bool((metrics or {}).get("registry")),
        },
        "library": known.identity,
        "library_note": known.note,
        "reuse": reuse_record,
        "aliases": aliases or {"saved": False,
                              "proposed": alias_proposals(reuse_plan, known.entries())},
        "routing": route_record,
        "llm": llm_summary,
        "routes_offered": {k: list(routing.routes_for(k)) for k in routing.KINDS},
        "counts": {
            "assets": len(assets),
            "models": sum(a["kind"] == "model" for a in assets),
            "textures": sum(a["kind"] == "texture" for a in assets),
            "by_status": dict(sorted(statuses.items())),
            "by_reuse": dict(sorted(decisions.items())),
            "by_route": dict(sorted(routes.items())),
            "by_form": dict(sorted(forms.items())),
        },
        # Assets the plan says are behaviour or can't be made: for a reviewer, who can have
        # them turned into components (#116). Nothing is provided for them.
        "review_items": review_items,
        "assets": assets,
        "provenance": provenance(),
        "created": _now(),
    }


def _step_line(label: str, r: Optional[dict]) -> Optional[str]:
    if not r or not r.get("calls"):
        return None
    cost = f"${r['cost_usd']:.4f}, " if "cost_usd" in r else ""
    return (f"{label}: {r['model']}, {cost}{r['seconds']} s, {r['calls']} call(s)"
            + (f", {r['transient_retries']} transient retries" if r.get("transient_retries") else ""))


def _print_plan(manifest: dict) -> None:
    for a in manifest["assets"]:
        if a["status"] == "available":
            what = f"available ({a['library']['asset_id']}" + \
                   (", built-in)" if a["library"]["builtin"] else ")")
        elif a["status"] == "reused":
            u = a["reuse"]
            base = u.get("asset_id") or u.get("shape", "")
            what = (f"reused     {base}" + (f" + {u['texture_id']}" if u.get("texture_id") else "")
                    + (f"  (same as {u['same_as']})" if u.get("same_as")
                       else f"  ({u.get('reason', '')[:60]})"))
        elif a["status"] in ("planned", "merged", "flagged"):
            p = a["plan"]
            if a["status"] == "merged":
                what = f"same as {p['same_as']}"
            elif a["status"] == "flagged":
                what = f"FLAGGED {p['route']:10} {p['reason'][:80]}"
            else:
                what = f"{p['route']:10} {p['form']:18} {p['description'][:70]}"
        else:
            what = a["status"]
        print(f"  {a['kind']:7} {a['name'][:32]:32} {what}")
    c = manifest["counts"]
    print(f"{c['assets']} assets ({c['models']} models, {c['textures']} textures): "
          + ", ".join(f"{n} {s}" for s, n in c["by_status"].items()))
    if c["by_route"]:
        print("routes: " + ", ".join(f"{n} {r}" for r, n in c["by_route"].items()))
    if manifest["review_items"]:
        print(f"{len(manifest['review_items'])} flagged for a reviewer (behaviour or impossible; "
              f"see review_items in the manifest)")
    al = manifest["aliases"]
    if al.get("saved"):
        saved = sum(x.get("saved", False) and not x.get("already") for x in al["proposed"])
        already = sum(bool(x.get("already")) for x in al["proposed"])
        print(f"aliases: {saved} of {len(al['proposed'])} saved to the library"
              + (f" ({already} already there)" if already else ""))
    elif al["proposed"]:
        print(f"aliases: {len(al['proposed'])} proposed (save them with --save-aliases)")
    for line in (_step_line("reuse", manifest.get("reuse")),
                 _step_line("describe and route", manifest.get("routing"))):
        if line:
            print(line)


def run_dir_for(input_path: Path, run_group: Optional[str], stamp: str) -> Path:
    stem = input_path.name
    for suffix in ("_output.json", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) or "lab"
    base = OUTPUT_ROOT / run_group if run_group else OUTPUT_ROOT
    return base / f"{stem}_{stamp}"


def main(argv: Optional[list[str]] = None, *, call_fn: Optional[llm.CallFn] = None,
         output_root: Optional[Path] = None) -> int:
    ap = argparse.ArgumentParser(description="Provision a lab's models and textures (so far: "
                                             "the plan, with --dry-run).")
    ap.add_argument("lab", help="a json_lab *_output.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: audit, library lookup, reuse, describe and route "
                         "(required for now)")
    ap.add_argument("--no-llm", action="store_true",
                    help="stop after the audit and library lookup (no API key needed)")
    ap.add_argument("--no-reuse", action="store_true",
                    help="skip the reuse judge: plan every asset the library has no name for")
    ap.add_argument("--save-aliases", action="store_true",
                    help="save the reuse judge's single-entry matches (not built-ins) as library aliases")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL,
                    help=f"benchmark_config.MODELS key for the reuse judge and describe and "
                         f"route (default {llm.DEFAULT_MODEL})")
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
    if args.save_aliases and (args.no_llm or args.no_reuse):
        return fail("--save-aliases needs the reuse judge (not --no-llm or --no-reuse)")
    input_path = Path(args.lab)
    metrics_problems: list[str] = []
    try:
        lab = audit.load_lab(input_path)
        if args.metrics:
            metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                return fail(f"--metrics {args.metrics} is not a metrics record (a JSON object)")
            metrics["_path"] = str(args.metrics)
        else:
            metrics = audit.paired_metrics(input_path, metrics_problems)
        known = Known(args.library)
    except (audit.InputError, library.LibraryError, OSError, ValueError) as exc:
        return fail(str(exc))
    if args.save_aliases and known.store is None:
        return fail("--save-aliases needs an asset library to save to (build one with library.py seed)")
    for problem in metrics_problems:
        print(f"note: {problem}")
    if known.note:
        print(f"note: {known.note}")

    requests = audit.audit_lab(lab, audit.purposes_from_metrics(metrics))
    hits = {(r.kind, r.key): known.find(r.name, r.kind) for r in requests}
    unmatched = [r for r in requests if hits[(r.kind, r.key)] is None]
    reuse_plan = reuse_record = plan = record = None
    aliases, llm_summary = None, None
    if not args.no_llm:
        # The model key is always checked, so a typo fails the same way on every lab; the
        # SDK and key only when a call will be made.
        try:
            if call_fn is None and unmatched:
                llm.check_environment(args.model)
            elif call_fn is None:
                llm.model_config_for(args.model)
        except KeyError as exc:
            return fail(exc.args[0])
        except SystemExit as exc:  # benchmark.require_environment prints its own message
            return fail(f"environment check failed ({exc.code})")
        if not unmatched:
            # Everything is in the library by name: nothing to ask, no call.
            plan, record = routing.LabAssetPlan(assets=[]), {"model": args.model, "calls": 0,
                                                             "note": "nothing to plan"}
        else:
            run_llm = llm.TrackedLLM(args.model, call_fn=call_fn)
            to_plan, taken = unmatched, {}
            if not args.no_reuse:
                print(f"checking {len(unmatched)} of {len(requests)} assets against the library "
                      f"with {args.model}...")
                reuse_plan, reuse_record = reuse.judge_reuse(run_llm, lab, unmatched, known.entries())
                if reuse_plan is not None:
                    taken = {(d.kind, library.name_key(d.name)): d for d in reuse_plan.decisions
                             if d.decision != "none"}
                    to_plan = [r for r in unmatched if (r.kind, r.key) not in taken]
            if args.no_reuse or reuse_plan is not None:
                print(f"planning {len(to_plan)} of {len(requests)} assets with {args.model}...")
                reused = [(r, _matched(taken[(r.kind, r.key)])) for r in unmatched
                          if (r.kind, r.key) in taken]
                plan, record = routing.plan_routes(run_llm, lab, to_plan, reused)
            if args.save_aliases and reuse_plan is not None:
                proposals = alias_proposals(reuse_plan, known.entries())
                aliases = {"saved": True, "proposed": save_aliases(known.store, proposals),
                           "library_after": known.store.identity()}
            llm_summary = run_llm.summary()
            if llm_summary and llm_summary.get("served_models"):
                llm._benchmark().warn_served_model(run_llm.model_config.model_id,
                                                   llm_summary["served_models"])

    manifest = build_manifest(input_path, lab, metrics, known, requests, hits, reuse_plan,
                              reuse_record, plan, record, no_llm=args.no_llm, aliases=aliases,
                              llm_summary=llm_summary, metrics_problems=metrics_problems)
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
        for label, rec in (("the reuse judge", reuse_record), ("describe and route", record)):
            if rec and rec.get("error"):
                print(f"error: {label} failed: {rec['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

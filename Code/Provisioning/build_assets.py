"""
Build the assets a provisioning plan routed to a builder (#111; spec: PROVISIONING.md,
"Builders"). A stepping stone until the full run (#113) does this itself: it reads a
``provision_lab.py --dry-run`` manifest, runs each ``builder:<name>`` asset's builder,
and writes the files and a results record into the run folder.

    python "Code/Provisioning/build_assets.py" "Artifacts/Data/Provisioning/testing/<run>"
        [--only NAME ...] [--save] [--library PATH]

- Files: ``<run>/built/<asset name>/model.glb`` and ``thumbnail.png`` (check them by
  eye; any glTF viewer opens the GLB).
- ``<run>/build_results.json``: per asset the builder, input and options, whether it
  built, the reason if not, the builder's details (for molecules: PubChem CID, geometry
  method, shape, lone pairs) and the library entry it would be stored as.
- ``--save`` stores each built asset in the library, with the lab's name (and the names
  merged into it) as aliases, so the next dry run finds them by name. An entry that is
  already there (the same molecule built for another lab) just gains the aliases.

Exit codes: 0 everything built (or nothing to build); 1 something didn't build or
couldn't be saved; 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import builders  # noqa: E402
import library  # noqa: E402

MANIFEST = "provisioning_manifest.json"
RESULTS = "build_results.json"


def _folder_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "asset"


def targets(manifest: dict) -> list[dict]:
    """The assets routed to a builder, each with the names merged into it."""
    assets = manifest.get("assets") or []
    out = []
    for a in assets:
        plan = a.get("plan") or {}
        if a.get("status") == "planned" and str(plan.get("route", "")).startswith("builder:"):
            merged = [m["name"] for m in assets if m.get("status") == "merged" and m["kind"] == a["kind"]
                      and library.name_key((m.get("plan") or {}).get("same_as") or "")
                      == library.name_key(a["name"])]
            out.append({**a, "merged_names": merged})
    return out


def build_one(asset: dict, out_root: Path, folder_name: Optional[str] = None) -> dict:
    plan = asset["plan"]
    rec = {"name": asset["name"], "kind": asset["kind"], "builder": plan["route"],
           "input": plan.get("builder_input"), "options": plan.get("builder_options") or [],
           "merged_names": asset["merged_names"], "ok": False}
    info = builders.get(plan["route"])
    if info is None or info.build is None:
        rec["reason"] = f"no builder named {plan['route']!r} is registered"
        return rec
    request = builders.BuildRequest(
        name=asset["name"], kind=asset["kind"], input=plan.get("builder_input") or "",
        options=tuple(plan.get("builder_options") or ()), description=plan.get("description", ""),
        must_be_right=tuple(plan.get("must_be_right") or ()))
    folder = out_root / (folder_name or _folder_name(asset["name"]))
    if folder.exists():   # an earlier build's files must not outlive a failed rebuild
        shutil.rmtree(folder)
    try:
        result = info.build(request, folder)
    except Exception as exc:  # a builder bug fails its asset, not the run
        rec["reason"] = f"the builder failed: {type(exc).__name__}: {exc}"
        return rec
    rec.update(ok=result.ok, details=result.details)
    if not result.ok:
        rec["reason"] = result.reason
        return rec
    rec["entry"] = result.entry
    rec["files"] = {role: str(Path(p).relative_to(out_root.parent)).replace("\\", "/")
                    for role, p in result.files.items()}
    rec["_files"] = result.files
    return rec


def _keep_reason(existing: library.LibraryEntry, new: library.LibraryEntry) -> Optional[str]:
    """Why the library's entry stays as it is ("same": made by this builder version),
    or None when the new build replaces it: only an entry an older version of the same
    builder made (an entry saved before versions were recorded counts as older), never
    one a reviewer has reviewed, a newer one, or one no builder made."""
    old, cur = existing.provenance, new.provenance
    if old is None or old.method != "builder":
        return "it wasn't made by a builder"
    if old.builder not in (None, cur.builder):
        return f"the {old.builder} builder made it"
    old_v, new_v = old.builder_version or 0, cur.builder_version or 0
    if old_v == new_v:
        return "same"
    if old_v > new_v:
        return f"a newer builder version ({old_v}) made it than this one ({new_v})"
    if existing.review == "reviewed":
        return (f"a reviewer reviewed it (made by builder version {old_v}; this build, "
                f"version {new_v}, is in the run folder)")
    return None


def save(store: library.LocalFolderStore, rec: dict) -> None:
    """Store a built asset in the library with the lab's names as aliases. An entry
    already there gains the aliases; if an older version of the same builder made it
    (its provenance note differs), its files and fields are replaced, keeping its
    aliases. Sets ``saved`` / ``save_error`` / ``save_notes`` on ``rec``."""
    names = [rec["name"], *rec["merged_names"]]
    entry = library.LibraryEntry(**rec["entry"])
    for attempt in (1, 2):
        try:
            rec.pop("save_notes", None)
            existing = store.get(entry.asset_id)
            target = existing.asset_id if existing else entry.asset_id
            free = []
            for n in names:
                hit = store.find(n)
                if hit is None and library.name_key(n) != library.name_key(entry.name):
                    free.append(n)
                elif hit is not None and hit.asset_id != target:
                    rec.setdefault("save_notes", []).append(
                        f"{n!r} already names {hit.asset_id}, so it isn't an alias of {target}")
            stale = False
            if existing is not None:
                keep = _keep_reason(existing, entry)
                stale = keep is None
                if keep and keep != "same":
                    rec.setdefault("save_notes", []).append(f"{target} kept as it is: {keep}")
            if existing is None or stale:
                old_names = [*existing.aliases, existing.name] if existing else []
                entry.aliases = [n for n in dict.fromkeys([*old_names, *free])
                                 if library.name_key(n) != library.name_key(entry.name)]
                store.put(entry, rec["_files"], keep_files=False)
            else:
                for n in free:
                    store.add_alias(target, n)
            rec["saved"] = {"asset_id": target, "added": existing is None, "replaced": stale,
                            "aliases": free}
            return
        except library.ConcurrentWriteError as exc:   # the store has reloaded: try again once
            if attempt == 2:
                rec["save_error"] = str(exc)
        except library.LibraryError as exc:
            rec["save_error"] = str(exc)
            return


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the assets a provisioning plan routed to a builder.")
    ap.add_argument("run", help="a provision_lab.py run folder, or its provisioning_manifest.json")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="build only these assets (lab names)")
    ap.add_argument("--save", action="store_true",
                    help="store what was built in the asset library, with the lab's names as aliases")
    ap.add_argument("--library", help=f"library folder (default: ${library.LIBRARY_ENV}, else "
                                      f"{library.DEFAULT_LIBRARY})")
    args = ap.parse_args(argv)

    def fail(msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr)
        return 2

    path = Path(args.run)
    manifest_path = path / MANIFEST if path.is_dir() else path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail(f"cannot read {manifest_path}: {exc}")
    if not isinstance(manifest, dict) or manifest.get("tool") != "provision_lab":
        return fail(f"{manifest_path} is not a provision_lab manifest")
    run_dir = manifest_path.parent
    todo = targets(manifest)
    if args.only:
        wanted = {library.name_key(n) for n in args.only}
        unknown = wanted - {library.name_key(a["name"]) for a in todo}
        if unknown:
            return fail(f"not routed to a builder in this plan: {', '.join(sorted(unknown))}")
        todo = [a for a in todo if library.name_key(a["name"]) in wanted]
    store = None
    if args.save:
        try:
            store = library.open_store(args.library)
        except (library.LibraryError, OSError) as exc:
            return fail(f"--save needs an asset library: {exc}")

    print(f"building {len(todo)} asset(s) routed to a builder...")
    results = []
    used: set[str] = set()   # folder names, compared ignoring case (Windows)
    for asset in todo:
        folder = base = _folder_name(asset["name"])
        n = 2
        while folder.lower() in used:
            folder, n = f"{base}_{n}", n + 1
        used.add(folder.lower())
        rec = build_one(asset, run_dir / "built", folder)
        if rec["ok"] and store is not None:
            save(store, rec)
        rec.pop("_files", None)
        results.append(rec)
        d = rec.get("details") or {}
        if rec["ok"]:
            what = ", ".join(str(x) for x in (d.get("shape"), d.get("geometry")) if x)
            saved = rec.get("saved")
            tail = (f" -> library {saved['asset_id']}" if saved else "") + \
                   (f" (not saved: {rec['save_error']})" if rec.get("save_error") else "")
            print(f"  built   {asset['name'][:32]:32} {rec['input']!r} -> {rec['entry']['asset_id']}"
                  f" ({what}){tail}")
        else:
            print(f"  FAILED  {asset['name'][:32]:32} {rec['input']!r}: {rec['reason']}")

    record = {
        "tool": "build_assets",
        "manifest": manifest_path.name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_after": store.identity() if store is not None else None,
        "counts": {"assets": len(results), "built": sum(r["ok"] for r in results),
                   "failed": sum(not r["ok"] for r in results),
                   "saved": sum("saved" in r for r in results)},
        "assets": results,
    }
    out = run_dir / RESULTS
    library.write_json_atomic(out, record)
    c = record["counts"]
    print(f"{c['built']} built, {c['failed']} failed" + (f", {c['saved']} saved" if store else "")
          + f"; results: {out}")
    return 1 if c["failed"] or any(r.get("save_error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

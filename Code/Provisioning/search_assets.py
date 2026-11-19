"""
Search open libraries for the assets a provisioning plan routed to ``search`` (#110;
spec: PROVISIONING.md, "Search"), rank the hits, auto-pick one, and keep the ranked
top 10 for a reviewer (#116). A stepping stone until the full run (#113) does this
itself, like ``build_assets.py`` for builders.

    python "Code/Provisioning/search_assets.py" "Artifacts/Data/Provisioning/testing/<run>"
        [--only NAME ...] [--sources polyhaven nasa] [--model KEY] [--no-vision]
        [--redo] [--save] [--library PATH]
    python "Code/Provisioning/search_assets.py" <run> --choose NAME=RANK [--save]

Per asset:
1. **Collect** up to ``MAX_HITS`` hits across the sources (``sources.py``) for the
   plan's search keywords.
2. **Filter**, counting what is dropped and why (never hidden): a license outside the
   tier (CC0, public domain, CC-BY with an author), an unusable format, a known
   triangle count over the budget.
3. **Text pass**: a hit sharing no word with the asset (name, description, keywords),
   that its source didn't rank as relevant either, is an obvious mismatch; the rest
   are ordered and the best ``TOP_FOR_VISION`` kept.
4. **One vision call** (``CandidateRanking``) ranks their thumbnails against the
   plan's description and must-be-right facts: best first, fits or not, one-line reason
   each. Every shown candidate is ranked exactly once (one reask for all problems).
5. **Auto-pick** the best candidate that fits, fetch it, convert (glTF to GLB; a
   texture to JPG within ``library.MAX_TEXTURE_SIDE``), center it and scale it to a
   unit cube, and check the triangle budget; if that fails, the next one that fits
   (up to ``FETCH_TRIES``). None fits: ``no_match`` (the composite builder, #112, or a
   reviewer takes over). ``--no-vision`` ranks by text only and picks nothing
   (``needs_review``).

Output, in the run folder: ``search_results.json`` (rewritten after every asset: the
top 10 with thumbnails, links, licenses, authors, fits and reasons; what was dropped;
which one was chosen and by whom, ``auto`` or ``reviewer``; the cost),
``search/<asset>/NN.png`` (the thumbnails) and ``found/<asset>/`` (the chosen file and
its thumbnail). A rerun skips assets already decided (``--redo`` searches again);
``--choose NAME=RANK`` takes another candidate from the saved ranking, with no new
search or LLM call. ``--save`` stores what was found in the library with the lab's
names as aliases (``build_assets.save``).

Exit codes: 0 every asset found; 1 some not found, failed or left for review; 2 bad
input or environment.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import build_assets  # noqa: E402
import library  # noqa: E402
import llm  # noqa: E402
import sources  # noqa: E402
from routing import strict_schema  # noqa: E402

RESULTS = "search_results.json"
MAX_HITS = 30          # across all sources, per asset
TOP_FOR_VISION = 12    # shown to the vision ranking
KEEP = 10              # ranked candidates kept for a reviewer
FETCH_TRIES = 3        # candidates that fit, tried in rank order
SOURCE_RELEVANT = 0.5  # a source's own score that keeps a hit through the text pass
FORMATS = {"model": ("glb", "gltf"), "texture": ("jpg", "jpeg", "png")}


# ── Filter and text pass ──────────────────────────────────────────────


def queries_for(asset: dict) -> list[str]:
    plan = asset.get("plan") or {}
    q = [k.strip() for k in plan.get("search_keywords") or [] if k.strip()]
    return q or [" ".join(sorted(sources.words(asset["name"])))]


def collect(asset: dict, chosen_sources: list) -> tuple[list[sources.Candidate], dict]:
    """Hits from every source (deduplicated, at most ``MAX_HITS``), and per source the
    number of hits or the error."""
    hits: dict[str, sources.Candidate] = {}
    per_source = {}
    for src in chosen_sources:
        try:
            found = src.search(queries_for(asset), asset["kind"])
            per_source[src.name] = {"hits": len(found)}
            if getattr(src, "truncated", False):
                per_source[src.name]["note"] = "the source's listing was cut short"
        except sources.SourceError as exc:
            per_source[src.name] = {"error": str(exc)}
            continue
        for c in found:
            hits.setdefault(c.key, c)
    ordered = sorted(hits.values(), key=lambda c: -c.source_score)[:MAX_HITS]
    return ordered, per_source


def filter_candidates(cands: list[sources.Candidate], kind: str):
    """(kept, dropped: reason -> candidate keys)."""
    kept, dropped = [], {}
    for c in cands:
        if c.license not in library.ALLOWED_LICENSES:
            why = "license"
        elif library.ALLOWED_LICENSES[c.license] and not c.author:
            why = "license_without_author"   # CC-BY needs the author to credit
        elif c.format.lower() not in FORMATS[kind]:
            why = "format"
        elif c.triangles and c.triangles > library.MAX_MODEL_TRIANGLES:
            why = "over_budget"
        else:
            kept.append(c)
            continue
        dropped.setdefault(why, []).append(c.key)
    return kept, dropped


def text_score(asset: dict, c: sources.Candidate) -> float:
    plan = asset.get("plan") or {}
    want = sources.words(" ".join([asset["name"], plan.get("description", ""), *queries_for(asset)]))
    have = sources.words(" ".join([c.name, *c.tags, c.description]))
    return len(want & have) / max(len(want), 1)


def text_pass(asset: dict, cands: list[sources.Candidate]):
    """(kept, best first, at most ``TOP_FOR_VISION``; scores by key; dropped keys)."""
    scores = {c.key: text_score(asset, c) for c in cands}
    kept = [c for c in cands if scores[c.key] > 0 or c.source_score >= SOURCE_RELEVANT]
    dropped = [c.key for c in cands if c not in kept]
    kept.sort(key=lambda c: (-(scores[c.key] + c.source_score), c.key))
    return kept[:TOP_FOR_VISION], scores, dropped


# ── Vision ranking ────────────────────────────────────────────────────

_SCOPE: contextvars.ContextVar = contextvars.ContextVar("ranking_scope", default=None)


@contextmanager
def ranking_scope(count: int):
    """Check ``CandidateRanking`` against candidates numbered 1..count."""
    token = _SCOPE.set(count)
    try:
        yield
    finally:
        _SCOPE.reset(token)


class RankedCandidate(BaseModel):
    model_config = ConfigDict(json_schema_extra=strict_schema(set()))
    number: int = Field(description="The candidate's number, as shown.")
    fits: bool = Field(False, description="True only if it clearly shows this asset as described, "
                                          "usable in the lab as it is.")
    reason: str = Field("", description="One sentence a reviewer can check.")


class CandidateRanking(BaseModel):
    ranking: list[RankedCandidate] = Field(default_factory=list,
                                           description="Every shown candidate once, best first.")

    @model_validator(mode="after")
    def _check(self):
        count = _SCOPE.get()
        if count is None:
            return self
        problems, seen = [], set()
        for r in self.ranking:
            if not 1 <= r.number <= count:
                problems.append(f"candidate {r.number} wasn't shown (the numbers are 1 to {count})")
            elif r.number in seen:
                problems.append(f"candidate {r.number} is ranked more than once")
            seen.add(r.number)
            if not r.reason.strip():
                problems.append(f"candidate {r.number}: the reason is empty")
        missing = [str(n) for n in range(1, count + 1) if n not in seen]
        if missing:
            problems.append("these candidates aren't ranked: " + ", ".join(missing))
        if problems:
            raise ValueError("Fix these problems and rank every candidate again:\n- " + "\n- ".join(problems))
        return self


def build_ranking_prompt(asset: dict, cands: list[sources.Candidate], thumbs: dict) -> list:
    """Text and thumbnail files, in order: the asset, then each numbered candidate."""
    plan = asset.get("plan") or {}
    facts = "\n".join(f"- {f}" for f in plan.get("must_be_right") or []) or "- (none given)"
    what = "3D model" if asset["kind"] == "model" else "texture (an image wrapped onto a model)"
    parts: list = [f"""You choose a {what} for an augmented-reality science lab on a headset, from \
candidates found in free libraries. Judge each candidate by its picture and its name.

THE ASSET
name in the lab: {asset['name']}
form: {plan.get('form', '')}
what it should look like: {plan.get('description', '')}
must be right:
{facts}

RULES
- fits = true only if the candidate clearly IS this asset as described (the same object or surface, \
recognizable), usable as it is: not a different thing, not a whole scene that merely contains it, not a \
cartoon when the lab needs a realistic one (or the reverse).
- When unsure, fits = false: a reviewer can still choose it from the saved ranking.
- Rank every candidate once, best first, whether it fits or not.
- reason: one sentence a reviewer can check.

CANDIDATES ({len(cands)})"""]
    for n, c in enumerate(cands, start=1):
        tags = f"; tags: {', '.join(c.tags[:12])}" if c.tags else ""
        desc = f"; {' '.join(c.description.split())[:160]}" if c.description else ""
        parts.append(f"Candidate {n}: {c.name} (from {c.source}{tags}{desc})"
                     + ("" if c.key in thumbs else " [no picture]"))
        if c.key in thumbs:
            parts.append(thumbs[c.key])
    return parts


# ── Fetch the chosen candidate ────────────────────────────────────────


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "asset"


def prepare(cand: sources.Candidate, asset: dict, work: Path, out: Path, thumb: Optional[Path]) -> dict:
    """Fetch, convert and check the candidate; write ``out/<role file>`` (and its
    thumbnail). Returns the library entry fields and files. Raises ``SourceError``."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    main = sources.SOURCES[cand.source].fetch(cand, work)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files: dict[str, Path] = {}
    entry: dict = {}
    mods = []
    if asset["kind"] == "model":
        try:
            import trimesh
            scene = trimesh.load(main, force="scene")
            lo, hi = scene.bounds
        except Exception as exc:   # an unreadable file fails this candidate only
            raise sources.SourceError(f"the model couldn't be read ({type(exc).__name__}: {exc})") from exc
        size = float((hi - lo).max())
        if size <= 0:
            raise sources.SourceError("the model has no size")
        triangles = int(sum(len(m.faces) for m in scene.dump()))
        if triangles > library.MAX_MODEL_TRIANGLES:
            raise sources.SourceError(f"the model has {triangles} triangles, more than the budget "
                                      f"of {library.MAX_MODEL_TRIANGLES}")
        scene.apply_transform(trimesh.transformations.scale_matrix(1.0 / size)
                              @ trimesh.transformations.translation_matrix(-(lo + hi) / 2))
        lo, hi = scene.bounds
        files["model"] = out / library.ROLE_FILES["model"]
        files["model"].write_bytes(scene.export(file_type="glb"))
        if main.suffix.lower() != ".glb":
            mods.append("converted from glTF to GLB")
        mods.append("centered and scaled to fit a unit cube")
        entry.update(bounds=[round(float(x), 6) for x in (hi - lo)], triangles=triangles)
    else:
        from PIL import Image
        try:
            with Image.open(main) as img:
                img = img.convert("RGB")
                if max(img.size) > library.MAX_TEXTURE_SIDE:
                    img.thumbnail((library.MAX_TEXTURE_SIDE, library.MAX_TEXTURE_SIDE))
                    mods.append(f"resized to at most {library.MAX_TEXTURE_SIDE} px")
                files["texture"] = out / library.ROLE_FILES["texture"]
                img.save(files["texture"], format="JPEG", quality=90)
                entry.update(pixels=list(img.size), mapping=cand.mapping or "uv")
        except OSError as exc:
            raise sources.SourceError(f"the texture couldn't be read ({exc})") from exc
        if main.suffix.lower() not in (".jpg", ".jpeg"):
            mods.append("converted to JPG")
    if thumb and thumb.is_file():
        files["thumbnail"] = out / library.ROLE_FILES["thumbnail"]
        shutil.copyfile(thumb, files["thumbnail"])
    elif asset["kind"] == "texture":
        files["thumbnail"] = out / library.ROLE_FILES["thumbnail"]
        library.make_thumbnail(files["texture"], files["thumbnail"])
    shutil.rmtree(work, ignore_errors=True)
    plan = asset.get("plan") or {}
    asset_id = f"{cand.source}_{_slug(cand.id)}"[:120]
    entry.update({
        "asset_id": asset_id,
        "kind": asset["kind"],
        "name": asset_id,   # unique: two sources (or two assets) can share a display name
        "form": plan.get("form") or None,
        "description": " ".join(f"{cand.name}. {cand.description}".split())[:600]
                       or cand.name,
        "tags": sorted({t.lower() for t in cand.tags})[:30],
        "provenance": {
            "method": "search", "source_site": cand.source, "page_url": cand.page_url,
            "record_id": cand.id, "author": cand.author, "title": cand.name,
            "license": cand.license, "license_url": cand.license_url,
            "modifications": mods, "notes": cand.notes or None,
        },
    })
    return {"entry": entry, "files": files}


# ── One asset ─────────────────────────────────────────────────────────


def _rel(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir)).replace("\\", "/")


def search_one(asset: dict, run_dir: Path, folder: str, chosen_sources, run_llm) -> dict:
    rec = {"name": asset["name"], "kind": asset["kind"], "merged_names": asset["merged_names"],
           "queries": queries_for(asset), "status": "no_match"}
    cands, rec["sources"] = collect(asset, chosen_sources)
    rec["hits"] = len(cands)
    kept, dropped = filter_candidates(cands, asset["kind"])
    kept, scores, mismatches = text_pass(asset, kept)
    if mismatches:
        dropped["text_mismatch"] = mismatches
    rec["dropped"] = {k: len(v) for k, v in dropped.items()}
    if not kept:
        errors = [s for s in rec["sources"].values() if "error" in s]
        if errors and len(errors) == len(rec["sources"]):
            rec.update(status="failed", reason="every source failed: "
                       + "; ".join(f"{n}: {s['error']}" for n, s in rec["sources"].items()))
        else:
            rec["reason"] = "no usable candidate in any source"
        return rec

    thumb_dir = run_dir / "search" / folder
    if thumb_dir.exists():
        shutil.rmtree(thumb_dir)
    thumb_dir.mkdir(parents=True)
    thumbs = {}
    for n, c in enumerate(kept, start=1):
        dest = thumb_dir / f"{n:02d}.png"
        if sources.fetch_thumbnail(c, dest):
            thumbs[c.key] = dest

    order = [(c, None, "") for c in kept]   # (candidate, fits, reason), best first
    if run_llm is not None:
        with ranking_scope(len(kept)):
            ranking, rec["ranking_call"] = run_llm.call(CandidateRanking,
                                                        build_ranking_prompt(asset, kept, thumbs))
        if ranking is None:
            rec.update(status="failed", reason="the ranking call failed")
            return rec
        order = [(kept[r.number - 1], r.fits, r.reason) for r in ranking.ranking]
    rec["ranked"] = [{
        "rank": i, **c.to_json(), "text_score": round(scores[c.key], 3),
        "thumbnail": _rel(thumbs[c.key], run_dir) if c.key in thumbs else None,
        "fits": fits, "reason": reason,
    } for i, (c, fits, reason) in enumerate(order[:KEEP], start=1)]

    if run_llm is None:
        rec.update(status="needs_review", reason="ranked by text only (--no-vision); a reviewer chooses")
        return rec
    tries = []
    for r, (c, fits, _) in zip(rec["ranked"], order):
        if not fits or len(tries) >= FETCH_TRIES:
            continue
        try:
            got = prepare(c, asset, run_dir / "search" / folder / "fetch", run_dir / "found" / folder,
                          thumbs.get(c.key))
        except sources.SourceError as exc:
            tries.append({"rank": r["rank"], "error": str(exc)})
            continue
        rec.update(status="found", chosen={"rank": r["rank"], "by": "auto"}, entry=got["entry"],
                   files={k: _rel(v, run_dir) for k, v in got["files"].items()}, _files=got["files"])
        break
    if tries:
        rec["fetch_errors"] = tries
    if rec["status"] != "found":
        rec["reason"] = ("no candidate fits" if not any(f for _, f, _ in order)
                         else "every candidate that fits failed to fetch or check")
        rec["status"] = "no_match" if not tries else "failed"
    return rec


def choose(rec: dict, rank: int, run_dir: Path, folder: str, asset: dict) -> dict:
    """A reviewer's choice from the saved ranking: no new search, no LLM call."""
    pick = next((r for r in rec.get("ranked") or [] if r["rank"] == rank), None)
    if pick is None:
        raise ValueError(f"{rec['name']} has no candidate ranked {rank}")
    fields = {k: pick[k] for k in sources.Candidate.__dataclass_fields__ if k in pick}
    cand = sources.Candidate(**fields)
    thumb = run_dir / pick["thumbnail"] if pick.get("thumbnail") else None
    got = prepare(cand, asset, run_dir / "search" / folder / "fetch", run_dir / "found" / folder, thumb)
    rec.update(status="found", chosen={"rank": rank, "by": "reviewer"}, entry=got["entry"],
               files={k: _rel(v, run_dir) for k, v in got["files"].items()}, _files=got["files"])
    rec.pop("reason", None)
    return rec


# ── CLI ───────────────────────────────────────────────────────────────


def _folders(assets: list[dict]) -> dict[str, str]:
    """A folder name per asset, unique ignoring case (as build_assets does)."""
    used, out = set(), {}
    for a in assets:
        folder = base = build_assets._folder_name(a["name"])
        n = 2
        while folder.lower() in used:
            folder, n = f"{base}_{n}", n + 1
        used.add(folder.lower())
        out[a["name"]] = folder
    return out


def _line(rec: dict) -> str:
    name = rec["name"][:32]
    if rec["status"] == "found":
        c = next(r for r in rec["ranked"] if r["rank"] == rec["chosen"]["rank"])
        saved = rec.get("saved")
        return (f"  found   {name:32} {c['source']}:{c['name']} (rank {c['rank']} of {len(rec['ranked'])}, "
                f"{c['license']}, by {rec['chosen']['by']})"
                + (f" -> library {saved['asset_id']}" if saved else "")
                + (f" (not saved: {rec['save_error']})" if rec.get("save_error") else ""))
    label = {"no_match": "NO MATCH", "needs_review": "REVIEW", "failed": "FAILED"}[rec["status"]]
    return f"  {label:7} {name:32} {rec.get('hits', 0)} hit(s): {rec.get('reason', '')}"


def main(argv: Optional[list[str]] = None, *, call_fn: Optional[llm.CallFn] = None) -> int:
    ap = argparse.ArgumentParser(description="Search open libraries for a plan's search-routed assets.")
    ap.add_argument("run", help="a provision_lab.py run folder, or its provisioning_manifest.json")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="these assets only (lab names)")
    ap.add_argument("--sources", nargs="+", choices=sorted(sources.SOURCES), help="these sources only")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL,
                    help=f"benchmark_config.MODELS key for the vision ranking (default {llm.DEFAULT_MODEL})")
    ap.add_argument("--no-vision", action="store_true",
                    help="rank by text only and pick nothing (no API key needed)")
    ap.add_argument("--redo", action="store_true", help="search again assets already decided")
    ap.add_argument("--choose", nargs="+", metavar="NAME=RANK",
                    help="take this candidate from the saved ranking (no new search)")
    ap.add_argument("--save", action="store_true",
                    help="store what was found in the asset library, with the lab's names as aliases")
    ap.add_argument("--library", help=f"library folder (default: ${library.LIBRARY_ENV}, else "
                                      f"{library.DEFAULT_LIBRARY})")
    args = ap.parse_args(argv)

    def fail(msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr)
        return 2

    path = Path(args.run)
    manifest_path = path / build_assets.MANIFEST if path.is_dir() else path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail(f"cannot read {manifest_path}: {exc}")
    if not isinstance(manifest, dict) or manifest.get("tool") != "provision_lab":
        return fail(f"{manifest_path} is not a provision_lab manifest")
    run_dir = manifest_path.parent
    todo = build_assets.targets(manifest, route=lambda r: r == "search")
    folders = _folders(todo)
    by_key = {library.name_key(a["name"]): a for a in todo}
    if args.only:
        unknown = {library.name_key(n) for n in args.only} - set(by_key)
        if unknown:
            return fail(f"not routed to search in this plan: {', '.join(sorted(unknown))}")
        todo = [a for a in todo if library.name_key(a["name"]) in {library.name_key(n) for n in args.only}]

    results_path = run_dir / RESULTS
    previous = {}
    if results_path.is_file():
        try:
            previous = {library.name_key(a["name"]): a
                        for a in json.loads(results_path.read_text(encoding="utf-8")).get("assets", [])}
        except (OSError, ValueError, AttributeError, KeyError) as exc:
            return fail(f"cannot read {results_path}: {exc}")

    choices = {}
    for c in args.choose or []:
        name, _, rank = c.rpartition("=")
        if not name or not rank.isdigit():
            return fail(f"--choose takes NAME=RANK, not {c!r}")
        key = library.name_key(name)
        if key not in previous or not previous[key].get("ranked"):
            return fail(f"{name} has no saved ranking to choose from (search it first)")
        choices[key] = int(rank)

    store = None
    if args.save:
        try:
            store = library.open_store(args.library)
        except (library.LibraryError, OSError) as exc:
            return fail(f"--save needs an asset library: {exc}")

    try:   # keys for sources such as Poly Pizza live in .env, like the LLM keys
        from dotenv import load_dotenv
        load_dotenv(HERE.parent.parent / ".env")
    except ImportError:
        pass
    chosen_sources = []
    for n in args.sources or sources.SOURCES:
        why = sources.SOURCES[n].unavailable()
        if why and args.sources:
            return fail(f"source {n} can't be used: {why}")
        if why:
            print(f"note: skipping {n}: {why}")
        else:
            chosen_sources.append(sources.SOURCES[n])
    to_search = [a for a in todo if library.name_key(a["name"]) not in choices
                 and (args.redo or previous.get(library.name_key(a["name"]), {}).get("status")
                      in (None, "failed"))]
    run_llm = None
    if to_search and not args.no_vision:
        if call_fn is None:
            try:
                llm.check_environment(args.model)
            except KeyError as exc:
                return fail(exc.args[0])
            except SystemExit as exc:
                return fail(f"environment check failed ({exc.code})")
        run_llm = llm.TrackedLLM(args.model, call_fn=call_fn)

    results = dict(previous)

    def write():
        record = {
            "tool": "search_assets", "manifest": manifest_path.name,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [s.name for s in chosen_sources],
            "vision_model": None if args.no_vision else args.model,
            "llm": run_llm.summary() if run_llm is not None else None,
            "library_after": store.identity() if store is not None else None,
            "counts": {s: sum(r["status"] == s for r in results.values())
                       for s in ("found", "no_match", "needs_review", "failed")},
            "assets": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results.values()],
        }
        library.write_json_atomic(results_path, record)

    print(f"searching {len(to_search)} of {len(todo)} asset(s) routed to search"
          + (f"; {len(choices)} chosen by a reviewer" if choices else "") + "...")
    for asset in todo:
        key = library.name_key(asset["name"])
        folder = folders[asset["name"]]
        if key in choices:
            rec = dict(previous[key])
            try:
                rec = choose(rec, choices[key], run_dir, folder, asset)
            except (ValueError, sources.SourceError) as exc:
                rec["choose_error"] = str(exc)
                print(f"  FAILED  {asset['name'][:32]:32} choosing rank {choices[key]}: {exc}")
                results[key] = rec
                write()
                continue
        elif asset in to_search:
            rec = search_one(asset, run_dir, folder, chosen_sources, run_llm)
        else:
            continue
        if rec["status"] == "found" and store is not None and "_files" in rec:
            build_assets.save(store, rec)
        results[key] = rec
        write()
        print(_line(rec))
    write()
    c = {s: sum(results[library.name_key(a["name"])]["status"] == s for a in todo
                if library.name_key(a["name"]) in results)
         for s in ("found", "no_match", "needs_review", "failed")}
    summary = run_llm.summary() if run_llm is not None else None
    cost = f", ranking ${summary['cost']['total_usd']:.4f}" if summary and summary.get("cost") else ""
    print(f"{c['found']} found, {c['no_match']} no match, {c['needs_review']} for review, "
          f"{c['failed']} failed{cost}; results: {results_path}")
    if summary and summary.get("served_models"):
        llm._benchmark().warn_served_model(run_llm.model_config.model_id, summary["served_models"])
    return 0 if c["found"] == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())

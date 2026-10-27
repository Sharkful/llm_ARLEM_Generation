"""
Narrate a finished lab (#93): one WAV per narrated clip, written to a working
folder and bundled into the lab resources zip (bundle.py, #92).

    python "Code/Narration/narrate_lab.py" <lab_output.json> --provider {openai,gemini}
        [--voice NAME] [--spoken-form none] [--topic NAME] [--bundle-name NAME]
        [--run-group NAME] [--overwrite] [--dry-run]
    python "Code/Narration/narrate_lab.py" --resume <run folder> [--overwrite]

Spec: NARRATION.md. In short:
- The input is a complete json_lab ``Lab`` (one-shot or hierarchical output). It is
  never modified; renames and drops apply to the bundled copy only.
- Each clip with narration becomes ``Audios/<audioClip>.wav``. Missing names are
  generated (``m{module:02d}_{clipName}``), unsafe ones sanitized, clashing ones
  suffixed, and a name on a clip with no narration is dropped. Every change is
  recorded.
- One request per clip, one voice and one style for the lab (the frozen defaults in
  tts.py). A clip longer than the provider's limit fails; it is never split.
- The manifest is rewritten after every clip. ``--resume`` reruns only clips whose
  audio is missing or whose inputs changed (source hash); ``--overwrite`` redoes all.
- The zip is built and validated only when every narrated clip has audio.

Output: Artifacts/Data/Narration/[<run group>/]<input stem>_<provider>_<timestamp>/
Exit codes: 0 complete and valid; 1 incomplete or invalid (resume it); 2 bad input or
environment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
for _p in (HERE, PROJECT_ROOT, PROJECT_ROOT / "Code" / "Schemas", PROJECT_ROOT / "Code" / "Benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bundle  # noqa: E402
import tts  # noqa: E402

NARRATION_ROOT = PROJECT_ROOT / "Artifacts" / "Data" / "Narration"
MANIFEST = "narration_manifest.json"
MANIFEST_SCHEMA = "narration_manifest/1"
AUDIO_FORMAT = {"format": "wav", "sample_rate": bundle.WAV_SAMPLE_RATE,
                "sample_width_bits": 8 * bundle.WAV_SAMPLE_WIDTH, "channels": bundle.WAV_CHANNELS}
SPOKEN_FORM_MODES = ("none",)   # "rules" and "llm" arrive with #94
FATAL_STATUS = {401, 403, 404}  # auth, permission, unknown model: every clip would fail
SHORT_TEXT, LONG_TEXT = 20, 1500  # characters; per-clip warnings only

_AUDIO_EXT = re.compile(r"\.(wav|mp3|ogg|m4a|aac|flac|opus)$", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


class InputError(Exception):
    """Bad input or environment: exit 2 with the message."""


# ── Planning: which clips get audio, under which names ───────────────


@dataclass
class ClipPlan:
    module: int                 # 1-based
    clip: int                   # 1-based within the module
    clip_name: str
    original_name: str | None   # audioClip as written in the input
    name: str | None            # audio file stem in the bundle (None: no audio)
    name_change: str | None     # None | generated | sanitized | deduplicated | dropped
    display_text: str | None    # Clip.narration as written
    action: str                 # narrate | shared | dropped | silent


def _safe(text: str) -> str:
    """A file-safe version of a name: no audio extension, no spaces, and only
    letters, digits, '_' and '-'."""
    return _UNSAFE.sub("_", _AUDIO_EXT.sub("", text.strip())).strip("_")


def plan_clips(lab: dict) -> list[ClipPlan]:
    """Decide every clip's audio name, in module then clip order (NARRATION.md,
    "Clip handling"). The first clip to use a name keeps it."""
    plans: list[ClipPlan] = []
    text_of: dict[str, str] = {}   # audio name -> the narration it holds

    def unique(base: str) -> str:
        n = 2
        while f"{base}_{n}" in text_of:
            n += 1
        return f"{base}_{n}"

    for m, module in enumerate(lab["modules"], 1):
        for c, clip in enumerate(module["clips"], 1):
            clip_name = str(clip.get("clipName") or f"clip{c}")
            raw = clip.get("audioClip")
            raw = raw if isinstance(raw, str) and raw != "" else None
            text = clip.get("narration")
            has_text = isinstance(text, str) and text.strip() != ""

            if not has_text:
                plans.append(ClipPlan(m, c, clip_name, raw, None,
                                      "dropped" if raw else None,
                                      text if isinstance(text, str) else None,
                                      "dropped" if raw else "silent"))
                continue

            change = None
            name = _safe(raw) if raw else ""
            if raw and name != raw:
                change = "sanitized"
            if not name:
                name = f"m{m:02d}_{_safe(clip_name) or f'clip{c}'}"
                change = "generated"
            action = "narrate"
            if name in text_of:
                if text_of[name] == text:
                    action = "shared"          # same words, same file
                else:
                    name, change = unique(name), "deduplicated"
            if action == "narrate":
                text_of[name] = text
            plans.append(ClipPlan(m, c, clip_name, raw, name, change, text, action))
    return plans


def bundled_lab(lab: dict, plans: list[ClipPlan]) -> dict:
    """The lab as it goes into the zip: renamed audioClips set, dropped ones removed."""
    out = copy.deepcopy(lab)
    for p in plans:
        clip = out["modules"][p.module - 1]["clips"][p.clip - 1]
        if p.name_change == "dropped":
            clip.pop("audioClip", None)
        elif p.name is not None and p.name != p.original_name:
            clip["audioClip"] = p.name
    return out


def spoken_text(display_text: str, mode: str) -> str:
    """The text sent to the voice service. ``none`` sends the narration as written
    (trimmed); the rules and llm modes are #94."""
    if mode != "none":
        raise InputError(f"--spoken-form {mode} is not available yet (#94)")
    return display_text.strip()


def source_hash(text: str, provider: str, model: str, voice: str, style: str) -> str:
    payload = {"text": text, "provider": provider, "model": model, "voice": voice,
               "style": style, **AUDIO_FORMAT}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ── Input, environment, provenance ────────────────────────────────────


def load_lab(path: Path) -> dict:
    try:
        lab = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {exc}")
    if not isinstance(lab, dict) or "modules" not in lab:
        raise InputError(
            f"{path.name} is not a full json_lab Lab (no 'modules'). L1 outlines, "
            "module-only outputs and ARLEM specs aren't supported."
        )
    from json_lab import Lab
    try:
        Lab.model_validate(lab)
    except Exception as exc:  # pydantic.ValidationError
        raise InputError(f"{path.name} is not a valid json_lab Lab: "
                         + " ".join(str(exc).splitlines()[:3]))
    return lab


def paired_metrics(path: Path) -> dict | None:
    """The benchmark/hierarchical metrics record for an ``*_output.json``, if any:
    beside it, or in the sibling Metrics/ folder."""
    if not path.name.endswith("_output.json"):
        return None
    name = path.name[: -len("_output.json")] + "_metrics.json"
    for cand in (path.with_name(name), path.parent.parent / "Metrics" / name):
        if cand.is_file():
            try:
                return {"_path": cand, **json.loads(cand.read_text(encoding="utf-8"))}
            except (OSError, ValueError):
                return None
    return None


def check_environment(spec: tts.ProviderSpec) -> str:
    """The API key, after checking the SDK and key exist (InputError otherwise)."""
    try:
        found = importlib.util.find_spec(spec.sdk_module) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        raise InputError(
            f"Python package '{spec.sdk_package}' is not installed ({spec.name} needs it). "
            "Activate the project venv and run: pip install -r requirements.txt"
        )
    for var in spec.api_key_env:
        if os.environ.get(var):
            return os.environ[var]
    raise InputError(f"no API key for {spec.name}: set {' or '.join(spec.api_key_env)} in .env")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    packages = {}
    for pkg in ("openai", "google-genai"):
        try:
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = None
    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD") or None,
            "git_dirty": bool(status) if status is not None else None,
            "packages": packages, "python": sys.version.split()[0]}


def write_json_atomic(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── The run ───────────────────────────────────────────────────────────


@dataclass
class Settings:
    input_path: Path
    provider: str
    voice: str
    style: str
    spoken_form: str
    topic: str | None
    bundle_name: str | None


def _bundle_names(lab: dict, settings: Settings) -> bundle.BundleNames:
    try:
        return bundle.bundle_names(lab.get("labId", ""), settings.bundle_name)
    except ValueError as exc:
        raise InputError(f"{exc} (--bundle-name NAME)") from exc


def _served_ok(served: str | None, model: str) -> bool:
    return served is None or served == model or served.startswith(model + "-")


def _is_fatal(exc: BaseException) -> bool:
    from benchmark import exception_chain, status_of
    return isinstance(exc, tts.TTSFormatError) or any(
        status_of(e) in FATAL_STATUS for e in exception_chain(exc))


def _clip_warnings(text: str) -> list[str]:
    if len(text) < SHORT_TEXT:
        return [f"very short narration ({len(text)} characters)"]
    if len(text) > LONG_TEXT:
        return [f"very long narration ({len(text)} characters)"]
    return []


def run(settings: Settings, run_dir: Path, *, overwrite: bool = False, provider=None,
        api_key: str | None = None, log=print) -> int:
    """Narrate ``settings.input_path`` into ``run_dir``. ``provider`` (anything with
    ``synthesize(text) -> tts.Synthesis``) replaces the real client in tests."""
    from benchmark import _transient_retryer
    from tracking.metrics import TokenUsage
    from tracking.pricing import PricingCalculator

    spec = tts.PROVIDERS[settings.provider]
    lab = load_lab(settings.input_path)
    plans = plan_clips(lab)
    names = _bundle_names(lab, settings)
    metrics = paired_metrics(settings.input_path) or {}
    topic = settings.topic or metrics.get("lab_name")

    run_dir.mkdir(parents=True, exist_ok=True)
    folder = bundle.create_layout(run_dir, names)
    audio_dir = folder / bundle.AUDIO_DIR
    manifest_path = run_dir / MANIFEST
    previous = {}
    old = None
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = {c["audio_name"]: c for c in old.get("clips", [])
                    if c.get("audio_name") and c.get("status") in ("generated", "skipped")}

    write_json_atomic(folder / names.json_file, bundled_lab(lab, plans))

    if provider is None:
        provider = tts.make_provider(settings.provider, settings.voice, settings.style,
                                     api_key=api_key)
    pricing = PricingCalculator(model=spec.model)

    started = _now()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "source": {
            "input_path": _rel(settings.input_path),
            "labId": lab.get("labId"),
            "source_model": metrics.get("model"),
            "level": metrics.get("level"),
            "structure": metrics.get("structure"),
            "lab_name": metrics.get("lab_name"),
            "metrics_file": _rel(metrics["_path"]) if metrics.get("_path") else None,
            "topic": topic,
        },
        "bundle": {"folder": names.folder, "zip_file": names.zip_file,
                   "json_file": names.json_file, "bundle_name_override": settings.bundle_name,
                   "validation": None},
        "voice_service": {"provider": spec.name, "model": spec.model, "served_models": [],
                          "voice": settings.voice, "style_instruction": settings.style,
                          **AUDIO_FORMAT},
        "spoken_form": {"mode": settings.spoken_form, "model": None, "glossary_version": None},
        "counts": {},
        "totals": {},
        "runs": (old or {}).get("runs", []) + [{"started_at": started, "finished_at": None,
                                                "overwrite": overwrite}],
        "provenance": provenance(),
        "warnings": [],
        "clips": [],
    }

    records: list[dict] = []
    fatal = False

    def save():
        manifest["clips"] = records + [_plan_record(p) for p in plans[len(records):]]
        _fill_counts(manifest)
        write_json_atomic(manifest_path, manifest)

    for p in plans:
        rec = _plan_record(p)
        records.append(rec)
        if p.action != "narrate":
            if p.action == "shared":
                rec["status"] = "shared"
            save()
            continue
        text = spoken_text(p.display_text, settings.spoken_form)
        h = source_hash(text, spec.name, spec.model, settings.voice, settings.style)
        rec.update(spoken_text=text, characters=len(text), source_hash=h,
                   warnings=_clip_warnings(text))
        wav = audio_dir / f"{p.name}.wav"
        prev = previous.get(p.name)

        if fatal:
            rec.update(status="failed", error="not attempted: the run stopped on an earlier error")
        elif not overwrite and prev and prev.get("source_hash") == h and wav.is_file():
            rec.update({k: prev.get(k) for k in ("audio_seconds", "input_tokens", "output_tokens",
                                                  "cost_usd", "wall_seconds", "served_model",
                                                  "transient_retries", "generated_at")})
            rec["status"] = "skipped"
            log(f"  skip  {p.name} (unchanged)")
        elif len(text) > spec.max_chars:
            rec.update(status="failed",
                       error=f"{len(text)} characters is over the {spec.name} limit of "
                             f"{spec.max_chars}; clips are never split")
            log(f"  FAIL  {p.name}: too long ({len(text)} characters)")
        else:
            retryer = _transient_retryer()
            t0 = time.monotonic()
            try:
                syn = retryer(provider.synthesize, text)
                bundle.write_wav(wav, syn.pcm)
                cost = None
                if syn.input_tokens is not None and syn.output_tokens is not None:
                    cost = pricing.calculate_cost(TokenUsage(
                        prompt_tokens=syn.input_tokens, completion_tokens=syn.output_tokens,
                        total_tokens=syn.input_tokens + syn.output_tokens))
                rec.update(status="generated", audio_seconds=round(syn.seconds, 3),
                           input_tokens=syn.input_tokens, output_tokens=syn.output_tokens,
                           cost_usd=cost, served_model=syn.served_model, generated_at=_now())
                if syn.seconds == 0:
                    rec["warnings"].append("the provider returned no audio")
                if cost is None:
                    rec["warnings"].append("the provider reported no token usage, so no cost")
                log(f"  ok    {p.name} ({syn.seconds:.1f} s audio"
                    + (f", ${cost:.4f}" if cost is not None else "") + ")")
            except Exception as exc:  # noqa: BLE001 - recorded per clip, run continues
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                log(f"  FAIL  {p.name}: {type(exc).__name__}: {exc}")
                if _is_fatal(exc):
                    fatal = True
                    manifest["warnings"].append(
                        f"stopped at {p.name}: {type(exc).__name__} means every clip would fail "
                        "(authentication, permission, unknown model or unusable audio)")
            rec["wall_seconds"] = round(time.monotonic() - t0, 3)
            rec["transient_retries"] = (retryer.statistics.get("attempt_number", 1) or 1) - 1
        save()

    served = sorted({r["served_model"] for r in records if r.get("served_model")})
    manifest["voice_service"]["served_models"] = served
    for s in served:
        if not _served_ok(s, spec.model):
            manifest["warnings"].append(f"served by {s}, not the requested {spec.model}")

    failed = [r for r in records if r["status"] == "failed"]
    if failed:
        manifest["status"] = "failed" if fatal else "incomplete"
        # A zip from an earlier run no longer matches the folder: don't leave it to ship.
        (run_dir / names.zip_file).unlink(missing_ok=True)
        log(f"\n{len(failed)} clip(s) failed; no zip built. Fix the cause, then rerun with:\n"
            f'  python "Code/Narration/narrate_lab.py" --resume "{_rel(run_dir)}"')
    else:
        # Audio this tool wrote for names the lab no longer uses (e.g. an edited input).
        planned = {p.name for p in plans if p.name}
        for f in sorted(audio_dir.glob("*.wav")):
            if f.stem not in planned:
                f.unlink()
                manifest["warnings"].append(f"removed {f.name}: no clip uses it any more")
        zip_path = bundle.zip_bundle(folder)
        report = bundle.validate_bundle(zip_path)
        manifest["bundle"]["validation"] = {"ok": report.ok, "errors": report.errors,
                                            "warnings": report.warnings}
        manifest["status"] = "complete" if report.ok else "invalid"
        log(f"\n{'OK  ' if report.ok else 'FAIL'} {_rel(zip_path)}")
        for e in report.errors:
            log(f"  error:   {e}")
        for w in report.warnings:
            log(f"  warning: {w}")
    manifest["runs"][-1]["finished_at"] = _now()
    save()
    for w in manifest["warnings"]:
        log(f"  WARNING: {w}")
    t = manifest["totals"]
    log(f"{manifest['counts'].get('with_audio', 0)} clips with audio, "
        f"{t.get('audio_seconds', 0):.0f} s, ${t.get('cost_usd', 0):.4f} "
        f"(this run ${t.get('cost_usd_this_run', 0):.4f}); manifest: {_rel(manifest_path)}")
    return 0 if manifest["status"] == "complete" else 1


def _plan_record(p: ClipPlan) -> dict:
    return {
        "module": p.module, "clip": p.clip, "clip_name": p.clip_name,
        "audio_name": p.name, "original_audio_name": p.original_name,
        "name_change": p.name_change, "display_text": p.display_text,
        "spoken_text": None, "characters": None,
        "status": {"silent": "silent", "dropped": "dropped"}.get(p.action, "pending"),
        "audio_seconds": None, "input_tokens": None, "output_tokens": None,
        "cost_usd": None, "wall_seconds": None, "served_model": None,
        "transient_retries": None, "generated_at": None, "source_hash": None,
        "error": None, "warnings": [],
    }


def _fill_counts(manifest: dict) -> None:
    clips = manifest["clips"]
    by = {s: sum(c["status"] == s for c in clips)
          for s in ("generated", "skipped", "shared", "failed", "silent", "dropped", "pending")}
    changes = {k: sum(c["name_change"] == k for c in clips)
               for k in ("generated", "sanitized", "deduplicated", "dropped")}
    audio = [c for c in clips if c["status"] in ("generated", "skipped")]
    this_run = [c for c in clips if c["status"] == "generated"]
    total = lambda rows, k: sum(c.get(k) or 0 for c in rows)  # noqa: E731  (unrounded)
    manifest["counts"] = {**by, "clips": len(clips),
                          "with_audio": by["generated"] + by["skipped"] + by["shared"],
                          "names": changes}
    manifest["totals"] = {
        "characters": sum(c.get("characters") or 0 for c in audio),
        "audio_seconds": round(total(audio, "audio_seconds"), 3),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in audio),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in audio),
        "cost_usd": total(audio, "cost_usd"),
        "cost_usd_this_run": total(this_run, "cost_usd"),
        "wall_seconds": round(total(audio, "wall_seconds"), 3),
        "transient_retries": sum(c.get("transient_retries") or 0 for c in clips),
        "spoken_form_cost_usd": 0.0,
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _dry_run(settings: Settings) -> int:
    spec = tts.PROVIDERS[settings.provider]
    lab = load_lab(settings.input_path)
    plans = plan_clips(lab)
    names = _bundle_names(lab, settings)
    narrate = [p for p in plans if p.action == "narrate"]
    chars = sum(len(spoken_text(p.display_text, settings.spoken_form)) for p in narrate)
    too_long = [p.name for p in narrate
                if len(spoken_text(p.display_text, settings.spoken_form)) > spec.max_chars]
    print(f"{names.zip_file}: {len(plans)} clips, {len(narrate)} to narrate with "
          f"{spec.model} / {settings.voice}")
    for p in plans:
        change = f"  [{p.name_change}]" if p.name_change else ""
        print(f"  m{p.module:02d} {p.clip_name:<12} {p.action:<8} {p.name or '-'}{change}")
    print(f"{chars} characters, roughly {chars / 15 / 60:.1f} min of audio. No requests sent.")
    if too_long:
        print(f"Over the {spec.max_chars}-character limit (would fail): {too_long}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Narrate a finished lab into its resources zip (#93).")
    ap.add_argument("lab", nargs="?", type=Path, help="a json_lab *_output.json")
    ap.add_argument("--provider", choices=sorted(tts.PROVIDERS))
    ap.add_argument("--voice", help="override the provider's frozen default voice")
    ap.add_argument("--spoken-form", choices=SPOKEN_FORM_MODES, default="none",
                    help="text sent to the voice service (rules/llm: #94)")
    ap.add_argument("--topic", help="lab YAML topic name (default: from the paired metrics file)")
    ap.add_argument("--bundle-name", help="zip/folder name, e.g. to match the server's lab id")
    ap.add_argument("--run-group", help="Artifacts/Data/Narration/<run group>/")
    ap.add_argument("--resume", type=Path, help="an existing run folder to finish or redo")
    ap.add_argument("--overwrite", action="store_true", help="regenerate every clip")
    ap.add_argument("--dry-run", action="store_true", help="show the plan; send nothing")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        if args.resume:
            given = [f"--{k.replace('_', '-')}" for k in
                     ("lab", "provider", "voice", "topic", "bundle_name", "run_group")
                     if getattr(args, k)]
            if given or args.spoken_form != "none" or args.dry_run:
                raise InputError(f"--resume takes its settings from the manifest; drop "
                                 f"{', '.join(given) or '--spoken-form/--dry-run'}")
            run_dir = args.resume
            path = run_dir / MANIFEST
            if not path.is_file():
                raise InputError(f"no {MANIFEST} in {run_dir}")
            m = json.loads(path.read_text(encoding="utf-8"))
            settings = Settings(
                input_path=PROJECT_ROOT / m["source"]["input_path"],
                provider=m["voice_service"]["provider"], voice=m["voice_service"]["voice"],
                style=m["voice_service"]["style_instruction"], spoken_form=m["spoken_form"]["mode"],
                topic=m["source"].get("topic"),
                bundle_name=m["bundle"].get("bundle_name_override"))
        else:
            if not args.lab or not args.provider:
                raise InputError("give a lab file and --provider (or --resume <run folder>)")
            spec = tts.PROVIDERS[args.provider]
            settings = Settings(input_path=args.lab, provider=args.provider,
                                voice=args.voice or spec.voice, style=tts.STYLE_INSTRUCTION,
                                spoken_form=args.spoken_form, topic=args.topic,
                                bundle_name=args.bundle_name)
            if args.dry_run:
                return _dry_run(settings)
            stem = args.lab.name.removesuffix(".json").removesuffix("_output")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = NARRATION_ROOT / args.run_group if args.run_group else NARRATION_ROOT
            run_dir = base / f"{stem}_{args.provider}_{stamp}"
        api_key = check_environment(tts.PROVIDERS[settings.provider])
        load_lab(settings.input_path)  # fail before creating a run folder
        print(f"Narrating {_rel(settings.input_path)} -> {_rel(run_dir)}")
        return run(settings, run_dir, overwrite=args.overwrite, api_key=api_key)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

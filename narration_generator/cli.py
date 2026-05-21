"""
AR Lab Narration Generator CLI

Usage (run from the narration_generator/ directory):
    python cli.py run "description text" [options]
    python cli.py run --file description.txt [options]
    python cli.py list
    python cli.py status <project_id>
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure narration_generator/ is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

import config
from models.log_models import SystemEntry, ValidationEntry
from models.manifest_models import AudioClipRecord, AudioManifest, ProjectManifest
from narration_definitions import TTSRuntimeConfig
from pipeline.brief_generator import generate_scene_brief
from pipeline.cost_tracker import CostTracker
from pipeline.filename_utils import enumerate_clips
from pipeline.script_generator import generate_narration_script
from pipeline.staleness import compute_clip_hash
from pipeline.tts_generator import OpenAITTSProvider, TTSGenerator
from pipeline.validation import validate_narration_script
from storage.project_storage import ProjectStorage


def _make_project_id(title: str | None) -> str:
    import re
    slug = ""
    if title:
        slug = re.sub(r"[^\w]+", "_", title.lower().strip())[:30].strip("_") + "_"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{slug}{ts}"


def _build_price_config() -> dict:
    return {"claude": config.CLAUDE_PRICES, "tts": config.OPENAI_TTS_PRICES}


def cmd_run(args: argparse.Namespace) -> int:
    if args.file:
        description = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        description = args.description.strip()

    if not description:
        print("Error: description is empty.", file=sys.stderr)
        return 1

    if not config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 1

    if not args.dry_run and not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set (required unless --dry-run).", file=sys.stderr)
        return 1

    project_id = _make_project_id(args.title)
    storage = ProjectStorage(config.PROJECTS_DIR / project_id)

    llm_model = args.model or config.DEFAULT_LLM_MODEL
    tts_model = args.tts_model or config.DEFAULT_TTS_MODEL
    voice = args.voice or config.DEFAULT_VOICE

    manifest = ProjectManifest(
        project_id=project_id,
        project_title=args.title,
        llm_model=llm_model,
        tts_provider=config.DEFAULT_TTS_PROVIDER,
        tts_model=tts_model,
        default_voice=voice,
    )
    storage.save_manifest(manifest)
    storage.save_description(description)
    storage.append_log_entry(SystemEntry(message=f"Project created: {project_id}"))

    print(f"\nProject: {project_id}")
    print(f"Output:  {storage.project_dir}\n")

    cost_tracker = CostTracker(_build_price_config())

    # Stage 1: Scene brief
    print("Generating scene brief...", end="", flush=True)
    t0 = time.monotonic()
    brief, brief_log = generate_scene_brief(description, llm_model, config.ANTHROPIC_API_KEY)
    print(f" done ({time.monotonic()-t0:.1f}s, {brief_log.input_tokens}+{brief_log.output_tokens} tokens)")
    storage.save_brief(brief)
    storage.append_log_entry(brief_log)
    cost_tracker._llm_entries.append(brief_log)
    manifest.status = "brief_generated"
    manifest.updated_at = datetime.utcnow()
    storage.save_manifest(manifest)

    # Stage 2: Narration script
    runtime_config = TTSRuntimeConfig(allowed_voices=config.OPENAI_TTS_VOICES, default_voice=voice)
    print("Generating narration script...", end="", flush=True)
    t0 = time.monotonic()
    script, script_log = generate_narration_script(
        brief, runtime_config, llm_model, config.ANTHROPIC_API_KEY
    )
    print(f" done ({time.monotonic()-t0:.1f}s, {script_log.input_tokens}+{script_log.output_tokens} tokens)")
    storage.save_script(script)
    storage.append_log_entry(script_log)
    cost_tracker._llm_entries.append(script_log)
    manifest.status = "script_generated"
    manifest.updated_at = datetime.utcnow()
    storage.save_manifest(manifest)

    # Validation
    validation_results = validate_narration_script(script, runtime_config)
    for r in validation_results:
        storage.append_log_entry(ValidationEntry(level=r.level, message=r.message, clip_ref=r.clip_ref))
    errors = [r for r in validation_results if r.level == "error"]
    warnings = [r for r in validation_results if r.level == "warning"]
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for e in errors:
            ref = f" [{e.clip_ref}]" if e.clip_ref else ""
            print(f"  ERROR{ref}: {e.message}")
        return 1
    if warnings:
        print(f"  {len(warnings)} warning(s) logged.")

    if args.dry_run:
        print("\n[dry-run] Skipping TTS. Narration script saved to:")
        print(f"  {storage.project_dir / 'narration_script.json'}")
        _print_script_summary(script)
        return 0

    # Stage 3: TTS
    provider = OpenAITTSProvider(api_key=config.OPENAI_API_KEY, model=tts_model)
    generator = TTSGenerator(provider, default_voice=voice)
    clips_list = enumerate_clips(script)
    audio_records: list[AudioClipRecord] = []

    print(f"\nGenerating audio for {len(clips_list)} clip(s)...")
    for m_idx, c_idx, clip, filename in clips_list:
        clip_ref = f"m{m_idx:02d}_c{c_idx:03d}"
        effective_voice = clip.voice or script.default_voice or voice
        char_count = generator.count_characters(clip)
        src_hash = compute_clip_hash(clip, effective_voice, tts_model)

        print(f"  {clip_ref} {filename:<45}", end="", flush=True)
        t0 = time.monotonic()
        try:
            audio = generator.generate_clip_audio(clip, effective_voice=effective_voice, style=clip.style)
            duration = generator.save_clip(audio, storage.audio_dir / filename)
            elapsed = time.monotonic() - t0
            tts_log = cost_tracker.record_tts_call(
                provider=provider.provider_name, model=provider.model_name,
                voice=effective_voice, character_count=char_count,
                clip_ref=clip_ref, duration_seconds=round(elapsed, 3), success=True,
            )
            storage.append_log_entry(tts_log)
            record = AudioClipRecord(
                module_index=m_idx, clip_index=c_idx, clip_title=clip.title,
                filename=filename, voice=effective_voice, tts_model=tts_model,
                source_hash=src_hash, status="generated",
                generated_at=datetime.utcnow(), duration_seconds=round(duration, 3),
                character_count=char_count,
            )
            print(f" {duration:.1f}s")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            tts_log = cost_tracker.record_tts_call(
                provider=provider.provider_name, model=provider.model_name,
                voice=effective_voice, character_count=char_count,
                clip_ref=clip_ref, duration_seconds=round(elapsed, 3),
                success=False, error_message=str(exc),
            )
            storage.append_log_entry(tts_log)
            record = AudioClipRecord(
                module_index=m_idx, clip_index=c_idx, clip_title=clip.title,
                filename=filename, voice=effective_voice, tts_model=tts_model,
                source_hash=src_hash, status="failed",
                character_count=char_count, error_message=str(exc),
            )
            print(f" FAILED: {exc}")
        audio_records.append(record)

    storage.save_audio_manifest(AudioManifest(
        project_id=project_id, tts_provider=config.DEFAULT_TTS_PROVIDER, clips=audio_records,
    ))
    manifest.status = "audio_generated"
    manifest.updated_at = datetime.utcnow()
    storage.save_manifest(manifest)
    cost_report = cost_tracker.build_cost_report(project_id)
    storage.save_cost_report(cost_report)

    generated = sum(1 for r in audio_records if r.status == "generated")
    failed = sum(1 for r in audio_records if r.status == "failed")
    print(f"\nDone: {generated} clip(s) generated, {failed} failed")
    print(f"Total cost: ${cost_report.total_cost_usd:.4f}"
          f" (LLM ${cost_report.total_llm_cost_usd:.4f} + TTS ${cost_report.total_tts_cost_usd:.4f})")
    print(f"Audio:  {storage.audio_dir}")
    return 0 if failed == 0 else 1


def _print_script_summary(script) -> None:
    total_clips = sum(len(m.clips) for m in script.modules)
    print(f"\nScript: {len(script.modules)} module(s), {total_clips} clip(s)")
    for m_i, module in enumerate(script.modules, start=1):
        print(f"  Module {m_i}: {module.title or f'Module {m_i}'} ({len(module.clips)} clips)")
        for c_i, clip in enumerate(module.clips, start=1):
            chars = sum(len(s.text) for s in clip.segments if s.type == "say")
            print(f"    {c_i}. {clip.title or f'Clip {c_i}'}  ({chars} chars)")


def cmd_list(args: argparse.Namespace) -> int:
    projects = sorted(config.PROJECTS_DIR.iterdir()) if config.PROJECTS_DIR.exists() else []
    if not projects:
        print("No projects found.")
        return 0
    print(f"{'Project ID':<45} {'Status':<20} {'Created'}")
    print("-" * 90)
    for p in projects:
        mp = p / "project_manifest.json"
        if mp.exists():
            m = ProjectManifest.model_validate_json(mp.read_text(encoding="utf-8"))
            print(f"{p.name:<45} {m.status:<20} {m.created_at.strftime('%Y-%m-%d %H:%M')}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    project_dir = config.PROJECTS_DIR / args.project_id
    if not project_dir.exists():
        print(f"Project not found: {args.project_id}", file=sys.stderr)
        return 1
    storage = ProjectStorage(project_dir)
    if not storage.manifest_exists():
        print(f"No manifest for project: {args.project_id}", file=sys.stderr)
        return 1
    m = storage.load_manifest()
    print(f"Project:  {m.project_id}")
    if m.project_title:
        print(f"Title:    {m.project_title}")
    print(f"Status:   {m.status}")
    print(f"LLM:      {m.llm_model}")
    print(f"TTS:      {m.tts_provider} / {m.tts_model} / {m.default_voice}")
    print(f"Created:  {m.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if storage.audio_manifest_exists():
        audio = storage.load_audio_manifest()
        print(f"Audio:    {sum(1 for c in audio.clips if c.status=='generated')} generated, "
              f"{sum(1 for c in audio.clips if c.status=='failed')} failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python cli.py", description="AR Lab Narration Generator")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Generate narration and audio from a description")
    run_p.add_argument("description", nargs="?", help="Lab description text")
    run_p.add_argument("--file", "-f", help="Read description from file")
    run_p.add_argument("--title", "-t", help="Project title")
    run_p.add_argument("--model", help=f"LLM model (default: {config.DEFAULT_LLM_MODEL})")
    run_p.add_argument("--voice", help=f"Default TTS voice (default: {config.DEFAULT_VOICE})")
    run_p.add_argument("--tts-model", dest="tts_model", help=f"TTS model (default: {config.DEFAULT_TTS_MODEL})")
    run_p.add_argument("--dry-run", action="store_true", help="Skip TTS, output script only")

    sub.add_parser("list", help="List all projects")

    status_p = sub.add_parser("status", help="Show project status")
    status_p.add_argument("project_id")

    args = parser.parse_args()

    if args.command == "run":
        if not args.description and not args.file:
            run_p.error("Provide description text or --file")
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "status":
        return cmd_status(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

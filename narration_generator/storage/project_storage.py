from __future__ import annotations

from pathlib import Path
from typing import Annotated, Union

from pydantic import Field

from models.cost_models import CostReport
from models.log_models import (
    LLMCallEntry,
    SystemEntry,
    TTSCallEntry,
    ValidationEntry,
)
from models.manifest_models import AudioManifest, ProjectManifest
from narration_definitions import ARLabNarrationScript
from script_definitions import ARSceneNarrationBrief

_LOG_ADAPTER_TYPE = Annotated[
    Union[LLMCallEntry, TTSCallEntry, ValidationEntry, SystemEntry],
    Field(discriminator="event_type"),
]


class ProjectStorage:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.audio_dir = project_dir / "audio"
        project_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Description ---

    def save_description(self, text: str) -> None:
        (self.project_dir / "description.txt").write_text(text, encoding="utf-8")

    def load_description(self) -> str:
        return (self.project_dir / "description.txt").read_text(encoding="utf-8")

    # --- Scene Brief ---

    def save_brief(self, brief: ARSceneNarrationBrief) -> None:
        (self.project_dir / "scene_brief.json").write_text(
            brief.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_brief(self) -> ARSceneNarrationBrief:
        text = (self.project_dir / "scene_brief.json").read_text(encoding="utf-8")
        return ARSceneNarrationBrief.model_validate_json(text)

    # --- Narration Script ---

    def save_script(self, script: ARLabNarrationScript) -> None:
        (self.project_dir / "narration_script.json").write_text(
            script.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_script(self) -> ARLabNarrationScript:
        text = (self.project_dir / "narration_script.json").read_text(encoding="utf-8")
        return ARLabNarrationScript.model_validate_json(text)

    # --- Project Manifest ---

    def save_manifest(self, manifest: ProjectManifest) -> None:
        (self.project_dir / "project_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_manifest(self) -> ProjectManifest:
        text = (self.project_dir / "project_manifest.json").read_text(encoding="utf-8")
        return ProjectManifest.model_validate_json(text)

    def manifest_exists(self) -> bool:
        return (self.project_dir / "project_manifest.json").exists()

    # --- Audio Manifest ---

    def save_audio_manifest(self, manifest: AudioManifest) -> None:
        (self.project_dir / "audio_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_audio_manifest(self) -> AudioManifest:
        text = (self.project_dir / "audio_manifest.json").read_text(encoding="utf-8")
        return AudioManifest.model_validate_json(text)

    def audio_manifest_exists(self) -> bool:
        return (self.project_dir / "audio_manifest.json").exists()

    # --- Generation Log (JSONL, append-only) ---

    def append_log_entry(self, entry: LLMCallEntry | TTSCallEntry | ValidationEntry | SystemEntry) -> None:
        log_path = self.project_dir / "generation_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def load_log(self) -> list:
        log_path = self.project_dir / "generation_log.jsonl"
        if not log_path.exists():
            return []
        from pydantic import TypeAdapter
        adapter = TypeAdapter(_LOG_ADAPTER_TYPE)
        entries = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(adapter.validate_json(line))
        return entries

    # --- Cost Report ---

    def save_cost_report(self, report: CostReport) -> None:
        (self.project_dir / "cost_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_cost_report(self) -> CostReport:
        text = (self.project_dir / "cost_report.json").read_text(encoding="utf-8")
        return CostReport.model_validate_json(text)

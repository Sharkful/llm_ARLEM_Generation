"""
Lab Metrics Analyzer

Extracts structural metrics from generated AR lab JSON output
for benchmarking comparison across models.
"""

from typing import Any


# ── Known (moon-lab) asset library ────────────────────────────────────
# The only assets that actually exist on the headset today. Anything an LLM
# references outside these sets is a *novel* asset it invented and that we would
# have to author to realize the generated lab. Kept lowercased; membership is
# tested case-insensitively so trivial casing drift ("TextPrefab") is not
# mis-counted as novel. Source of truth: the field descriptions in
# Code/Tools/json_lab.py (SceneObject.prefab / .texture) — keep these in sync.
KNOWN_TEXTURES = {"2k_earth_daymap", "2k_moon", "2k_sun", "balldimpled"}
KNOWN_PREFABS = {
    "sunprefab", "moveablesphere", "clickablesphere", "tinysphere",
    "textprefab", "robotidle",
    # System prefab auto-filled on every DemoModule (SkipJsonSchema default); it
    # is not an LLM-invented asset, so it must never count as novel.
    "demoprefab",
}


def _is_novel(name: str | None, known: set[str]) -> bool:
    """True if `name` is a non-empty asset reference outside the known set."""
    if not name or not name.strip():
        return False
    return name.strip().lower() not in known


def _iter_components(module: dict):
    """Yield every component dict in a module, from both initial objects and the
    sparse clip deltas (audio can be introduced via CheckAngleComponent in either
    place)."""
    for obj in module.get("objects", []):
        yield from obj.get("components") or []
    for clip in module.get("clips", []):
        for change in clip.get("changes") or []:
            yield from change.get("components") or []


def analyze_assets(lab_json: dict) -> dict:
    """Count the assets a generated lab references, separating *novel* assets
    (invented by the model — not in the moon-lab library) from reuse of known
    ones, plus audio (no known audio library exists, so audio is reported as
    volume/density, not novelty).

    Walks the same module-normalized structure as ``analyze_json_lab`` so it works
    across Lab / single-module / module-only outputs and returns all-zero for L1
    ``LabOutline`` outputs. Asset reference points are bounded:
      - prefab/texture: only ``SceneObject`` carries them (``ObjectChange`` cannot
        introduce new visual assets — it has no prefab/texture field);
      - audio: ``Clip.audioClip`` plus ``CheckAngleComponent.audioClipSuccess`` in
        either an object's components or a clip delta's components.
    Returns per-lab counts; novel names are deduped across the whole lab (a prefab
    reused in three modules is one asset to build).
    """
    if "modules" in lab_json:
        modules = lab_json.get("modules", [])
    else:
        is_module = lab_json.get("moduleType") == "demo" or "clips" in lab_json
        modules = [lab_json] if is_module else []

    novel_textures: set[str] = set()
    novel_prefabs: set[str] = set()
    audio_names: set[str] = set()
    texture_refs = prefab_refs = audio_refs = 0
    novel_texture_refs = novel_prefab_refs = 0

    for module in modules:
        for obj in module.get("objects", []):
            prefab = obj.get("prefab")
            if prefab and prefab.strip():
                prefab_refs += 1
                if _is_novel(prefab, KNOWN_PREFABS):
                    novel_prefabs.add(prefab.strip())
                    novel_prefab_refs += 1
            texture = obj.get("texture")
            if texture and texture.strip():
                texture_refs += 1
                if _is_novel(texture, KNOWN_TEXTURES):
                    novel_textures.add(texture.strip())
                    novel_texture_refs += 1

        for clip in module.get("clips", []):
            audio = clip.get("audioClip")
            if audio and audio.strip():
                audio_refs += 1
                audio_names.add(audio.strip())

        for comp in _iter_components(module):
            audio = comp.get("audioClipSuccess")
            if audio and audio.strip():
                audio_refs += 1
                audio_names.add(audio.strip())

    return {
        # Headline: distinct invented assets we'd have to author for this lab.
        "novel_textures": len(novel_textures),
        "novel_prefabs": len(novel_prefabs),
        # Reuse intensity: object-instances that point at a novel asset.
        "novel_texture_refs": novel_texture_refs,
        "novel_prefab_refs": novel_prefab_refs,
        # Denominators / context: total references regardless of novelty.
        "texture_refs": texture_refs,
        "prefab_refs": prefab_refs,
        # Audio has no known library — report volume, not novelty.
        "audio_refs": audio_refs,
        "audio_unique": len(audio_names),
        # Kept for qualitative spot-checking; not flattened into the CSV.
        "novel_texture_names": sorted(novel_textures),
        "novel_prefab_names": sorted(novel_prefabs),
    }


def analyze_json_lab(lab_json: dict) -> dict:
    """
    Analyze a generated JSON Lab (v2.0 spec) and return structural metrics.

    Expects the output of Lab.model_dump(mode="json", exclude_none=True).
    """
    metrics = {}

    # Normalize the module list. A Lab wrapper (single- or multi-module) carries a
    # "modules" key; a "module-only" output is a bare DemoModule dump with no wrapper
    # (top-level moduleType/objects/clips). Treat the bare module as a one-element list
    # so the rest of the analysis is identical across all three structures.
    if "modules" in lab_json:
        modules = lab_json.get("modules", [])
        metrics["lab_id"] = lab_json.get("labId", "")
        metrics["num_objectives"] = len(lab_json.get("objectives", []))
    else:
        is_module = lab_json.get("moduleType") == "demo" or "clips" in lab_json
        modules = [lab_json] if is_module else []  # L1 LabOutline -> [] (stays all-zero)
        metrics["lab_id"] = ""
        metrics["num_objectives"] = 0
    metrics["num_modules"] = len(modules)

    # Aggregate across modules (1 for single/module-only, N for multi-module)
    total_objects = 0
    total_clips = 0
    total_components = 0
    total_changes = 0
    total_text_labels = 0
    total_edu_objectives = 0
    object_names = set()

    for module in modules:
        objects = module.get("objects", [])
        clips = module.get("clips", [])
        total_objects += len(objects)
        total_clips += len(clips)
        total_edu_objectives += len(module.get("educationalObjectives", []))

        # module_type is uniform ("demo"); take it from the first module
        metrics.setdefault("module_type", module.get("moduleType", ""))

        # Object analysis
        for obj in objects:
            object_names.add(obj.get("name", ""))
            components = obj.get("components", [])
            total_components += len(components)
            for comp in components:
                if comp.get("componentType") == "textMeshPro":
                    total_text_labels += 1

        # Clip analysis
        for clip in clips:
            changes = clip.get("changes", [])
            if changes:
                total_changes += len(changes)

    metrics["num_objects"] = total_objects
    metrics["num_clips"] = total_clips
    metrics["num_components"] = total_components
    metrics["num_object_changes"] = total_changes
    metrics["num_text_labels"] = total_text_labels
    metrics["num_educational_objectives"] = total_edu_objectives
    metrics["unique_object_names"] = sorted(object_names)

    # Component type breakdown
    comp_types = _count_component_types(modules)
    metrics["component_types"] = comp_types

    # Prefab diversity
    prefabs = set()
    for module in modules:
        for obj in module.get("objects", []):
            prefabs.add(obj.get("prefab", ""))
    metrics["unique_prefabs"] = sorted(prefabs)
    metrics["num_unique_prefabs"] = len(prefabs)

    return metrics


def _count_component_types(modules: list[dict]) -> dict[str, int]:
    """Count occurrences of each component type across all objects in the modules."""
    counts: dict[str, int] = {}
    for module in modules:
        for obj in module.get("objects", []):
            for comp in obj.get("components", []):
                ct = comp.get("componentType", "unknown")
                counts[ct] = counts.get(ct, 0) + 1
    return counts


def analyze_arlem(scenario_json: dict) -> dict:
    """
    Analyze a generated ARLEM scenario and return structural metrics.

    Expects the output of ARLEMScenario.model_dump(mode="json", exclude_none=True).
    """
    metrics = {}

    workplace = scenario_json.get("workplace", {})
    activity = scenario_json.get("activity", {})

    # Workplace metrics
    metrics["workplace_id"] = workplace.get("id", "")
    metrics["num_things"] = len(workplace.get("things", []))
    metrics["num_places"] = len(workplace.get("places", []))
    metrics["num_persons"] = len(workplace.get("persons", []))
    metrics["num_sensors"] = len(workplace.get("sensors", []))
    metrics["num_devices"] = len(workplace.get("devices", []))
    metrics["num_apps"] = len(workplace.get("apps", []))
    metrics["num_detectables"] = len(workplace.get("detectables", []))
    metrics["num_primitives"] = len(workplace.get("primitives", []))
    metrics["num_predicates"] = len(workplace.get("predicates", []))
    metrics["num_warnings"] = len(workplace.get("warnings", []))

    # Activity metrics
    metrics["activity_id"] = activity.get("id", "")
    metrics["activity_name"] = activity.get("name", "")
    metrics["language"] = activity.get("language", "")
    metrics["num_actions"] = len(activity.get("actions", []))

    # Action flow analysis
    total_activates = 0
    total_deactivates = 0
    total_messages = 0
    total_triggers = 0
    trigger_modes = {}

    for action in activity.get("actions", []):
        # Enter flow
        enter = action.get("enter", {})
        if enter:
            total_activates += len(enter.get("activates", []))
            total_deactivates += len(enter.get("deactivate", []))
            total_messages += len(enter.get("messages", []))

        # Exit flow
        exit_flow = action.get("exit", {})
        if exit_flow:
            total_activates += len(exit_flow.get("activates", []))
            total_deactivates += len(exit_flow.get("deactivate", []))
            total_messages += len(exit_flow.get("messages", []))

        # Triggers
        triggers = action.get("triggers", [])
        total_triggers += len(triggers)
        for trigger in triggers:
            mode = trigger.get("mode", "unknown")
            trigger_modes[mode] = trigger_modes.get(mode, 0) + 1

    metrics["total_activates"] = total_activates
    metrics["total_deactivates"] = total_deactivates
    metrics["total_messages"] = total_messages
    metrics["total_triggers"] = total_triggers
    metrics["trigger_modes"] = trigger_modes

    # POI analysis
    total_pois = 0
    for tangible_list_key in ["things", "places", "persons"]:
        for tangible in workplace.get(tangible_list_key, []):
            total_pois += len(tangible.get("pois", []))
    metrics["total_pois"] = total_pois

    return metrics

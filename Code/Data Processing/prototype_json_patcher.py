#!/usr/bin/env python3
"""
prototype_json_patcher.py
==========================
Ingests a new AR lab JSON and replaces any resource references not present in
the Moon Phase Lab asset library with safe placeholder stand-ins so the lab
can be loaded and inspected even without the correct assets.

PLACEHOLDER RULES
-----------------
| Resource type  | Available check           | Replacement                          |
|----------------|---------------------------|--------------------------------------|
| Prefab (type)  | not in KNOWN_PREFABS      | Prefabs/moveableSphere               |
| Texture        | not in KNOWN_TEXTURES     | 2k_moon                              |
| Audio clip     | not in KNOWN_AUDIO        | moonlab_1_1_0                        |
| Component name | not in KNOWN_COMPONENTS   | removed; replaced with display note  |

Usage
-----
    python3 prototype_json_patcher.py <input_lab.json> [output_lab.json]

    If no output path is given the patched JSON is written to
    <input_lab>_patched.json next to the original file.

The script also prints a substitution report so you can see at a glance what
was swapped.
"""

import json
import sys
import os
from collections import defaultdict
from copy import deepcopy

# ---------------------------------------------------------------------------
# ── MOON LAB ASSET INDEX (auto-extracted from Full_Lab_Transmission.json) ──
# ---------------------------------------------------------------------------

KNOWN_PREFABS = {
    "Prefabs/SunPrefab",
    "Prefabs/clickableSphere",
    "Prefabs/demoButtonPrefab",
    "Prefabs/moveableSphere",
    "Prefabs/robotIdle",
    "Prefabs/textPrefab",
    "Prefabs/tinySphere",
}

KNOWN_TEXTURES = {
    "2k_earth_daymap",
    "2k_moon",
    "balldimpled",
}

KNOWN_AUDIO = {
    "moonlab_1_1_0", "moonlab_1_2_0", "moonlab_1_3_0", "moonlab_1_4_0",
    "moonlab_1_5_0", "moonlab_1_6_0", "moonlab_1_7_0", "moonlab_1_8_0",
    "moonlab_1_9_0", "moonlab_1_10_0", "moonlab_1_11_0",
    "moonlab_2_0_0", "moonlab_2_1_0", "moonlab_2_2_0", "moonlab_2_3_0",
    "moonlab_2_4_0", "moonlab_2_5_0", "moonlab_2_6_0", "moonlab_2_7_0",
    "moonlab_2_8_0", "moonlab_2_9_0", "moonlab_2_10_0", "moonlab_2_11_0",
    "moonlab_2_20_0",
    "moonlab_3_0_0", "moonlab_3_1_0",
    "moonlab_4_0_0",
    "moonlab_5_0_0",
    "moonlab_6_0_0",
    "moonlab_7_0_0", "moonlab_7_1_0", "moonlab_7_2_0", "moonlab_7_3_0",
    "moonlab_7_4_0", "moonlab_7_5_0", "moonlab_7_6_0",
}

KNOWN_COMPONENTS = {
    "checkAngle",
    "demoButtonActions",
    "simpleOrbit",
    "simpleRotation",
}

# ---------------------------------------------------------------------------
# ── PLACEHOLDER CONSTANTS ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

PH_PREFAB    = "Prefabs/moveableSphere"   # sphere stand-in (real Moon Lab prefab)
PH_TEXTURE   = "2k_moon"                  # real Moon Lab texture
PH_AUDIO     = "moonlab_1_1_0"            # real Moon Lab audio clip

# For an unknown component we inject a simpleRotation at near-zero speed so
# the object still appears in the scene but doesn't do anything distracting.
# We also tag the placeholder so it is easy to search for in the app logs.
def ph_component(original_name: str) -> str:
    """Return a JSON string for a harmless placeholder component."""
    payload = {
        "name": "simpleRotation",
        "timeRate": 0.0,
        "rotationTime": 99999,
        "_placeholder_for": original_name   # informational field
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# ── REVERT HELPERS (NEW v2.0 → OLD Transmission format) ────────────────────
# ---------------------------------------------------------------------------
#
# These mirror the forward direction in
# `Code/Data Processing/convert_lab_json.py`. We need to revert because the
# AR headset prototype consumes only the old `Transmission`-wrapped format,
# but `generate_lab.py` produces the new v2.0 format from the LLM.
#
# After reverting we still need to re-inject the navigation buttons that
# `convert_lab_json.py` strips out (see EXCLUDED_OBJECT_NAMES there).

# Module-level `moduleType` discriminator → old `prefabName` string.
# Inverse of PREFAB_TO_MODULE_TYPE in convert_lab_json.py.
MODULE_TYPE_TO_PREFAB = {
    "demo": "demoPrefab",
}

# Defaults for module envelope fields that the old format requires but the
# new schema doesn't carry. Lifted from Full_Lab_Transmission.json so the
# headset's module loader sees a familiar shape.
MODULE_ENVELOPE_DEFAULTS = {
    "prerequisiteActivities": [],
    "numRepeatsAllowed": 0,
    "numGradableRepeatsAllowed": 0,
    "gradingCriteria": "",
    "currentScore": 0.0,
    "bestScore": 0.0,
    "completed": False,
    "currentSubphase": 0,
    "subphaseNames": [],
    "urlJson": "",
    "json": "",
    "timeToEnd": 2000,
    "endUsingButton": True,
    "createObjects": True,
    "destroyObjects": True,
    "restoreLights": True,
    "useSunlight": True,
}

# Hardcoded brb (be-right-back nav) button templates extracted from
# Full_Lab_Transmission.json. These are appended to every reverted module
# so the headset's clip-runner can advance/retreat through the lab.
# All referenced prefabs and components are in KNOWN_PREFABS / KNOWN_COMPONENTS,
# so they survive the placeholder pass without substitution.
BRB_OBJECT_TEMPLATES = [
    {
        "type": "Prefabs/demoButtonPrefab",
        "parentName": "sectionLabel",
        "position": {"x": 1.2, "y": 0, "z": 0},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "name": "brbNextModule",
        "color": [255, 0, 0, 255],
        "componentsToAdd": [
            '{"name": "demoButtonActions", "callBackObjects": ["demoModule"]}'
        ],
        "childColor": [0, 255, 255, 255],
        "childName": "button",
    },
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "Next\\n Module", "color": [255, 255, 0, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.0, "z": 0.13},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "brbNextModule",
        "name": "brbNextModuleLabel",
    },
    {
        "type": "Prefabs/demoButtonPrefab",
        "parentName": "sectionLabel",
        "position": {"x": -1.2, "y": 0, "z": 0},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "name": "brbPreviousModule",
        "color": [255, 0, 0, 255],
        "componentsToAdd": [
            '{"name": "demoButtonActions", "callBackObjects": ["demoModule"]}'
        ],
        "childColor": [0, 255, 255, 255],
        "childName": "button",
    },
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "Previous \\n Module", "color": [255, 255, 0, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.0, "z": 0.13},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "brbPreviousModule",
        "name": "brbPreviousModuleLabel",
    },
    {
        "type": "Prefabs/demoButtonPrefab",
        "parentName": "taskLabel",
        "position": {"x": 1, "y": 0, "z": 0},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 1, "y": 1, "z": 1},
        "name": "brbNextClip",
        "color": [255, 0, 0, 255],
        "componentsToAdd": [
            '{"name": "demoButtonActions", "callBackObjects": ["demoModule"]}'
        ],
        "childColor": [0, 255, 0, 255],
        "childName": "button",
    },
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "Next", "color": [255, 255, 0, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.0, "z": 0.1},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "brbNextClip",
        "name": "brbNextClipLabel",
    },
    {
        "type": "Prefabs/demoButtonPrefab",
        "parentName": "taskLabel",
        "position": {"x": -1, "y": 0, "z": 0},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 1, "y": 1, "z": 1},
        "name": "brbPreviousClip",
        "color": [255, 0, 0, 255],
        "componentsToAdd": [
            '{"name": "demoButtonActions", "callBackObjects": ["demoModule"]}'
        ],
        "childColor": [0, 255, 0, 255],
        "childName": "button",
    },
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "Previous", "color": [255, 255, 0, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.0, "z": 0.1},
        "eulerAngles": {"x": 90, "y": 180.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "brbPreviousClip",
        "name": "brbPreviousClipLabel",
    },
]

# Names of nav buttons whose visibility we toggle per-clip. Labels follow
# their parents automatically, so we don't need to inject changes for them.
BRB_BUTTON_NAMES = [
    "brbNextModule",
    "brbPreviousModule",
    "brbNextClip",
    "brbPreviousClip",
]


def _zero_vec():
    return {"x": 0.0, "y": 0.0, "z": 0.0}


def _one_vec():
    return {"x": 1, "y": 1, "z": 1}


def revert_vec3(arr):
    """Inverse of convert_vec3: [x,y,z] → {x,y,z}. Pass through dicts."""
    if arr is None:
        return None
    if isinstance(arr, dict):
        return arr
    if isinstance(arr, (list, tuple)) and len(arr) >= 3:
        return {"x": arr[0], "y": arr[1], "z": arr[2]}
    return _zero_vec()


def serialize_component(comp):
    """Inverse of parse_component_string: component dict → JSON string with `name` field."""
    out = {}
    comp_type = comp.get("componentType", "")
    out["name"] = comp_type
    for k, v in comp.items():
        if k == "componentType":
            continue
        # Re-expand vector arrays that originally came from {x,y,z} dicts.
        if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            out[k] = {"x": v[0], "y": v[1], "z": v[2]}
        else:
            out[k] = v
    return json.dumps(out)


def serialize_tmp(comp):
    """Inverse of parse_tmp_string: textMeshPro component → tmp JSON string."""
    payload = {}
    if "text" in comp:
        payload["textField"] = comp["text"]
    if "color" in comp:
        payload["color"] = comp["color"]
    if "fontSize" in comp:
        payload["fontSize"] = comp["fontSize"]
    if "wrapText" in comp:
        payload["wrapText"] = comp["wrapText"]
    return json.dumps(payload)


def revert_scene_object(obj):
    """Inverse of convert_scene_object: new SceneObject → old object dict."""
    result = {}

    # prefab → type (re-add Prefabs/ prefix if missing)
    prefab = obj.get("prefab", "")
    if prefab:
        result["type"] = prefab if prefab.startswith("Prefabs/") else f"Prefabs/{prefab}"

    if "name" in obj:
        result["name"] = obj["name"]

    if "parent" in obj and obj["parent"]:
        result["parentName"] = obj["parent"]

    if "position" in obj:
        result["position"] = revert_vec3(obj["position"])
    if "eulerAngles" in obj:
        result["eulerAngles"] = revert_vec3(obj["eulerAngles"])
    if "scale" in obj:
        result["scale"] = revert_vec3(obj["scale"])

    if "enabled" in obj:
        result["enabled"] = obj["enabled"]

    # Visual properties pass through.
    for k in ("texture", "textureByURL", "material", "color", "tag",
              "transmittable", "childColor", "childName"):
        if k in obj:
            result[k] = obj[k]

    # Walk components: textMeshPro → tmp string on parent;
    # rigidBody / pointerReceiver → top-level RigidBody / PointerReceiver
    # blocks; everything else → entry in componentsToAdd.
    components_to_add = []
    for comp in obj.get("components", []):
        ctype = comp.get("componentType", "")
        if ctype == "textMeshPro":
            result["tmp"] = serialize_tmp(comp)
        elif ctype == "rigidBody":
            rb = {k: v for k, v in comp.items() if k != "componentType"}
            result["RigidBody"] = rb
        elif ctype == "pointerReceiver":
            pr = {k: v for k, v in comp.items() if k != "componentType"}
            result["PointerReceiver"] = pr
        else:
            components_to_add.append(serialize_component(comp))

    # The old format always emits componentsToAdd, even when empty.
    result["componentsToAdd"] = components_to_add

    return result


def revert_object_change(change):
    """Inverse of convert_object_change: sparse delta → full delta with new* flags."""
    result = {}
    result["name"] = change.get("target", "")
    result["parentObject"] = "[CURRENT_LAB]"
    result["activationConditions"] = 0
    result["reactiveObject"] = False
    # `enable` defaults to True unless the new format explicitly disables.
    result["enable"] = change.get("enabled", True)

    has_position = "position" in change
    result["position"] = revert_vec3(change["position"]) if has_position else _zero_vec()
    result["newPosition"] = has_position

    has_euler = "eulerAngles" in change
    result["eulerAngles"] = revert_vec3(change["eulerAngles"]) if has_euler else _zero_vec()
    result["newEulerAngles"] = has_euler

    has_scale = "scale" in change
    result["scale"] = revert_vec3(change["scale"]) if has_scale else _zero_vec()
    result["newScale"] = has_scale

    if "color" in change:
        result["color"] = change["color"]
    if "childColor" in change:
        result["childColor"] = change["childColor"]
    if "childName" in change:
        result["childName"] = change["childName"]

    # Components inside a change: textMeshPro becomes tmp on the change itself,
    # other components are appended to componentsToAdd on the change.
    components_to_add = []
    for comp in change.get("components", []):
        ctype = comp.get("componentType", "")
        if ctype == "textMeshPro":
            result["tmp"] = serialize_tmp(comp)
        else:
            components_to_add.append(serialize_component(comp))
    if components_to_add:
        result["componentsToAdd"] = components_to_add

    return result


def revert_clip(clip):
    """Inverse of convert_clip: new clip → old clip with audioClipString/objectChanges."""
    result = {}
    result["clipName"] = clip.get("clipName", "")
    if "audioClip" in clip:
        result["audioClipString"] = clip["audioClip"]
    if "autoAdvance" in clip:
        result["autoAdvance"] = clip["autoAdvance"]
    result["objectChanges"] = [
        revert_object_change(c) for c in clip.get("changes", [])
    ]
    return result


def _default_brb_change(name):
    """An always-visible objectChange for one brb button (scale 1, enabled)."""
    return {
        "name": name,
        "parentObject": "[CURRENT_LAB]",
        "activationConditions": 0,
        "reactiveObject": False,
        "enable": True,
        "position": _zero_vec(),
        "newPosition": False,
        "eulerAngles": _zero_vec(),
        "newEulerAngles": False,
        "scale": _one_vec(),
        "newScale": True,
    }


def inject_nav_buttons(module):
    """
    Append the brb nav buttons (objects + per-clip objectChanges) to a
    reverted module dict so the AR headset can navigate clips/modules.

    Mutates in place. All buttons stay visible across every clip — the
    simplest behavior for prototype testing.
    """
    module.setdefault("objects", [])
    for tpl in BRB_OBJECT_TEMPLATES:
        module["objects"].append(deepcopy(tpl))

    for clip in module.get("clips", []):
        clip.setdefault("objectChanges", [])
        for name in BRB_BUTTON_NAMES:
            clip["objectChanges"].append(_default_brb_change(name))


def revert_module(mod):
    """Inverse of convert_module: new module dict → old module dict (not yet stringified)."""
    result = {}

    # Required identity / display fields.
    module_name = mod.get("moduleName", "")
    result["moduleName"] = module_name
    result["specificName"] = module_name
    result["jsonFileName"] = (module_name.replace(" ", "_") + ".json") if module_name else ""

    module_type = mod.get("moduleType", "")
    result["prefabName"] = MODULE_TYPE_TO_PREFAB.get(module_type, mod.get("prefab") or module_type or "demoPrefab")

    # Educational content.
    result["educationalObjectives"] = mod.get("educationalObjectives", [])
    result["instructions"] = mod.get("instructions", [])

    # Author / institution.
    result["description"] = mod.get("description", "")
    result["author"] = mod.get("author", "")
    if mod.get("institution"):
        result["authorInstitution"] = mod["institution"]
    if mod.get("dateCreated"):
        result["dateCreated"] = mod["dateCreated"]

    # Audio.
    result["introAudio"] = mod.get("introAudio", "")

    # Envelope defaults required by the old format.
    for k, v in MODULE_ENVELOPE_DEFAULTS.items():
        result[k] = deepcopy(v)

    # Objects + clips.
    result["objects"] = [revert_scene_object(o) for o in mod.get("objects", [])]
    result["clips"] = [revert_clip(c) for c in mod.get("clips", [])]

    return result


def revert_lab(data):
    """
    Inverse of convert_lab: new v2.0 lab dict → old Transmission dict.
    Each module is reverted, nav buttons injected, then re-stringified into
    ActivityModules per the old schema.
    """
    modules = []
    for mod in data.get("modules", []):
        reverted = revert_module(mod)
        inject_nav_buttons(reverted)
        modules.append(json.dumps(reverted))

    return {
        "Transmission": "true",
        "Lab_ID": data.get("labId", ""),
        "Author": data.get("author", ""),
        "CourseName": data.get("courseName", ""),
        "EstimatedLength": data.get("estimatedLength", ""),
        "NumModules": str(len(modules)),
        "Objectives": data.get("objectives", []),
        "ActivityModules": modules,
        "Assets": [],
    }


def is_new_format(raw):
    """Detect v2.0 lab JSON (versus old Transmission wrapper or bare module)."""
    if not isinstance(raw, dict):
        return False
    return "modules" in raw or "labId" in raw or raw.get("version") == "2.0"


# ---------------------------------------------------------------------------
# ── PATCHING HELPERS ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def patch_components(components_list: list, report: dict) -> list:
    """Replace unknown components, keep known ones intact."""
    patched = []
    for comp_str in components_list:
        try:
            comp = json.loads(comp_str)
        except (json.JSONDecodeError, TypeError):
            patched.append(comp_str)
            continue

        name = comp.get("name", "")
        if name and name not in KNOWN_COMPONENTS:
            report["components"].append(name)
            patched.append(ph_component(name))
        else:
            patched.append(comp_str)
    return patched


def patch_object(obj: dict, report: dict) -> dict:
    """Patch a single scene object dict."""
    obj = deepcopy(obj)

    # ── Prefab type ──────────────────────────────────────────────────────
    t = obj.get("type", "")
    if t and t not in KNOWN_PREFABS:
        report["prefabs"].append(t)
        obj["type"] = PH_PREFAB

    # ── Texture ──────────────────────────────────────────────────────────
    tx = obj.get("texture", "")
    if tx and tx not in KNOWN_TEXTURES:
        report["textures"].append(tx)
        obj["texture"] = PH_TEXTURE

    # ── Components ───────────────────────────────────────────────────────
    if "componentsToAdd" in obj:
        obj["componentsToAdd"] = patch_components(obj["componentsToAdd"], report)

    return obj


def patch_clip(clip: dict, report: dict) -> dict:
    """Patch a single clip dict."""
    clip = deepcopy(clip)

    # Audio on the clip itself
    a = clip.get("audioClipString", "")
    if a and a not in KNOWN_AUDIO:
        report["audio"].append(a)
        clip["audioClipString"] = PH_AUDIO

    # objectChanges inside the clip can also carry componentsToAdd
    for change in clip.get("objectChanges", []):
        if "componentsToAdd" in change:
            change["componentsToAdd"] = patch_components(
                change["componentsToAdd"], report
            )

    return clip


def patch_module(mod: dict) -> tuple:
    """
    Patch all objects and clips inside a module dict.
    Returns (patched_module, report_dict).
    """
    report = defaultdict(list)
    mod = deepcopy(mod)

    # introAudio
    ia = mod.get("introAudio", "")
    if ia and ia not in KNOWN_AUDIO:
        report["audio"].append(ia)
        mod["introAudio"] = PH_AUDIO

    # Scene objects
    mod["objects"] = [patch_object(o, report) for o in mod.get("objects", [])]

    # Clips
    mod["clips"] = [patch_clip(c, report) for c in mod.get("clips", [])]

    return mod, dict(report)


# ---------------------------------------------------------------------------
# ── TRANSMISSION-LEVEL PATCHER ──────────────────────────────────────────────
# ---------------------------------------------------------------------------

def patch_transmission(transmission: dict) -> tuple:
    """
    Patch an entire Transmission JSON (outer wrapper with ActivityModules).
    Handles both pre-parsed and raw-string module formats.
    Returns (patched_transmission, full_report).
    """
    transmission = deepcopy(transmission)
    full_report = {}

    patched_modules = []
    for i, mod_entry in enumerate(transmission.get("ActivityModules", [])):
        # Modules may be stored as raw JSON strings
        if isinstance(mod_entry, str):
            try:
                mod = json.loads(mod_entry)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Could not parse ActivityModule[{i}]: {e}")
                patched_modules.append(mod_entry)
                continue
            patched_mod, report = patch_module(mod)
            patched_modules.append(json.dumps(patched_mod))
        elif isinstance(mod_entry, dict):
            patched_mod, report = patch_module(mod_entry)
            patched_modules.append(patched_mod)
            report = report
        else:
            patched_modules.append(mod_entry)
            report = {}

        label = ""
        if isinstance(mod_entry, str):
            try:
                label = json.loads(mod_entry).get("moduleName", f"Module[{i}]")
            except Exception:
                label = f"Module[{i}]"
        elif isinstance(mod_entry, dict):
            label = mod_entry.get("moduleName", f"Module[{i}]")

        if any(report.values()):
            full_report[label] = report

    transmission["ActivityModules"] = patched_modules
    return transmission, full_report


# ---------------------------------------------------------------------------
# ── STANDALONE LAB JSON PATCHER ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

def patch_lab_json(lab: dict) -> tuple:
    """
    Patch a standalone module JSON (no Transmission wrapper).
    Returns (patched_lab, report).
    """
    patched, report = patch_module(lab)
    return patched, report


# ---------------------------------------------------------------------------
# ── REPORT PRINTER ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def print_report(full_report: dict):
    if not full_report:
        print("\n✅  No substitutions needed — all resources are available in the Moon Lab library.")
        return

    print("\n" + "=" * 65)
    print("  PLACEHOLDER SUBSTITUTION REPORT")
    print("=" * 65)

    for module_name, report in full_report.items():
        print(f"\n  Module: {module_name}")
        print("  " + "-" * 55)

        if report.get("prefabs"):
            unique = sorted(set(report["prefabs"]))
            print(f"  🟠  Unknown Prefabs  → replaced with {PH_PREFAB}")
            for p in unique:
                print(f"        • {p}")

        if report.get("textures"):
            unique = sorted(set(report["textures"]))
            print(f"  🟣  Unknown Textures → replaced with '{PH_TEXTURE}'")
            for t in unique:
                print(f"        • {t}")

        if report.get("audio"):
            unique = sorted(set(report["audio"]))
            print(f"  🔊  Unknown Audio    → replaced with '{PH_AUDIO}'")
            for a in unique:
                print(f"        • {a}")

        if report.get("components"):
            unique = sorted(set(report["components"]))
            print(f"  🔧  Unknown Components → replaced with no-op simpleRotation")
            for c in unique:
                print(f"        • {c}")

    print("\n" + "=" * 65)

    # Summary counts
    all_prefabs    = sum(len(set(r.get("prefabs", [])))    for r in full_report.values())
    all_textures   = sum(len(set(r.get("textures", [])))   for r in full_report.values())
    all_audio      = sum(len(set(r.get("audio", [])))      for r in full_report.values())
    all_components = sum(len(set(r.get("components", []))) for r in full_report.values())

    print(f"\n  Total unique substitutions:")
    print(f"    Prefabs    : {all_prefabs}")
    print(f"    Textures   : {all_textures}")
    print(f"    Audio clips: {all_audio}")
    print(f"    Components : {all_components}")
    print()


# ---------------------------------------------------------------------------
# ── INDEX PRINTER ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def print_moon_lab_index():
    """Print the full Moon Lab asset index to stdout."""
    print("\n" + "=" * 65)
    print("  MOON LAB ASSET INDEX")
    print("=" * 65)

    print(f"\n  Prefabs ({len(KNOWN_PREFABS)}):")
    for p in sorted(KNOWN_PREFABS):
        print(f"    • {p}")

    print(f"\n  Textures ({len(KNOWN_TEXTURES)}):")
    for t in sorted(KNOWN_TEXTURES):
        print(f"    • {t}")

    print(f"\n  Audio Clips ({len(KNOWN_AUDIO)}):")
    for a in sorted(KNOWN_AUDIO):
        print(f"    • {a}")

    print(f"\n  Scriptable Components ({len(KNOWN_COMPONENTS)}):")
    for c in sorted(KNOWN_COMPONENTS):
        print(f"    • {c}")

    print("\n  Placeholders:")
    print(f"    • Unknown prefab    → {PH_PREFAB}")
    print(f"    • Unknown texture   → {PH_TEXTURE}")
    print(f"    • Unknown audio     → {PH_AUDIO}")
    print(f"    • Unknown component → no-op simpleRotation (rotationTime=99999)")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# ── CLI ENTRY POINT ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    # Windows consoles default to cp1252 and choke on the emoji glyphs in
    # print_report. Force UTF-8 so the report prints cleanly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("\nUsage: python3 lab_placeholder_patcher.py [--index] <input.json> [output.json]")
        sys.exit(0)

    if sys.argv[1] == "--index":
        print_moon_lab_index()
        sys.exit(0)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_patched" + (ext or ".json")

    print(f"\n  Input : {input_path}")
    print(f"  Output: {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Auto-detect input format and route accordingly.
    if is_new_format(raw):
        print("  Format: New v2.0 lab — reverting to old Transmission format\n")
        raw = revert_lab(raw)
        patched, full_report = patch_transmission(raw)
    elif "ActivityModules" in raw:
        print("  Format: Transmission (outer wrapper detected)\n")
        patched, full_report = patch_transmission(raw)
    else:
        print("  Format: Standalone module JSON\n")
        patched, report = patch_lab_json(raw)
        module_name = raw.get("moduleName", "module")
        full_report = {module_name: report} if any(report.values()) else {}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2, ensure_ascii=False)

    print_report(full_report)
    print(f"  Patched JSON written to: {output_path}\n")


if __name__ == "__main__":
    main()
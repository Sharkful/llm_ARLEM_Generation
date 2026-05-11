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

# Anchor scene objects that the brb buttons parent to. The brb templates use
# parentName "sectionLabel" / "taskLabel", so without these the buttons float
# at world origin and the demo callbacks don't resolve. The moduleName /
# clipName fields are filled in at injection time.
#
# sectionLabel shows the module title; taskLabel shows the current clip title
# and is updated per clip via an injected objectChange. Both are textPrefab
# objects parented to "[CURRENT_LAB]" — exact shape lifted from
# Full_Lab_Transmission.json.
ANCHOR_OBJECT_TEMPLATES = [
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "{moduleName}", "color": [255, 255, 255, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.75, "z": 2.5},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "[CURRENT_LAB]",
        "name": "sectionLabel",
    },
    {
        "type": "Prefabs/textPrefab",
        "tmp": '{"textField": "{firstClipName}", "color": [255, 255, 255, 255], "fontSize": 1.0, "wrapText": false}',
        "position": {"x": 0.0, "y": 0.6, "z": 2.5},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 0.6, "y": 0.6, "z": 0.6},
        "enabled": True,
        "parentName": "[CURRENT_LAB]",
        "name": "taskLabel",
    },
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

    # prefab → type (re-add Prefabs/ prefix if missing). The old format
    # requires `type` to be a non-empty string; fall back to PH_PREFAB so
    # the headset doesn't hit a null-instantiation error.
    prefab = obj.get("prefab") or ""
    if prefab:
        result["type"] = prefab if prefab.startswith("Prefabs/") else f"Prefabs/{prefab}"
    else:
        result["type"] = PH_PREFAB

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


def _anchor_change(name, text):
    """An objectChange that updates the TMP text on an anchor label."""
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
        "scale": _zero_vec(),
        "newScale": False,
        "tmp": json.dumps({
            "textField": text,
            "color": [255, 255, 255, 255],
            "fontSize": 1.0,
            "wrapText": False,
        }),
    }


def inject_nav_buttons(module):
    """
    Inject the anchor labels (sectionLabel, taskLabel) and brb nav buttons
    into a reverted module dict so the AR headset can render the module
    title, per-clip task captions, and the navigation buttons.

    Mutates in place. All buttons stay visible across every clip — the
    simplest behavior for prototype testing. Skips any anchor or brb object
    that already exists by name (idempotent).
    """
    module.setdefault("objects", [])
    module.setdefault("clips", [])

    existing_names = {o.get("name") for o in module["objects"]}

    module_name = module.get("moduleName", "")
    clips = module["clips"]
    first_clip_name = clips[0].get("clipName", "") if clips else ""

    # 1. Anchor labels: sectionLabel (module title) + taskLabel (clip caption).
    for tpl in ANCHOR_OBJECT_TEMPLATES:
        if tpl["name"] in existing_names:
            continue
        obj = deepcopy(tpl)
        obj["tmp"] = (
            obj["tmp"]
            .replace("{moduleName}", module_name)
            .replace("{firstClipName}", first_clip_name)
        )
        module["objects"].append(obj)
        existing_names.add(obj["name"])

    # 2. brb nav buttons.
    for tpl in BRB_OBJECT_TEMPLATES:
        if tpl["name"] in existing_names:
            continue
        module["objects"].append(deepcopy(tpl))
        existing_names.add(tpl["name"])

    # 3. Per-clip changes: keep all 4 brb buttons visible, and update
    # taskLabel's TMP text to the current clip's clipName.
    for clip in module["clips"]:
        clip.setdefault("objectChanges", [])
        clip_name = clip.get("clipName", "")
        clip["objectChanges"].append(_anchor_change("taskLabel", clip_name))
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


def revert_lab(data, inject=True):
    """
    Inverse of convert_lab: new v2.0 lab dict → old Transmission dict.
    Each module is reverted (and nav buttons + anchors optionally injected),
    then re-stringified into ActivityModules per the old schema.

    Set inject=False when round-trip-validating against the moon lab — the
    optimized moon lab already round-trips its original anchors, so injecting
    would double-add them and pollute the diff.
    """
    modules = []
    for mod in data.get("modules", []):
        reverted = revert_module(mod)
        if inject:
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
# ── ROUND-TRIP VALIDATION (moon lab → revert → compare with raw) ────────────
# ---------------------------------------------------------------------------
#
# Sanity check that `revert_lab(optimized_moon_lab)` reproduces the raw moon
# lab. Diffs that are an unavoidable consequence of the forward converter's
# strips are whitelisted; anything else is a real bug in the revert path.

DEFAULT_VALIDATE_OPTIMIZED = "Artifacts/Data/Original Moon Lab/Processed/Optomized_moon_lab_final.json"
DEFAULT_VALIDATE_RAW = "Artifacts/Data/Original Moon Lab/Raw/Full_Lab_Transmission.json"

import re

# Fields the forward converter strips that the patcher cannot recover from
# new-format data alone — these aren't revert bugs.
_FORWARD_STRIPPED_MODULE_FIELDS = {
    "specificName", "jsonFileName", "authorInstitution",
}
_FORWARD_STRIPPED_CLIP_FIELDS = {
    "timeToEnd",
}
# Keys inside a tmp JSON string that the forward converter drops.
_FORWARD_STRIPPED_TMP_KEYS = {"parentObject"}

# Top-level path prefixes for fields the forward converter doesn't preserve.
# `NumModules` is whitelisted because the raw moon lab has a data error
# ("1" despite 8 ActivityModules) — our re-derived value is more correct.
_EXPECTED_LOSS_PREFIXES = (
    "Assets",
    "Objectives",
    "NumModules",
)

_RE_MODULE_FIELD = re.compile(r"^ActivityModules\[\d+\]\.([^.\[]+)$")
_RE_CLIP_FIELD = re.compile(r"^ActivityModules\[\d+\]\.clips\[\d+\]\.([^.\[]+)$")
_RE_OC_FIELD = re.compile(r"^ActivityModules\[\d+\]\.clips\[\d+\]\.objectChanges\[[^\]]+\]\.([^.\[]+)$")
# objectChange fields the forward converter strips.
_FORWARD_STRIPPED_OC_FIELDS = {"activationConditions"}

# Names whose entries the forward converter strips entirely (see
# EXCLUDED_OBJECT_NAMES in convert_lab_json.py). Mismatches involving these
# aren't revert bugs.
_FORWARD_STRIPPED_OC_NAMES = {
    "brbNextClip", "brbNextClipLabel",
    "brbNextModule", "brbNextModuleLabel",
    "brbPreviousClip", "brbPreviousClipLabel",
    "brbPreviousModule", "brbPreviousModuleLabel",
    "MainInstructions",
}

# Old-format field aliases the new schema collapses. Both sides of each pair
# mean the same thing to the headset. `euleriAngles` is a typo in the raw
# moon lab that the forward converter normalizes to `eulerAngles`.
_FIELD_ALIASES = {
    "parentName": "parent",
    "rotation": "eulerAngles",
    "euleriAngles": "eulerAngles",
}

# Fields whose default value is dropped by the forward converter. When
# `produced` is missing a key and `expected` has the default, they match.
_FIELD_DEFAULTS = {
    "active": True,
    "material": "",
    "tag": "",
    "textureByURL": "",
    "transmittable": False,
    "componentsToAdd": [],
}


def _normalize_keys(d):
    """Rename alias keys to a canonical name so both sides compare equal."""
    return {_FIELD_ALIASES.get(k, k): v for k, v in d.items()}


def _matches_default(key, value):
    if key in _FIELD_DEFAULTS and value == _FIELD_DEFAULTS[key]:
        return True
    # `instructions` and `educationalObjectives` can be [], [""], or absent
    # — all mean "no instructions". The forward converter at line 305-306
    # drops [""] but the patcher emits whatever's in the new format ([] or
    # nothing). Treat all three as equivalent.
    if key in ("instructions", "educationalObjectives") and value in ([], [""]):
        return True
    return False


def _is_noop_change(change):
    """An objectChange is a no-op manifest entry if it has no new*: True flags
    and no tmp / component / color override."""
    if not isinstance(change, dict):
        return False
    if any(change.get(f) for f in ("newPosition", "newEulerAngles", "newScale")):
        return False
    if change.get("tmp") or change.get("componentsToAdd"):
        return False
    if "color" in change or "childColor" in change or "childName" in change:
        return False
    return True


def _is_expected_loss(path):
    if any(path.startswith(p) for p in _EXPECTED_LOSS_PREFIXES):
        return True
    m = _RE_MODULE_FIELD.match(path)
    if m and m.group(1) in _FORWARD_STRIPPED_MODULE_FIELDS:
        return True
    m = _RE_CLIP_FIELD.match(path)
    if m and m.group(1) in _FORWARD_STRIPPED_CLIP_FIELDS:
        return True
    m = _RE_OC_FIELD.match(path)
    if m and m.group(1) in _FORWARD_STRIPPED_OC_FIELDS:
        return True
    return False


def _tmp_strings_equivalent(a, b):
    """
    Two tmp JSON strings are equivalent if their parsed payloads differ only
    by keys the forward converter strips (e.g. parentObject).
    """
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    try:
        pa, pb = json.loads(a), json.loads(b)
    except (json.JSONDecodeError, TypeError):
        return False
    for k in _FORWARD_STRIPPED_TMP_KEYS:
        pa.pop(k, None)
        pb.pop(k, None)
    return pa == pb


def _named_list_diff(produced, expected, path, diffs, skip_names):
    """
    Name-aware diff for object / objectChange lists. Reports extra/missing
    entries semantically by `name` field; skips entries whose name is in
    skip_names (forward-converter-stripped entries).
    """
    by_name_p = {c.get("name"): c for c in produced if isinstance(c, dict)}
    by_name_e = {c.get("name"): c for c in expected if isinstance(c, dict)}
    produced_real = {n: v for n, v in by_name_p.items() if n not in skip_names}
    expected_real = {n: v for n, v in by_name_e.items() if n not in skip_names}

    for name in sorted(set(expected_real) | set(produced_real), key=lambda x: x or ""):
        sub_path = f"{path}[name={name!r}]"
        if name not in produced_real:
            # No-op manifest entries get stripped by the forward converter
            # (convert_clip drops len(change) <= 1 changes after sparse-delta
            # conversion). Treat them as equivalent to missing here.
            if _is_noop_change(expected_real[name]):
                continue
            diffs.append((sub_path, "<missing>", _short(expected_real[name])))
        elif name not in expected_real:
            diffs.append((sub_path, _short(produced_real[name]), "<missing>"))
        else:
            _diff(produced_real[name], expected_real[name], sub_path, diffs)


def _diff(produced, expected, path, diffs):
    """Recursive structural diff. Order-insensitive, alias-aware."""
    if path.endswith(".tmp") and _tmp_strings_equivalent(produced, expected):
        return

    # [""] ≡ [] for instructions/educationalObjectives (the forward
    # converter strips [""] but both mean "no content").
    if path.endswith(".instructions") or path.endswith(".educationalObjectives"):
        if produced in ([], [""]) and expected in ([], [""]):
            return

    if type(produced) is not type(expected):
        if not (isinstance(produced, (int, float)) and isinstance(expected, (int, float))):
            diffs.append((path, f"type {type(produced).__name__}", f"type {type(expected).__name__}"))
            return

    if isinstance(produced, dict):
        p = _normalize_keys(produced)
        e = _normalize_keys(expected)
        for k in sorted(set(p) | set(e)):
            sub_path = f"{path}.{k}"
            if k not in p:
                if _is_expected_loss(sub_path) or _matches_default(k, e[k]):
                    continue
                diffs.append((sub_path, "<missing>", _short(e[k])))
            elif k not in e:
                if _matches_default(k, p[k]):
                    continue
                diffs.append((sub_path, _short(p[k]), "<missing>"))
            else:
                _diff(p[k], e[k], sub_path, diffs)
        return

    if isinstance(produced, list):
        if path.endswith(".objectChanges") or path.endswith(".objects"):
            _named_list_diff(produced, expected, path, diffs, _FORWARD_STRIPPED_OC_NAMES)
            return
        if len(produced) != len(expected):
            diffs.append((path, f"len={len(produced)}", f"len={len(expected)}"))
        for i in range(min(len(produced), len(expected))):
            _diff(produced[i], expected[i], f"{path}[{i}]", diffs)
        return

    if produced != expected:
        diffs.append((path, _short(produced), _short(expected)))


def _short(v):
    """Compact repr of a value for diff display."""
    s = repr(v) if not isinstance(v, str) else f"'{v}'"
    return s if len(s) <= 60 else s[:57] + "..."


def _parse_module_strings(lab_dict):
    """Return a copy of the lab with ActivityModules parsed from JSON strings."""
    out = deepcopy(lab_dict)
    parsed = []
    for entry in out.get("ActivityModules", []):
        if isinstance(entry, str):
            try:
                parsed.append(json.loads(entry))
            except json.JSONDecodeError:
                parsed.append(entry)
        else:
            parsed.append(entry)
    out["ActivityModules"] = parsed
    return out


def validate_roundtrip(optimized_path, raw_path):
    """
    Revert the optimized moon lab and compare against the raw transmission.
    Print a categorized diff. Return 0 if no unexpected diffs, 1 otherwise.
    """
    print(f"\n  Optimized (new format) : {optimized_path}")
    print(f"  Raw       (old format) : {raw_path}\n")

    with open(optimized_path, "r", encoding="utf-8") as f:
        optimized = json.load(f)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Revert without injecting brb/anchors — the optimized moon lab already
    # round-trips its own objects.
    produced = revert_lab(optimized, inject=False)

    # Parse ActivityModules into dicts so JSON whitespace doesn't pollute.
    produced_parsed = _parse_module_strings(produced)
    raw_parsed = _parse_module_strings(raw)

    diffs = []
    _diff(produced_parsed, raw_parsed, "", diffs)

    # Trim the leading "." each path picks up from the top-level call.
    diffs = [(p.lstrip("."), a, b) for (p, a, b) in diffs]

    expected = [d for d in diffs if _is_expected_loss(d[0])]
    unexpected = [d for d in diffs if not _is_expected_loss(d[0])]

    print("=" * 65)
    print(f"  ROUND-TRIP VALIDATION:  {len(unexpected)} unexpected diff(s), "
          f"{len(expected)} expected loss(es)")
    print("=" * 65)

    if expected:
        print("\n  Expected losses (forward converter strips these):")
        for path, a, b in expected[:10]:
            print(f"    • {path}")
            print(f"        produced: {a}")
            print(f"        expected: {b}")
        if len(expected) > 10:
            print(f"    ... and {len(expected) - 10} more")

    if unexpected:
        print("\n  ⚠️  Unexpected diffs (real revert bugs):")
        for path, a, b in unexpected[:20]:
            print(f"    • {path}")
            print(f"        produced: {a}")
            print(f"        expected: {b}")
        if len(unexpected) > 20:
            print(f"    ... and {len(unexpected) - 20} more")
        print()
        return 1

    print("\n  ✅  No unexpected diffs — revert is faithful for the moon lab.\n")
    return 0


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
        print("\nUsage: python3 prototype_json_patcher.py [--index | --validate-roundtrip] <input.json> [output.json]")
        sys.exit(0)

    if sys.argv[1] == "--index":
        print_moon_lab_index()
        sys.exit(0)

    if sys.argv[1] == "--validate-roundtrip":
        optimized = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_VALIDATE_OPTIMIZED
        raw = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_VALIDATE_RAW
        sys.exit(validate_roundtrip(optimized, raw))

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
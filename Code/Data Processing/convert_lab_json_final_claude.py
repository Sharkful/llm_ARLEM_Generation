#!/usr/bin/env python3
"""
Converts old AR Lab JSON format to the new restructured format.

Changes applied:
1. ActivityModules: string[] -> object[] with explicit moduleType discriminator
2. Unity structs (position, eulerAngles, scale, color) -> compact arrays
3. TextMeshPro (tmp) -> parsed object stored as a component in the components array
4. componentsToAdd strings -> parsed objects with componentType discriminator
5. Nav buttons (brb*) stripped from objects and objectChanges
6. objectChanges -> sparse delta format (only changed fields included)
7. Field renames for clarity
8. RigidBody/PointerReceiver moved into components array
"""

import json
import sys
import copy

# Object names to strip entirely (nav buttons + deprecated components)
EXCLUDED_OBJECT_NAMES = {
    "brbNextClip", "brbNextClipLabel",
    "brbNextModule", "brbNextModuleLabel",
    "brbPreviousClip", "brbPreviousClipLabel",
    "brbPreviousModule", "brbPreviousModuleLabel",
    "MainInstructions",
}

# Prefab name -> moduleType mapping
PREFAB_TO_MODULE_TYPE = {
    "demoPrefab": "demo",
}

# ── Vector / Color helpers ────────────────────────────────────────────

def convert_vec3(obj):
    """Convert {"x": 0, "y": 1, "z": 2} -> [0, 1, 2]"""
    if obj is None:
        return None
    if isinstance(obj, list):
        return obj  # already an array
    return [obj.get("x", 0), obj.get("y", 0), obj.get("z", 0)]


def convert_color(c):
    """Colors are already arrays in the old format, just pass through."""
    if c is None:
        return None
    return c


# ── Component parsing ─────────────────────────────────────────────────

def parse_component_string(s):
    """Parse a serialized component JSON string into an object with componentType."""
    comp = json.loads(s)
    # Rename "name" to "componentType"
    comp_type = comp.pop("name", None)
    if comp_type:
        comp["componentType"] = comp_type

    # Convert any nested vector-like fields
    for key in list(comp.keys()):
        val = comp[key]
        if isinstance(val, dict) and set(val.keys()) <= {"x", "y", "z", "w"}:
            comp[key] = convert_vec3(val)

    # Move componentType to front for readability
    ordered = {"componentType": comp.pop("componentType")}
    ordered.update(comp)
    return ordered


def parse_tmp_string(s):
    """Parse a serialized TextMeshPro JSON string into a component object."""
    tmp_data = json.loads(s)
    result = {"componentType": "textMeshPro"}

    if "textField" in tmp_data:
        result["text"] = tmp_data["textField"]
    if "color" in tmp_data:
        result["color"] = convert_color(tmp_data["color"])
    if "fontSize" in tmp_data:
        result["fontSize"] = tmp_data["fontSize"]
    if "wrapText" in tmp_data:
        result["wrapText"] = tmp_data["wrapText"]

    return result


# ── Object conversion ─────────────────────────────────────────────────

def convert_scene_object(obj):
    """Convert an object from the objects[] array."""
    result = {}

    # Type -> prefab (strip "Prefabs/" prefix)
    obj_type = obj.get("type", "")
    if obj_type.startswith("Prefabs/"):
        result["prefab"] = obj_type[len("Prefabs/"):]
    elif obj_type:
        result["prefab"] = obj_type

    # Name
    if "name" in obj:
        result["name"] = obj["name"]

    # Parent (standardize parentName/parent)
    parent = obj.get("parent") or obj.get("parentName")
    if parent:
        result["parent"] = parent

    # Transform: convert to arrays
    if "position" in obj:
        result["position"] = convert_vec3(obj["position"])

    # Handle rotation/eulerAngles/euleriAngles (typo) -> eulerAngles
    euler = obj.get("eulerAngles") or obj.get("euleriAngles") or obj.get("rotation")
    if euler:
        result["eulerAngles"] = convert_vec3(euler)

    if "scale" in obj:
        result["scale"] = convert_vec3(obj["scale"])

    # Enabled / active
    if "enabled" in obj:
        result["enabled"] = obj["enabled"]
    if "active" in obj and not obj["active"]:
        result["enabled"] = False

    # Visual properties
    if "texture" in obj and obj["texture"]:
        result["texture"] = obj["texture"]
    if "textureByURL" in obj and obj["textureByURL"]:
        result["textureByURL"] = obj["textureByURL"]
    if "material" in obj and obj["material"]:
        result["material"] = obj["material"]
    if "color" in obj:
        result["color"] = convert_color(obj["color"])
    if "tag" in obj and obj["tag"]:
        result["tag"] = obj["tag"]
    if "transmittable" in obj and obj["transmittable"]:
        result["transmittable"] = obj["transmittable"]

    # Child overrides
    if "childColor" in obj:
        result["childColor"] = convert_color(obj["childColor"])
    if "childName" in obj:
        result["childName"] = obj["childName"]

    # ── Build components array ────────────────────────────────────
    components = []

    # TextMeshPro (tmp) -> component
    if "tmp" in obj and obj["tmp"]:
        try:
            tmp_comp = parse_tmp_string(obj["tmp"])
            components.append(tmp_comp)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse tmp on '{obj.get('name', '?')}': {e}", file=sys.stderr)

    # RigidBody -> component
    if "RigidBody" in obj:
        rb = {"componentType": "rigidBody"}
        rb.update(obj["RigidBody"])
        components.append(rb)

    # PointerReceiver -> component
    if "PointerReceiver" in obj:
        pr = {"componentType": "pointerReceiver"}
        pr.update(obj["PointerReceiver"])
        components.append(pr)

    # componentsToAdd strings -> parsed components
    for comp_str in obj.get("componentsToAdd", []):
        if not comp_str:
            continue
        try:
            comp = parse_component_string(comp_str)
            components.append(comp)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse component on '{obj.get('name', '?')}': {e}", file=sys.stderr)

    if components:
        result["components"] = components

    return result


# ── Object change (clip delta) conversion ─────────────────────────────

def convert_object_change(oc):
    """Convert an objectChange to sparse delta format."""
    result = {}

    # Target name
    result["target"] = oc["name"]

    # Only include enable if it's explicitly false (disabling something)
    if "enable" in oc and not oc["enable"]:
        result["enabled"] = False

    # Sparse transforms: only include if the corresponding flag is true
    if oc.get("newPosition"):
        result["position"] = convert_vec3(oc["position"])

    if oc.get("newEulerAngles"):
        result["eulerAngles"] = convert_vec3(oc["eulerAngles"])

    if oc.get("newScale"):
        result["scale"] = convert_vec3(oc["scale"])

    # Color changes
    if "color" in oc:
        result["color"] = convert_color(oc["color"])
    if "childColor" in oc:
        result["childColor"] = convert_color(oc["childColor"])
    if "childName" in oc:
        result["childName"] = oc["childName"]

    # TextMeshPro updates -> component in components array
    components = []

    if "tmp" in oc and oc["tmp"]:
        try:
            tmp_comp = parse_tmp_string(oc["tmp"])
            components.append(tmp_comp)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse tmp in objectChange '{oc.get('name', '?')}': {e}", file=sys.stderr)

    # componentsToAdd in changes
    for comp_str in oc.get("componentsToAdd", []):
        if not comp_str:
            continue
        try:
            comp = parse_component_string(comp_str)
            components.append(comp)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  WARNING: Failed to parse component in objectChange '{oc.get('name', '?')}': {e}", file=sys.stderr)

    if components:
        result["components"] = components

    return result


# ── Clip conversion ───────────────────────────────────────────────────

def convert_clip(clip):
    """Convert a clip, filtering out nav button changes."""
    result = {}

    result["clipName"] = clip["clipName"]

    if clip.get("audioClipString"):
        result["audioClip"] = clip["audioClipString"]

    if "autoAdvance" in clip:
        result["autoAdvance"] = clip["autoAdvance"]

    # Convert objectChanges, filtering out nav buttons
    changes = []
    for oc in clip.get("objectChanges", []):
        name = oc.get("name", "")
        if name in EXCLUDED_OBJECT_NAMES:
            continue
        change = convert_object_change(oc)
        # Skip if the change is target-only (nothing actually changed)
        if len(change) <= 1:
            continue
        changes.append(change)

    if changes:
        result["changes"] = changes

    return result


# ── Module conversion ─────────────────────────────────────────────────

def convert_module(mod_str):
    """Convert a serialized module string to the new format."""
    mod = json.loads(mod_str)
    result = {}

    # Module type discriminator
    prefab = mod.get("prefabName", "")
    result["moduleType"] = PREFAB_TO_MODULE_TYPE.get(prefab, prefab)

    # Keep prefab reference for instantiation
    result["prefab"] = prefab

    # Core metadata
    result["moduleName"] = mod.get("moduleName", "")
    result["description"] = mod.get("description", "")
    result["author"] = mod.get("author", "")
    if mod.get("authorInstitution"):
        result["institution"] = mod["authorInstitution"]
    if mod.get("dateCreated"):
        result["dateCreated"] = mod["dateCreated"]

    # Educational content
    if mod.get("educationalObjectives") and mod["educationalObjectives"] != [""]:
        result["educationalObjectives"] = mod["educationalObjectives"]
    if mod.get("instructions") and mod["instructions"] != [""]:
        result["instructions"] = mod["instructions"]

    # Audio
    if mod.get("introAudio"):
        result["introAudio"] = mod["introAudio"]

    # Objects: filter out nav buttons
    objects = []
    for obj in mod.get("objects", []):
        name = obj.get("name", "")
        if name in EXCLUDED_OBJECT_NAMES:
            continue
        converted = convert_scene_object(obj)
        objects.append(converted)

    if objects:
        result["objects"] = objects

    # Clips
    clips = []
    for clip in mod.get("clips", []):
        converted = convert_clip(clip)
        clips.append(converted)

    if clips:
        result["clips"] = clips

    return result


# ── Top-level conversion ─────────────────────────────────────────────

def convert_lab(data):
    """Convert the entire lab JSON."""
    result = {}

    result["version"] = "2.0"
    result["labId"] = data.get("Lab_ID", "")
    result["author"] = data.get("Author", "")
    result["courseName"] = data.get("CourseName", "")
    result["estimatedLength"] = data.get("EstimatedLength", "")
    result["objectives"] = data.get("Objectives", [])

    # Convert all modules
    modules = []
    for mod_str in data.get("ActivityModules", []):
        converted = convert_module(mod_str)
        modules.append(converted)

    result["modules"] = modules

    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else  "Artifacts/Data/Raw/Full_Lab_Transmission.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else  "Artifacts/Data/Processed/Optomized_moon_lab_final.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    result = convert_lab(data)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Stats
    input_size = len(json.dumps(data))
    output_size = len(json.dumps(result))
    print(f"Input size:  {input_size:>8,} chars")
    print(f"Output size: {output_size:>8,} chars")
    print(f"Reduction:   {(1 - output_size/input_size)*100:.1f}%")
    print(f"Written to:  {output_path}")


if __name__ == "__main__":
    main()

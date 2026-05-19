"""
arlem_to_preview_scene.py

Small utility to inspect/normalize an ARLEM-style Unity JSON lab file.
This does not try to reproduce Unity perfectly. It extracts the pieces
needed for a quick static 3D preview: modules, objects, transforms,
text labels, textures, and clip changes.

Usage:
    python arlem_to_preview_scene.py Optomized_moon_lab_final.json
    python arlem_to_preview_scene.py Optomized_moon_lab_final.json --module 1 --clip 5 --out preview_scene.json
"""

import argparse
import copy
import json
from pathlib import Path


def merge_components(base, changes):
    out = copy.deepcopy(base or [])
    for change in changes or []:
        ctype = change.get("componentType")
        idx = next((i for i, c in enumerate(out) if c.get("componentType") == ctype), None)
        if idx is None:
            out.append(copy.deepcopy(change))
        else:
            out[idx].update(copy.deepcopy(change))
    return out


def apply_clip(objects, clip):
    objects = copy.deepcopy(objects)
    by_name = {obj.get("name"): obj for obj in objects}
    for change in clip.get("changes", []) or []:
        target = by_name.get(change.get("target"))
        if not target:
            continue
        for key, value in change.items():
            if key == "target":
                continue
            if key == "components":
                target["components"] = merge_components(target.get("components", []), value)
            else:
                target[key] = copy.deepcopy(value)
    return objects


def text_from_components(components):
    for comp in components or []:
        if comp.get("componentType") == "textMeshPro":
            return {
                "text": comp.get("text", ""),
                "color": comp.get("color", [255, 255, 255, 255]),
                "fontSize": comp.get("fontSize", 1),
                "wrapText": comp.get("wrapText", False),
            }
    return None


def normalize_object(obj, flip_z=False):
    position = obj.get("position", [0, 0, 0])
    euler = obj.get("eulerAngles", [0, 0, 0])
    if flip_z:
        position = [position[0], position[1], -position[2]]
        euler = [euler[0], -euler[1], euler[2]]

    return {
        "name": obj.get("name"),
        "prefab": obj.get("prefab"),
        "parent": obj.get("parent", "[CURRENT_LAB]"),
        "position": position,
        "eulerAnglesDegrees": euler,
        "scale": obj.get("scale", [1, 1, 1]),
        "texture": obj.get("texture"),
        "color": obj.get("color"),
        "text": text_from_components(obj.get("components", [])),
        "components": obj.get("components", []),
    }


def convert(path, module_index=0, clip_index=None, flip_z=False):
    lab = json.loads(Path(path).read_text(encoding="utf-8"))
    module = lab["modules"][module_index]
    objects = module.get("objects", [])
    selected_clip = None

    if clip_index is not None:
        selected_clip = module.get("clips", [])[clip_index]
        objects = apply_clip(objects, selected_clip)

    return {
        "source": str(path),
        "labId": lab.get("labId"),
        "courseName": lab.get("courseName"),
        "moduleIndex": module_index,
        "moduleName": module.get("moduleName"),
        "clipIndex": clip_index,
        "clipName": selected_clip.get("clipName") if selected_clip else None,
        "objects": [normalize_object(o, flip_z=flip_z) for o in objects],
    }


def summarize(path):
    lab = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Lab: {lab.get('labId')}")
    print(f"Version: {lab.get('version')}")
    print(f"Modules: {len(lab.get('modules', []))}")
    for i, m in enumerate(lab.get("modules", [])):
        prefabs = sorted({o.get("prefab") for o in m.get("objects", [])})
        textures = sorted({o.get("texture") for o in m.get("objects", []) if o.get("texture")})
        print(f"\n[{i}] {m.get('moduleName')}")
        print(f"  objects: {len(m.get('objects', []))}, clips: {len(m.get('clips', []))}")
        print(f"  prefabs: {', '.join(prefabs)}")
        print(f"  textures: {', '.join(textures) if textures else '(none)'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("--module", type=int, default=None)
    parser.add_argument("--clip", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--flip-z", action="store_true")
    args = parser.parse_args()

    if args.module is None:
        summarize(args.json_file)
        return

    preview = convert(args.json_file, args.module, args.clip, flip_z=args.flip_z)
    output = json.dumps(preview, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()

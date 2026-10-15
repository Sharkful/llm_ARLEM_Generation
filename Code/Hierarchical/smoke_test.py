"""
Offline smoke test for the hierarchical pipeline — no API keys, no paid calls.

    python "Code/Hierarchical/smoke_test.py"

Part 1 swaps the LLM for a fake call_fn that replays a committed wave-2
multi-module output one module at a time (plan <- its module list, asset plan <-
each module's objects/components, module <- the module itself), validated through
the real response models and validation scopes. It covers the auto-approved run,
name/description normalization, Revise/Replace review decisions (including a
rejected edit and accumulated feedback), failure + resume, an interrupted run,
fail-fast errors, the scene cap, per-module minimums, all-errors-at-once asset-plan
validation, run groups, PascalCase names, soft metrics, the NewComponent check,
declare-once new components, structured inputs, one error per bad component entry,
and the --no-new-components switch.

Part 2 runs the real _llm_call path (instructor client, tracker, transient
retryer, provider kwargs) against a local mock OpenAI server that replays part
1's step files, with one invalid module response to force an instructor reask.

Part 3 builds the real Anthropic client against a local mock Anthropic server and
checks the request shape we send (tool_choice auto + disable_parallel_tool_use,
the system line asking for the tool, max_tokens), including instructor's reask
after a plain-text reply.

Outputs go to a temp dir that is removed afterwards. Exit code 1 on any failure.
"""

import atexit
import json
import os
import re
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402
import prompts  # noqa: E402
from _paths import PROJECT_ROOT  # noqa: E402
from json_lab import DemoModule, Lab, NewComponent  # noqa: E402
from module_asset_plan import asset_plan_validation  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from lab_outline import LabOutline  # noqa: E402
from module_asset_plan import ModuleAssetPlan  # noqa: E402
from review import Approve, Replace, Revise  # noqa: E402

# A 9-module wave-2 output with 13 distinct NewComponents, several reused across modules.
SAMPLE = (PROJECT_ROOT / "Artifacts" / "Data" / "Benchmark" / "Outputs"
          / "claude-opus-4-8_json_lab_L4_20260710_183648_output.json")
LAB_NAME = "apparent_retrograde_motion"
SOURCE = json.loads(SAMPLE.read_text(encoding="utf-8"))
MODS = SOURCE["modules"]
TMP = Path(tempfile.mkdtemp(prefix="hier_smoke_"))
atexit.register(shutil.rmtree, TMP, ignore_errors=True)  # also on a crash

_BUILTIN_BY_COMPONENT_TYPE = {
    "textMeshPro": "TextMeshProComponent", "simpleRotation": "SimpleRotationComponent",
    "simpleOrbit": "SimpleOrbitComponent", "checkAngle": "CheckAngleComponent",
    "rigidBody": "RigidBodyComponent", "pointerReceiver": "PointerReceiverComponent",
}

FAILED = False
# The fake asset plans declare this one input on every new component (#74).
FAKE_INPUT = {"name": "target_object", "type": "GameObject", "description": "object it acts on"}


def check(cond, msg):
    global FAILED
    print(("PASS " if cond else "FAIL ") + msg)
    FAILED |= not cond


def _comp_name(c):
    if c["componentType"] == "newscript":
        return c["scriptName"]
    return _BUILTIN_BY_COMPONENT_TYPE[c["componentType"]]


class FakeLLM:
    """call_fn replaying SOURCE; `pipe` is attached after construction."""

    def __init__(self, fail_module=None, fail_exc=None):
        self.pipe = None
        self.fail_module = fail_module
        self.fail_exc = fail_exc        # exception (instance) to raise at fail_module
        self.calls = []
        self.last_user = []             # last user message of every call

    def _index(self):
        return len(self.pipe.registry.modules) + 1

    def __call__(self, model, messages, max_retries):
        self.calls.append((model.__name__, len(messages)))
        self.last_user.append(messages[-1]["content"])
        if issubclass(model, LabOutline):
            return model.model_validate({
                "lab_title": "Apparent Retrograde Motion",
                "top_level_objectives": SOURCE["objectives"],
                "scenes": [{"scene_name": m["moduleName"], "brief_purpose": m["description"],
                            "key_visuals": [o["name"] for o in m["objects"]][:3],
                            "student_actions": ["advance through the clips"]} for m in MODS],
            })
        m = MODS[self._index() - 1]
        if model is ModuleAssetPlan:
            pairs = [(o["name"], c) for o in m["objects"] for c in o.get("components") or []]
            pairs += [(ch["target"], c) for cl in m["clips"] for ch in cl.get("changes") or []
                      for c in ch.get("components") or []]
            comps, new_comps, seen = [], [], set()
            for obj_name, c in pairs:
                name = _comp_name(c)
                if (c["componentType"] == "newscript"
                        and self.pipe.registry.find_component(name) is None
                        and name.lower() not in seen):
                    new_comps.append({"name": name, "behavior_description": c["scriptDescription"],
                                      "why_existing_insufficient": "no built-in component does this",
                                      "inputs": [FAKE_INPUT]})
                seen.add(name.lower())
                comps.append({"object_name": obj_name, "component_type": name,
                              "rationale": "needed for this scene's behavior"})
            beats = [cl["changeMeaning"] for cl in m["clips"]]
            return model.model_validate({
                "module_name": m["moduleName"],
                "design_notes": "What the student must see and do in this scene, and why.",
                "objects": [{"name": o["name"], "prefab": o["prefab"], "texture": o.get("texture"),
                             "purpose": f"Shows {o['name']} in this scene", "is_new_asset": False}
                            for o in m["objects"]],
                "new_components": new_comps,
                "components": comps,
                "interactions": [{"object_name": m["objects"][0]["name"],
                                  "action": "advances with Next",
                                  "effect": "the next clip plays"}],
                "clip_beats": beats if len(beats) >= 2 else beats + ["recap"],
            })
        if self.fail_module == self._index():
            raise self.fail_exc or ValueError(f"forced failure on module {self._index()}")
        return model.model_validate(m)


# The replayed wave-2 modules were generated under lab-wide minimums, so they are
# smaller than the pipeline's per-module defaults; replay runs use minimums the
# source data meets (the minimum checks themselves are tested separately).
# The 9-module source also exceeds the default 8-scene cap.
REPLAY_MINIMUMS = {"min_objects": 1, "min_clips": 2, "max_scenes": 12}


def make(fake, run_name, cfg=None, **kw):
    config = P.PipelineConfig(**{**REPLAY_MINIMUMS, **(cfg or {})})
    pipe = P.HierarchicalPipeline("claude-haiku-4.5", LAB_NAME, config,
                                  output_dir=TMP / "fake", run_name=run_name, call_fn=fake, **kw)
    fake.pipe = pipe
    return pipe


# ── Part 1: fake LLM ─────────────────────────────────────────────────
FAKE = TMP / "fake"

# 1. full auto-approved run
rec = make(FakeLLM(), "t1_auto").run()
check(rec["success"], "auto run succeeds")
lab = Lab.model_validate_json(Path(rec["output_path"]).read_text(encoding="utf-8"))
check(len(lab.modules) == len(MODS), f"assembled Lab has {len(MODS)} modules")
check(rec["lab_metrics"]["num_clips"] == sum(len(m["clips"]) for m in MODS), "clip count preserved")
out = json.loads((FAKE / "Outputs" / "t1_auto_output.json").read_text(encoding="utf-8"))
# Compare with the source as the current schema normalizes it (PascalCase
# scriptNames), so the pipeline itself must not alter a module.
SRC_NORMALIZED = [P._dump(DemoModule.model_validate(m)) for m in MODS]


def _strip_scripts(node):
    """Drop dateCreated and NewComponent name/description/inputs, which the
    pipeline deliberately normalizes; everything else must come through untouched."""
    if isinstance(node, dict):
        return {k: _strip_scripts(v) for k, v in node.items()
                if k not in ("dateCreated", "scriptName", "scriptDescription", "inputs")}
    if isinstance(node, list):
        return [_strip_scripts(v) for v in node]
    return node


def _script_variants(lab_json):
    """scriptName -> set of descriptions used for it across the whole lab."""
    variants = {}
    for m in lab_json["modules"]:
        for c in re.findall(r'"scriptName": "([^"]+)", "scriptDescription": "((?:[^"\\]|\\.)*)"',
                            json.dumps(m)):
            variants.setdefault(c[0], set()).add(" ".join(c[1].split()))
    return variants


check(all(_strip_scripts(a) == _strip_scripts(b) for a, b in zip(out["modules"], SRC_NORMALIZED)),
      "modules come through unchanged apart from NewComponent normalization")
variants = _script_variants(out)
check(variants and all(len(d) == 1 for d in variants.values()),
      f"one description per new component across the lab ({len(variants)} components)")
src_variants = _script_variants({"modules": SRC_NORMALIZED})
check(any(len(d) > 1 for d in src_variants.values()) and rec["normalizations"],
      f"source had divergent copies; {len(rec['normalizations'])} normalizations logged")
run_dir = FAKE / "Runs" / "t1_auto"
files = {p.name for p in run_dir.iterdir()}
check({"run_config.json", "review_log.json", "00_plan.json", "00_plan_prompt.txt",
       "01_asset_plan.json", "01_module.json", "01_module_prompt.txt"} <= files, "run dir files")
check(all(a["planned_objects"] == a["planned_objects_present"]
          and a["planned_components"] == a["planned_components_present"]
          for a in rec["plan_adherence"]), "plan adherence 100% on replayed modules")
newc = rec["registry"]["new_components"]
check(len(newc) > 0, f"registry collected {len(newc)} new components")
first = newc[0]
later = [p for p in sorted(run_dir.glob("*_asset_plan_prompt.txt"))
         if int(p.name[:2]) > first["origin_module"]]
check(bool(later) and first["name"] in later[0].read_text(encoding="utf-8"),
      f"'{first['name']}' (module {first['origin_module']}) is in a later asset-plan catalog")
check(len(rec["steps"]) == 1 + 2 * len(MODS) and all(s["success"] for s in rec["steps"]),
      "one successful step record per step")
# structured inputs (#74): the plan's definitions reach the module, the catalog
# and later prompts, and their types are tallied for the metrics
check(first["inputs"] == [{"name": "targetObject", "type": "GameObject",
                           "description": "object it acts on"}]
      and "targetObject (GameObject): object it acts on" in later[0].read_text(encoding="utf-8"),
      "planned inputs (name camelCased) carried by the catalog into later prompts")
check(all(c.get("inputs") == [{k: v for k, v in first["inputs"][0].items()}]
          for m in out["modules"] for o in m["objects"] for c in o.get("components") or []
          if c["componentType"] == "newscript"),
      "every NewComponent instance carries its canonical input definitions")
check(rec["new_component_input_types"] == {"GameObject": len(newc)},
      "input types tallied across the lab (new_component_input_types)")
check("its inputs: every inspector field it exposes\n  (object references and values) as "
      "{name, type, description}" in later[0].read_text(encoding="utf-8"),
      "asset-plan prompt asks for each new component's inputs (they are required)")
# the module prompt shows each usable new component as a ready-to-copy entry (#77)
_mp = (run_dir / f"{first['origin_module']:02d}_module_prompt.txt").read_text(encoding="utf-8")
check(f'{{"type": "NewComponent", "scriptName": "{first["name"]}", "scriptDescription": ' in _mp
      and '"value": "<value for this instance>"' in _mp,
      "module prompt lists each allowed new component as a NewComponent entry to copy")
check(rec["pipeline"] == "hierarchical" and rec["structure"] == "multi-module", "metrics keys")

# 2. scripted reviewer: revise the plan once, replace asset plan 1


class ScriptedReviewer:
    name = "scripted"

    def review(self, cp):
        if cp.step == "plan" and cp.revision == 0:
            return Revise("Add a recap scene.")
        if cp.step == "asset_plan" and cp.index == 1:
            edited = P._dump(cp.output)
            edited["interactions"] = [{"object_name": edited["objects"][0]["name"],
                                       "action": "EDITED BY HUMAN", "effect": "edited"}]
            return Replace(edited, note="human edit")
        return Approve()


fake = FakeLLM()
rec = make(fake, "t2_review", reviewer=ScriptedReviewer()).run()
check(rec["success"], "scripted-review run succeeds")
check(fake.calls[:2] == [("LabPlan", 2), ("LabPlan", 4)],
      "Revise re-calls the plan with previous output + feedback appended")
ap1 = json.loads((FAKE / "Runs" / "t2_review" / "01_asset_plan.json").read_text(encoding="utf-8"))
check(ap1["interactions"][0]["action"] == "EDITED BY HUMAN", "Replace output is the one saved")
check([e["decision"] for e in rec["review_log"][:3]] == ["Revise", "Approve", "Replace"],
      "review log decisions")
check(rec["steps"][0]["revisions"] == 1, "plan step records 1 revision")

# 3. failure at module 2, then resume
rec = make(FakeLLM(fail_module=2), "t3_resume", cfg={"step_attempts": 2}).run()
check(not rec["success"] and rec["failed_step"] == "module 2", "run fails at module 2")
check(rec["steps"][-1]["attempts"] == 2 and rec["steps"][-1]["failed_attempts"] == 2,
      "module 2 regenerated step_attempts times")
check(not (FAKE / "Runs" / "t3_resume" / "02_module.json").exists(), "failed module not saved")
fake = FakeLLM()
pipe = P.HierarchicalPipeline.resume(FAKE / "Runs" / "t3_resume", call_fn=fake)
fake.pipe = pipe
rec = pipe.run()
check(rec["success"], "resumed run succeeds")
check(fake.calls[0] == ("PlannedDemoModule", 2), "resume starts at module 2")
check(sum(s["resumed"] for s in rec["steps"]) == 4, "4 steps loaded from disk")
m2 = next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 2)
check(m2["attempts"] == 3 and m2["failed_attempts"] == 2 and not m2["resumed"]
      and len(m2["errors"]) == 2,
      "regenerated step keeps the earlier session's attempts and errors")
check(m2["errors"] == [f"step attempt {n} failed: ValueError: forced failure on module 2"
                       for n in (1, 2)],
      "one error line per failed step attempt, no duplicates")
t3 = json.loads((FAKE / "Outputs" / "t3_resume_output.json").read_text(encoding="utf-8"))
check(t3["modules"] == out["modules"], "resumed lab identical to the uninterrupted lab")

# 4. PlannedDemoModule rejects an unplanned NewComponent
mod = next(m for m in MODS if "newscript" in json.dumps(m))
with P.module_validation([]):
    try:
        P.PlannedDemoModule.model_validate(mod)
        check(False, "unplanned NewComponent rejected")
    except Exception as e:  # noqa: BLE001
        check("were not planned" in str(e), "unplanned NewComponent rejected")

# 4b. an invented component written as its own type gets one clear error line per
# entry in the module step (#77), not every union member's failure
import copy  # noqa: E402
_bad_mod = copy.deepcopy(MODS[0])
_bad_mod["objects"][1]["components"] = [{"type": "ToggleOnClick", "target": "x"},
                                        {"type": "NewComponent", "scriptName": "ToggleOnClick"}]


def _module_errors(scope):
    with scope:
        try:
            P.PlannedDemoModule.model_validate(_bad_mod)
            return []
        except ValidationError as e:
            return [ln for err in e.errors()
                    for ln in str(err["msg"]).removeprefix("Value error, ").splitlines()]


_bad_mod["objects"][1]["components"] += [{"type": "PulseGlow"}]
_lines = _module_errors(P.module_validation(["ToggleOnClick"]))
check(len(_lines) == 4 and "'ToggleOnClick' is not a built-in component type" in _lines[0]
      and "components[0]" in _lines[0] and "type: NewComponent, scriptName: 'ToggleOnClick'" in _lines[0]
      and _lines[1].startswith("components[1] (NewComponent) scriptDescription")
      and "'PulseGlow' is not a built-in" in _lines[2]
      and _lines[3].startswith("Built-in component types: TextMeshProComponent")
      and sum("SimpleOrbitComponent" in ln for ln in _lines) == 1,
      "module step: one error line per bad component entry, built-ins listed once")
_bad_mod["objects"][1]["components"] = [{"type": "simple orbitComponent"}, {"type": "newComponent"}]
_lines = _module_errors(P.module_validation([]))
check(_lines == ["components[0]: type 'simple orbitComponent' is misspelled; did you mean "
                 "'SimpleOrbitComponent'?",
                 "components[1]: type 'newComponent' is misspelled; did you mean 'NewComponent'?"],
      "a miscased built-in tag gets 'did you mean', not the invented-component hint")
_bad_mod["objects"][1]["components"] = [{"type": ["NewComponent"], "scriptName": "Toggle"}]
_lines = _module_errors(P.module_validation([]))  # must be a ValidationError (a reask), not a crash
check(bool(_lines), "a malformed (non-string) type is reported by the union, not a crash")
_bad_mod["objects"][1]["components"] = [{"type": "ToggleOnClick", "target": "x"}]
check(len(_module_errors(P.plan_validation(12))) > 10,
      "outside the module step (one-shot runs) the union's full error list is unchanged")

# 5. --no-new-components: an asset plan proposing a new component fails its step
rec = make(FakeLLM(), "t5_nonew", cfg={"allow_new_components": False, "step_attempts": 1}).run()
first_new = next(i for i, m in enumerate(MODS, 1) if "newscript" in json.dumps(m))
check(not rec["success"] and rec["failed_step"] == f"asset_plan {first_new}",
      f"no-new-components fails at asset_plan {first_new}")
check("disabled" in (rec["error"] or ""), "error says new components are disabled")
prompt = (FAKE / "Runs" / "t5_nonew" / "01_asset_plan_prompt.txt").read_text(encoding="utf-8")
check("New components are disabled" in prompt, "prompt states the switch")


# 6. --run-group resolves to one folder under Artifacts/Data/Hierarchical/
check(P.output_dir_for("pilot_oct") == P.DEFAULT_OUTPUT_DIR / "pilot_oct"
      and P.output_dir_for(None) == P.DEFAULT_OUTPUT_DIR, "run group folder")
import benchmark_config as _bc  # noqa: E402
check(_bc.BenchmarkRunConfig().output_dir == "Artifacts/Data/OneShot"
      and P.run_group_dir(_bc.ONESHOT_OUTPUT_DIR, "trial") == Path("Artifacts/Data/OneShot/trial"),
      "one-shot runs go to Artifacts/Data/OneShot/[<run group>/], never the frozen Benchmark/")
for bad in ("../x", "a/b", "", "has space"):
    try:
        P.output_dir_for(bad)
        check(False, f"run group {bad!r} rejected")
    except ValueError:
        pass
check(True, "invalid run group names rejected")

# 7. PascalCase script names, shared schema and asset plan
check(NewComponent(scriptName="orbit_trail", scriptDescription="d" * 25).scriptName == "OrbitTrail",
      "json_lab NewComponent.scriptName normalized to PascalCase")

# 8. every asset-plan problem reported in one error
_obj = {"prefab": "SunPrefab", "purpose": "reference body in the scene", "is_new_asset": False}
_bad_plan = {
    "module_name": "M", "design_notes": "d" * 40, "clip_beats": ["a", "b"],
    "objects": [{"name": "sun", **_obj}, {"name": "sun", **_obj}],
    "components": [
        {"object_name": "moon", "component_type": "SimpleOrbitComponent", "rationale": "r" * 20},
        {"object_name": "sun", "component_type": "SimpleRotationComponent", "rationale": "r" * 20},
        {"object_name": "sun", "component_type": "SimpleRotationComponent", "rationale": "r" * 20},
    ],
    "interactions": [{"object_name": "earth", "action": "drags it", "effect": "it moves"}],
}
with asset_plan_validation(["SimpleOrbitComponent", "SimpleRotationComponent"],
                           min_objects=4, min_clip_beats=5):
    try:
        ModuleAssetPlan.model_validate(_bad_plan)
        check(False, "all asset-plan problems reported at once")
    except ValidationError as e:
        msg = str(e)
        check("6 problem(s)" in msg and all(t in msg for t in (
            "Duplicate planned object", "at least 4 scene objects", "at least 5 clip beats",
            "unknown object 'earth'", "unknown object", "planned more than once")),
            "all asset-plan problems reported at once (duplicates, minimums, interactions)")

# 8b. declare once, attach many (#77): attachments resolve against the catalog plus
# new_components; a declaration is checked once, not per attachment
_decl = {"name": "pulse_glow", "behavior_description": "pulses the object's emission color",
         "why_existing_insufficient": "no catalog component changes emission", "inputs": []}
_glow_plan = {
    "module_name": "M", "design_notes": "d" * 40, "clip_beats": ["a", "b"],
    "objects": [{"name": "sun", **_obj}, {"name": "moon", **_obj}],
    "components": [
        {"object_name": "sun", "component_type": "PulseGlow", "rationale": "r" * 20},
        {"object_name": "moon", "component_type": "pulseGlow", "rationale": "r" * 20},
    ],
}
_catalog = ["SimpleOrbitComponent", "SimpleRotationComponent"]
with asset_plan_validation(_catalog):
    try:
        ModuleAssetPlan.model_validate(_glow_plan)
        check(False, "attachment to an undeclared new component rejected")
    except ValidationError as e:
        check("2 problem(s)" in str(e) and "declare it in new_components" in str(e),
              "attachment to an undeclared new component rejected")
    ok = ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [_decl]})
    check(ok.new_components[0].name == "PulseGlow",
          "one declaration covers every attachment; its name normalized to PascalCase")
    try:
        ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [_decl, _decl]})
        check(False, "a new component declared twice is rejected")
    except ValidationError as e:
        check("declared more than once" in str(e), "a new component declared twice is rejected")
    try:
        ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [
            _decl, {**_decl, "name": "SimpleOrbitComponent"}]})
        check(False, "a new component re-declaring a catalog entry is rejected")
    except ValidationError as e:
        check("already exists in the component catalog" in str(e),
              "a new component re-declaring a catalog entry is rejected")
with asset_plan_validation(_catalog, allow_new=False):
    try:
        ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [_decl]})
        check(False, "--no-new-components rejects a non-empty new_components")
    except ValidationError as e:
        check("leave new_components empty" in str(e),
              "--no-new-components rejects a non-empty new_components")
    try:
        ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [{**_decl, "inputs": [
            {"name": "pulseSpeed", "type": "float", "description": "pulses per second"},
            {"name": "pulsespeed", "type": "float", "description": "pulses per second"}]}]})
        check(False, "input names differing only in case are duplicates")
    except ValidationError as e:
        check("listed more than once" in str(e), "input names differing only in case are duplicates")
# declared but attached nowhere: a soft metric, not a reask
with asset_plan_validation(_catalog):
    _unused = ModuleAssetPlan.model_validate({**_glow_plan, "components": [], "new_components": [
        _decl, {**_decl, "name": "OrbitTrail"}]})
_adh = P.plan_adherence(_unused, DemoModule.model_validate(MODS[0]), min_objects=1, min_clips=1)
check(_adh["unattached_new_components"] == ["PulseGlow", "OrbitTrail"],
      "declared-but-unattached new components recorded in plan_adherence (no reask)")

# 8c. structured inputs: required on a declaration; the module step rejects an input
# the component doesn't declare (one line per mistake); accepted instances keep their
# values but take the canonical definitions
with asset_plan_validation(_catalog):
    try:
        ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [
            {k: v for k, v in _decl.items() if k != "inputs"}]})
        check(False, "a new component declared without inputs is rejected")
    except ValidationError as e:
        check("inputs" in str(e) and "Field required" in str(e),
              "a new component declared without inputs is rejected")
    _glow = ModuleAssetPlan.model_validate({**_glow_plan, "new_components": [{**_decl, "inputs": [
        {"name": "PulseSpeed", "type": "float", "description": "pulses per second, 0.1-5"},
        {"name": "glowColor", "type": "Color", "description": "emission color"}]}]})
check([i.name for i in _glow.new_components[0].inputs] == ["pulseSpeed", "glowColor"],
      "input names normalized to camelCase")
from json_lab import NewComponentInput, to_camel_case  # noqa: E402
check([to_camel_case(n) for n in ("URL", "XMLSource", "IDLabel", "ID2", "pulse_speed", "PulseSpeed")]
      == ["url", "xmlSource", "idLabel", "id2", "pulseSpeed", "pulseSpeed"],
      "camelCase lowercases a leading acronym whole (URL -> url, XMLSource -> xmlSource)")
check([NewComponentInput(name="a", type="float", description="speed x", value=v).value
       for v in (2.5, True, 3, [255, 0, 0, 255], "Mars", None)]
      == ["2.5", "true", "3", "[255, 0, 0, 255]", "Mars", None],
      "input values written as numbers, bools or arrays are stored as text, not reasked")
_speed = {"name": "pulse_speed", "type": "Single", "description": "speed!", "value": "2.5"}
_wobble = {"name": "wobble", "type": "float", "description": "not planned", "value": "1"}


def _glow_module(inputs):
    """MODS[0] with its own components removed and one PulseGlow on its first object."""
    inst = NewComponent(scriptName="PulseGlow", scriptDescription="pulses the emission color",
                        inputs=inputs)
    mod = copy.deepcopy(MODS[0])
    for o in mod["objects"]:
        o.pop("components", None)
    for cl in mod["clips"]:
        for ch in cl.get("changes") or []:
            ch.pop("components", None)
    mod["objects"][0]["components"] = [inst.model_dump()]
    return mod


_scope = lambda: P.module_validation(  # noqa: E731
    prompts.allowed_new_component_names(_glow, P.AssetRegistry()),
    prompts.allowed_new_component_inputs(_glow, P.AssetRegistry()))
with _scope():
    try:
        P.PlannedDemoModule.model_validate(_glow_module([_speed, _wobble]))
        check(False, "module step rejects an undeclared input name")
    except ValidationError as e:
        _msgs = [ln for err in e.errors() for ln in str(err["msg"]).splitlines()]
        check(len(_msgs) == 1 and "input 'wobble' isn't one of its inputs ['pulseSpeed', 'glowColor']"
              in _msgs[0], "module step rejects an undeclared input name, in one line")
    try:
        P.PlannedDemoModule.model_validate(_glow_module([{**_speed, "value": prompts.VALUE_PLACEHOLDER}]))
        check(False, "module step rejects the prompt's value placeholder copied verbatim")
    except ValidationError as e:
        check("still has the placeholder value" in str(e),
              "module step rejects the prompt's value placeholder copied verbatim")
    _mod = P.PlannedDemoModule.model_validate(_glow_module([_speed]))
_changes = P.normalize_module(_mod, _glow, P.AssetRegistry())
_after = _mod.objects[0].components[0].inputs
check([(i.name, i.type, i.value) for i in _after]
      == [("pulseSpeed", "float", "2.5"), ("glowColor", "Color", None)],
      "instance inputs take the planned definitions and keep their values")
check({"field": "inputs", "component": "PulseGlow", "redefined": ["pulseSpeed"]} in _changes,
      "redefined inputs logged as normalizations")
check("inputs" not in P._dump(NewComponent(scriptName="OrbitTrail", scriptDescription="d" * 25)),
      "a NewComponent without inputs dumps exactly as before (wave-2 outputs unchanged)")

# 9. per-module minimums are enforced on the asset plan (defaults 4 objects / 5 beats)
rec = make(FakeLLM(), "t9_minimums",
           cfg={"min_objects": 4, "min_clips": 5, "step_attempts": 1}).run()
check(not rec["success"] and rec["failed_step"].startswith("asset_plan")
      and "clip beats" in rec["error"], "per-module minimums enforced on the asset plan")

# 10. scene cap
rec = make(FakeLLM(), "t10_cap", cfg={"max_scenes": 8, "step_attempts": 1}).run()
check(not rec["success"] and rec["failed_step"] == "plan" and "at most 8 scenes" in rec["error"],
      f"plan with {len(MODS)} scenes rejected by the 8-scene cap")


# 11. invalid Replace comes back with every error; feedback accumulates; context
class TrickyReviewer:
    name = "tricky"

    def __init__(self):
        self.seen = []

    def review(self, cp):
        self.seen.append(cp)
        if cp.step == "plan" and cp.revision < 2:
            return Revise(["Add a recap scene.", "Shorten scene 2."][cp.revision])
        if cp.step == "asset_plan" and cp.index == 1 and cp.revision == 0:
            bad = P._dump(cp.output)
            bad["interactions"] = [{"object_name": "nope", "action": "pokes it", "effect": "nothing"}]
            bad["objects"] = bad["objects"] + [bad["objects"][0]]
            return Replace(bad)
        return Approve()


reviewer, fake = TrickyReviewer(), FakeLLM()
rec = make(fake, "t11_review", reviewer=reviewer).run()
check(rec["success"], "tricky-review run succeeds")
second_revision = fake.last_user[2]
check("1. Add a recap scene." in second_revision and "2. Shorten scene 2." in second_revision,
      "second revision request carries both rounds of feedback")
after_bad = next(cp for cp in reviewer.seen if cp.step == "asset_plan" and cp.last_error)
check("2 problem(s)" in after_bad.last_error and "unknown object 'nope'" in after_bad.last_error
      and "Duplicate planned object" in after_bad.last_error and after_bad.rejected_output,
      "rejected Replace returns all its errors and the rejected edit to the reviewer")
check(after_bad.revision == 1 and rec["steps"][1]["decision"] == "Approve",
      "rejected edit counts as a revision round; last valid output kept")
mod_cp = next(cp for cp in reviewer.seen if cp.step == "module")
check({"scene", "plan", "previous_modules", "asset_plan", "run_dir"} <= set(mod_cp.context),
      "module checkpoint context has plan, earlier modules, asset plan, run_dir")


# 12. errors a regeneration can't fix fail the step at once
class FakeAuthError(Exception):
    status_code = 401


rec = make(FakeLLM(fail_module=1, fail_exc=FakeAuthError("bad key")), "t12_failfast",
           cfg={"step_attempts": 3}).run()
m1 = rec["steps"][-1]
check(not rec["success"] and m1["attempts"] == 1 and m1["non_retryable"]
      and "not retryable" in rec["error"], "401 fails the step without regenerating")

# 13. metrics survive an interruption, and resume keeps them
try:
    make(FakeLLM(fail_module=2, fail_exc=KeyboardInterrupt()), "t13_interrupt").run()
except KeyboardInterrupt:
    pass
partial = json.loads((FAKE / "Metrics" / "t13_interrupt_metrics.json").read_text(encoding="utf-8"))
check(partial["status"] == "interrupted" and partial["failed_step"] == "module 2"
      and [s["step"] for s in partial["steps"]]
      == ["plan", "asset_plan", "module", "asset_plan", "module"]
      and not partial["steps"][-1]["success"],
      "an interruption leaves an 'interrupted' record naming the step")
fake = FakeLLM()
pipe = P.HierarchicalPipeline.resume(FAKE / "Runs" / "t13_interrupt", call_fn=fake)
fake.pipe = pipe
rec = pipe.run()
check(rec["success"] and rec["status"] == "complete"
      and sum(s["resumed"] for s in rec["steps"]) == 4, "interrupted run resumes to completion")
check(next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 2)["attempts"] == 2,
      "the interrupted attempt stays counted after resume")

# 14. soft metrics recorded per module
adh = rec["plan_adherence"][0]
check({"below_min_objects", "below_min_clips", "interaction_objects_without_pointer",
       "is_new_asset_mismatches"} <= set(adh), "soft metrics recorded in plan_adherence")
check(any(a["is_new_asset_mismatches"] for a in rec["plan_adherence"]),
      "is_new_asset claims checked against the registry (fake always claims False)")


# 15. environment preflight: a missing provider SDK stops the run with a fix-it message
import benchmark  # noqa: E402
import importlib.util  # noqa: E402
_real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a: None if name == "anthropic" else _real_find_spec(name, *a)
try:
    errors, _ = benchmark.check_environment([P.MODELS["claude-haiku-4.5"], P.MODELS["gpt-5.4-mini"]])
finally:
    importlib.util.find_spec = _real_find_spec
check(len(errors) == 1 and "'anthropic' is not installed" in errors[0],
      "preflight reports a missing provider SDK")
check(benchmark._pinned_version("instructor") == "1.17.0", "preflight reads the instructor pin")
# The #44 Gemini patch must wrap the parse_genai_tools the GENAI_TOOLS handler calls
# (a module-global lookup) and turn the text + one-functionCall assert into a
# retryable ResponseParsingError; re-check on every instructor upgrade (issue #70).
try:
    import instructor.v2.providers.genai.handlers as _genai_handlers
    from google.genai import types as _gt
    from instructor.v2.core.errors import ResponseParsingError as _RPE
    benchmark._patch_genai_parallel_call_retry()
    _shape = _gt.GenerateContentResponse(candidates=[_gt.Candidate(content=_gt.Content(parts=[
        _gt.Part(text="Here is the plan."),
        _gt.Part(function_call=_gt.FunctionCall(name="LabPlan", args={})),
    ]))])
    try:
        _genai_handlers.parse_genai_tools(P.LabPlan, _shape)
        _patched = False
    except _RPE:
        _patched = True
    _calls_module_global = "model = parse_genai_tools(" in Path(_genai_handlers.__file__).read_text(
        encoding="utf-8")
    check(_patched and _calls_module_global,
          "Gemini text + functionCall parse error is retryable under the patch")
except Exception as _e:  # noqa: BLE001 — report, don't crash the offline test
    check(False, f"Gemini parallel-call retry patch check ({type(_e).__name__}: {_e})")
_req = TMP / "requirements_marker.txt"
_req.write_text('instructor[anthropic]==1.15.4 ; python_version >= "3.10"  # pinned\n', encoding="utf-8")
check(benchmark._pinned_version("instructor", _req) == "1.15.4",
      "preflight pin ignores environment markers and comments")
# The transient retryer is the only transport retry (issue #54): the SDK clients
# the factory builds don't re-send on their own (dummy keys; nothing is sent).
_env = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
os.environ.update(OPENAI_API_KEY=_env["OPENAI_API_KEY"] or "test",
                  ANTHROPIC_API_KEY=_env["ANTHROPIC_API_KEY"] or "test")
try:
    _sdk_retries = {m: benchmark.create_instructor_client(P.MODELS[m], P.SpecType.JSON_LAB)
                    .client.max_retries for m in ("gpt-5.4-mini", "claude-haiku-5.5")}
except Exception as _e:  # noqa: BLE001 — report, don't crash the offline test
    _sdk_retries = {"error": f"{type(_e).__name__}: {_e}"}
finally:
    for _k, _v in _env.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v
check(_sdk_retries == {"gpt-5.4-mini": 0, "claude-haiku-5.5": 0},
      f"OpenAI and Anthropic SDK retries are off; the transient retryer re-sends ({_sdk_retries})")


# 16. one-line console errors; new-component counts in the metrics
try:
    ModuleAssetPlan.model_validate({"module_name": "M"})
except ValidationError as _ve:
    _validation_error = _ve


class _FakeAttempt:
    def __init__(self, n, exc):
        self.attempt_number, self.exception = n, exc


class InstructorRetryException(Exception):
    def __init__(self):
        super().__init__("<failed_attempts>\n<generation number=\"1\">... long XML ...")
        self.n_attempts = 4
        self.failed_attempts = [_FakeAttempt(i, _validation_error) for i in range(1, 5)]


line = P.brief_error(InstructorRetryException())
check(line.startswith("InstructorRetryException after 4 attempt(s); last: ValidationError")
      and "\n" not in line and "<failed_attempts>" not in line and len(line) <= 220,
      f"brief_error summarizes a retry exception on one line: {line[:90]}...")
t1 = json.loads((FAKE / "Metrics" / "t1_auto_metrics.json").read_text(encoding="utf-8"))
check(t1["new_components_total"] == len(t1["registry"]["new_components"])
      == sum(a["new_components_added"] for a in t1["plan_adherence"]) > 0
      and all(a["new_components_planned"] >= a["new_components_added"] for a in t1["plan_adherence"]),
      f"new-component counts recorded per module and in total ({t1['new_components_total']})")


# 17. registry and lab_metrics agree on which assets are known (one rule)
from lab_metrics import _is_novel, KNOWN_PREFABS, KNOWN_TEXTURES  # noqa: E402
from registry import AssetRegistry  # noqa: E402
_reg = AssetRegistry()
_names = ["SunPrefab", " sunprefab ", "DemoPrefab", "marsPrefab", "textPrefab"]
check(all(_reg.is_known_asset(prefab=n) == (not _is_novel(n, KNOWN_PREFABS)) for n in _names)
      and all(_reg.is_known_asset(texture=t) == (not _is_novel(t, KNOWN_TEXTURES))
              for t in ["2k_moon", "2K_MOON", "mars_texture"]),
      "registry known-asset rule matches lab_metrics (incl. demoPrefab, casing, spaces)")


# 18. --max-scenes reaches the plan prompt; impossible settings are rejected up front
import prompts as _prompts  # noqa: E402
check("Use 3-5 scenes" in _prompts.build_plan_prompt("ctx", has_script=False, max_scenes=5),
      "plan prompt asks for the configured scene range")
_rejected = []
for _bad in ({"max_scenes": 2}, {"max_scenes": 13}, {"min_clips": 1}, {"step_attempts": 0},
             {"module_retries": -1}):
    try:
        P.PipelineConfig(**_bad)
    except ValueError:
        _rejected.append(_bad)
check(len(_rejected) == 5, "settings no plan could satisfy are rejected before any call")
check((_prompts.MIN_SCENES, _prompts.MAX_SCENES_LIMIT) == (
           LabOutline.model_fields["scenes"].metadata[0].min_length,
           LabOutline.model_fields["scenes"].metadata[1].max_length)
      and _prompts.MIN_CLIP_BEATS == 2,
      "scene range and clip-beat minimum are read from the schemas")
_plan5 = (FAKE / "Runs" / "t10_cap" / "00_plan_prompt.txt").read_text(encoding="utf-8")
check("Use 3-8 scenes" in _plan5, "default run asks for 3-8 scenes, matching the 8-scene cap")


# 19. a revision that can't be generated keeps the last valid output
class FailRevisionLLM(FakeLLM):
    def __call__(self, model, messages, max_retries):
        if model is P.PlannedDemoModule and len(messages) > 2:
            raise ValueError("revision call failed")
        return super().__call__(model, messages, max_retries)


class ReviseModuleOnce:
    name = "revise-module"

    def review(self, cp):
        return Revise("Add a recap clip.") if cp.step == "module" and cp.revision == 0 else Approve()


rec = make(FailRevisionLLM(), "t19_revfail", cfg={"step_attempts": 1},
           reviewer=ReviseModuleOnce()).run()
m1 = next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 1)
check(rec["success"] and m1["success"] and m1["decision"] == "Approve (revision failed)"
      and any((e.get("note") or "").startswith("revision failed") for e in rec["review_log"]),
      "failed revision keeps the last valid output instead of failing the run")
check(m1["revisions"] == 0, "a revision round that produced nothing isn't counted")


class AuthFailRevisionLLM(FakeLLM):
    def __call__(self, model, messages, max_retries):
        if model is P.PlannedDemoModule and len(messages) > 2:
            raise FakeAuthError("key revoked")
        return super().__call__(model, messages, max_retries)


rec = make(AuthFailRevisionLLM(), "t19_revauth", cfg={"step_attempts": 3},
           reviewer=ReviseModuleOnce()).run()
check(not rec["success"] and rec["failed_step"] == "module 1"
      and rec["steps"][-1]["non_retryable"] and not rec["steps"][-1]["success"]
      and "not retryable" in rec["error"],
      "an auth error during a revision fails the run instead of being approved")


# 20. a broken reviewer (or any unexpected error) leaves a 'failed' record, not 'in_progress'
class NoneReviewer:
    name = "none"

    def review(self, cp):
        return None


rec = make(FakeLLM(), "t20_badreviewer", reviewer=NoneReviewer()).run()
saved = json.loads((FAKE / "Metrics" / "t20_badreviewer_metrics.json").read_text(encoding="utf-8"))
check(not rec["success"] and saved["status"] == "failed" and saved["failed_step"] == "plan"
      and saved["error"].startswith("TypeError"),
      "unexpected error recorded as a failed run at its step")
_errors_txt = (FAKE / "Runs" / "t20_badreviewer" / "errors.txt").read_text(encoding="utf-8")
check("Traceback (most recent call last)" in (saved["traceback"] or "")
      and "TypeError" in saved["traceback"] and "========== traceback" in _errors_txt
      and "Traceback (most recent call last)" in _errors_txt,
      "unexpected error keeps its traceback in the metrics and errors.txt")


# 21. --resume continues where the run directory is now, not where it was made
MOVED = TMP / "moved"
shutil.copytree(FAKE / "Runs" / "t1_auto", MOVED / "Runs" / "t1_auto")
fake = FakeLLM()
pipe = P.HierarchicalPipeline.resume(MOVED / "Runs" / "t1_auto", call_fn=fake)
fake.pipe = pipe
rec = pipe.run()
check(rec["success"] and fake.calls == [] and (MOVED / "Outputs" / "t1_auto_output.json").exists(),
      "a moved run resumes in place without regenerating")
_rejected = 0
for _bad_dir in (MOVED / "Runs", MOVED):
    for _read in (P.HierarchicalPipeline.read_run_config,
                  lambda d: P.HierarchicalPipeline.resume(d, call_fn=FakeLLM())):
        try:
            _read(_bad_dir)
        except ValueError:
            _rejected += 1
check(_rejected == 4 and P.HierarchicalPipeline.read_run_config(MOVED / "Runs" / "t1_auto")
      ["model_key"] == "claude-haiku-4.5",
      "one run-directory check for the CLI and resume(); non-run directories rejected")


# 22. tracker edge cases, through the public TrackedClient / track_call: a stub
#     client fires the hook sequence a script describes (issue #71, PR #80 review)
from types import SimpleNamespace  # noqa: E402
from tracking import InstructorTracker, TrackedClient  # noqa: E402


class _StubInstructor:
    """Fires instructor-style hooks from a script, then returns or raises."""

    def __init__(self):
        self.handlers = {}

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def off(self, event, handler):
        self.handlers[event].remove(handler)

    def _emit(self, event, *args):
        for h in list(self.handlers.get(event, [])):
            h(*args)

    def create(self, script):
        for step, *arg in script:
            if step == "kwargs":
                self._emit("completion:kwargs")
            elif step == "response":
                self._emit("completion:response", arg[0])
            elif step == "parse_error":
                self._emit("parse:error", arg[0])
            elif step == "raise":
                raise arg[0]
        return "ok"


def _resp(tokens):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=tokens, completion_tokens=0,
                                                 total_tokens=tokens))


class _RetryWrapper(Exception):
    """Shape-alike of instructor's InstructorRetryException: it has failed_attempts
    and last_completion (the previous attempt's), and is raised from the cause."""

    def __init__(self, msg, last_completion=None):
        super().__init__(msg)
        self.last_completion = last_completion
        self.failed_attempts = None


class IncompleteOutputException(Exception):  # name-alike of instructor's truncation error
    def __init__(self, last_completion=None):
        super().__init__("The output is incomplete due to a max_tokens length limit.")
        self.last_completion = last_completion


def _tracked(script, tracker=None):
    tracker = tracker or InstructorTracker(model="m", provider="openai")
    try:
        TrackedClient(_StubInstructor(), tracker).create(script=script)
    except Exception:  # noqa: BLE001
        pass
    return tracker.aggregate.calls[-1]


_c = _tracked([("raise", TypeError("bad kwargs, nothing sent"))])
check(_c.total_attempts == 0 and _c.completion_errors == 0 and _c.final_error is None,
      "an error before any request adds no attempt and no error count")

_t = InstructorTracker(model="m", provider="openai")
_stub = _StubInstructor()
try:
    with _t.track_call(_stub):
        _stub.create(script=[("kwargs",), ("response", _resp(10))])
        raise KeyError("caller's own post-processing")
except KeyError:
    pass
_c = _t.aggregate.calls[-1]
check(_c.total_attempts == 1 and _c.attempts[0].success and not _c.errors
      and _c.completion_errors == 0,
      "an error in the caller's own code isn't pinned on the request")

_wrapper = _RetryWrapper("<failed_attempts>\n<generation number=\"1\">...",
                         last_completion=_resp(10))
_wrapper.__cause__ = AssertionError("Instructor does not support multiple function calls")
_c = _tracked([("kwargs",), ("response", _resp(10)), ("parse_error", ValueError("validation error")),
               ("kwargs",), ("response", _resp(0)), ("raise", _wrapper)])
_a2 = _c.attempts[-1].error
check(_c.total_attempts == 2 and _c.retry_count == 1 and _a2 is not None
      and _a2.exception_class == "AssertionError" and "<failed_attempts>" not in _a2.message
      and _c.final_error is _a2 and _c.total_usage.total_tokens == 10,
      "a wrapped non-retryable failure is recorded as its cause, with no usage counted twice")

_c = _tracked([("kwargs",), ("raise", IncompleteOutputException(last_completion=_resp(7)))])
check(_c.total_attempts == 1 and _c.total_usage.total_tokens == 7
      and _c.final_error.exception_class == "IncompleteOutputException",
      "a truncation with no response hook keeps its own request's usage")

_c = _tracked([("kwargs",), ("response", _resp(5)), ("parse_error", ValueError("validation error")),
               ("response", _resp(5))])
check(_c.total_attempts == 2 and _c.retry_count == 1 and _c.attempts[-1].success
      and _c.attempts[0].error is not None,
      "a request whose kwargs hook didn't fire still gets its own attempt")

_c = _tracked([("kwargs",), ("parse_error", ValueError("validation error")),
               ("parse_error", ValueError("validation error again")), ("raise", KeyError("end"))])
check(_c.total_attempts == 1 and _c.retry_count == 0 and _c.parse_errors == 2
      and _c.attempts[0].error.message == "validation error",
      "two error hooks for one request stay on one attempt (no phantom retry)")

_chained = RuntimeError("config error")
_chained.__cause__ = KeyError("x")
_c = _tracked([("kwargs",), ("response", _resp(3)), ("raise", _chained)])
check(_c.attempts[0].error.exception_class == "RuntimeError"
      and _c.attempts[0].error.message == "config error",
      "only instructor's retry wrapper is unwrapped; other chained errors are recorded as is")

_trunc_cause = IncompleteOutputException(last_completion=_resp(9))
_wrapped_trunc = _RetryWrapper("<failed_attempts>...")
_wrapped_trunc.__cause__ = _trunc_cause
_c = _tracked([("kwargs",), ("raise", _wrapped_trunc)])
check(_c.total_usage.total_tokens == 9
      and _c.final_error.exception_class == "IncompleteOutputException",
      "a wrapped truncation still keeps its request's usage")


# 23. which errors the transient retryer re-sends (benchmark._is_transient_api_error)
#     and which fail a step at once (pipeline.is_non_retryable), on stub exceptions
class OverloadedError(Exception):
    """Name-alike of anthropic's 529 error (an APIStatusError, not an
    InternalServerError); no status attribute, so only the name can match."""


class _Status(Exception):
    def __init__(self, status):
        super().__init__(f"Error code: {status}")
        self.status_code = status


def _wrapped(cause):
    w = _RetryWrapper("<failed_attempts>...")
    w.__cause__ = cause
    return w


_trunc_under_503 = _Status(503)
_trunc_under_503.__cause__ = IncompleteOutputException(last_completion=None)


def _raised_while_handling(inner):
    """A ValueError whose only link to ``inner`` is implicit context."""
    try:
        try:
            raise inner
        except Exception:
            raise ValueError("validation error")
    except ValueError as e:
        return e


check(benchmark._is_transient_api_error(_wrapped(OverloadedError("overloaded")))
      and benchmark._is_transient_api_error(_wrapped(_Status(529))),
      "an Anthropic 529 is transient, by class name or by status")
check(not benchmark._is_transient_api_error(_wrapped(_trunc_under_503)),
      "a truncation anywhere in the cause chain is never re-sent")
_ctx503, _ctx401 = _raised_while_handling(_Status(503)), _raised_while_handling(_Status(401))
check(isinstance(_ctx503.__context__, _Status)
      and not benchmark._is_transient_api_error(_ctx503) and not P.is_non_retryable(_ctx401),
      "an error raised while handling a 503 or 401 is classified as itself (only causes are followed)")
check(P.is_non_retryable(_wrapped(_Status(401)))
      and not P.is_non_retryable(_wrapped(_Status(503))),
      "a wrapped 401 still fails the step at once; a wrapped 503 does not")


# 24. step error lines, through the real tracker (TrackedClient around a stub):
#     an error the tracker recorded is listed once, in full, and the step summary
#     is shortened; an error the tracker never saw keeps its full text
LONG_TAIL = "provider said: " + "x" * 400 + " END-OF-ERROR"


class TrackedThenFailLLM(FakeLLM):
    def __init__(self, inside_call):
        super().__init__()
        self.inside_call = inside_call   # raise inside the instructor call, or after it

    def __call__(self, model, messages, max_retries):
        if model is P.PlannedDemoModule and self._index() == 1:
            client = TrackedClient(_StubInstructor(), self.pipe.tracker)
            if self.inside_call:   # a hooked parse error, then instructor raises again
                client.create(script=[("kwargs",), ("response", _resp(10)),
                                      ("parse_error", ValueError("validation error")),
                                      ("raise", RuntimeError(LONG_TAIL))])
            client.create(script=[("kwargs",), ("response", _resp(10))])
            raise RuntimeError(LONG_TAIL)    # outside any tracked call
        return super().__call__(model, messages, max_retries)


rec = make(TrackedThenFailLLM(inside_call=True), "t24_tracked", cfg={"step_attempts": 1}).run()
m1 = rec["steps"][-1]
check(m1["calls"] == 1 and m1["parse_errors"] == 1 and m1["retries"] == 0
      and m1["errors"][0] == "attempt 1 ValueError: validation error"
      and m1["errors"][1] == f"attempt 1 RuntimeError: {LONG_TAIL}"
      and sum("END-OF-ERROR" in e for e in m1["errors"]) == 1,
      "an error raised after a hooked error is recorded as the call's terminal error, once")
rec = make(TrackedThenFailLLM(inside_call=False), "t24_untracked", cfg={"step_attempts": 1}).run()
m1 = rec["steps"][-1]
check(m1["calls"] == 1 and m1["errors"][-1].endswith("END-OF-ERROR"),
      "an error the tracker didn't record keeps its full text in errors.txt")


# 25. cached input is priced at its own rate (issue #72): usage parsing per provider
#     shape, and the cost formula
from tracking import PricingCalculator, TokenUsage  # noqa: E402
_oa = TokenUsage.from_completion(SimpleNamespace(usage=SimpleNamespace(
    prompt_tokens=1000, completion_tokens=100, total_tokens=1100,
    prompt_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=300),
    completion_tokens_details=None)))
_an = TokenUsage.from_completion(SimpleNamespace(usage=SimpleNamespace(
    input_tokens=100, cache_read_input_tokens=600, cache_creation_input_tokens=300,
    output_tokens=100)))
_ge = TokenUsage.from_completion(SimpleNamespace(usage_metadata=SimpleNamespace(
    prompt_token_count=1000, candidates_token_count=100, total_token_count=1100,
    cached_content_token_count=600, thoughts_token_count=0)))
check((_oa.prompt_tokens, _oa.cached_tokens, _oa.cache_write_tokens) == (1000, 600, 300)
      and (_an.prompt_tokens, _an.cached_tokens, _an.cache_write_tokens, _an.total_tokens)
      == (1000, 600, 300, 1100)
      and (_ge.prompt_tokens, _ge.cached_tokens, _ge.cache_write_tokens) == (1000, 600, 0),
      "cache reads and writes parsed from OpenAI, Anthropic and Gemini usage, all inside prompt_tokens")
check(abs(PricingCalculator(model="gpt-6-luna").calculate_cost(_oa)
          - (100 * 0.10 + 600 * 0.01 + 300 * 0.125 + 100 * 0.50) / 1e6) < 1e-12
      and abs(PricingCalculator(model="gemini-3.5-flash-lite").calculate_cost(_ge)
              - (400 * 0.30 + 600 * 0.03 + 100 * 2.50) / 1e6) < 1e-12,
      "uncached input, cache reads and cache writes each billed at their own rate")
check(abs(PricingCalculator(model="gpt-5.4-mini").calculate_cost(_oa)
          - (1000 * 0.75 + 100 * 4.50) / 1e6) < 1e-12,
      "a model without cache rates (wave 2) bills all input at the input rate, as before")


# 26. the model that actually answered is recorded (issue #72): Google serves
#     gemini-3.5-flash requests with gemini-3.6-flash since 2026-10-08
import contextlib  # noqa: E402
import io  # noqa: E402
from tracking.metrics import served_model_of  # noqa: E402
check(served_model_of(SimpleNamespace(model_version="models/gemini-3.6-flash", usage_metadata=None))
      == "gemini-3.6-flash"
      and served_model_of(SimpleNamespace(model="gpt-6-luna-2026-09-01")) == "gpt-6-luna-2026-09-01"
      and served_model_of(SimpleNamespace(model=None)) is None,
      "served model read from Gemini model_version and OpenAI/Anthropic model")
_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    benchmark.warn_served_model("gpt-6-luna", ["gpt-6-luna-2026-09-01"])
    _quiet = _out.getvalue()
    benchmark.warn_served_model("gemini-3.5-flash", ["gemini-3.6-flash"])
check(_quiet == "" and "answered with gemini-3.6-flash" in _out.getvalue(),
      "a dated snapshot of the requested model is fine; a redirect to another model warns")


# 27. reasoning effort pinned per provider (current roster), provider default for
#     wave 2 (#72); Responses usage parsed; the Responses reask falls back to
#     instructor's own when the reply has no function call
_pk = lambda m: benchmark.provider_create_kwargs(P.MODELS[m], P.LabPlan)
check(_pk("gemini-3.8-flash").get("thinking_config") == {"thinking_level": "medium"}
      and "thinking_config" not in _pk("gemini-3.5-flash")
      and _pk("claude-sonnet-5.5").get("output_config") == {"effort": "medium"}
      and "output_config" not in _pk("claude-sonnet-5")
      and _pk("gpt-6-luna").get("reasoning") == {"effort": "medium"}
      and not {"reasoning", "store", "include"} & set(_pk("gpt-5.4-mini")),
      "effort medium on the current roster, nothing extra on wave-2 models")
_ru = TokenUsage.from_completion(SimpleNamespace(usage=SimpleNamespace(
    input_tokens=1000, output_tokens=200, total_tokens=1200,
    input_tokens_details=SimpleNamespace(cached_tokens=600, cache_write_tokens=100),
    output_tokens_details=SimpleNamespace(reasoning_tokens=50))))
check((_ru.prompt_tokens, _ru.cached_tokens, _ru.cache_write_tokens, _ru.completion_tokens,
       _ru.reasoning_tokens) == (1000, 600, 100, 200, 50),
      "Responses usage: cached and cache writes inside input_tokens, reasoning inside output")
try:
    import instructor.v2.providers.openai.handlers as _oh
    benchmark._patch_openai_responses()
    _text_only = SimpleNamespace(output=[SimpleNamespace(type="message")])
    _fallback = _oh.reask_responses_tools({"messages": [{"role": "user", "content": "q"}]},
                                          _text_only, ValueError("bad"))
    check(_fallback["messages"][-1]["role"] == "user"
          and "Validation Error found" in str(_fallback["messages"][-1]["content"]),
          "a Responses reply with no function call gets instructor's own reask")
except Exception as _e:  # noqa: BLE001 — report, don't crash the offline test
    check(False, f"Responses reask fallback ({type(_e).__name__}: {_e})")


# 28. PR #83 review fixes: served-model match, empty-arguments reask, OpenAI
#     timeout, price override keeps cache rates, probe reads Responses system text
check(benchmark.served_model_mismatches("gemini-3.5-flash", ["gemini-3.5-flash-lite"])
      == ["gemini-3.5-flash-lite"]
      and benchmark.served_model_mismatches("claude-x", ["claude-x-20261001", "claude-x"]) == []
      and benchmark.served_model_mismatches("gpt-6-luna", ["gpt-6-luna-2026-09-01"]) == [],
      "only the requested id or a dated snapshot counts as the same served model")
try:
    import instructor.v2.providers.openai.handlers as _oh
    benchmark._patch_openai_responses()
    _empty = SimpleNamespace(output=[SimpleNamespace(type="function_call", call_id="c1",
                                                     arguments="{}")])
    _out = _oh.reask_responses_tools({"messages": []}, _empty, ValueError("bad"))["messages"][-1]
    check(_out["type"] == "function_call_output" and "empty arguments" in _out["output"],
          "an empty-arguments call gets instructor's empty-arguments feedback")
except Exception as _e:  # noqa: BLE001 — report, don't crash the offline test
    check(False, f"empty-arguments reask ({type(_e).__name__}: {_e})")
_env = {k: os.environ.get(k) for k in ("OPENAI_API_KEY",)}
os.environ["OPENAI_API_KEY"] = _env["OPENAI_API_KEY"] or "test"
try:
    _timeouts = {m: benchmark.create_instructor_client(P.MODELS[m], P.SpecType.JSON_LAB)
                 .client.timeout.read for m in ("gpt-6-luna", "gpt-5.4-mini")}
finally:
    if _env["OPENAI_API_KEY"] is None:
        os.environ.pop("OPENAI_API_KEY", None)
check(_timeouts == {"gpt-6-luna": 900.0, "gpt-5.4-mini": 900.0},
      f"OpenAI clients get the same 900 s timeout as Anthropic ({_timeouts})")
check(PricingCalculator(config={"input": 1.0, "output": 2.0}, model="gpt-6-luna").cached_input_price
      == 0.01,
      "a price override keeps the model's cache rates")
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Benchmark" / "probes"))
from probe_tool_schemas import system_text  # noqa: E402
check(system_text({"input": [{"role": "system", "content": "be brief"},
                             {"role": "user", "content": "hi"}]}) == "be brief",
      "the schema probe reads the system line from a Responses request's input")


# ── Part 2: real _llm_call path against a local mock OpenAI server ──────
print("\n-- part 2: mock OpenAI server --")
REPLAY = FAKE / "Runs" / "t1_auto"
state = {"asset": 0, "bad_sent": False, "requests": []}


class MockOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tool = body["tools"][0]["function"]["name"]
        state["requests"].append((tool, len(body["messages"])))
        state.setdefault("body_keys", set()).update(body)
        if tool == "LabPlan":
            args = (REPLAY / "00_plan.json").read_text(encoding="utf-8")
        elif tool == "ModuleAssetPlan":
            state["asset"] += 1
            args = (REPLAY / f"{state['asset']:02d}_asset_plan.json").read_text(encoding="utf-8")
        elif tool == "PlannedDemoModule" and self._should_503(state.get("mode")):
            return self._reply(503, {"error": {"message": "The server is overloaded",
                                               "type": "server_error"}})
        elif state.get("mode") == "reask400" and state["module_requests"] >= 1:
            # the reask after an invalid module is rejected: an API error ends the call
            state["module_requests"] += 1
            return self._reply(400, {"error": {"message": "Invalid request: bad reask",
                                               "type": "invalid_request_error"}})
        elif (state.get("always_bad") or state.get("mode") == "reask400"
              or (state.get("mode") == "503reask" and state["module_requests"] == 0)
              or (state["asset"] == 2 and not state["bad_sent"])):
            state["bad_sent"] = True
            state["module_requests"] = state.get("module_requests", 0) + 1
            args = json.dumps({"moduleName": "broken"})  # missing required fields -> reask
        else:
            args = (REPLAY / f"{state['asset']:02d}_module.json").read_text(encoding="utf-8")
        finish = "length" if state.get("mode") == "truncate" and tool == "PlannedDemoModule" \
            else "tool_calls"   # "length": max_tokens truncation, no error hook fires
        self._reply(200, {
            "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": tool, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                      "prompt_tokens_details": {"cached_tokens": 600}},
        })

    @staticmethod
    def _should_503(mode):
        """fail503: every module request. 503once: the first one. 503reask: module
        1's reask (its first response is invalid, below), once."""
        if mode == "fail503":
            return True
        if (mode == "503once" or (mode == "503reask" and state["module_requests"] == 1)) \
                and not state["sent_503"]:
            state["sent_503"] = True
            return True
        return False

    def _reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1"]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]
srv = HTTPServer(("127.0.0.1", 0), MockOpenAI)
threading.Thread(target=srv.serve_forever, daemon=True).start()

import instructor  # noqa: E402
import tenacity  # noqa: E402

benchmark._TRANSIENT_WAIT = tenacity.wait_none()   # real retry policy, no sleeping

_real_factory = P.create_instructor_client
P.create_instructor_client = lambda mc, spec: instructor.from_provider(
    f"openai/{mc.model_id}", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="test",
    max_retries=0)   # SDK retries off, as in the real factory (the 503 cases)
try:
    MOCK = TMP / "mock"
    class CallingReviewer:
        """Stands in for an LLM reviewer (#69): one tracked call during the plan review."""
        name = "calling"
        pipe = None

        def review(self, cp):
            if cp.step == "plan":
                self.pipe._llm_call(P.LabPlan, [{"role": "user", "content": "critique"}], 0)
            return Approve()

    calling = CallingReviewer()
    mock_pipe = P.HierarchicalPipeline("gpt-5.4-mini", LAB_NAME, P.PipelineConfig(**REPLAY_MINIMUMS),
                                       output_dir=MOCK, run_name="mock", reviewer=calling)
    calling.pipe = mock_pipe
    rec = mock_pipe.run()
    # every module response broken: instructor exhausts its retries on module 1
    n_good_requests = len(state["requests"])
    state.update(asset=0, always_bad=True)
    bad = P.HierarchicalPipeline(
        "gpt-5.4-mini", LAB_NAME,
        P.PipelineConfig(**REPLAY_MINIMUMS, module_retries=1, step_attempts=1),
        output_dir=TMP / "mock_bad", run_name="mock_bad",
    ).run()
    # module 1's reask is rejected with a 400, and module 1 is truncated (max_tokens)
    edge = {}
    for _mode in ("reask400", "truncate", "fail503", "503once", "503reask"):
        state.update(asset=0, always_bad=False, mode=_mode, module_requests=0, sent_503=False)
        edge[_mode] = P.HierarchicalPipeline(
            "gpt-5.4-mini", LAB_NAME,
            P.PipelineConfig(**REPLAY_MINIMUMS, module_retries=1, step_attempts=1),
            output_dir=TMP / f"mock_{_mode}", run_name=f"mock_{_mode}",
        ).run()
    state["mode"] = None
finally:
    P.create_instructor_client = _real_factory
    srv.shutdown()

check(rec["success"], "mock-server run succeeds")
check(rec["served_models"] == ["gpt-5.4-mini"] == rec["tracking"]["served_models"],
      "the record lists the model that answered")
check(rec["totals"]["calls"] == len(rec["steps"]) + 1 and rec["steps"][0]["calls"] == 2,
      "one tracked call per step, plus the reviewer's call charged to the plan step")
m2 = next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 2)
check(m2["parse_errors"] == 1 and m2["errors"], "module 2 records its one parse error")
check(m2["retries"] == m2["parse_errors"] == 1
      and rec["totals"]["retries"] == rec["totals"]["parse_errors"]
      and rec["tracking"]["retries"]["total"] == rec["totals"]["parse_errors"],
      "one reask counts as one retry (issue #71)")
check(m2["prompt_tokens"] == 2000 and m2["completion_tokens"] == 400,
      "module 2 tokens include the reask")
check(rec["totals"]["total_tokens"] == rec["tracking"]["tokens"]["total"]
      and rec["totals"]["retries"] == rec["tracking"]["retries"]["total"],
      "per-step totals match the tracker")
check(rec["totals"]["cost_usd"] > 0
      and abs(rec["totals"]["cost_usd"] - rec["tracking"]["cost"]["total_usd"]) < 1e-9,
      "per-step costs sum to the tracker cost")
check((MOCK / "Runs" / "mock" / "errors.txt").exists(), "errors.txt written")
check(rec["totals"]["cached_tokens"] == 600 * n_good_requests and m2["cached_tokens"] == 1200,
      "cached prompt tokens recorded per step and in totals")
check(state["requests"][0] == ("LabPlan", 2), "first request is the plan (system + user)")
check(not {"reasoning", "reasoning_effort", "store", "include", "output_config"} & state["body_keys"]
      and rec["reasoning_effort"] is None,
      "a wave-2 model sends no reasoning settings (provider default, as in wave 2)")
bad_m1 = bad["steps"][-1]
check(not bad["success"] and bad_m1["step"] == "module" and len(bad_m1["errors"]) == 3
      and [e.split(" ", 2)[:2] for e in bad_m1["errors"][:2]] == [["attempt", "1"], ["attempt", "2"]]
      and bad_m1["errors"][2].startswith("step attempt 1 failed: InstructorRetryException after 2")
      and "<failed_attempts>" not in "".join(bad_m1["errors"]),
      "exhausted retries: one line per instructor attempt plus one summary, no duplicate dump")
# One tracker attempt per request (issue #71), on the paths instructor actually takes.
check(bad_m1["retries"] == 1 and bad_m1["parse_errors"] == 2,
      "ran out of attempts: 2 requests, 1 reask, 2 parse errors (was counted as 3 retries)")
_r400 = edge["reask400"]["steps"][-1]
check(not edge["reask400"]["success"] and _r400["step"] == "module" and _r400["calls"] == 1
      and _r400["retries"] == 1 and _r400["parse_errors"] == 1
      and any(e.startswith("attempt 2 ") and "400" in e for e in _r400["errors"]),
      "a reask rejected with a 400: 2 requests, 1 retry, the API error on attempt 2")
_trunc = edge["truncate"]["steps"][-1]
check(not edge["truncate"]["success"] and _trunc["retries"] == 0 and _trunc["prompt_tokens"] == 1000
      and any(e.startswith("attempt 1 IncompleteOutputException") for e in _trunc["errors"])
      and sum("IncompleteOutputException" in e for e in _trunc["errors"]) == 2
      and _trunc["errors"][-1].startswith("step attempt 1 failed: IncompleteOutputException")
      and len(_trunc["errors"][-1]) <= len("step attempt 1 failed: ") + 220,
      "a truncated response is recorded on its attempt (no error hook fires) and not repeated")
_f503 = edge["fail503"]["steps"][-1]
_n = benchmark._TRANSIENT_ATTEMPTS
check(not edge["fail503"]["success"] and _f503["calls"] == _n and _f503["retries"] == 0
      and edge["fail503"]["transient_retries"] == _f503["transient_retries"] == _n - 1
      and _f503["reask_restarts"] == 0 and len(_f503["errors"]) == _n + 1
      and all(e.startswith("attempt 1 InternalServerError") for e in _f503["errors"][:_n])
      and _f503["errors"][-1].startswith("step attempt 1 failed: InstructorRetryException")
      and len(_f503["errors"][-1]) <= len("step attempt 1 failed: ") + 220,
      f"a persistent 5xx is re-sent {_n - 1}x by the transient retryer (issue #54), "
      "each call's error recorded once, step summary shortened")
_ok503 = edge["503once"]
_m503 = next(s for s in _ok503["steps"] if s["step"] == "module" and s["index"] == 1)
check(_ok503["success"] and _ok503["transient_retries"] == 1 and _m503["calls"] == 2
      and _m503["success"] and _m503["attempts"] == 1 and _m503["retries"] == 0
      and _m503["transient_retries"] == 1 and _m503["reask_restarts"] == 0
      and _ok503["totals"]["transient_retries"] == 1,
      "a one-off 5xx is re-sent once and the step succeeds (wrapped by instructor 1.17, #54)")
_rr = edge["503reask"]
_mrr = next(s for s in _rr["steps"] if s["step"] == "module" and s["index"] == 1)
check(_rr["success"] and _rr["transient_retries"] == 1 and _mrr["calls"] == 2
      and _mrr["success"] and _mrr["parse_errors"] == 1 and _mrr["retries"] == 1
      and _mrr["transient_retries"] == 1 and _mrr["reask_restarts"] == 1
      and _rr["totals"]["reask_restarts"] == 1
      and any("InternalServerError" in e for e in _mrr["errors"]),
      "a 5xx on a reask restarts the call with a fresh parse-retry budget, counted per "
      "step in reask_restarts (documented caveat, #54)")
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Analysis"))
from benchmark_dataframe import load_runs  # noqa: E402
df = load_runs(MOCK)
from benchmark_dataframe import fail_mode  # noqa: E402
_no_call = "1 validation error for list[any]\n  List should have at least 1 item after validation, not 0"
_schema = "1 validation error for Lab\nmodules\n  Field required"
check(fail_mode(_no_call) == "no tool call (auto)"
      and fail_mode(_no_call + "\n" + _schema) == "schema validation"
      and fail_mode("1 validation error for list[any]\n  List should have at most 1 item "
                    "after validation, not 2") == "schema validation",
      "a run that never called the tool gets its own fail mode")
check(benchmark.tool_choice_for(P.MODELS["claude-haiku-5.5"]) == "auto"
      and benchmark.tool_choice_for(P.MODELS["gpt-5.4-mini"]) == "forced"
      and benchmark.tool_choice_for(P.MODELS["gemini-3.5-flash"]) == "forced",
      "recorded tool_choice is read from what provider_create_kwargs sends")
check(len(df) == 1 and bool(df.iloc[0]["success"]) and df.iloc[0]["structure"] == "multi-module",
      "Code/Analysis load_runs() reads hierarchical metrics")


# ── Part 3: the exact request we send Anthropic (mock Anthropic server) ──
# Builds the client through the real create_instructor_client (pointed at the
# mock via ANTHROPIC_BASE_URL), so this checks the shape of what we actually send:
# tool_choice "auto" with disable_parallel_tool_use (#44, #70; Claude Sonnet 5.5,
# Opus 5.5 and Fable 5.1 reject a forced choice), the system line asking for the
# tool, and
# max_tokens. The first module request is answered in plain text, as "auto"
# allows, to check instructor's "no tool invocation" reask.
print("\n-- part 3: mock Anthropic server --")
anth = {"asset": 0, "requests": [], "text_sent": False}


class MockAnthropic(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tool = body["tools"][0]["name"]
        anth["requests"].append(body)
        if tool == "LabPlan":
            data = (REPLAY / "00_plan.json").read_text(encoding="utf-8")
        elif tool == "ModuleAssetPlan":
            anth["asset"] += 1
            data = (REPLAY / f"{anth['asset']:02d}_asset_plan.json").read_text(encoding="utf-8")
        else:
            data = (REPLAY / f"{anth['asset']:02d}_module.json").read_text(encoding="utf-8")
        if tool == "PlannedDemoModule" and not anth["text_sent"]:
            anth["text_sent"] = True
            content, stop = [{"type": "text", "text": "Here is the module you asked for."}], "end_turn"
        else:
            content = [{"type": "tool_use", "id": "toolu_mock", "name": tool,
                        "input": json.loads(data)}]
            stop = "tool_use"
        payload = json.dumps({
            "id": "msg_mock", "type": "message", "role": "assistant", "model": body["model"],
            "content": content, "stop_reason": stop, "stop_sequence": None,
            # input_tokens excludes cache reads/writes; the tracker folds them in
            "usage": {"input_tokens": 600, "cache_read_input_tokens": 300,
                      "cache_creation_input_tokens": 100, "output_tokens": 200},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


asrv = HTTPServer(("127.0.0.1", 0), MockAnthropic)
threading.Thread(target=asrv.serve_forever, daemon=True).start()
_saved_env = {k: os.environ.get(k) for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")}
os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{asrv.server_port}"
os.environ["ANTHROPIC_API_KEY"] = "test"
try:
    rec = P.HierarchicalPipeline("claude-haiku-5.5", LAB_NAME, P.PipelineConfig(**REPLAY_MINIMUMS),
                                 output_dir=TMP / "anthropic", run_name="anthropic").run()
finally:
    asrv.shutdown()
    for k, v in _saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

check(rec["success"], "mock-Anthropic run succeeds")
reqs = anth["requests"]
check(len(reqs) == 2 + 2 * len(MODS),
      f"one Anthropic request per step, plus one reask ({len(reqs)})")
check(all(r["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True} for r in reqs),
      "every request sends tool_choice auto with disable_parallel_tool_use")


def _system_text(r):
    sys_ = r.get("system") or ""
    return sys_ if isinstance(sys_, str) else " ".join(b.get("text", "") for b in sys_)


check(all(f"calling the `{r['tools'][0]['name']}` tool" in _system_text(r)
          and P.SYSTEM_PROMPT[:40] in _system_text(r) for r in reqs),
      "every request asks for its step's tool and keeps the system prompt")
_reask = next((r for r in reqs if len(r["messages"]) > 2), None)
check(_reask is not None and _reask["tools"][0]["name"] == "PlannedDemoModule"
      and "no tool invocation" in json.dumps(_reask["messages"][-1]),
      "a plain-text reply is reasked ('no tool invocation')")
_m1 = next((s for s in rec["steps"] if s["step"] == "module" and s["index"] == 1), None)
check(_m1 is not None and _m1["retries"] == 1, "a plain-text reply's reask counts as one retry")
check(_m1 is not None and _m1["success"] and _m1["parse_errors"] == 1
      and rec.get("tool_choice") == "auto",
      "the reask costs one attempt; the record says tool_choice auto")
check(all(r["max_tokens"] == 32000 for r in reqs), "every request sets max_tokens=32000")
check(all(r.get("output_config") == {"effort": "medium"} for r in reqs)
      and rec["reasoning_effort"] == "medium",
      "every request pins reasoning effort medium (output_config.effort)")
check([r["tools"][0]["name"] for r in reqs[:3]] == ["LabPlan", "ModuleAssetPlan", "PlannedDemoModule"],
      "tool names follow the step response models")
check(all(r["model"] == "claude-haiku-5-5" for r in reqs) and rec["served_models"] == ["claude-haiku-5-5"],
      "model id sent and served (Claude Haiku 5.5)")
check(rec["totals"]["prompt_tokens"] == 1000 * len(reqs)
      and rec["totals"]["cached_tokens"] == 300 * len(reqs)
      and rec["totals"]["cache_write_tokens"] == 100 * len(reqs),
      "Anthropic usage tracked per step, cache reads and writes folded into the input")
check(abs(rec["totals"]["cost_usd"]
          - len(reqs) * (600 * 0.10 + 300 * 0.01 + 100 * 0.125 + 200 * 0.50) / 1e6) < 1e-9,
      "Claude Haiku 5.5 priced at $0.10 / $0.50, cache reads $0.01, writes $0.125 per MTok")

# ── Part 4: GPT-6 on the OpenAI Responses API (mock /v1/responses server) ──
# Real create_instructor_client (pointed at the mock via OPENAI_BASE_URL). Checks the
# request we send (reasoning effort, stateless encrypted reasoning, forced tool, no
# strict text.format) and that a reask sends the model's reasoning back with the
# error as the function call's output (_patch_openai_responses, #72).
print("\n-- part 4: mock OpenAI Responses server --")
resp = {"asset": 0, "requests": [], "bad_sent": False}


class MockResponses(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        resp["requests"].append((self.path, body))
        tool = body["tools"][0]["name"]
        if tool == "LabPlan":
            data = (REPLAY / "00_plan.json").read_text(encoding="utf-8")
        elif tool == "ModuleAssetPlan":
            resp["asset"] += 1
            data = (REPLAY / f"{resp['asset']:02d}_asset_plan.json").read_text(encoding="utf-8")
        elif not resp["bad_sent"]:
            resp["bad_sent"] = True
            data = json.dumps({"moduleName": "broken"})   # invalid -> reask
        else:
            data = (REPLAY / f"{resp['asset']:02d}_module.json").read_text(encoding="utf-8")
        n = len(resp["requests"])
        payload = json.dumps({
            "id": f"resp_{n}", "object": "response", "created_at": 0, "status": "completed",
            "model": body["model"],
            "output": [
                {"type": "reasoning", "id": f"rs_{n}", "summary": [],
                 "encrypted_content": f"enc-{n}"},
                {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}",
                 "name": tool, "arguments": data, "status": "completed"},
            ],
            "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
                      "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 100},
                      "output_tokens_details": {"reasoning_tokens": 50}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


rsrv = HTTPServer(("127.0.0.1", 0), MockResponses)
threading.Thread(target=rsrv.serve_forever, daemon=True).start()
_saved_env = {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{rsrv.server_port}/v1"
os.environ["OPENAI_API_KEY"] = "test"
try:
    rec = P.HierarchicalPipeline("gpt-6-luna", LAB_NAME, P.PipelineConfig(**REPLAY_MINIMUMS),
                                 output_dir=TMP / "responses", run_name="responses").run()
finally:
    rsrv.shutdown()
    for k, v in _saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

rreqs = resp["requests"]
check(rec["success"] and rec["decode_mode"] == "responses_tools"
      and all(path == "/v1/responses" for path, _ in rreqs),
      "GPT-6 runs on the Responses API (every request to /v1/responses)")
check(all(b.get("store") is False and b.get("include") == ["reasoning.encrypted_content"]
          and b.get("reasoning") == {"effort": "medium"} for _, b in rreqs)
      and rec["reasoning_effort"] == "medium",
      "every request: effort medium, store false, encrypted reasoning requested")
check(all(b.get("tool_choice") == {"type": "function", "name": b["tools"][0]["name"]}
          and "text" not in b for _, b in rreqs),
      "the step's tool is forced and no strict text.format is sent")
_rr = next((b for _, b in rreqs if any(isinstance(i, dict) and i.get("type") == "function_call_output"
                                       for i in b["input"])), None)
_tail = _rr["input"][-3:] if _rr else []
check(_rr is not None and [i.get("type") for i in _tail] == ["reasoning", "function_call",
                                                            "function_call_output"]
      and _tail[0].get("encrypted_content", "").startswith("enc-")
      and _tail[2]["call_id"] == _tail[1]["call_id"]
      and "Validation Error found" in _tail[2]["output"],
      "a reask sends the reasoning and the call back, with the error as the call's output")
_m1 = next((s for s in rec["steps"] if s["step"] == "module" and s["index"] == 1), None)
check(_m1 is not None and _m1["success"] and _m1["parse_errors"] == 1 and _m1["retries"] == 1,
      "the reask costs one attempt and the step succeeds")
check(rec["served_models"] == ["gpt-6-luna"]
      and rec["totals"]["prompt_tokens"] == 1000 * len(rreqs)
      and rec["totals"]["cached_tokens"] == 600 * len(rreqs)
      and rec["totals"]["cache_write_tokens"] == 100 * len(rreqs)
      and abs(rec["totals"]["cost_usd"]
              - len(rreqs) * (300 * 0.10 + 600 * 0.01 + 100 * 0.125 + 200 * 0.50) / 1e6) < 1e-9,
      "Responses usage (cached, cache writes) tracked and priced at GPT-6 Luna rates")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)

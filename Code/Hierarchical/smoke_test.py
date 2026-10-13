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
validation, run groups, PascalCase names, soft metrics, the NewComponent check, and
the --no-new-components switch.

Part 2 runs the real _llm_call path (instructor client, tracker, transient
retryer, provider kwargs) against a local mock OpenAI server that replays part
1's step files, with one invalid module response to force an instructor reask.

Part 3 builds the real Anthropic client against a local mock Anthropic server and
checks the request shape we send (forced tool_choice + disable_parallel_tool_use,
max_tokens).

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
            comps, seen = [], set()
            for obj_name, c in pairs:
                name = _comp_name(c)
                new = (c["componentType"] == "newscript"
                       and self.pipe.registry.find_component(name) is None
                       and name.lower() not in seen)
                seen.add(name.lower())
                extra = {"behavior_description": c["scriptDescription"],
                         "why_existing_insufficient": "no built-in component does this"} if new else {}
                comps.append({"object_name": obj_name, "component_type": name,
                              "rationale": "needed for this scene's behavior", "is_new": new, **extra})
            beats = [cl["changeMeaning"] for cl in m["clips"]]
            return model.model_validate({
                "module_name": m["moduleName"],
                "design_notes": "What the student must see and do in this scene, and why.",
                "objects": [{"name": o["name"], "prefab": o["prefab"], "texture": o.get("texture"),
                             "purpose": f"Shows {o['name']} in this scene", "is_new_asset": False}
                            for o in m["objects"]],
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
    """Drop dateCreated and NewComponent name/description, which the pipeline
    deliberately normalizes; everything else must come through untouched."""
    if isinstance(node, dict):
        return {k: _strip_scripts(v) for k, v in node.items()
                if k not in ("dateCreated", "scriptName", "scriptDescription")}
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
# (a module-global lookup); re-check on every instructor upgrade (issue #70).
import instructor.v2.providers.genai.handlers as _genai_handlers  # noqa: E402
benchmark._patch_genai_parallel_call_retry()
_genai_src = Path(_genai_handlers.__file__).read_text(encoding="utf-8")
check(getattr(_genai_handlers.parse_genai_tools, "_arlem_retryable_patch", False)
      and "model = parse_genai_tools(" in _genai_src,
      "Gemini parallel-call retry patch wraps the parse the TOOLS handler calls")
_req = TMP / "requirements_marker.txt"
_req.write_text('instructor[anthropic]==1.15.4 ; python_version >= "3.10"  # pinned\n', encoding="utf-8")
check(benchmark._pinned_version("instructor", _req) == "1.15.4",
      "preflight pin ignores environment markers and comments")


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


# 22. a final error the tracker never saw keeps its full text, even when earlier
#     attempts in the same call were tracked
from datetime import datetime  # noqa: E402
from tracking.metrics import AttemptMetrics, CallMetrics, ErrorRecord, ErrorType, TokenUsage  # noqa: E402

LONG_TAIL = "provider said: " + "x" * 400 + " END-OF-ERROR"


class TrackedThenFailLLM(FakeLLM):
    def __call__(self, model, messages, max_retries):
        if model is P.PlannedDemoModule and self._index() == 1:
            now = datetime.now()
            err = ErrorRecord(ErrorType.PARSE_ERROR, "ValidationError", "parse error", now, 1)
            self.pipe.tracker.aggregate.calls.append(CallMetrics(
                call_id="fake", model="fake", provider="fake", start_time=now,
                total_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                attempts=[AttemptMetrics(1, now, TokenUsage(), False, err)], parse_errors=1,
            ))
            raise RuntimeError(LONG_TAIL)
        return super().__call__(model, messages, max_retries)


rec = make(TrackedThenFailLLM(), "t22_untracked", cfg={"step_attempts": 1}).run()
m1 = rec["steps"][-1]
check(m1["calls"] == 1 and m1["errors"][0] == "attempt 1 ValidationError: parse error"
      and m1["errors"][1].endswith("END-OF-ERROR"),
      "an error the tracker didn't record keeps its full text in errors.txt")


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
        if tool == "LabPlan":
            args = (REPLAY / "00_plan.json").read_text(encoding="utf-8")
        elif tool == "ModuleAssetPlan":
            state["asset"] += 1
            args = (REPLAY / f"{state['asset']:02d}_asset_plan.json").read_text(encoding="utf-8")
        elif state.get("always_bad") or (state["asset"] == 2 and not state["bad_sent"]):
            state["bad_sent"] = True
            args = json.dumps({"moduleName": "broken"})  # missing required fields -> reask
        else:
            args = (REPLAY / f"{state['asset']:02d}_module.json").read_text(encoding="utf-8")
        data = json.dumps({
            "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": tool, "arguments": args}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                      "prompt_tokens_details": {"cached_tokens": 600}},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1"]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]
srv = HTTPServer(("127.0.0.1", 0), MockOpenAI)
threading.Thread(target=srv.serve_forever, daemon=True).start()

import instructor  # noqa: E402

_real_factory = P.create_instructor_client
P.create_instructor_client = lambda mc, spec: instructor.from_provider(
    f"openai/{mc.model_id}", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="test")
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
finally:
    P.create_instructor_client = _real_factory
    srv.shutdown()

check(rec["success"], "mock-server run succeeds")
check(rec["totals"]["calls"] == len(rec["steps"]) + 1 and rec["steps"][0]["calls"] == 2,
      "one tracked call per step, plus the reviewer's call charged to the plan step")
m2 = next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 2)
check(m2["parse_errors"] == 1 and m2["errors"], "module 2 records its one parse error")
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
bad_m1 = bad["steps"][-1]
check(not bad["success"] and bad_m1["step"] == "module" and len(bad_m1["errors"]) == 3
      and [e.split(" ", 2)[:2] for e in bad_m1["errors"][:2]] == [["attempt", "1"], ["attempt", "2"]]
      and bad_m1["errors"][2].startswith("step attempt 1 failed: InstructorRetryException after 2")
      and "<failed_attempts>" not in "".join(bad_m1["errors"]),
      "exhausted retries: one line per instructor attempt plus one summary, no duplicate dump")
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Analysis"))
from benchmark_dataframe import load_runs  # noqa: E402
df = load_runs(MOCK)
check(len(df) == 1 and bool(df.iloc[0]["success"]) and df.iloc[0]["structure"] == "multi-module",
      "Code/Analysis load_runs() reads hierarchical metrics")


# ── Part 3: the exact request we send Anthropic (mock Anthropic server) ──
# Builds the client through the real create_instructor_client (pointed at the
# mock via ANTHROPIC_BASE_URL), so this checks the shape of what we actually send:
# tool_choice "auto" with disable_parallel_tool_use (#44, #70; current Claude
# models reject a forced choice), the system line asking for the tool, and
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
            "usage": {"input_tokens": 1000, "output_tokens": 200},
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
_reask = next(r for r in reqs if len(r["messages"]) > 2)
check(_reask["tools"][0]["name"] == "PlannedDemoModule"
      and "no tool invocation" in json.dumps(_reask["messages"][-1]),
      "a plain-text reply is reasked ('no tool invocation')")
_m1 = next(s for s in rec["steps"] if s["step"] == "module" and s["index"] == 1)
check(_m1["success"] and _m1["parse_errors"] == 1 and rec["tool_choice"] == "auto",
      "the reask costs one attempt; the record says tool_choice auto")
check(all(r["max_tokens"] == 32000 for r in reqs), "every request sets max_tokens=32000")
check([r["tools"][0]["name"] for r in reqs[:3]] == ["LabPlan", "ModuleAssetPlan", "PlannedDemoModule"],
      "tool names follow the step response models")
check(all(r["model"] == "claude-haiku-5-5" for r in reqs), "model id sent (Claude Haiku 5.5)")
check(rec["totals"]["prompt_tokens"] == 1000 * len(reqs), "Anthropic usage tracked per step")
check(abs(rec["totals"]["cost_usd"] - len(reqs) * (1000 * 0.10 + 200 * 0.50) / 1e6) < 1e-9,
      "Claude Haiku 5.5 priced at $0.10 / $0.50 per MTok")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)

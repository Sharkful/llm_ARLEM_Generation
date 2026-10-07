"""
Offline smoke test for the hierarchical pipeline — no API keys, no paid calls.

    python "Code/Hierarchical/smoke_test.py"

Part 1 swaps the LLM for a fake call_fn that replays a committed wave-2
multi-module output one module at a time (plan <- its module list, asset plan <-
each module's objects/components, module <- the module itself), validated through
the real response models and validation scopes. It covers the auto-approved run,
Revise/Replace review decisions, failure + resume, the NewComponent check, and
the --no-new-components switch.

Part 2 runs the real _llm_call path (instructor client, tracker, transient
retryer, provider kwargs) against a local mock OpenAI server that replays part
1's step files, with one invalid module response to force an instructor reask.

Outputs go to a temp dir that is removed afterwards. Exit code 1 on any failure.
"""

import json
import os
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
from json_lab import Lab  # noqa: E402
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

    def __init__(self, fail_module=None):
        self.pipe = None
        self.fail_module = fail_module
        self.calls = []

    def _index(self):
        return len(self.pipe.registry.modules) + 1

    def __call__(self, model, messages, max_retries):
        self.calls.append((model.__name__, len(messages)))
        if model is LabOutline:
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
                "interactions": ["advance with Next"],
                "clip_beats": beats if len(beats) >= 2 else beats + ["recap"],
            })
        if self.fail_module == self._index():
            raise ValueError(f"forced failure on module {self._index()}")
        return model.model_validate(m)


def make(fake, run_name, cfg=None, **kw):
    pipe = P.HierarchicalPipeline("claude-haiku-4.5", LAB_NAME, P.PipelineConfig(**(cfg or {})),
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
check(all({**a, "dateCreated": None} == {**b, "dateCreated": None}
          for a, b in zip(out["modules"], MODS)),
      "every module serializes identically to the source (besides dateCreated)")
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
            edited["interactions"] = ["EDITED BY HUMAN"]
            return Replace(edited, note="human edit")
        return Approve()


fake = FakeLLM()
rec = make(fake, "t2_review", reviewer=ScriptedReviewer()).run()
check(rec["success"], "scripted-review run succeeds")
check(fake.calls[:2] == [("LabOutline", 2), ("LabOutline", 4)],
      "Revise re-calls the plan with previous output + feedback appended")
ap1 = json.loads((FAKE / "Runs" / "t2_review" / "01_asset_plan.json").read_text(encoding="utf-8"))
check(ap1["interactions"] == ["EDITED BY HUMAN"], "Replace output is the one saved")
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
t3 = json.loads((FAKE / "Outputs" / "t3_resume_output.json").read_text(encoding="utf-8"))
check(t3["modules"] == out["modules"], "resumed lab identical to the uninterrupted lab")

# 4. PlannedDemoModule rejects an unplanned NewComponent
mod = next(m for m in MODS if "newscript" in json.dumps(m))
with P.module_validation([]):
    try:
        P.PlannedDemoModule.model_validate(mod)
        check(False, "unplanned NewComponent rejected")
    except Exception as e:  # noqa: BLE001
        check("was not planned" in str(e), "unplanned NewComponent rejected")

# 5. --no-new-components: an asset plan proposing a new component fails its step
rec = make(FakeLLM(), "t5_nonew", cfg={"allow_new_components": False, "step_attempts": 1}).run()
first_new = next(i for i, m in enumerate(MODS, 1) if "newscript" in json.dumps(m))
check(not rec["success"] and rec["failed_step"] == f"asset_plan {first_new}",
      f"no-new-components fails at asset_plan {first_new}")
check("disabled" in (rec["error"] or ""), "error says new components are disabled")
prompt = (FAKE / "Runs" / "t5_nonew" / "01_asset_plan_prompt.txt").read_text(encoding="utf-8")
check("New components are disabled" in prompt, "prompt states the switch")


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
        if tool == "LabOutline":
            args = (REPLAY / "00_plan.json").read_text(encoding="utf-8")
        elif tool == "ModuleAssetPlan":
            state["asset"] += 1
            args = (REPLAY / f"{state['asset']:02d}_asset_plan.json").read_text(encoding="utf-8")
        elif state["asset"] == 2 and not state["bad_sent"]:
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
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
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
    rec = P.HierarchicalPipeline("gpt-5.4-mini", LAB_NAME, P.PipelineConfig(),
                                 output_dir=MOCK, run_name="mock").run()
finally:
    P.create_instructor_client = _real_factory
    srv.shutdown()

check(rec["success"], "mock-server run succeeds")
check(rec["totals"]["calls"] == len(rec["steps"]), "one tracked call per step")
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
check(state["requests"][0] == ("LabOutline", 2), "first request is the plan (system + user)")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL PASSED" if not FAILED else "\nSOME FAILED")
sys.exit(1 if FAILED else 0)

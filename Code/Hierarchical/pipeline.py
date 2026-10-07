"""
Hierarchical (module-at-a-time) JSON Lab generation pipeline.

    Lab YAML ──► 1. plan (LabOutline, one scene per module)          ─► checkpoint
             ──► for each scene i:
                   2. asset plan (ModuleAssetPlan: objects, components, why)  ─► checkpoint
                   3. module (DemoModule implementing the approved plan)      ─► checkpoint
             ──► 4. assemble Lab in code (no LLM) ─► analyze ─► save

Each step is its own instructor call with its own retry budget. If a step still
fails after instructor's retries it is regenerated from scratch, up to
``step_attempts`` times, without touching the steps before it. Every accepted step
is written to the run directory, so ``resume()`` continues a failed or interrupted
run from the first missing step without paying for the earlier ones again.

Checkpoints go through a ``Reviewer`` (review.py). The default auto-approves, so a
run is fully unattended; a human or LLM reviewer can approve, request a revision
with feedback, or replace the output with an edited one.

LLM calls go through ``benchmark.create_instructor_client`` (Gemini parse-retry
patch), ``benchmark.provider_create_kwargs`` (Anthropic tool_choice guard), and
``benchmark._transient_retryer`` (bounded 429/5xx retry), the same guards as the
one-shot benchmark runner.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Optional, Type

from pydantic import BaseModel, model_validator

import _paths  # noqa: F401
from _paths import PROJECT_ROOT

from benchmark import (
    _transient_retryer,
    create_instructor_client,
    decode_mode_for,
    get_provenance,
    provider_create_kwargs,
)
from benchmark_config import MODELS, SYSTEM_PROMPT, ModelConfig, SpecType
from json_lab import DemoModule, Lab, NewComponent
from lab_metrics import analyze_assets, analyze_json_lab
from lab_outline import LabOutline
from module_asset_plan import ModuleAssetPlan, asset_plan_validation
from prompt_builder import LabDescription, Level, _render_context, discover_labs
from tracking import InstructorTracker

import prompts
from registry import AssetRegistry
from review import Approve, AutoApproveReviewer, Checkpoint, Replace, Reviewer, Revise


DEFAULT_OUTPUT_DIR = Path("Artifacts") / "Data" / "Hierarchical"

# A step function: (response_model, messages, max_retries) -> validated model.
CallFn = Callable[[Type[BaseModel], list[dict], int], BaseModel]


# ── Module-step validation ───────────────────────────────────────────

_ALLOWED_NEW_COMPONENTS: ContextVar[Optional[frozenset[str]]] = ContextVar(
    "planned_module_allowed_new_components", default=None
)


@contextmanager
def module_validation(allowed_new_components: Iterable[str]) -> Iterator[None]:
    """Activate PlannedDemoModule's NewComponent check in this scope.

    A ContextVar rather than instructor's ``context=``, which would also render
    every message as a Jinja template (see module_asset_plan.py)."""
    token = _ALLOWED_NEW_COMPONENTS.set(frozenset(n.lower() for n in allowed_new_components))
    try:
        yield
    finally:
        _ALLOWED_NEW_COMPONENTS.reset(token)


class PlannedDemoModule(DemoModule):
    """DemoModule plus one check for the module step: every NewComponent must be
    one this module's approved asset plan introduces or one already in the
    catalog. Serializes exactly like DemoModule (no extra fields)."""

    @model_validator(mode="after")
    def _validate_new_components(self) -> "PlannedDemoModule":
        allowed = _ALLOWED_NEW_COMPONENTS.get()
        if allowed is None:
            return self
        comps = [c for o in self.objects for c in (o.components or [])]
        comps += [c for cl in self.clips for ch in (cl.changes or []) for c in (ch.components or [])]
        for comp in comps:
            if isinstance(comp, NewComponent) and comp.scriptName.lower() not in allowed:
                raise ValueError(
                    f"NewComponent '{comp.scriptName}' was not planned for this module. "
                    f"Use a built-in component, or a NewComponent with one of these "
                    f"scriptNames: {sorted(allowed) or 'none'}"
                )
        return self


# ── Config / records ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    level: Level = Level.L4
    plan_retries: int = 3          # instructor max_retries per step type
    asset_plan_retries: int = 3
    module_retries: int = 3
    step_attempts: int = 2         # fresh regenerations of a step after a terminal failure
    max_revisions: int = 3         # reviewer Revise rounds per step before accepting
    allow_new_components: bool = True
    min_objects: int = 4           # lab-wide minima, spread across modules
    min_clips: int = 5

    def to_json(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    @classmethod
    def from_json(cls, d: dict) -> "PipelineConfig":
        d = dict(d)
        d["level"] = Level(d["level"])
        return cls(**d)


@dataclass
class StepRecord:
    step: str
    index: Optional[int]
    calls: int = 0                 # tracked create() calls, incl. regenerations & revisions
    attempts: int = 0              # step-level generations (fresh or revision)
    failed_attempts: int = 0
    retries: int = 0               # instructor parse retries summed over calls
    revisions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    wall_time_s: float = 0.0
    success: bool = False
    resumed: bool = False          # loaded from the run dir, not generated this session
    decision: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class StepFailed(RuntimeError):
    def __init__(self, step: str, index: Optional[int], message: str):
        super().__init__(message)
        self.step = step
        self.index = index


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "lab"


def _rel(path: Path) -> str:
    """Project-relative POSIX path when possible (absolute otherwise)."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _dump(model: BaseModel) -> dict:
    return model.model_dump(mode="json", exclude_none=True)


# ── Plan adherence (soft metric) ─────────────────────────────────────

def plan_adherence(asset_plan: ModuleAssetPlan, module: DemoModule) -> dict:
    """How much of the approved asset plan the module actually implemented."""
    obj_names = {o.name for o in module.objects}
    attached: dict[str, set[str]] = {o.name: set() for o in module.objects}

    def _comp_name(c) -> str:
        return (c.scriptName if isinstance(c, NewComponent) else type(c).__name__).lower()

    for o in module.objects:
        attached[o.name].update(_comp_name(c) for c in o.components or [])
    for clip in module.clips:
        for ch in clip.changes or []:
            attached.setdefault(ch.target, set()).update(_comp_name(c) for c in ch.components or [])

    planned_objs = [o.name for o in asset_plan.objects]
    present_objs = [n for n in planned_objs if n in obj_names]
    planned_comps = [(c.object_name, c.component_type.lower()) for c in asset_plan.components]
    present_comps = [p for p in planned_comps if p[1] in attached.get(p[0], set())]
    return {
        "planned_objects": len(planned_objs),
        "planned_objects_present": len(present_objs),
        "missing_objects": [n for n in planned_objs if n not in obj_names],
        "planned_components": len(planned_comps),
        "planned_components_present": len(present_comps),
        "missing_components": [f"{o}:{c}" for o, c in planned_comps if (o, c) not in present_comps],
        "clip_beats": len(asset_plan.clip_beats),
        "clips": len(module.clips),
    }


# ── Pipeline ─────────────────────────────────────────────────────────

class HierarchicalPipeline:
    def __init__(
        self,
        model_key: str,
        lab_name: str,
        config: PipelineConfig,
        *,
        reviewer: Optional[Reviewer] = None,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        run_name: Optional[str] = None,
        save: bool = True,
        call_fn: Optional[CallFn] = None,
    ):
        if model_key not in MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Available: {sorted(MODELS)}")
        if config.level not in (Level.L3, Level.L4):
            raise ValueError("Hierarchical generation needs a full-spec level: L3 or L4")
        labs = discover_labs()
        if lab_name not in labs:
            raise ValueError(f"Unknown lab '{lab_name}'. Available: {sorted(labs)}")

        self.model_key = model_key
        self.model_config: ModelConfig = MODELS[model_key]
        self.lab_name = lab_name
        self.lab = LabDescription.from_yaml(labs[lab_name])
        self.config = config
        self.reviewer: Reviewer = reviewer or AutoApproveReviewer()
        self.save = save
        self.lab_context = _render_context(self.lab, config.level)

        safe_model = self.model_config.model_id.replace("/", "_").replace(".", "-")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_name = run_name or f"{safe_model}_json_lab_{config.level.value}_{ts}"
        self.output_dir = PROJECT_ROOT / output_dir
        self.run_dir = self.output_dir / "Runs" / self.base_name

        self.tracker = InstructorTracker(
            model=self.model_config.model_id,
            provider=self.model_config.provider.value,
            pricing_config=self.model_config.pricing_config,
        )
        if call_fn is not None:
            self._call_fn = call_fn
            self._tracked_client = None
        else:
            client = create_instructor_client(self.model_config, SpecType.JSON_LAB)
            self._tracked_client = self.tracker.wrap(client)
            self._call_fn = self._llm_call

        self.registry = AssetRegistry()
        self.steps: list[StepRecord] = []
        self.review_log: list[dict] = []
        self.transient_retries = 0
        self._prior_steps: dict[tuple[str, Optional[int]], dict] = {}

    # ── resume ───────────────────────────────────────────────────────
    @classmethod
    def resume(
        cls,
        run_dir: Path,
        *,
        reviewer: Optional[Reviewer] = None,
        call_fn: Optional[CallFn] = None,
        config_overrides: Optional[dict] = None,
    ) -> "HierarchicalPipeline":
        """Rebuild a pipeline for an existing run directory. Steps already on disk
        are loaded instead of regenerated; their earlier per-step metrics are kept."""
        run_dir = Path(run_dir).resolve()
        meta = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        cfg = meta["config"] | (config_overrides or {})
        pipe = cls(
            meta["model_key"], meta["lab_name"], PipelineConfig.from_json(cfg),
            reviewer=reviewer, output_dir=Path(meta["output_dir"]),
            run_name=run_dir.name, call_fn=call_fn,
        )
        metrics_path = pipe.output_dir / "Metrics" / f"{pipe.base_name}_metrics.json"
        if metrics_path.exists():
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            pipe._prior_steps = {(s["step"], s["index"]): s for s in prior.get("steps", [])}
            pipe.review_log = prior.get("review_log", [])
        return pipe

    # ── LLM call ─────────────────────────────────────────────────────
    def _llm_call(self, response_model, messages, max_retries) -> BaseModel:
        kwargs = dict(response_model=response_model, messages=messages, max_retries=max_retries)
        kwargs.update(provider_create_kwargs(self.model_config, response_model))
        retryer = _transient_retryer()
        try:
            return retryer(self._tracked_client.create, **kwargs)
        finally:
            self.transient_retries += retryer.statistics.get("attempt_number", 1) - 1

    def _generate(self, response_model, messages, max_retries, scope, rec: StepRecord) -> BaseModel:
        """One step generation, regenerated from scratch up to step_attempts times."""
        last_error = None
        for _ in range(self.config.step_attempts):
            rec.attempts += 1
            try:
                with scope():
                    return self._call_fn(response_model, messages, max_retries)
            except Exception as e:  # noqa: BLE001 — any failure regenerates the step
                rec.failed_attempts += 1
                last_error = f"{type(e).__name__}: {e}"
                rec.errors.append(last_error)
                print(f"    attempt failed: {last_error[:200]}")
        raise StepFailed(rec.step, rec.index, last_error or "step failed")

    # ── one checkpointed step ────────────────────────────────────────
    def _run_step(
        self,
        step: str,
        index: Optional[int],
        response_model: Type[BaseModel],
        prompt: str,
        max_retries: int,
        file_stem: str,
        what: str,
        scope: Callable[[], ContextManager] = nullcontext,
        context: Optional[dict[str, Any]] = None,
    ) -> BaseModel:
        label = f"{step} {index}" if index else step
        out_path = self.run_dir / f"{file_stem}.json"

        if out_path.exists():
            with scope():
                output = response_model.model_validate_json(out_path.read_text(encoding="utf-8"))
            prior = self._prior_steps.get((step, index))
            rec = StepRecord(**prior) if prior else StepRecord(step=step, index=index, success=True)
            rec.resumed = True
            self.steps.append(rec)
            print(f"  [{label}] loaded from {out_path.name}")
            return output

        if self.save:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / f"{file_stem}_prompt.txt").write_text(prompt, encoding="utf-8")

        print(f"  [{label}] generating...")
        rec = StepRecord(step=step, index=index)
        self.steps.append(rec)
        calls_before = len(self.tracker.aggregate.calls)
        t0 = time.time()
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        messages = base_messages
        try:
            while True:
                output = self._generate(response_model, messages, max_retries, scope, rec)
                decision = self.reviewer.review(Checkpoint(
                    step=step, index=index, output=output,
                    revision=rec.revisions, context=context or {},
                ))
                entry = {"step": step, "index": index, "revision": rec.revisions,
                         "reviewer": self.reviewer.name, "decision": type(decision).__name__}

                if isinstance(decision, Revise):
                    entry["feedback"] = decision.feedback
                    if rec.revisions >= self.config.max_revisions:
                        entry["note"] = "max revisions reached; accepted as is"
                        self.review_log.append(entry)
                        rec.decision = "Approve (max revisions)"
                        break
                    self.review_log.append(entry)
                    rec.revisions += 1
                    messages = base_messages + [
                        {"role": "assistant", "content": json.dumps(_dump(output), indent=2)},
                        {"role": "user", "content": prompts.build_revision_message(what, decision.feedback)},
                    ]
                    continue

                if isinstance(decision, Replace):
                    edited = decision.output
                    data = _dump(edited) if isinstance(edited, BaseModel) else edited
                    with scope():
                        output = response_model.model_validate(data)
                    entry["note"] = decision.note
                elif isinstance(decision, Approve):
                    entry["note"] = decision.note
                else:
                    raise TypeError(f"Reviewer returned {decision!r}")
                self.review_log.append(entry)
                rec.decision = type(decision).__name__
                break
            rec.success = True
        finally:
            rec.wall_time_s = round(time.time() - t0, 2)
            for call in self.tracker.aggregate.calls[calls_before:]:
                rec.calls += 1
                rec.retries += call.retry_count
                rec.prompt_tokens += call.total_usage.prompt_tokens
                rec.completion_tokens += call.total_usage.completion_tokens
                rec.total_tokens += call.total_usage.total_tokens
                rec.cost_usd += self.tracker.pricing.calculate_cost(call.total_usage)
                rec.errors.extend(f"attempt {e.attempt_number} {e.exception_class}: {e.message}"
                                  for e in call.errors)

        if self.save:
            out_path.write_text(json.dumps(_dump(output), indent=2), encoding="utf-8")
        return output

    # ── full run ─────────────────────────────────────────────────────
    def run(self) -> dict:
        cfg = self.config
        print(f"\n  Hierarchical: {self.model_config.display_name} | Lab: {self.lab_name} "
              f"| Level: {cfg.level.value} | Reviewer: {self.reviewer.name}")
        if self.save:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "run_config.json").write_text(json.dumps({
                "model_key": self.model_key,
                "lab_name": self.lab_name,
                "output_dir": _rel(self.output_dir),
                "config": cfg.to_json(),
            }, indent=2), encoding="utf-8")

        t0 = time.time()
        plan: Optional[LabOutline] = None
        lab: Optional[Lab] = None
        modules: list[DemoModule] = []
        adherence: list[dict] = []
        failed_step = None
        error = None
        try:
            plan = self._run_step(
                "plan", None, LabOutline,
                prompts.build_plan_prompt(self.lab_context, has_script=cfg.level == Level.L4),
                cfg.plan_retries, "00_plan", "lab plan",
            )
            n = len(plan.scenes)
            min_objects = prompts.per_module_minimum(cfg.min_objects, n, floor=1)
            min_beats = prompts.per_module_minimum(cfg.min_clips, n, floor=2)

            for i in range(1, n + 1):
                catalog = self.registry.catalog_names()
                asset_plan = self._run_step(
                    "asset_plan", i, ModuleAssetPlan,
                    prompts.build_asset_plan_prompt(
                        self.lab_context, plan, i, self.registry,
                        allow_new_components=cfg.allow_new_components,
                        min_objects=min_objects, min_beats=min_beats,
                    ),
                    cfg.asset_plan_retries, f"{i:02d}_asset_plan", "module asset plan",
                    scope=lambda: asset_plan_validation(catalog, allow_new=cfg.allow_new_components),
                    context={"scene": plan.scenes[i - 1]},
                )
                allowed = prompts.allowed_new_component_names(asset_plan, self.registry)
                module = self._run_step(
                    "module", i, PlannedDemoModule,
                    prompts.build_module_prompt(self.lab_context, plan, i, self.registry, asset_plan),
                    cfg.module_retries, f"{i:02d}_module", "module",
                    scope=lambda: module_validation(allowed),
                    context={"scene": plan.scenes[i - 1], "asset_plan": asset_plan},
                )
                self.registry.register_module(i, module)
                modules.append(module)
                adherence.append({"index": i, **plan_adherence(asset_plan, module)})

            try:
                lab = Lab(
                    labId=_slug(plan.lab_title),
                    courseName=self.lab.course_context,
                    objectives=plan.top_level_objectives,
                    modules=modules,
                )
            except Exception as e:  # noqa: BLE001
                raise StepFailed("assemble", None, f"{type(e).__name__}: {e}") from e
        except StepFailed as e:
            failed_step = f"{e.step}" + (f" {e.index}" if e.index else "")
            error = str(e)
            print(f"  FAILED at {failed_step}: {error[:200]}")

        return self._finish(lab, plan, adherence, failed_step, error, time.time() - t0)

    # ── results ──────────────────────────────────────────────────────
    def _finish(self, lab, plan, adherence, failed_step, error, wall_time_s) -> dict:
        output_json = _dump(lab) if lab is not None else None
        lab_metrics = analyze_json_lab(output_json) if output_json else {}
        asset_metrics = analyze_assets(output_json) if output_json else {}
        steps = [asdict(s) for s in self.steps]
        totals = {
            k: sum(s[k] for s in steps)
            for k in ("calls", "retries", "prompt_tokens", "completion_tokens",
                      "total_tokens", "cost_usd", "wall_time_s")
        }
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_config.model_id,
            "provider": self.model_config.provider.value,
            "display_name": self.model_config.display_name,
            "spec_type": SpecType.JSON_LAB.value,
            "decode_mode": decode_mode_for(self.model_config, SpecType.JSON_LAB).value,
            "topic": None,
            "lab_name": self.lab_name,
            "level": self.config.level.value,
            "structure": "multi-module",
            "pipeline": "hierarchical",
            "granularity": "module",
            "reviewer": self.reviewer.name,
            "config": self.config.to_json(),
            "run_dir": _rel(self.run_dir),
            "success": lab is not None,
            "failed_step": failed_step,
            "error": error,
            "transient_retries": self.transient_retries,
            "wall_time_seconds": round(wall_time_s, 2),
            "provenance": get_provenance(),
            # This session's tracker; `totals` also counts steps resumed from disk.
            "tracking": self.tracker.summary(),
            "totals": totals,
            "steps": steps,
            "review_log": self.review_log,
            "plan_adherence": adherence,
            "registry": {
                "new_components": [asdict(c) for c in self.registry.components if not c.builtin],
                "invented_prefabs": self.registry.invented_prefabs,
                "invented_textures": self.registry.invented_textures,
            },
            "lab_metrics": lab_metrics,
            "asset_metrics": asset_metrics,
        }

        if self.save:
            self._save(record, output_json)
        self._print_summary(record)
        return record

    def _save(self, record: dict, output_json: Optional[dict]) -> None:
        outputs_dir = self.output_dir / "Outputs"
        metrics_dir = self.output_dir / "Metrics"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        errors = [
            f"========== {s['step']}{' ' + str(s['index']) if s['index'] else ''} ==========\n"
            + "\n\n".join(s["errors"])
            for s in record["steps"] if s["errors"]
        ]
        if errors:
            path = self.run_dir / "errors.txt"
            path.write_text("\n\n".join(errors) + "\n", encoding="utf-8")
            record["errors_file"] = _rel(path)

        (self.run_dir / "review_log.json").write_text(
            json.dumps(self.review_log, indent=2, default=str), encoding="utf-8"
        )
        if output_json is not None:
            out = outputs_dir / f"{self.base_name}_output.json"
            out.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
            record["output_path"] = _rel(out)
        (metrics_dir / f"{self.base_name}_metrics.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )

    @staticmethod
    def _print_summary(record: dict) -> None:
        t = record["totals"]
        status = "OK" if record["success"] else f"FAILED at {record['failed_step']}"
        print(f"\n  [{status}] {record['display_name']} — hierarchical json_lab {record['level']}")
        print(f"  Steps: {len(record['steps'])} | calls: {t['calls']} | retries: {t['retries']}")
        print(f"  Tokens: {t['prompt_tokens']:,} in / {t['completion_tokens']:,} out "
              f"| Cost: ${t['cost_usd']:.6f} | Wall time: {record['wall_time_seconds']}s")
        lab = record.get("lab_metrics") or {}
        if lab:
            print(f"  Lab: modules: {lab.get('num_modules')} | objects: {lab.get('num_objects')} "
                  f"| clips: {lab.get('num_clips')} | components: {lab.get('num_components')}")
        print(f"  Run dir: {record['run_dir']}")

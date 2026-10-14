"""
Hierarchical (module-at-a-time) JSON Lab generation pipeline.

    Lab YAML ──► 1. plan (LabPlan, one scene per module)              ─► checkpoint
             ──► for each scene i:
                   2. asset plan (ModuleAssetPlan: objects, components, why)  ─► checkpoint
                   3. module (DemoModule implementing the approved plan)      ─► checkpoint
             ──► 4. assemble Lab in code (no LLM) ─► analyze ─► save

Each step is its own instructor call with its own retry budget. If a step still
fails after instructor's retries it is regenerated from scratch, up to
``step_attempts`` times, without touching the steps before it; errors a retry
can't fix (authentication, permission, unknown model, configuration) fail the
step at once. Every accepted step is written to the run directory, and the
metrics file is rewritten after every step, so ``resume()`` continues a failed or
interrupted run from the first missing step without paying for, or losing the
cost records of, the earlier ones.

Checkpoints go through a ``Reviewer`` (review.py). The default auto-approves, so a
run is fully unattended; a human or LLM reviewer can approve, request a revision
with feedback, or replace the output with an edited one.

After a step is accepted, new-component names are normalized to their catalog
spelling (and reused components to their catalog description), so the final lab
has one version of each script. Every rewrite is logged in ``normalizations``.

LLM calls go through ``benchmark.create_instructor_client`` (Gemini parse-retry
patch), ``benchmark.provider_create_kwargs`` (Anthropic tool_choice guard), and
``benchmark._transient_retryer`` (bounded 429/5xx retry), the same guards as the
one-shot benchmark runner.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Optional, Type

from pydantic import BaseModel, ValidationError, model_validator

import _paths  # noqa: F401
from _paths import PROJECT_ROOT

from benchmark import (
    _transient_retryer,
    create_instructor_client,
    decode_mode_for,
    exception_chain,
    get_provenance,
    provider_create_kwargs,
    tool_choice_for,
)
from benchmark_config import MODELS, SYSTEM_PROMPT, ModelConfig, SpecType
from json_lab import DemoModule, Lab, NewComponent, PointerReceiverComponent
from lab_metrics import analyze_assets, analyze_json_lab
from lab_outline import LabOutline
from module_asset_plan import ModuleAssetPlan, asset_plan_validation, component_key
from prompt_builder import LabDescription, Level, _render_context, discover_labs
from tracking import InstructorTracker

import prompts
from registry import AssetRegistry, iter_module_components
from review import Approve, AutoApproveReviewer, Checkpoint, Replace, Reviewer, Revise


DEFAULT_OUTPUT_DIR = Path("Artifacts") / "Data" / "Hierarchical"
_RUN_GROUP_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def output_dir_for(run_group: Optional[str]) -> Path:
    """``Artifacts/Data/Hierarchical/[<run_group>/]``. A run group is a single folder
    name, so test runs and data-collection runs can be kept apart."""
    if run_group is None:
        return DEFAULT_OUTPUT_DIR
    if not _RUN_GROUP_RE.match(run_group):
        raise ValueError(
            f"Invalid run group {run_group!r}: use a single folder name made of "
            "letters, digits, '_' or '-'"
        )
    return DEFAULT_OUTPUT_DIR / run_group

# A step function: (response_model, messages, max_retries) -> validated model.
CallFn = Callable[[Type[BaseModel], list[dict], int], BaseModel]


# ── Step-specific validation scopes ──────────────────────────────────
# ContextVars rather than instructor's ``context=``, which would also render every
# message as a Jinja template (see module_asset_plan.py).

_MAX_SCENES: ContextVar[Optional[int]] = ContextVar("lab_plan_max_scenes", default=None)
_ALLOWED_NEW_COMPONENTS: ContextVar[Optional[frozenset[str]]] = ContextVar(
    "planned_module_allowed_new_components", default=None
)


@contextmanager
def plan_validation(max_scenes: int) -> Iterator[None]:
    """Activate LabPlan's scene cap in this scope."""
    token = _MAX_SCENES.set(max_scenes)
    try:
        yield
    finally:
        _MAX_SCENES.reset(token)


@contextmanager
def module_validation(allowed_new_components: Iterable[str]) -> Iterator[None]:
    """Activate PlannedDemoModule's NewComponent check in this scope."""
    token = _ALLOWED_NEW_COMPONENTS.set(
        frozenset(component_key(n) for n in allowed_new_components)
    )
    try:
        yield
    finally:
        _ALLOWED_NEW_COMPONENTS.reset(token)


class LabPlan(LabOutline):
    """LabOutline for the plan step, where every scene becomes a module: adds a
    scene cap so the number of calls per lab stays bounded. The shared L1 schema
    (lab_outline.py) is unchanged."""

    @model_validator(mode="after")
    def _cap_scenes(self) -> "LabPlan":
        cap = _MAX_SCENES.get()
        if cap is not None and len(self.scenes) > cap:
            raise ValueError(
                f"Plan at most {cap} scenes (got {len(self.scenes)}); merge closely "
                "related scenes so each one is a single clear teaching step."
            )
        return self


class PlannedDemoModule(DemoModule):
    """DemoModule plus one check for the module step: every NewComponent must be
    one this module's approved asset plan introduces or one already in the
    catalog. Serializes exactly like DemoModule (no extra fields)."""

    @model_validator(mode="after")
    def _validate_new_components(self) -> "PlannedDemoModule":
        allowed = _ALLOWED_NEW_COMPONENTS.get()
        if allowed is None:
            return self
        unplanned = sorted({
            c.scriptName for c in iter_module_components(self)
            if isinstance(c, NewComponent) and component_key(c.scriptName) not in allowed
        })
        if unplanned:
            raise ValueError(
                f"NewComponent(s) {unplanned} were not planned for this module. Use a "
                "built-in component, or a NewComponent with one of these scriptNames: "
                f"{sorted(allowed) or 'none'}"
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
    max_revisions: int = 3         # reviewer revision rounds per step before accepting
    allow_new_components: bool = True
    min_objects: int = 4           # per-module minimums (lab-wide in the one-shot runs)
    min_clips: int = 5
    max_scenes: int = 8            # scenes (= modules) the plan may contain

    def __post_init__(self):
        """Reject settings no plan or step could ever satisfy, before any call."""
        problems = []
        if not prompts.MIN_SCENES <= self.max_scenes <= prompts.MAX_SCENES_LIMIT:
            problems.append(
                f"max_scenes must be {prompts.MIN_SCENES}-{prompts.MAX_SCENES_LIMIT} "
                f"(the plan schema accepts {prompts.MIN_SCENES}-{prompts.MAX_SCENES_LIMIT} "
                f"scenes); got {self.max_scenes}"
            )
        if self.min_objects < 1:
            problems.append(f"min_objects must be >= 1; got {self.min_objects}")
        if self.min_clips < prompts.MIN_CLIP_BEATS:
            problems.append(f"min_clips must be >= {prompts.MIN_CLIP_BEATS} (an asset plan "
                            f"needs {prompts.MIN_CLIP_BEATS}+ clip beats); got {self.min_clips}")
        if self.step_attempts < 1:
            problems.append(f"step_attempts must be >= 1; got {self.step_attempts}")
        for name in ("plan_retries", "asset_plan_retries", "module_retries", "max_revisions"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be >= 0; got {getattr(self, name)}")
        if problems:
            raise ValueError("Invalid pipeline settings: " + "; ".join(problems))

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
    retries: int = 0               # reasks: tracker retry_count (requests - 1) summed over calls
    parse_errors: int = 0          # failed parse/validation attempts (= retries, +1 per call
                                   # that ran out of attempts: its last failure isn't reasked)
    revisions: int = 0             # reviewer revision rounds (Revise, or a rejected Replace)
    prompt_tokens: int = 0
    cached_tokens: int = 0         # of prompt_tokens, served from the provider's prompt cache
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    wall_time_s: float = 0.0
    success: bool = False
    resumed: bool = False          # loaded from the run dir, not generated this session
    non_retryable: bool = False    # failed on an error regeneration can't fix
    decision: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class StepFailed(RuntimeError):
    def __init__(self, step: str, index: Optional[int], message: str):
        super().__init__(message)
        self.step = step
        self.index = index


# Errors no regeneration can fix. A provider 400 is deliberately *not* here:
# Gemini's parallel-functionCall reask 400 (#44) is often survived by a fresh
# regeneration, which is exactly what step_attempts is for.
_NON_RETRYABLE_STATUS = {401, 403, 404}
_NON_RETRYABLE_EXC_NAMES = {
    "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConfigurationError",
}


def is_non_retryable(exc: BaseException) -> bool:
    """True if ``exc`` (or anything in its cause chain) is an error that
    regenerating the step can't fix."""
    for e in exception_chain(exc):
        if type(e).__name__ in _NON_RETRYABLE_EXC_NAMES:
            return True
        for attr in ("status_code", "code"):
            val = getattr(e, attr, None)
            if isinstance(val, int) and val in _NON_RETRYABLE_STATUS:
                return True
    return False


def _first_error_line(exc: BaseException) -> str:
    """The informative part of one exception: for a pydantic ValidationError the
    first error's location and message (its str() starts with a generic header),
    otherwise the first non-empty line."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            errs = errors()
            first = errs[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            lines = [ln.strip() for ln in str(first.get("msg", "")).splitlines() if ln.strip()]
            msg = lines[0].removeprefix("Value error, ") if lines else ""
            if msg.endswith(":") and len(lines) > 1:   # aggregated list: show the first item
                msg = f"{msg} {lines[1].lstrip('- ')}"
            more = f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""
            return f"{loc + ': ' if loc else ''}{msg}{more}"
        except Exception:  # noqa: BLE001 — fall back to the plain text
            pass
    return next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), type(exc).__name__)


def brief_error(exc: BaseException, limit: int = 220) -> str:
    """One readable console line for a failed step attempt. The full text still
    goes to the step record / errors.txt. Instructor's retry exception wraps the
    individual attempts; report the count and the last attempt's error."""
    attempts = getattr(exc, "failed_attempts", None)
    if attempts:
        last = attempts[-1]
        inner = getattr(last, "exception", None) or last
        n = getattr(exc, "n_attempts", None) or len(attempts)
        text = (f"{type(exc).__name__} after {n} attempt(s); last: "
                f"{type(inner).__name__} {_first_error_line(inner)}")
    else:
        text = f"{type(exc).__name__}: {_first_error_line(exc)}"
    return text if len(text) <= limit else text[: limit - 1] + "…"


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


def _same_text(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


# ── Normalization (one spelling per component) ───────────────────────

def normalize_asset_plan(asset_plan: ModuleAssetPlan, registry: AssetRegistry) -> list[dict]:
    """Rewrite component references to their canonical spelling: the catalog's,
    or the one this plan declared a new component with. Returns the changes."""
    declared = {component_key(c.component_type): c.component_type
                for c in asset_plan.components if c.is_new}
    changes = []
    for comp in asset_plan.components:
        if comp.is_new:
            continue
        entry = registry.find_component(comp.component_type)
        canonical = entry.name if entry else declared.get(component_key(comp.component_type))
        if canonical and canonical != comp.component_type:
            changes.append({"field": "component_type", "object": comp.object_name,
                            "from": comp.component_type, "to": canonical})
            comp.component_type = canonical
    return changes


def normalize_module(
    module: DemoModule, asset_plan: ModuleAssetPlan, registry: AssetRegistry
) -> list[dict]:
    """Give every NewComponent one canonical name and description: a component
    already in the catalog takes the catalog's; one introduced here takes the
    asset plan's name and its first description in this module."""
    planned = {component_key(c.component_type): c.component_type
               for c in asset_plan.components if c.is_new}
    first_description: dict[str, str] = {}
    changes = []
    for comp in iter_module_components(module):
        if not isinstance(comp, NewComponent):
            continue
        key = component_key(comp.scriptName)
        entry = registry.find_component(comp.scriptName)
        if entry is not None and not entry.builtin:
            name, description = entry.name, entry.description
        else:
            name = planned.get(key, comp.scriptName)
            description = first_description.setdefault(key, comp.scriptDescription)
        if comp.scriptName != name:
            changes.append({"field": "scriptName", "from": comp.scriptName, "to": name})
            comp.scriptName = name
        if not _same_text(comp.scriptDescription, description):
            changes.append({"field": "scriptDescription", "component": name,
                            "from": comp.scriptDescription[:80], "to": description[:80]})
            comp.scriptDescription = description
    return changes


# ── Plan adherence (soft metric) ─────────────────────────────────────

def plan_adherence(
    asset_plan: ModuleAssetPlan,
    module: DemoModule,
    *,
    min_objects: int,
    min_clips: int,
) -> dict:
    """How much of the approved asset plan the module actually implemented."""
    obj_names = {o.name for o in module.objects}
    attached: dict[str, set[str]] = {o.name: set() for o in module.objects}

    def _comp_name(c) -> str:
        return component_key(c.scriptName if isinstance(c, NewComponent) else type(c).__name__)

    for o in module.objects:
        attached[o.name].update(_comp_name(c) for c in o.components or [])
    for clip in module.clips:
        for ch in clip.changes or []:
            attached.setdefault(ch.target, set()).update(_comp_name(c) for c in ch.components or [])

    planned_objs = [o.name for o in asset_plan.objects]
    present_objs = [n for n in planned_objs if n in obj_names]
    planned_comps = [(c.object_name, component_key(c.component_type)) for c in asset_plan.components]
    present_comps = [p for p in planned_comps if p[1] in attached.get(p[0], set())]
    pointer = component_key(PointerReceiverComponent.__name__)
    return {
        "planned_objects": len(planned_objs),
        "planned_objects_present": len(present_objs),
        "missing_objects": [n for n in planned_objs if n not in obj_names],
        "planned_components": len(planned_comps),
        "planned_components_present": len(present_comps),
        "missing_components": [f"{o}:{c}" for o, c in planned_comps if (o, c) not in present_comps],
        "clip_beats": len(asset_plan.clip_beats),
        "clips": len(module.clips),
        "objects": len(module.objects),
        "below_min_objects": len(module.objects) < min_objects,
        "below_min_clips": len(module.clips) < min_clips,
        # Interactions whose object can't receive pointer input in the module.
        "interaction_objects_without_pointer": sorted({
            i.object_name for i in asset_plan.interactions
            if pointer not in attached.get(i.object_name, set())
        }),
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
        run_group: Optional[str] = None,
        output_dir: Optional[Path] = None,
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
        # ``output_dir`` overrides the run-group folder; it is for programmatic use
        # (the smoke test points it at a temp dir), not exposed on the CLI.
        self.run_group = run_group
        self.output_dir = PROJECT_ROOT / (output_dir or output_dir_for(run_group))
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
        self.normalizations: list[dict] = []
        self.adherence: list[dict] = []
        self.transient_retries = 0
        self._prior_steps: dict[tuple[str, Optional[int]], dict] = {}
        self._t0 = time.time()
        self._current_step: Optional[str] = None
        self._traceback: Optional[str] = None   # set when run() catches an unexpected error
        self._calls_absorbed = 0                # tracker calls already charged to a step

    # ── resume ───────────────────────────────────────────────────────
    @staticmethod
    def read_run_config(run_dir: Path) -> dict:
        """The run_config.json of a run directory; ValueError if ``run_dir`` isn't one."""
        run_dir = Path(run_dir).resolve()
        config_path = run_dir / "run_config.json"
        if run_dir.parent.name != "Runs" or not config_path.exists():
            raise ValueError(
                f"{run_dir} is not a hierarchical run directory "
                "(expected <output root>/Runs/<run name>/run_config.json)"
            )
        return json.loads(config_path.read_text(encoding="utf-8"))

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
        meta = cls.read_run_config(run_dir)
        cfg = meta["config"] | (config_overrides or {})
        # Continue where the run directory actually is: a copied or moved run
        # must not silently regenerate (and re-bill) everything at its old path.
        output_root = run_dir.parent.parent
        run_group = meta.get("run_group")
        if output_root != (PROJECT_ROOT / meta["output_dir"]).resolve():
            default_root = (PROJECT_ROOT / DEFAULT_OUTPUT_DIR).resolve()
            if output_root == default_root:
                run_group = None
            elif output_root.parent == default_root:
                run_group = output_root.name
            print(f"  NOTE: run directory moved from {meta['output_dir']}; "
                  f"continuing at {_rel(output_root)}")
        pipe = cls(
            meta["model_key"], meta["lab_name"], PipelineConfig.from_json(cfg),
            reviewer=reviewer, run_group=run_group, output_dir=output_root,
            run_name=run_dir.name, call_fn=call_fn,
        )
        metrics_path = pipe.output_dir / "Metrics" / f"{pipe.base_name}_metrics.json"
        if metrics_path.exists():
            prior = json.loads(metrics_path.read_text(encoding="utf-8"))
            pipe._prior_steps = {(s["step"], s["index"]): s for s in prior.get("steps", [])}
            pipe.review_log = prior.get("review_log", [])
            pipe.normalizations = prior.get("normalizations", [])
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

    def _absorb_calls(self, rec: StepRecord) -> list:
        """Charge every tracker call not yet charged to a step to ``rec`` (tokens,
        cost, retries, per-attempt errors). Returns the calls it charged."""
        calls = self.tracker.aggregate.calls
        new, self._calls_absorbed = calls[self._calls_absorbed:], len(calls)
        for call in new:
            rec.calls += 1
            rec.retries += call.retry_count
            rec.parse_errors += call.parse_errors
            rec.prompt_tokens += call.total_usage.prompt_tokens
            rec.cached_tokens += call.total_usage.cached_tokens
            rec.completion_tokens += call.total_usage.completion_tokens
            rec.total_tokens += call.total_usage.total_tokens
            rec.cost_usd += self.tracker.pricing.calculate_cost(call.total_usage)
            rec.errors.extend(f"attempt {err.attempt_number} {err.exception_class}: {err.message}"
                              for err in call.errors)
        return new

    def _generate(self, response_model, messages, max_retries, scope, rec: StepRecord) -> BaseModel:
        """One step generation, regenerated from scratch up to step_attempts times
        (immediately failed for errors a regeneration can't fix).

        Each instructor attempt's error is recorded once, from the tracker; a
        failed step attempt then adds a summary line. When the call that failed
        has a ``final_error`` in the tracker (the error that ended it: a parse
        error that ran out of attempts, an API error such as a 5xx or 400, a
        max_tokens truncation), that text is already in the step's errors, so the
        line is shortened to one. Any other exception (one raised before a
        request was sent, or outside the instructor call) keeps its full text."""
        last_error = None
        for _ in range(self.config.step_attempts):
            rec.attempts += 1
            try:
                with scope():
                    return self._call_fn(response_model, messages, max_retries)
            except Exception as e:  # noqa: BLE001 — any failure regenerates the step
                rec.failed_attempts += 1
                absorbed = self._absorb_calls(rec)
                last_error = f"{type(e).__name__}: {e}"
                repeats_tracker = bool(absorbed) and absorbed[-1].final_error is not None
                rec.errors.append(f"step attempt {rec.attempts} failed: "
                                  + (brief_error(e) if repeats_tracker else last_error))
                where = " (details: errors.txt)" if self.save else ""
                print(f"    attempt failed: {brief_error(e)}{where}")
                if is_non_retryable(e):
                    rec.non_retryable = True
                    raise StepFailed(rec.step, rec.index, f"{last_error} (not retryable)") from e
            finally:
                self._absorb_calls(rec)
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
        normalize: Optional[Callable[[BaseModel], list[dict]]] = None,
    ) -> BaseModel:
        label = f"{step} {index}" if index else step
        self._current_step = label
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
        prior = self._prior_steps.get((step, index))
        if prior:
            # Regenerating a step an earlier session already spent tokens on (it
            # failed or was interrupted): keep its counts, cost and errors.
            rec = StepRecord(**{**prior, "success": False, "resumed": False,
                                "decision": None, "non_retryable": False})
        else:
            rec = StepRecord(step=step, index=index)
        self.steps.append(rec)
        t0 = time.time()
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        context = dict(context or {})
        context.setdefault("run_dir", str(self.run_dir))
        feedback_history: list[str] = []
        last_error: Optional[str] = None
        rejected: Optional[dict] = None
        try:
            output = self._generate(response_model, base_messages, max_retries, scope, rec)
            while True:
                decision = self.reviewer.review(Checkpoint(
                    step=step, index=index, output=output, revision=rec.revisions,
                    context=context, feedback_history=list(feedback_history),
                    last_error=last_error, rejected_output=rejected,
                ))
                last_error = rejected = None
                entry = {"step": step, "index": index, "revision": rec.revisions,
                         "reviewer": self.reviewer.name, "decision": type(decision).__name__}
                out_of_rounds = rec.revisions >= self.config.max_revisions

                if isinstance(decision, Approve):
                    entry["note"] = decision.note
                    self.review_log.append(entry)
                    rec.decision = "Approve"
                    break

                if isinstance(decision, Revise):
                    entry["feedback"] = decision.feedback
                    if out_of_rounds:
                        entry["note"] = "max revisions reached; accepted as is"
                        self.review_log.append(entry)
                        rec.decision = "Approve (max revisions)"
                        break
                    self.review_log.append(entry)
                    rec.revisions += 1
                    feedback_history.append(decision.feedback)
                    messages = base_messages + [
                        {"role": "assistant", "content": json.dumps(_dump(output), indent=2)},
                        {"role": "user",
                         "content": prompts.build_revision_message(what, feedback_history)},
                    ]
                    try:
                        output = self._generate(response_model, messages, max_retries, scope, rec)
                    except StepFailed as e:
                        # An auth or config error would fail every later call too.
                        if rec.non_retryable:
                            raise
                        # Otherwise the step already has a valid output; a revision that
                        # can't be generated shouldn't fail the whole run, and it doesn't
                        # count as a revision round.
                        rec.revisions -= 1
                        self.review_log.append({
                            "step": step, "index": index, "revision": rec.revisions,
                            "reviewer": self.reviewer.name, "decision": "Revise",
                            "note": "revision failed; kept the last valid output",
                            "error": str(e),
                        })
                        print(f"    revision failed; keeping the last valid {what}")
                        rec.decision = "Approve (revision failed)"
                        break
                    continue

                if isinstance(decision, Replace):
                    edited = decision.output
                    data = _dump(edited) if isinstance(edited, BaseModel) else edited
                    try:
                        with scope():
                            candidate = response_model.model_validate(data)
                    except ValidationError as e:
                        # Back to the reviewer with every error at once, plus the
                        # edit itself so it can be fixed rather than redone.
                        entry["note"] = "edit rejected: failed validation"
                        entry["error"] = str(e)
                        if out_of_rounds:
                            entry["note"] += "; max revisions reached, kept the last valid output"
                            self.review_log.append(entry)
                            rec.decision = "Approve (max revisions)"
                            break
                        self.review_log.append(entry)
                        rec.revisions += 1
                        last_error, rejected = str(e), data
                        continue
                    output = candidate
                    entry["note"] = decision.note
                    self.review_log.append(entry)
                    rec.decision = "Replace"
                    break

                raise TypeError(f"Reviewer returned {decision!r}")
            rec.success = True
        finally:
            # Calls made during the step outside generation (e.g. an LLM reviewer
            # using the tracked client) are charged to it too.
            self._absorb_calls(rec)
            rec.wall_time_s = round(rec.wall_time_s + time.time() - t0, 2)

        if normalize is not None:
            for change in normalize(output):
                self.normalizations.append({"step": step, "index": index, **change})
        if self.save:
            out_path.write_text(json.dumps(_dump(output), indent=2), encoding="utf-8")
            # Keep the metrics current so an interrupted run keeps its cost records.
            self._save(self._build_record(None, None, None, status="in_progress"), None)
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
                "run_group": self.run_group,
                "output_dir": _rel(self.output_dir),
                "config": cfg.to_json(),
            }, indent=2), encoding="utf-8")

        self._t0 = time.time()
        lab: Optional[Lab] = None
        failed_step = None
        error = None
        try:
            plan = self._run_step(
                "plan", None, LabPlan,
                prompts.build_plan_prompt(self.lab_context, has_script=cfg.level == Level.L4,
                                          max_scenes=cfg.max_scenes),
                cfg.plan_retries, "00_plan", "lab plan",
                scope=lambda: plan_validation(cfg.max_scenes),
            )
            modules: list[DemoModule] = []
            for i in range(1, len(plan.scenes) + 1):
                scene = plan.scenes[i - 1]
                shared_context = {
                    "scene": scene,
                    "plan": plan,
                    "previous_modules": self.registry.render_previous_modules(),
                }
                catalog = self.registry.catalog_names()
                asset_plan = self._run_step(
                    "asset_plan", i, ModuleAssetPlan,
                    prompts.build_asset_plan_prompt(
                        self.lab_context, plan, i, self.registry,
                        allow_new_components=cfg.allow_new_components,
                        min_objects=cfg.min_objects, min_beats=cfg.min_clips,
                    ),
                    cfg.asset_plan_retries, f"{i:02d}_asset_plan", "module asset plan",
                    scope=lambda: asset_plan_validation(
                        catalog, allow_new=cfg.allow_new_components,
                        min_objects=cfg.min_objects, min_clip_beats=cfg.min_clips,
                    ),
                    context=shared_context,
                    normalize=lambda ap: normalize_asset_plan(ap, self.registry),
                )
                allowed = prompts.allowed_new_component_names(asset_plan, self.registry)
                module = self._run_step(
                    "module", i, PlannedDemoModule,
                    prompts.build_module_prompt(
                        self.lab_context, plan, i, self.registry, asset_plan,
                        min_objects=cfg.min_objects, min_clips=cfg.min_clips,
                    ),
                    cfg.module_retries, f"{i:02d}_module", "module",
                    scope=lambda: module_validation(allowed),
                    context={**shared_context, "asset_plan": asset_plan},
                    normalize=lambda m: normalize_module(m, asset_plan, self.registry),
                )
                adherence = plan_adherence(
                    asset_plan, module, min_objects=cfg.min_objects, min_clips=cfg.min_clips
                )
                # Checked against the registry *before* this module registers.
                adherence["is_new_asset_mismatches"] = [
                    o.name for o in asset_plan.objects
                    if o.is_new_asset != self._is_novel_asset(o.prefab, o.texture)
                ]
                adherence["new_components_planned"] = sum(c.is_new for c in asset_plan.components)
                before = len(self.registry.new_component_names())
                self.registry.register_module(i, module, asset_plan, scene_purpose=scene.brief_purpose)
                adherence["new_components_added"] = len(self.registry.new_component_names()) - before
                self.adherence.append({"index": i, **adherence})
                modules.append(module)

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
        except KeyboardInterrupt:
            # Leave an honest record (metrics say where it stopped), then stop.
            if self.save:
                self._save(self._build_record(None, self._current_step, "interrupted",
                                              status="interrupted"), None)
            raise
        except Exception as e:  # noqa: BLE001 — record any failure instead of leaving in_progress
            # Not an API failure but a bug (pipeline, reviewer, schema): keep the
            # traceback on the console, in the metrics and in errors.txt.
            failed_step = self._current_step or "unknown"
            error = f"{type(e).__name__}: {e}"
            self._traceback = traceback.format_exc()
            print(self._traceback, end="")
            print(f"  FAILED at {failed_step}: {error[:200]}")

        record = self._build_record(lab, failed_step, error,
                                    status="complete" if lab is not None else "failed")
        output_json = _dump(lab) if lab is not None else None
        if self.save:
            self._save(record, output_json)
        self._print_summary(record)
        return record

    def _is_novel_asset(self, prefab: str, texture: Optional[str]) -> bool:
        return (not self.registry.is_known_asset(prefab=prefab)) or bool(
            texture and not self.registry.is_known_asset(texture=texture)
        )

    # ── results ──────────────────────────────────────────────────────
    def _build_record(self, lab: Optional[Lab], failed_step, error, *, status: str) -> dict:
        output_json = _dump(lab) if lab is not None else None
        steps = [asdict(s) for s in self.steps]
        totals = {
            k: sum(s[k] for s in steps)
            for k in ("calls", "retries", "parse_errors", "prompt_tokens", "cached_tokens",
                      "completion_tokens",
                      "total_tokens", "cost_usd", "wall_time_s")
        }
        return {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_config.model_id,
            "provider": self.model_config.provider.value,
            "display_name": self.model_config.display_name,
            "spec_type": SpecType.JSON_LAB.value,
            "decode_mode": decode_mode_for(self.model_config, SpecType.JSON_LAB).value,
            "tool_choice": tool_choice_for(self.model_config),
            "topic": None,
            "lab_name": self.lab_name,
            "level": self.config.level.value,
            "structure": "multi-module",
            "pipeline": "hierarchical",
            "granularity": "module",
            "run_group": self.run_group,
            "reviewer": self.reviewer.name,
            "config": self.config.to_json(),
            "run_dir": _rel(self.run_dir),
            # "in_progress" while running, then "complete" / "failed" / "interrupted"
            "status": status,
            "success": lab is not None,
            "failed_step": failed_step,
            "error": error,
            "traceback": self._traceback,   # only for unexpected (non-API) errors
            "transient_retries": self.transient_retries,
            "wall_time_seconds": round(time.time() - self._t0, 2),
            "provenance": get_provenance(),
            # This session's tracker; `totals` also counts steps resumed from disk.
            "tracking": self.tracker.summary(),
            "totals": totals,
            "steps": steps,
            "review_log": self.review_log,
            "normalizations": self.normalizations,
            "plan_adherence": self.adherence,
            "new_components_total": len(self.registry.new_component_names()),
            "registry": {
                "new_components": [asdict(c) for c in self.registry.components if not c.builtin],
                "invented_prefabs": {k: asdict(v) for k, v in self.registry.invented_prefabs.items()},
                "invented_textures": {k: asdict(v) for k, v in self.registry.invented_textures.items()},
            },
            "lab_metrics": analyze_json_lab(output_json) if output_json else {},
            "asset_metrics": analyze_assets(output_json) if output_json else {},
        }

    def _save(self, record: dict, output_json: Optional[dict]) -> None:
        outputs_dir = self.output_dir / "Outputs"
        metrics_dir = self.output_dir / "Metrics"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        errors = [
            f"========== {s['step']}{' ' + str(s['index']) if s['index'] else ''} ==========\n"
            + "\n\n".join(s["errors"])
            for s in record["steps"] if s["errors"]
        ]
        if record.get("traceback"):
            errors.append("========== traceback ==========\n" + record["traceback"].rstrip())
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
        print(f"  Tokens: {t['prompt_tokens']:,} in ({t['cached_tokens']:,} cached) / "
              f"{t['completion_tokens']:,} out "
              f"| Cost: ${t['cost_usd']:.6f} | Wall time: {record['wall_time_seconds']}s")
        lab = record.get("lab_metrics") or {}
        if lab:
            print(f"  Lab: modules: {lab.get('num_modules')} | objects: {lab.get('num_objects')} "
                  f"| clips: {lab.get('num_clips')} | components: {lab.get('num_components')}")
        print(f"  New components: {record['new_components_total']}")
        if record.get("normalizations"):
            print(f"  Normalized names/descriptions: {len(record['normalizations'])}")
        print(f"  Run dir: {record['run_dir']}")

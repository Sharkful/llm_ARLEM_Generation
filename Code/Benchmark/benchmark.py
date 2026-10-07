"""
AR Lab Generation Benchmark Runner

Benchmarks LLM providers on structured AR lab generation quality,
tracking tokens, cost, retries, and output metrics.

Usage:
    # Single model, default topic (JSON Lab spec)
    python benchmark.py --model gpt-4o-mini

    # Multiple models, custom topic
    python benchmark.py --model gpt-4o-mini claude-haiku-4.5 gemini-2.5-flash \
        --topic "Volcanic Eruption Mechanics"

    # ARLEM spec instead of JSON Lab (YAML-driven path, L1/L3/L4)
    python benchmark.py --model claude-haiku-4.5 --lab phases_of_the_moon --level L3 --spec arlem

    # Sweep multiple output formats together (json_lab + ARLEM full + simplified)
    python benchmark.py --model gpt-4o-mini --lab phases_of_the_moon --level L3 \
        --spec json_lab arlem arlem_simple

    # Quick benchmark suite (all default topics × cheap models)
    python benchmark.py --suite quick

    # Full benchmark suite (all default topics × all models)
    python benchmark.py --suite full
"""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path so we can import from Code/Schemas and tracking/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Schemas"))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Local imports
from benchmark_config import (
    ARLEM_PROMPT_TEMPLATE,
    BenchmarkRunConfig,
    DEFAULT_TOPICS,
    FULL_BENCHMARK_MODELS,
    MODELS,
    ModelConfig,
    ModelSize,
    Provider,
    QUICK_BENCHMARK_MODELS,
    SYSTEM_PROMPT,
    SpecType,
    models_by_size,
)
from prompt_builder import (
    LabDescription,
    Level,
    Structure,
    build_prompt,
    discover_labs,
)
from lab_metrics import analyze_json_lab, analyze_arlem

from tracking import InstructorTracker, BenchmarkExporter


# ── Client Factory ───────────────────────────────────────────────────

_GENAI_PATCH_WARNED = False


def _patch_genai_parallel_call_retry():
    """Make Gemini's ``parse_genai_tools`` assertion failures retryable (issue #44).

    ``parse_genai_tools`` guards its parse with four asserts. The one that fires
    in practice is a *text part + one functionCall* response, where the count
    guard ("Instructor does not support multiple function calls") trips even
    though there is a single call to decode. instructor 1.15.4's retry loop only
    retries ValidationError / JSONDecodeError / ResponseParsingError and re-raises
    the AssertionError, so that shape kills the run on attempt 1 — retries=0, the
    exact fairness problem #44 exists to fix. Re-raising as ResponseParsingError
    (retryable, message fed back to the model on reask) restores retryability for
    it.

    Scope: this only rescues the 1:1 (text + single functionCall) shape. A *true*
    ≥2-functionCall response is still a hard fail, and the root cause is upstream in
    ``instructor.v2.providers.genai.handlers.reask_genai_tools`` (~L75-82 in
    1.15.4): the reask replays the model turn with all N functionCall parts but
    appends a single functionResponse, and Gemini requires the counts to match, so
    it 400s (INVALID_ARGUMENT). This is not fixable here — the malformed request is
    built inside the reask handler, after this patch runs — and not preventable by
    config: instructor already forces the strongest constraint the API offers,
    ``FunctionCallingConfig(mode=ANY, allowed_function_names=[<tool>])`` (same
    module, ~L400-409), yet Gemini can still emit parallel calls to that one tool,
    and google-genai exposes no "disable parallel calls" flag (unlike Anthropic's
    ``disable_parallel_tool_use``). So it stays a documented hard-fail (issue #53 /
    review F2 / threats §7), tracked upstream, and is *detected* in the run stats
    as the ``fail_mode`` "gemini parallel-call reask 400"
    (``benchmark_dataframe.fail_mode``) so a re-run wave can spot it when it occurs.

    It is *not* a re-creation of 1.14.4 behavior: 1.14.4 did not give corrective
    feedback for this shape — its generic branch blind-replayed the request
    unchanged (reask feedback existed only for ValidationError / JSONDecodeError /
    InstructorValidationError). Remove this patch if upstream makes the parse error
    retryable.
    """
    import instructor.v2.providers.genai.handlers as genai_handlers
    from instructor.v2.core.errors import ResponseParsingError

    original_parse = genai_handlers.parse_genai_tools
    if getattr(original_parse, "_arlem_retryable_patch", False):
        return

    def parse_genai_tools_retryable(*args, **kwargs):
        try:
            return original_parse(*args, **kwargs)
        except AssertionError as e:
            # Three of the four asserts in parse_genai_tools are bare (empty
            # text): a content-free ResponseParsingError yields an uninformative
            # reask and an empty error string that fail-mode/quarantine
            # classification can't match — substitute a concrete message. The one
            # messaged assert steers the model toward "use List[Model] instead",
            # advice it can't act on inside a single-object response schema; swap
            # that clause for feedback it can (review F8).
            msg = str(e).strip()
            if not msg:
                msg = (
                    "The model's response could not be parsed as a single function "
                    "call. Return exactly one function call matching the requested "
                    "schema."
                )
            else:
                msg = msg.replace(
                    "use List[Model] instead", "return exactly one function call"
                )
            raise ResponseParsingError(msg) from e

    parse_genai_tools_retryable._arlem_retryable_patch = True
    genai_handlers.parse_genai_tools = parse_genai_tools_retryable


# ── bounded transient-error retry (issue #54 / review F5) ──────────────
#
# instructor 1.15.4 retries only parse errors (ValidationError / JSONDecodeError /
# ResponseParsingError) and re-raises everything else on attempt 1, dropping the
# blind transient-retry that 1.14.4's unfiltered tenacity loop provided. google-genai
# has no transport-level retry of its own, so a self-healing 503 "high demand" burst
# (seen live on gemini-2.5-flash-lite in the July sweep) would otherwise kill a cell
# that a re-send would have completed. We wrap create() in a bounded, transient-ONLY
# retry, kept distinct from instructor's parse-retry loop.
_TRANSIENT_ATTEMPTS = 3          # 1 initial try + 2 retries
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
# Provider-SDK exception class names that denote a transient transport condition
# with no HTTP status to inspect, plus 5xx/429 name backups. Matched by name so we
# don't hard-import anthropic/openai/google-genai/httpx just to build a tuple.
# NOTE: google's ``ClientError`` is intentionally excluded — it also covers 4xx
# (e.g. 400 bad request), which must not be retried; google 429/5xx is caught by the
# status-code check below via its int ``code`` attribute.
_TRANSIENT_EXC_NAMES = {
    "APIConnectionError", "APITimeoutError",
    "ConnectError", "ConnectTimeout", "ReadTimeout", "PoolTimeout",
    "RemoteProtocolError", "TransportError", "TimeoutException",
    "InternalServerError", "ServiceUnavailableError", "ServerError",
    "RateLimitError",
}


def _is_transient_api_error(exc: BaseException) -> bool:
    """True for retryable transport / 5xx / 429 errors from any provider SDK.

    Deliberately does NOT match instructor's ``IncompleteOutputException``
    (``max_tokens`` truncation, review F9): re-sending the same request just
    re-truncates against the same cap, so that stays a hard fail documented in
    ``threats_to_validity.md`` §7. Parse errors are handled by instructor's own
    retry loop and never reach here.
    """
    # anthropic / openai expose `status_code`; google-genai errors expose `code`.
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and val in _TRANSIENT_STATUS:
            return True
    return type(exc).__name__ in _TRANSIENT_EXC_NAMES


def _transient_retryer():
    """A bounded ``tenacity.Retrying`` that retries transient API errors only.

    Read ``.statistics['attempt_number']`` after the call to recover how many
    transient re-sends happened (attempts - 1). ``reraise=True`` surfaces the
    underlying provider error (not tenacity's ``RetryError``) once attempts are
    exhausted, so the caller's error handling and fail-mode classification are
    unchanged.
    """
    import tenacity

    return tenacity.Retrying(
        retry=tenacity.retry_if_exception(_is_transient_api_error),
        stop=tenacity.stop_after_attempt(_TRANSIENT_ATTEMPTS),
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )


def decode_mode_for(model_config: ModelConfig, spec_type: SpecType):
    """Return the instructor decode Mode used for this provider × spec.

    Single source of truth for the enforcement path, so the client factory and the
    metrics record agree.

    Every spec (JSON_LAB, ARLEM, ARLEM_SIMPLIFIED) now uses a unified,
    provider-agnostic schema (plain smart-union + const->enum via
    ConstToEnumSchemaMixin), so all providers run the same *free-decode* path
    (generate + validate/retry): OpenAI ``TOOLS``, Anthropic ``ANTHROPIC_TOOLS``,
    and Gemini ``GENAI_TOOLS``. The ARLEM Gemini twins and the constrained
    ``GENAI_STRUCTURED_OUTPUTS`` path were retired once the ARLEM schemas accepted
    the same const->enum treatment as json_lab (see issue #29). ``spec_type`` no
    longer changes the mode; it is kept for interface stability.

    instructor 1.15 deprecates the provider-prefixed modes (normalized internally
    to ``Mode.TOOLS``; removal in v3.0, DeprecationWarning until then). We keep
    them deliberately: metrics records store ``decode_mode`` as the enum value
    ("anthropic_tools"/"genai_tools"), and switching to TOOLS would fork that
    column mid-matrix. Revisit when the instructor pin moves past 2.x (#44).
    """
    import instructor

    if model_config.provider == Provider.ANTHROPIC:
        return instructor.Mode.ANTHROPIC_TOOLS
    if model_config.provider == Provider.GOOGLE:
        return instructor.Mode.GENAI_TOOLS
    return instructor.Mode.TOOLS  # OpenAI (from_provider default)


def create_instructor_client(model_config: ModelConfig, spec_type: SpecType):
    """
    Create an instructor-patched client for the given provider
    using ``instructor.from_provider("provider/model")``.

    Every spec rides the free-decode path (see ``decode_mode_for``): Gemini uses
    GENAI_TOOLS with the unified, provider-agnostic schema for JSON_LAB and ARLEM
    alike.
    """
    import instructor

    # Apply the Gemini parse-retry patch lazily, here, rather than at import time.
    # It reaches into instructor's private ``v2`` genai handler module, which only
    # exists on the pinned ``instructor==1.15.4`` (see requirements.txt / issue
    # #44). Guarding the import keeps a different instructor version — and cheap
    # entry points like ``--list-models``, which never build a client — from
    # crashing with a raw ModuleNotFoundError, and keeps the ~1.6s instructor+genai
    # import chain out of every invocation. The patch is idempotent, so calling it
    # per-run is free.
    try:
        _patch_genai_parallel_call_retry()
    except ImportError:
        global _GENAI_PATCH_WARNED
        if not _GENAI_PATCH_WARNED:
            print(
                "  WARNING: Gemini parallel-call retry patch not applied — it "
                "targets instructor==1.15.4's private v2 genai handlers (issue "
                "#44); Gemini single-functionCall parse failures will not be "
                "retried on this instructor version.",
                file=sys.stderr,
            )
            _GENAI_PATCH_WARNED = True

    provider_prefix = {
        Provider.OPENAI: "openai",
        Provider.ANTHROPIC: "anthropic",
        Provider.GOOGLE: "google",
    }
    prefix = provider_prefix.get(model_config.provider)
    if prefix is None:
        raise ValueError(f"Unknown provider: {model_config.provider}")

    # Anthropic: build the client ourselves with an explicit timeout. A non-default
    # timeout makes client.timeout != DEFAULT_TIMEOUT, which disables the SDK's
    # non-streaming guard (it otherwise raises "Streaming is required..." once
    # max_tokens > ~21,333). This lets us request full labs (~22.5K tokens) without
    # switching to streaming, leaving instructor's I/O, hooks, and tracking unchanged.
    if model_config.provider == Provider.ANTHROPIC:
        import anthropic
        import httpx

        client = anthropic.Anthropic(  # reads ANTHROPIC_API_KEY from env (load_dotenv'd)
            timeout=httpx.Timeout(900.0, connect=5.0),
        )
        return instructor.from_anthropic(
            client,
            model=model_config.model_id,           # baked into create() like from_provider does
            mode=instructor.Mode.ANTHROPIC_TOOLS,  # explicit; matches from_provider default
            max_tokens=32000,                      # default; per-request value still overrides
        )

    kwargs = {}
    # from_provider expects GOOGLE_API_KEY but our .env uses GEMINI_API_KEY.
    # All specs use free-decode GENAI_TOOLS on the unified schema (see decode_mode_for).
    if model_config.provider == Provider.GOOGLE:
        kwargs["api_key"] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        kwargs["mode"] = decode_mode_for(model_config, spec_type)

    return instructor.from_provider(
        f"{prefix}/{model_config.model_id}",
        **kwargs,
    )


def provider_create_kwargs(model_config: ModelConfig, response_model) -> dict:
    """Provider-specific ``create()`` kwargs every entry point must pass.

    Shared by ``run_single_benchmark`` and the hierarchical pipeline
    (``Code/Hierarchical/``) so no entry point can drop the Anthropic guard and
    reintroduce the issue #44 retry-death.
    """
    kwargs: dict = {}
    # Anthropic requires an explicit max_tokens. 8192 truncated full L3-L4 labs
    # (IncompleteOutputException); a rich lab runs ~18-22K completion tokens, so we
    # cap at 32000 (~45% head-room). Exceeding the SDK's ~21,333 non-streaming
    # threshold is allowed here because create_instructor_client() builds the
    # Anthropic client with an explicit timeout, which disables that guard.
    if model_config.provider == Provider.ANTHROPIC:
        kwargs["max_tokens"] = 32000
        # Forbid parallel tool calls: under instructor a parallel call always
        # fails parsing, and its reask replays the assistant turn with only one
        # tool_result, which the API 400s (issue #44 — Haiku 4.5 retry-death;
        # unfixed upstream as of 1.15.4). Requires instructor>=1.15: earlier
        # versions overwrite a caller-supplied tool_choice. The name must equal
        # the tool instructor registers, which is model_json_schema()["title"];
        # that equals __name__ only while no response model sets a custom title
        # (ConfigDict(title=...)). None of our response models do, so this
        # holds today — but a future custom title would 400 every Anthropic run
        # (latent; review F13).
        kwargs["tool_choice"] = {
            "type": "tool",
            "name": response_model.__name__,
            "disable_parallel_tool_use": True,
        }
    return kwargs


# ── Model Selection ──────────────────────────────────────────────────

def get_response_model(spec_type: SpecType):
    """
    Return the Pydantic response model class for the spec.

    Every spec now uses one provider-agnostic schema (const->enum mixin; no
    field-level discriminated unions), so the same model goes to every provider —
    the Gemini twins (``json_lab_gemini.py``, ``arlem_*_gemini.py``) were retired.
    """
    if spec_type == SpecType.JSON_LAB:
        from json_lab import Lab
        return Lab
    if spec_type == SpecType.ARLEM:
        from arlem_full import ARLEMScenario
        return ARLEMScenario
    # ARLEM_SIMPLIFIED
    from arlem_simplified import ARLEMScenario
    return ARLEMScenario


# ── Run Provenance ───────────────────────────────────────────────────
#
# Environment fingerprint stamped into every metrics record. The formative
# 2026-06/07 dataset mixed three validator eras and two retry regimes that are
# distinguishable only by run timestamp against remembered cutoff dates;
# stamping the repo SHA and library versions makes the era an explicit,
# queryable column instead.

_PROVENANCE_PACKAGES = ("instructor", "anthropic", "openai", "google-genai", "pydantic")
_PROVENANCE_CACHE: Optional[dict] = None


def get_provenance() -> dict:
    """Repo + library fingerprint for metrics records; computed once per process."""
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            # Untracked files excluded: dirty means the *tracked* code the run
            # executed differs from what git_sha says it was.
            dirty = bool(subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip())
        except Exception:
            sha, dirty = None, None
        versions: dict[str, Optional[str]] = {}
        for pkg in _PROVENANCE_PACKAGES:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = None
        _PROVENANCE_CACHE = {
            "git_sha": sha,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "packages": versions,
        }
    return _PROVENANCE_CACHE


# ── Single Run ───────────────────────────────────────────────────────

def run_single_benchmark(
    model_config: ModelConfig,
    run_config: BenchmarkRunConfig,
    *,
    save_prompts: bool = True,
    overwrite_prompts: bool = False,
) -> dict:
    """
    Execute a single benchmark run: one model × one topic × one spec.

    Returns a dict with all metrics, the generated output, and lab analysis.
    """
    # Build prompt + response model. YAML-driven path takes precedence.
    prompt_file_rel: Optional[str] = None
    if run_config.lab_name and run_config.level:
        labs = discover_labs()
        if run_config.lab_name not in labs:
            raise ValueError(
                f"Unknown lab '{run_config.lab_name}'. "
                f"Available: {sorted(labs.keys())}"
            )
        lab = LabDescription.from_yaml(labs[run_config.lab_name])
        user_prompt, response_model = build_prompt(
            lab,
            run_config.level,
            run_config.spec_type.value,
            structure=run_config.structure,
            min_objects=run_config.min_objects,
            min_clips=run_config.min_clips,
            min_things=run_config.min_things,
            min_places=run_config.min_places,
            min_actions=run_config.min_actions,
        )
        if save_prompts:
            prompt_file_rel = _save_prompt_artifact(
                run_config, user_prompt, overwrite=overwrite_prompts
            )
    else:
        response_model = get_response_model(run_config.spec_type)
        user_prompt = run_config.get_prompt()

    # Create instructor client (model is baked into the client via from_provider).
    # decode_mode is the enforcement path actually used — recorded below and used
    # to gate Gemini's response-schema token recovery.
    decode_mode = decode_mode_for(model_config, run_config.spec_type)
    client = create_instructor_client(model_config, run_config.spec_type)

    # Set up tracking
    tracker = InstructorTracker(
        model=model_config.model_id,
        provider=model_config.provider.value,
        pricing_config=model_config.pricing_config,
    )
    tracked_client = tracker.wrap(client)

    # Shared base name for output / metrics / errors artifacts. Include the
    # specificity level (L1-L4) on the YAML-driven path so runs that differ only
    # by level are distinguishable by filename, not just timestamp.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_config.model_id.replace("/", "_").replace(".", "-")
    name_parts = [safe_model, run_config.spec_type.value]
    if run_config.level is not None:
        name_parts.append(run_config.level.value)
    name_parts.append(ts)
    base_name = "_".join(name_parts)

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Model is set on the client; create() just needs response_model + messages
    create_kwargs = dict(
        response_model=response_model,
        messages=messages,
        max_retries=run_config.max_retries,
    )

    create_kwargs.update(provider_create_kwargs(model_config, response_model))

    # Execute generation
    print(f"\n  Generating with {model_config.display_name}...")
    if run_config.lab_name and run_config.level:
        print(
            f"  Spec: {run_config.spec_type.value} | Lab: {run_config.lab_name} "
            f"| Level: {run_config.level.value} | Structure: {run_config.structure.value}"
        )
    else:
        print(f"  Spec: {run_config.spec_type.value} | Topic: {run_config.topic[:50]}...")

    start_time = time.time()
    result = None
    generation_error = None
    transient_retries = 0

    # Bounded retry on transient API errors only (429/5xx/connection), separate
    # from instructor's parse-retry loop; see _is_transient_api_error (issue #54).
    # Read the count in `finally` so it is captured on both success and failure.
    retryer = _transient_retryer()
    try:
        result = retryer(tracked_client.create, **create_kwargs)
    except Exception as e:
        generation_error = str(e)
        print(f"  ERROR: {type(e).__name__}: {generation_error[:200]}")
    finally:
        transient_retries = retryer.statistics.get("attempt_number", 1) - 1
    if transient_retries:
        print(f"  (recovered after {transient_retries} transient-error retr"
              f"{'y' if transient_retries == 1 else 'ies'})"
              if result is not None else
              f"  (failed after {transient_retries} transient-error retries)")

    wall_time_s = time.time() - start_time

    # Analyze the generated lab
    lab_metrics = {}
    output_json = None
    if result is not None:
        try:
            output_json = result.model_dump(mode="json", exclude_none=True)
            if run_config.spec_type == SpecType.JSON_LAB:
                lab_metrics = analyze_json_lab(output_json)
            else:
                lab_metrics = analyze_arlem(output_json)
        except Exception as e:
            lab_metrics = {"analysis_error": str(e)}

    # Collect tracking metrics
    tracker_summary = tracker.summary()

    # Save any retry / terminal errors next to the metrics file
    errors_file_rel: Optional[str] = None
    if run_config.save_output:
        errors_file_rel = _save_errors_file(tracker, base_name, generation_error)

    # Build result record
    record = {
        "timestamp": datetime.now().isoformat(),
        "model": model_config.model_id,
        "provider": model_config.provider.value,
        "display_name": model_config.display_name,
        "spec_type": run_config.spec_type.value,
        "decode_mode": decode_mode.value,  # instructor enforcement path actually used
        "topic": run_config.topic,
        "lab_name": run_config.lab_name,
        "level": run_config.level.value if run_config.level else None,
        # `structure` only shapes json_lab output; it is meaningless for ARLEM
        # (single ARLEMScenario shape), so record None there to avoid misleading reports.
        "structure": (
            run_config.structure.value
            if (run_config.lab_name and run_config.level
                and run_config.spec_type == SpecType.JSON_LAB)
            else None
        ),
        "prompt_file": prompt_file_rel,
        "errors_file": errors_file_rel,
        "success": result is not None,
        "error": generation_error,
        # Transient API-error re-sends (429/5xx/connection), tracked separately from
        # tracking.retries (instructor parse-error self-correction) so infra noise
        # does not inflate the self-correction signal (issue #54).
        "transient_retries": transient_retries,
        "wall_time_seconds": round(wall_time_s, 2),
        "provenance": get_provenance(),
        "tracking": tracker_summary,
        "lab_metrics": lab_metrics,
    }

    # Save generated output
    if run_config.save_output and output_json is not None:
        output_path = _save_output(run_config, output_json, record, base_name)
        record["output_path"] = str(output_path)

    # Print summary
    _print_run_summary(record)

    return record


# ── Output Saving ────────────────────────────────────────────────────

def _save_prompt_artifact(
    run_config: BenchmarkRunConfig,
    user_prompt: str,
    *,
    overwrite: bool = False,
) -> str:
    """Write the assembled prompt to Artifacts/Data/Benchmark/prompts/, deduped.

    The same prompt is shared across all models for a given
    (lab, level, spec, structure) tuple, so we write it once and let
    every run's metrics record reference the same file by relative path.
    ``structure`` only applies to json_lab L3-L4; ARLEM and L1 omit it.

    Returns the prompt file path relative to PROJECT_ROOT.
    """
    prompts_dir = PROJECT_ROOT / run_config.output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    parts = [run_config.lab_name, run_config.level.value, run_config.spec_type.value]
    if run_config.spec_type == SpecType.JSON_LAB and run_config.level != Level.L1:
        parts.append(run_config.structure.value)
    fname = "_".join(parts) + ".txt"

    path = prompts_dir / fname
    if overwrite or not path.exists():
        path.write_text(user_prompt, encoding="utf-8")
    return str(path.relative_to(PROJECT_ROOT))


def _save_output(
    run_config: BenchmarkRunConfig,
    output_json: dict,
    record: dict,
    base_name: str,
) -> Path:
    """Save generated JSON and metrics to the benchmark output directory.

    Outputs and metrics live in sibling ``Outputs/`` and ``Metrics/``
    subfolders of the benchmark dir; ``suite_results_*.json`` and ``prompts/``
    stay at the root.
    """
    base_dir = PROJECT_ROOT / run_config.output_dir
    outputs_dir = base_dir / "Outputs"
    metrics_dir = base_dir / "Metrics"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save the generated lab JSON
    lab_path = outputs_dir / f"{base_name}_output.json"
    lab_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")

    # Save the metrics record
    metrics_path = metrics_dir / f"{base_name}_metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print(f"  Saved: {lab_path.relative_to(PROJECT_ROOT)}")
    return lab_path


def _save_errors_file(
    tracker: InstructorTracker,
    base_name: str,
    generation_error: Optional[str],
) -> Optional[str]:
    """Write all per-attempt validation/API errors for the most recent call to a
    sibling artifact under ``Artifacts/Data/Errors/{base_name}_errors.txt``.

    Multiple errors from the same run are appended to one file with a banner
    delimiter so it's easy to see where one error ends and the next begins.
    Returns the relative path (POSIX-style) or ``None`` if nothing was written.
    """
    call = tracker.get_last_call_metrics()
    attempt_errors = list(call.errors) if call else []

    if not attempt_errors and not generation_error:
        return None

    errors_dir = PROJECT_ROOT / "Artifacts" / "Data" / "Errors"
    errors_dir.mkdir(parents=True, exist_ok=True)
    path = errors_dir / f"{base_name}_errors.txt"

    sections: list[str] = []
    for err in attempt_errors:
        banner = (
            f"========== Attempt {err.attempt_number} — "
            f"{err.exception_class} @ {err.timestamp.isoformat()} =========="
        )
        sections.append(f"{banner}\n{err.message}")

    if generation_error:
        banner = (
            f"========== Terminal failure @ {datetime.now().isoformat()} =========="
        )
        sections.append(f"{banner}\n{generation_error}")

    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix()


def _print_run_summary(record: dict):
    """Print a concise summary of a benchmark run."""
    tracking = record["tracking"]
    tokens = tracking["tokens"]
    cost = tracking["cost"]
    retries = tracking["retries"]
    lab = record.get("lab_metrics", {})

    status = "OK" if record["success"] else "FAILED"
    print(f"\n  [{status}] {record['display_name']} — {record['spec_type']}")
    print(f"  Tokens: {tokens['prompt']:,} in / {tokens['completion']:,} out / {tokens['total']:,} total")
    print(f"  Cost: {cost['formatted']} | Wall time: {record['wall_time_seconds']}s")
    print(f"  Retries: {retries['total']} (parse errors: {tracking['errors']['parse_errors']})")
    if record.get("errors_file"):
        print(f"  Errors saved: {record['errors_file']}")

    if lab:
        # Print key lab metrics
        metric_parts = []
        for key in ["num_objects", "num_clips", "num_components", "num_things",
                     "num_places", "num_actions", "num_predicates"]:
            if key in lab:
                label = key.replace("num_", "").replace("_", " ")
                metric_parts.append(f"{label}: {lab[key]}")
        if metric_parts:
            print(f"  Lab: {' | '.join(metric_parts)}")


# ── Suite Runner ─────────────────────────────────────────────────────

def run_benchmark_suite(
    model_ids: list[str],
    *,
    topics: Optional[list[str]] = None,
    lab_names: Optional[list[str]] = None,
    levels: Optional[list[Level]] = None,
    spec_types: Optional[list[SpecType]] = None,
    max_retries: int = 3,
    save_output: bool = True,
    structure: Structure = Structure.MULTI_MODULE,
    save_prompts: bool = True,
    overwrite_prompts: bool = False,
) -> list[dict]:
    """
    Run benchmarks across multiple models and work items.

    A work item is one (lab_name, level, topic, spec_type) tuple. The YAML-driven
    path (lab_names × levels cross-product) takes precedence; otherwise each topic
    string becomes a work item for the legacy free-form path. Every work item is
    produced for each requested spec_type, so a single sweep can cover json_lab,
    arlem, and arlem_simple together.

    Returns list of all result records.
    """
    spec_types = spec_types or [SpecType.JSON_LAB]

    # Build the ordered work-item list: (lab_name, level, topic, spec_type)
    work_items: list[tuple[Optional[str], Optional[Level], Optional[str], SpecType]] = []
    yaml_path = bool(lab_names and levels)
    for spec_type in spec_types:
        if yaml_path:
            for lab_name in lab_names:
                for level in levels:
                    work_items.append((lab_name, level, None, spec_type))
        else:
            for topic in (topics or [DEFAULT_TOPICS[0]]):
                work_items.append((None, None, topic, spec_type))

    results = []
    total_runs = len(model_ids) * len(work_items)
    run_num = 0

    specs_label = ", ".join(s.value for s in spec_types)
    print(f"\n{'=' * 70}")
    if yaml_path:
        print(
            f"BENCHMARK SUITE: {len(model_ids)} models × {len(lab_names)} labs "
            f"× {len(levels)} levels × {len(spec_types)} specs = {total_runs} runs"
        )
        print(f"Labs: {', '.join(lab_names)}")
        print(f"Levels: {', '.join(lvl.value for lvl in levels)}")
        print(f"Structure: {structure.value}")
    else:
        print(
            f"BENCHMARK SUITE: {len(model_ids)} models × {len(topics or [DEFAULT_TOPICS[0]])} "
            f"topics × {len(spec_types)} specs = {total_runs} runs"
        )
    print(f"Specs: {specs_label}")
    print(f"{'=' * 70}")

    for lab_name, level, topic, spec_type in work_items:
        for model_id in model_ids:
            run_num += 1
            print(f"\n--- Run {run_num}/{total_runs} ---")

            if model_id not in MODELS:
                print(f"  SKIPPED: Unknown model '{model_id}'")
                continue

            model_config = MODELS[model_id]
            rc_kwargs = dict(
                spec_type=spec_type,
                lab_name=lab_name,
                level=level,
                structure=structure,
                max_retries=max_retries,
                save_output=save_output,
            )
            # Only override the dataclass default topic on the legacy path,
            # so YAML runs keep BenchmarkRunConfig's default topic string.
            if topic is not None:
                rc_kwargs["topic"] = topic
            run_config = BenchmarkRunConfig(**rc_kwargs)

            try:
                record = run_single_benchmark(
                    model_config,
                    run_config,
                    save_prompts=save_prompts,
                    overwrite_prompts=overwrite_prompts,
                )
                results.append(record)
            except Exception as e:
                print(f"  FATAL ERROR: {type(e).__name__}: {e}")
                results.append({
                    "model": model_id,
                    "spec_type": spec_type.value,
                    "lab_name": lab_name,
                    "level": level.value if level else None,
                    "topic": topic,
                    "success": False,
                    "error": f"Fatal: {e}",
                })

    # Print comparison summary
    _print_suite_summary(results)

    # Save combined results
    if save_output:
        output_dir = PROJECT_ROOT / "Artifacts" / "Data" / "Benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = output_dir / f"suite_results_{ts}.json"
        summary_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nSuite results saved: {summary_path.relative_to(PROJECT_ROOT)}")

    return results


def _print_suite_summary(results: list[dict]):
    """Print a comparison table of all benchmark runs."""
    print(f"\n{'=' * 70}")
    print("SUITE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'Spec':<12} {'Level':<6} {'Structure':<14} {'Status':<8} {'Tokens':>8} {'Cost':>10} {'Time':>7} {'Retries':>8}")
    print("-" * 106)

    for r in results:
        if "tracking" not in r:
            print(f"{r.get('model', '?'):<25} {(r.get('spec_type') or '-'):<12} {(r.get('level') or '-'):<6} {(r.get('structure') or '-'):<14} {'FATAL':<8}")
            continue

        t = r["tracking"]
        print(
            f"{r['display_name']:<25} "
            f"{(r.get('spec_type') or '-'):<12} "
            f"{(r.get('level') or '-'):<6} "
            f"{(r.get('structure') or '-'):<14} "
            f"{'OK' if r['success'] else 'FAIL':<8} "
            f"{t['tokens']['total']:>8,} "
            f"{t['cost']['formatted']:>10} "
            f"{r['wall_time_seconds']:>6.1f}s "
            f"{t['retries']['total']:>8}"
        )


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM providers on AR lab generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Model ID(s) to benchmark (e.g., gpt-4o-mini claude-haiku-4.5)",
    )
    parser.add_argument(
        "--small-models",
        action="store_true",
        help="Include all SMALL-tier models (combinable with other tiers / --model)",
    )
    parser.add_argument(
        "--medium-models",
        action="store_true",
        help="Include all MEDIUM-tier models",
    )
    parser.add_argument(
        "--large-models",
        action="store_true",
        help="Include all LARGE-tier models",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include every registered model (all three tiers)",
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Topic for the AR lab (default: Solar System)",
    )
    parser.add_argument(
        "--spec", "-s",
        nargs="+",
        choices=["json_lab", "arlem", "arlem_simple"],
        default=["json_lab"],
        help="Specification type(s) to generate; pass 1, 2, or all 3 to sweep formats "
             "(default: json_lab). Composes with lab/level sweeping.",
    )
    parser.add_argument(
        "--suite",
        choices=["quick", "full"],
        default=None,
        help="Run a pre-defined benchmark suite",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max instructor retries per call (default: 3)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output files",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model IDs and exit",
    )
    parser.add_argument(
        "--lab",
        type=str,
        default=None,
        help="topic_name of a lab YAML in Artifacts/Lab Descriptions/",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["L1", "L3", "L4"],
        default=None,
        help="Specificity level for the YAML-driven prompt (requires --lab)",
    )
    parser.add_argument(
        "--all-labs",
        action="store_true",
        help="Iterate over ALL discovered lab YAMLs (excludes the Example/ subfolder)",
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Iterate over all specificity levels (L1, L3, L4; L2 retired)",
    )
    parser.add_argument(
        "--structure",
        type=str,
        choices=["single-module", "multi-module", "module-only"],
        default="multi-module",
        help="Output structure for L3-L4 (default: multi-module; ignored for L1)",
    )
    parser.add_argument(
        "--list-labs",
        action="store_true",
        help="List available lab YAMLs and exit",
    )
    parser.add_argument(
        "--no-save-prompts",
        action="store_true",
        help="Skip writing the prompt artifact alongside benchmark outputs",
    )
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Rewrite the prompt artifact file even if it already exists",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        print(f"  {'model':<25} {'tier':<8} {'provider':<12} name")
        for model_id, config in MODELS.items():
            print(
                f"  {model_id:<25} {config.size.value:<8} "
                f"{config.provider.value:<12} {config.display_name}"
            )
        return

    if args.list_labs:
        labs = discover_labs()
        if not labs:
            print(f"\nNo lab YAMLs found in Artifacts/Lab Descriptions/")
        else:
            print("\nAvailable labs:")
            for topic_name, path in labs.items():
                print(f"  {topic_name:<40} {path.relative_to(PROJECT_ROOT)}")
        return

    # De-dupe specs, preserving CLI order.
    _seen_specs: set[str] = set()
    spec_types = [
        SpecType(s) for s in args.spec
        if not (s in _seen_specs or _seen_specs.add(s))
    ]
    save_output = not args.no_save
    structure = Structure(args.structure)
    save_prompts = not args.no_save_prompts

    # ── Resolve model selection (tier flags ∪ explicit --model) ──────────
    tier_sizes: list[ModelSize] = []
    if args.all_models:
        tier_sizes = [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]
    else:
        if args.small_models:
            tier_sizes.append(ModelSize.SMALL)
        if args.medium_models:
            tier_sizes.append(ModelSize.MEDIUM)
        if args.large_models:
            tier_sizes.append(ModelSize.LARGE)

    resolved_models: list[str] = []
    if tier_sizes:
        resolved_models.extend(models_by_size(*tier_sizes))
    if args.model:
        resolved_models.extend(args.model)
    # De-dupe, preserving order
    seen: set[str] = set()
    resolved_models = [m for m in resolved_models if not (m in seen or seen.add(m))]

    # ── Resolve labs and levels ──────────────────────────────────────────
    if args.all_labs:
        lab_names = sorted(discover_labs().keys())
        if not lab_names:
            print("Error: --all-labs found no lab YAMLs in Artifacts/Lab Descriptions/.")
            sys.exit(1)
    elif args.lab:
        lab_names = [args.lab]
    else:
        lab_names = None

    if args.all_levels:
        levels = [Level.L1, Level.L3, Level.L4]
    elif args.level:
        levels = [Level(args.level)]
    else:
        levels = None

    # ── Validation ───────────────────────────────────────────────────────
    used_tier_flags = (
        args.small_models or args.medium_models or args.large_models or args.all_models
    )
    if args.suite:
        if used_tier_flags or args.model:
            print("Error: --suite cannot be combined with --model or the tier flags.")
            sys.exit(1)
        if args.all_labs or args.all_levels:
            print("Error: --suite cannot be combined with --all-labs or --all-levels "
                  "(use --suite with a single --lab/--level if needed).")
            sys.exit(1)
    else:
        # Lab/level coherence on the non-suite path
        if args.all_labs and not levels:
            print("Error: --all-labs requires --level or --all-levels.")
            sys.exit(1)
        if args.all_levels and not lab_names:
            print("Error: --all-levels requires --lab or --all-labs.")
            sys.exit(1)
        if args.lab and not levels:
            print("Error: --lab requires --level or --all-levels.")
            sys.exit(1)
        if args.level and not lab_names:
            print("Error: --level requires --lab or --all-labs.")
            sys.exit(1)
        if not resolved_models:
            print("Error: Specify --model, a tier flag (--small/medium/large/all-models), "
                  "or --suite. Use --list-models or --list-labs to see options.")
            sys.exit(1)

    # ── Dispatch ─────────────────────────────────────────────────────────
    if args.suite:
        model_ids = (
            QUICK_BENCHMARK_MODELS if args.suite == "quick"
            else FULL_BENCHMARK_MODELS
        )
    else:
        model_ids = resolved_models

    common = dict(
        spec_types=spec_types,
        max_retries=args.max_retries,
        save_output=save_output,
        structure=structure,
        save_prompts=save_prompts,
        overwrite_prompts=args.overwrite_prompts,
    )

    if lab_names and levels:
        run_benchmark_suite(
            model_ids=model_ids,
            lab_names=lab_names,
            levels=levels,
            **common,
        )
    else:
        if args.suite:
            topics = [args.topic] if args.topic else DEFAULT_TOPICS
        else:
            topics = [args.topic or DEFAULT_TOPICS[0]]
        run_benchmark_suite(
            model_ids=model_ids,
            topics=topics,
            **common,
        )


if __name__ == "__main__":
    main()

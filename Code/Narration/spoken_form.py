"""
Spoken-form pass (#94): the text each clip sends to the voice service.

``Clip.narration`` is written to be read: it has symbols, numbers, formulas and
acronyms (``5'→3'``, ``109.5°``, ``XeF4``, ``vx = v cosθ``). This pass writes the
same narration the way it should be said aloud. The display text is never
changed; only what is spoken is.

Modes (``narrate_lab.py --spoken-form``):

- ``none``: the narration as written.
- ``rules``: deterministic rules and a layered glossary, no LLM (``spoken_rules.py``).
  Notation the rules can't decide is left as written and flagged.
- ``llm``: one call per lab to a cheap model (``MODELS``), through the same
  guarded client path as every LLM entry point (``create_instructor_client`` +
  ``provider_create_kwargs``, tracked by ``tracking/``). Every narrated clip goes
  in one prompt, tagged with its audio name, and comes back keyed by that name.

The model is told to expand only symbols, numbers, units, formulas and acronyms.
``check_spoken()`` then compares each clip's spoken text with its narration, word
by word, in code. Anything beyond that (a changed, added, dropped or reordered
ordinary word) is **flagged, not rejected**: the run carries on with the model's
text, and a reviewer can edit the cached text or rerun later. A clip the model
didn't return falls back to its narration as written, also flagged.

Results are cached in ``spoken_forms.json`` in the run folder, written straight
after the call. A rerun reuses a clip's cached spoken text while its narration,
mode, model and prompt version are unchanged, so the LLM's run-to-run wording
differences never re-bill audio. A reviewer can edit ``spoken_text`` in that
file and rerun: the edit is kept (marked ``edited``) and only that clip's audio
is regenerated.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

MODES = ("none", "llm", "rules")
# Benchmark registry keys (benchmark_config.MODELS) allowed for the LLM mode.
MODELS = ("claude-haiku-5.5", "gpt-6-luna", "gemini-3.5-flash-lite")
DEFAULT_MODEL = "claude-haiku-5.5"   # provisional; the #94 measurement picks the default
PROMPT_VERSION = 1                   # bump when the prompt changes, so caches refresh
CACHE_FILE = "spoken_forms.json"
MAX_RETRIES = 2                      # instructor reasks for a malformed reply (schema only)
# Expanding symbols needs no deliberation, so the call asks for "low" reasoning instead
# of the roster's "medium" (#72). Every provider accepts "low" (Anthropic effort, OpenAI
# reasoning.effort, Gemini thinking_level).
REASONING_EFFORT = "low"


# ── Response model ────────────────────────────────────────────────────


class ClipSpokenForm(BaseModel):
    audio_name: str = Field(description="The clip's tag from the prompt, copied exactly")
    spoken_text: str = Field(
        description="The clip's narration as it should be said aloud: symbols, numbers, "
                    "units, formulas and acronyms written out; every other word unchanged")


class LabSpokenForms(BaseModel):
    clips: list[ClipSpokenForm] = Field(description="One entry per clip in the prompt, in order")


# ── Prompt ────────────────────────────────────────────────────────────

_INSTRUCTIONS = """\
You prepare narration for a text-to-speech voice in an educational AR lab.
Rewrite each clip's narration exactly as a teacher would say it aloud, changing
ONLY what a voice could misread:

- symbols and arrows: "5'->3'" or "5'→3'" -> "five prime to three prime";
  "=" -> "equals"; "/" between words -> "or" / "and" as the sentence means it
- numbers with units or symbols: "109.5°" -> "one hundred nine point five degrees";
  "1.52 AU" -> "one point five two A U"; "150-200" -> "one hundred fifty to two hundred"
- chemical formulas as letters and numbers: "XeF4" -> "X e F four", "CH₄" -> "C H four",
  "H2O" -> "H two O"
- acronyms read letter by letter as "D N A", "S S Bs"; acronyms that are said as a
  word may be written that way ("VSEPR" -> "vesper")
- math: "vx = v cosθ" -> "v x equals v cos theta"
- base or letter sequences: "TACGGTCA" or "T-A-C-G" -> "T, A, C, G, G, T, C, A"
- words in capitals only for emphasis ("press NEXT") -> normal case

Do not change, add, remove or reorder any other word. Keep sentences and
punctuation. If a clip needs no changes, return it exactly as given.
Return every clip once, with its tag copied exactly."""


def build_prompt(lab: dict, topic: str | None, clips: list[tuple[str, str]]) -> str:
    """The user message: the lab's context, then each clip as ``[tag] text``."""
    context = [f"Lab: {lab.get('labId') or ''}"]
    if lab.get("courseName"):
        context.append(f"Course: {lab['courseName']}")
    if topic:
        context.append(f"Topic: {topic}")
    body = "\n\n".join(f"[{name}]\n{text}" for name, text in clips)
    return f"{_INSTRUCTIONS}\n\n" + "\n".join(context) + f"\n\nClips:\n\n{body}"


# ── The word check (flags, never rejects) ─────────────────────────────

_WORD = re.compile(r"^[A-Za-z]+(?:['’][A-Za-z]+)?$")
# Punctuation around a token. Symbols (5', θ, 45°, +) stay: they are what may change.
_EDGE = re.compile(r"^[.,;:!?()\[\]{}\"“”‘’—–…*-]+|[.,;:!?()\[\]{}\"“”‘’—–…*-]+$")


def _tokens(text: str) -> list[str]:
    out = []
    for raw in text.split():
        tok = _EDGE.sub("", raw)
        if tok:
            out.append(tok)
    return out


def is_ordinary(token: str) -> bool:
    """A plain word that should reach the voice unchanged: letters only, and not an
    all-caps acronym or emphasis word (``DNA``, ``SSBs``, ``DNA's``, ``NEXT`` may
    change). Capitalized words like ``It`` or ``In`` are ordinary."""
    if not _WORD.match(token):
        return False
    core = token.replace("'", "").replace("’", "")
    if len(core) > 2 and core.endswith("s"):   # plural or possessive: SSBs, DNA's
        core = core[:-1]
    return not (len(core) >= 2 and core.isupper())


def _subsequence(needles: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needles)


def check_spoken(original: str, spoken: str) -> tuple[list[str], list[dict]]:
    """(flags, changes) for one clip.

    ``changes``: every place the spoken text differs, as ``{"original", "spoken"}``.
    ``flags``: changes that go beyond the allowed kind: an ordinary word changed,
    dropped, reordered, or a word added where nothing was replaced.
    """
    a, b = _tokens(original), _tokens(spoken)
    fa, fb = [t.casefold() for t in a], [t.casefold() for t in b]
    flags, changes = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=fa, b=fb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        changes.append({"original": before, "spoken": after})
        # A short letters-only token spelled out nearby ("vx" -> "v x") is a math
        # variable or an acronym being read, not a reworded word.
        near = " " + " ".join(fb[max(0, j1 - 4): j2 + 8]) + " "
        ordinary = [fa[i] for i in range(i1, i2) if is_ordinary(a[i])
                    and not (2 <= len(a[i]) <= 4 and f" {' '.join(fa[i])} " in near)]
        if op == "insert":
            flags.append(f"added words: '{after}'")
        elif not _subsequence(ordinary, fb[j1:j2]):
            lost = [w for w in ordinary if w not in fb[j1:j2]]
            flags.append(f"changed ordinary words {lost or ordinary}: '{before}' -> '{after or '(removed)'}'")
    if not spoken.strip():
        flags.append("empty spoken text")
    return flags, changes


# ── Cache ─────────────────────────────────────────────────────────────


@dataclass
class SpokenResult:
    spoken_text: str
    source: str                 # none | llm | cache | edited | fallback
    flags: list[str] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)


def _cache_key(mode: str, model: str | None) -> dict:
    return {"mode": mode, "model": model, "prompt_version": PROMPT_VERSION}


def load_cache(run_dir: Path) -> dict:
    path = Path(run_dir) / CACHE_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("clips", {})
    except (OSError, ValueError):
        return {}


def save_cache(run_dir: Path, entries: dict, write_json) -> None:
    write_json(Path(run_dir) / CACHE_FILE, {
        "note": "Spoken text per audio name. Edit spoken_text and rerun narrate_lab.py "
                "--resume to regenerate just that clip; the edit is kept.",
        "prompt_version": PROMPT_VERSION,
        "clips": entries,
    })


# ── The pass ──────────────────────────────────────────────────────────


@dataclass
class PassReport:
    mode: str
    model: str | None = None              # registry key
    model_id: str | None = None
    served_models: list[str] = field(default_factory=list)
    calls: int = 0
    clips_sent: int = 0
    clips_cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    transient_retries: int = 0
    retries: int = 0
    reasoning_effort: str | None = None
    error: str | None = None
    # One record per request sent: first attempt, instructor reasks, transient re-sends.
    attempts: list[dict] = field(default_factory=list)
    # Rules mode: the merged glossary's hash and layers, and the terms it couldn't decide.
    glossary_version: str | None = None
    glossary_files: list[str] = field(default_factory=list)
    unresolved: dict[str, int] = field(default_factory=dict)


def run_pass(mode: str, model_key: str | None, lab: dict, topic: str | None,
             clips: list[tuple[str, str]], run_dir: Path, write_json,
             call_fn=None, log=print, glossary_files: list | None = None,
             ) -> tuple[dict[str, SpokenResult], PassReport]:
    """Spoken text for each ``(audio_name, display_text)``.

    ``call_fn(prompt) -> (LabSpokenForms, usage_dict)`` replaces the real LLM call in
    tests. A failed LLM call (after transient retries) falls back to the narration
    as written for every clip it covered, flagged, and is reported in the pass
    report; the run continues.
    """
    report = PassReport(mode=mode)
    if mode == "none":
        return {n: SpokenResult(t.strip(), "none") for n, t in clips}, report
    if mode == "rules":
        return _rules_pass(lab, topic, clips, run_dir, glossary_files, report, log)

    report.model = model_key
    key = _cache_key(mode, model_key)
    cache = load_cache(run_dir)
    results: dict[str, SpokenResult] = {}
    todo: list[tuple[str, str]] = []
    for name, text in clips:
        hit = cache.get(name)
        edited = bool(hit) and hit.get("spoken_text") != hit.get("llm_text")
        # A fallback (failed call, missing clip, empty reply) is not the model's answer:
        # send that clip again, unless a reviewer has written its text by hand.
        usable = hit and (hit.get("source") != "fallback" or edited)
        if usable and hit.get("display_text") == text and all(hit.get(k) == v for k, v in key.items()):
            flags, changes = check_spoken(text, hit["spoken_text"])
            results[name] = SpokenResult(hit["spoken_text"], "edited" if edited else "cache",
                                         flags, changes)
            report.clips_cached += 1
        else:
            todo.append((name, text))

    if todo:
        report.clips_sent = len(todo)
        prompt = build_prompt(lab, topic, todo)
        report.calls += 1
        try:
            forms, usage = (call_fn or _make_llm_call(model_key, report))(prompt)
            _add_usage(report, usage)
            report.attempts += (usage["attempts"] if "attempts" in usage
                                else [_one_attempt(usage, "ok", None)])
        except Exception as exc:  # noqa: BLE001 - the run continues with the narration as written
            usage = getattr(exc, "spoken_form_usage", None) or {}
            _add_usage(report, usage)
            # Real calls list their attempts (none if nothing was sent); a caller
            # that reported nothing counts as one failed request of unknown cost.
            report.attempts += (usage["attempts"] if "attempts" in usage else [
                _one_attempt(usage, "error", exc, cost_known="cost_usd" in usage)])
            report.error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
            log(f"  spoken form: the {model_key} call failed ({report.error}); "
                "using the narration as written for these clips")
            forms = LabSpokenForms(clips=[])

        returned: dict[str, str] = {}
        extra, normalized = [], 0
        sent = {name.casefold(): name for name, _ in todo}
        for c in forms.clips:
            name = _match_tag(c.audio_name, sent)
            if name is None:
                extra.append(f"{c.audio_name!r} (not in the prompt)")
            elif name in returned:
                extra.append(f"{c.audio_name!r} (returned twice; first kept)")
            else:
                normalized += name != c.audio_name
                returned[name] = c.spoken_text
        if normalized:
            log(f"  spoken form: matched {normalized} tag(s) after trimming brackets, spaces or case")
        for name, text in todo:
            spoken = returned.get(name, "").strip()
            if spoken:
                flags, changes = check_spoken(text, spoken)
                results[name] = SpokenResult(spoken, "llm", flags, changes)
            else:
                why = ("the spoken-form call failed" if report.error
                       else "the model returned an empty text" if name in returned
                       else "the model didn't return this clip")
                results[name] = SpokenResult(text.strip(), "fallback", [f"{why}; narration used as written"])
            cache[name] = {**key, "display_text": text, "llm_text": results[name].spoken_text,
                           "spoken_text": results[name].spoken_text,
                           "source": results[name].source, "flags": results[name].flags}
        if extra:
            log(f"  spoken form: ignored {', '.join(extra)}")
        save_cache(run_dir, cache, write_json)

    flagged = sum(1 for r in results.values() if r.flags)
    if mode == "llm":
        log(f"  spoken form ({model_key}): {report.clips_sent} sent, {report.clips_cached} cached, "
            f"{flagged} flagged for review" + (f", ${report.cost_usd:.4f}" if report.calls else ""))
    return results, report


def _rules_pass(lab, topic, clips, run_dir, glossary_files, report, log):
    """Rules mode: free and deterministic, so nothing is cached; changing the glossary
    and rerunning redoes only the clips whose spoken text changed (source hash)."""
    import spoken_rules
    g = spoken_rules.load_glossary(topic, glossary_files, run_dir)
    ctx = spoken_rules.LabContext.from_lab(lab, topic)
    report.glossary_version, report.glossary_files = g.version, g.files
    results = {}
    for name, text in clips:
        r = spoken_rules.apply_rules(text, g, ctx)
        # The word check is a safety net: a rule that changed an ordinary word is a bug.
        check_flags, _ = check_spoken(text, r.spoken_text)
        results[name] = SpokenResult(r.spoken_text, "rules", r.flags + check_flags, r.changes)
        for term in r.unresolved:
            report.unresolved[term] = report.unresolved.get(term, 0) + 1
    changed = sum(1 for r in results.values() if r.changes)
    flagged = sum(1 for r in results.values() if r.flags)
    log(f"  spoken form (rules, glossary {g.version}): {changed} of {len(clips)} clips changed, "
        f"{flagged} flagged for review"
        + (f" ({', '.join(sorted(report.unresolved))})" if report.unresolved else ""))
    return results, report


def _match_tag(tag: str, sent: dict[str, str]) -> str | None:
    """The prompt's audio name for a returned tag, trimming what a model may add when
    copying it: surrounding brackets, quotes or spaces, and a change of case."""
    t = tag.strip().strip("[]\"'`").strip()
    return sent.get(t.casefold())


def _one_attempt(usage: dict, status: str, exc, cost_known: bool = True) -> dict:
    """An attempt record when the caller reported only totals (tests, or a failure
    before any request was tracked)."""
    return {"attempt": 1, "kind": "first", "status": status,
            "error_type": type(exc).__name__ if exc else None,
            "error": " ".join(str(exc).split())[:300] if exc else None,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "cost_usd": usage.get("cost_usd", 0.0) if cost_known else None,
            "seconds": None, "served_model": (usage.get("served_models") or [None])[0],
            "started_at": None}


def _add_usage(report: PassReport, usage: dict) -> None:
    for k in ("input_tokens", "output_tokens", "cost_usd", "transient_retries", "retries"):
        setattr(report, k, getattr(report, k) + (usage.get(k) or 0))
    report.served_models = sorted(set(report.served_models) | set(usage.get("served_models") or []))


def _make_llm_call(model_key: str, report: PassReport):
    """The real call: guarded client, tracked, transient-only retries."""
    from benchmark import (_transient_retryer, create_instructor_client, exception_chain,
                           provider_create_kwargs, status_of)
    from benchmark_config import MODELS as REGISTRY, SpecType
    from tracking import InstructorTracker

    import dataclasses
    mc = dataclasses.replace(REGISTRY[model_key], reasoning_effort=REASONING_EFFORT)
    report.model_id = mc.model_id
    report.reasoning_effort = REASONING_EFFORT
    tracker = InstructorTracker(model=mc.model_id, provider=mc.provider.value,
                                pricing_config=mc.pricing_config,
                                store_raw_responses=True)   # keeps each error's exception
    client = tracker.wrap(create_instructor_client(mc, SpecType.JSON_LAB))

    def call(prompt: str):
        kwargs = dict(response_model=LabSpokenForms, max_retries=MAX_RETRIES,
                      messages=[{"role": "user", "content": prompt}])
        kwargs.update(provider_create_kwargs(mc, LabSpokenForms))
        retryer = _transient_retryer()

        def unanswered_cost(err):
            # No response: an HTTP error reply was not billed ($0); a request cut off
            # mid-reply reported no usage, so its cost is unknown (None).
            raw = err.raw_exception if err else None
            if raw is not None and any(status_of(e) is not None for e in exception_chain(raw)):
                return 0.0
            return None

        def attempts() -> list[dict]:
            # Each create() is one tracked call (a transient re-send is a new call);
            # each instructor attempt inside it is one request (attempt 2+ = a reask).
            out = []
            for ci, tc in enumerate(tracker.aggregate.calls):
                for ai, att in enumerate(tc.attempts):
                    # The tracker marks when each request starts; a request ends when
                    # the next one starts (a reask) or the call ends.
                    end = (tc.attempts[ai + 1].timestamp if ai + 1 < len(tc.attempts)
                           else tc.end_time)
                    kind = ("reask" if att.attempt_number > 1
                            else "first" if ci == 0 else "transient re-send")
                    err = att.error
                    out.append({
                        "attempt": len(out) + 1, "kind": kind,
                        "status": "ok" if att.success else "error",
                        "error_type": err.exception_class if err else None,
                        "error": " ".join(err.message.split())[:300] if err else None,
                        "input_tokens": att.usage.prompt_tokens,
                        "output_tokens": att.usage.completion_tokens,
                        "cost_usd": (tracker.pricing.calculate_cost(att.usage)
                                     if att.responded else unanswered_cost(err)),
                        "seconds": (round(att.duration_ms / 1000, 3) if att.duration_ms
                                    else round((end - att.timestamp).total_seconds(), 3)
                                    if end else None),
                        "served_model": att.served_model,
                        "started_at": att.timestamp.isoformat(timespec="seconds"),
                    })
            return out

        def usage() -> dict:
            agg = tracker.aggregate
            return {
                "attempts": attempts(),
                "input_tokens": agg.total_usage.prompt_tokens,
                "output_tokens": agg.total_usage.completion_tokens,
                "cost_usd": tracker.pricing.calculate_cost(agg.total_usage),
                "retries": agg.total_retries,
                "transient_retries": (retryer.statistics.get("attempt_number", 1) or 1) - 1,
                "served_models": tracker.served_models(),
            }

        try:
            forms = retryer(client.create, **kwargs)
        except Exception as exc:
            # The attempts were billed even though no answer came back.
            exc.spoken_form_usage = usage()
            raise
        return forms, usage()

    return call

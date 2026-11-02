"""
Spoken-form pass (#94): the text each clip sends to the voice service.

``Clip.narration`` is written to be read: it has symbols, numbers, formulas and
acronyms (``5'→3'``, ``109.5°``, ``XeF4``, ``vx = v cosθ``). This pass writes the
same narration the way it should be said aloud. The display text is never
changed; only what is spoken is.

Modes (``narrate_lab.py --spoken-form``):

- ``none``: the narration as written.
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

MODES = ("none", "llm")
# Benchmark registry keys (benchmark_config.MODELS) allowed for the LLM mode.
MODELS = ("claude-haiku-5.5", "gpt-6-luna", "gemini-3.5-flash-lite")
DEFAULT_MODEL = "claude-haiku-5.5"   # provisional; the #94 measurement picks the default
PROMPT_VERSION = 1                   # bump when the prompt changes, so caches refresh
CACHE_FILE = "spoken_forms.json"
MAX_RETRIES = 2                      # instructor reasks for a malformed reply (schema only)


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
_EDGE = re.compile(r"^[^\w']+|[^\w']+$")   # punctuation around a token (keeps 5', θ)


def _tokens(text: str) -> list[str]:
    out = []
    for raw in text.split():
        tok = _EDGE.sub("", raw)
        if tok:
            out.append(tok)
    return out


def is_ordinary(token: str) -> bool:
    """A plain word that should reach the voice unchanged: letters only, not an
    all-caps acronym (``DNA``, ``SSBs`` and emphasis caps may change)."""
    if not _WORD.match(token):
        return False
    letters = token.replace("'", "").replace("’", "")
    upper = sum(ch.isupper() for ch in letters)
    return not (len(letters) >= 2 and upper >= len(letters) - 1)


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
                    and not (len(a[i]) <= 4 and f" {' '.join(fa[i])} " in near)]
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
    error: str | None = None


def run_pass(mode: str, model_key: str | None, lab: dict, topic: str | None,
             clips: list[tuple[str, str]], run_dir: Path, write_json,
             call_fn=None, log=print) -> tuple[dict[str, SpokenResult], PassReport]:
    """Spoken text for each ``(audio_name, display_text)``.

    ``call_fn(prompt) -> (LabSpokenForms, usage_dict)`` replaces the real LLM call in
    tests. A failed LLM call (after transient retries) falls back to the narration
    as written for every clip it covered, flagged, and is reported in the pass
    report; the run continues.
    """
    report = PassReport(mode=mode)
    if mode == "none":
        return {n: SpokenResult(t.strip(), "none") for n, t in clips}, report

    report.model = model_key
    key = _cache_key(mode, model_key)
    cache = load_cache(run_dir)
    results: dict[str, SpokenResult] = {}
    todo: list[tuple[str, str]] = []
    for name, text in clips:
        hit = cache.get(name)
        if hit and hit.get("display_text") == text and all(hit.get(k) == v for k, v in key.items()):
            edited = hit.get("spoken_text") != hit.get("llm_text")
            flags, changes = check_spoken(text, hit["spoken_text"])
            results[name] = SpokenResult(hit["spoken_text"], "edited" if edited else "cache",
                                         flags, changes)
            report.clips_cached += 1
        else:
            todo.append((name, text))

    if todo:
        report.clips_sent = len(todo)
        prompt = build_prompt(lab, topic, todo)
        try:
            forms, usage = (call_fn or _make_llm_call(model_key, report))(prompt)
            report.calls += 1
            for k in ("input_tokens", "output_tokens", "cost_usd", "transient_retries", "retries"):
                setattr(report, k, getattr(report, k) + (usage.get(k) or 0))
            report.served_models = sorted(set(report.served_models) | set(usage.get("served_models") or []))
        except Exception as exc:  # noqa: BLE001 - the run continues with the narration as written
            report.error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
            log(f"  spoken form: the {model_key} call failed ({report.error}); "
                "using the narration as written for these clips")
            forms = LabSpokenForms(clips=[])

        returned: dict[str, str] = {}
        extra = []
        for c in forms.clips:
            if c.audio_name in returned:
                extra.append(f"{c.audio_name} (returned twice; first kept)")
            elif c.audio_name not in dict(todo):
                extra.append(f"{c.audio_name} (not in the prompt)")
            else:
                returned[c.audio_name] = c.spoken_text
        for name, text in todo:
            if name in returned:
                spoken = returned[name].strip()
                flags, changes = check_spoken(text, spoken)
                results[name] = SpokenResult(spoken, "llm", flags, changes)
            else:
                why = "the spoken-form call failed" if report.error else "the model didn't return this clip"
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


def _make_llm_call(model_key: str, report: PassReport):
    """The real call: guarded client, tracked, transient-only retries."""
    from benchmark import (_transient_retryer, create_instructor_client,
                           provider_create_kwargs)
    from benchmark_config import MODELS as REGISTRY, SpecType
    from tracking import InstructorTracker

    mc = REGISTRY[model_key]
    report.model_id = mc.model_id
    tracker = InstructorTracker(model=mc.model_id, provider=mc.provider.value,
                                pricing_config=mc.pricing_config)
    client = tracker.wrap(create_instructor_client(mc, SpecType.JSON_LAB))

    def call(prompt: str):
        kwargs = dict(response_model=LabSpokenForms, max_retries=MAX_RETRIES,
                      messages=[{"role": "user", "content": prompt}])
        kwargs.update(provider_create_kwargs(mc, LabSpokenForms))
        retryer = _transient_retryer()
        try:
            forms = retryer(client.create, **kwargs)
        finally:
            report.transient_retries += (retryer.statistics.get("attempt_number", 1) or 1) - 1
        agg = tracker.aggregate
        return forms, {
            "input_tokens": agg.total_usage.prompt_tokens,
            "output_tokens": agg.total_usage.completion_tokens,
            "cost_usd": tracker.pricing.calculate_cost(agg.total_usage),
            "retries": agg.total_retries,
            "served_models": tracker.served_models(),
        }

    return call

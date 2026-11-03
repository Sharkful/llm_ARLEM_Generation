"""
Rules mode of the spoken-form pass (#94): deterministic, no LLM.

    python "Code/Narration/spoken_rules.py" preview <lab_output.json> [--topic NAME] [--glossary FILE]
    python "Code/Narration/spoken_rules.py" terms <lab files or folders...> [--glossary FILE]

The narration is written to be read; this module rewrites the parts a voice
misreads, and nothing else. It is built to work on topics we have never seen:

- **Rules cover notation, not topics.** Chemical formulas are found by checking a
  token splits into real element symbols plus digits (``XeF4``, ``C6H12O6``);
  number + unit uses a general unit table; Greek letters, math operators, arrows,
  ranges, letter chains (``A-T-G-C``) and letter sequences are one rule each.
- **Ambiguous notation is decided by evidence in the lab itself.** ``5'`` can be
  "prime", "feet" or "arcminutes". Each meaning lists clue words; the meaning with
  the most clues anywhere in the lab wins, and strong local patterns (``5'10"``,
  ``23°26'``, ``5'→3'``) win outright. A topic glossary can pin a meaning.
- **When unsure, leave it and flag it.** An acronym the glossary doesn't know is
  left for the voice; a capitalized word that also appears in lowercase in the
  narration is emphasis (``press NEXT`` -> "press next"). The worst case on a new
  topic is the narration as written plus a list of terms to review.
- **Numbers keep their digits.** Only what voices misread changes: a decimal point
  ("104.5" -> "104 point 5"), units, degrees, ranges and signs.

The glossary is layered, most specific first: ``glossary.yaml`` in the run folder,
``--glossary FILE``, ``glossary/topics/<topic>.yaml``, ``glossary/shared.yaml``.
Its content hash is the ``glossary_version`` recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLOSSARY_DIR = HERE / "glossary"
SHARED_GLOSSARY = GLOSSARY_DIR / "shared.yaml"
RUN_GLOSSARY = "glossary.yaml"   # in a run folder: picked up automatically, also on --resume

# Every element symbol: a formula is a token made only of these plus counts.
ELEMENTS = frozenset("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge
As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm
Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U
Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())


class GlossaryError(ValueError):
    """A glossary file that can't be read or has the wrong shape."""


# ── Glossary ──────────────────────────────────────────────────────────


@dataclass
class Glossary:
    terms: dict[str, str | None] = field(default_factory=dict)    # token -> say (None: as written)
    units: dict[str, dict] = field(default_factory=dict)          # symbol -> {say: [one, many], cues}
    symbols: dict[str, str] = field(default_factory=dict)         # character -> words
    senses: dict[str, dict] = field(default_factory=dict)         # notation -> meaning -> {say, cues}
    pins: dict[str, str] = field(default_factory=dict)            # notation -> meaning
    files: list[str] = field(default_factory=list)                # layers used, least specific first

    @property
    def version(self) -> str:
        payload = {k: getattr(self, k) for k in ("terms", "units", "symbols", "senses", "pins")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _say_pair(value, where: str) -> list[str]:
    if isinstance(value, str):
        return [value, value]
    if (isinstance(value, list) and len(value) == 2 and all(isinstance(v, str) for v in value)):
        return list(value)
    raise GlossaryError(f"{where}: 'say' must be a string or [singular, plural]")


def _merge(g: Glossary, data, name: str) -> None:
    if data is None:
        return
    if not isinstance(data, dict):
        raise GlossaryError(f"{name}: expected a mapping at the top level")
    unknown = set(data) - {"version", "terms", "units", "symbols", "senses", "pins", "note"}
    if unknown:
        raise GlossaryError(f"{name}: unknown section(s) {sorted(unknown)}")
    for key, value in (data.get("terms") or {}).items():
        if value is not None and not isinstance(value, str):
            raise GlossaryError(f"{name}: terms.{key} must be a string or empty")
        g.terms[str(key)] = value
    for key, value in (data.get("units") or {}).items():
        spec = value if isinstance(value, dict) else {"say": value}
        cues = spec.get("cues") or []
        if not isinstance(cues, list):
            raise GlossaryError(f"{name}: units.{key}.cues must be a list")
        g.units[str(key)] = {"say": _say_pair(spec.get("say"), f"{name}: units.{key}"),
                             "cues": [str(c).casefold() for c in cues]}
    for key, value in (data.get("symbols") or {}).items():
        if not isinstance(value, str):
            raise GlossaryError(f"{name}: symbols.{key} must be a string")
        g.symbols[str(key)] = value
    for notation, meanings in (data.get("senses") or {}).items():
        if not isinstance(meanings, dict):
            raise GlossaryError(f"{name}: senses.{notation} must map meanings to {{say, cues}}")
        merged = g.senses.setdefault(str(notation), {})
        for meaning, spec in meanings.items():
            if not isinstance(spec, dict):
                raise GlossaryError(f"{name}: senses.{notation}.{meaning} must have say and cues")
            merged[str(meaning)] = {
                "say": _say_pair(spec.get("say", meaning), f"{name}: senses.{notation}.{meaning}"),
                "cues": [str(c).casefold() for c in spec.get("cues") or []]}
    for notation, meaning in (data.get("pins") or {}).items():
        g.pins[str(notation)] = str(meaning)


def load_glossary(topic: str | None = None, extra: list[Path] | None = None,
                  run_dir: Path | None = None) -> Glossary:
    """The merged glossary: shared, then the topic's file (if any), then each extra
    file, then the run folder's ``glossary.yaml`` (if any). Later layers win."""
    import yaml
    layers = [SHARED_GLOSSARY]
    if topic and (GLOSSARY_DIR / "topics" / f"{topic}.yaml").is_file():
        layers.append(GLOSSARY_DIR / "topics" / f"{topic}.yaml")
    layers += [Path(p) for p in extra or []]
    if run_dir and (Path(run_dir) / RUN_GLOSSARY).is_file():
        layers.append(Path(run_dir) / RUN_GLOSSARY)
    g = Glossary()
    for path in layers:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise GlossaryError(f"cannot read glossary {path}: {exc}") from exc
        _merge(g, data, path.name)
        g.files.append(_display_path(path))
    for notation, meaning in g.pins.items():
        if meaning not in g.senses.get(notation, {}):
            raise GlossaryError(f"pins.{notation}: '{meaning}' is not one of "
                                f"{sorted(g.senses.get(notation, {}))}")
    return g


def _display_path(path: Path) -> str:
    root = HERE.parent.parent
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


# ── Lab context: the evidence ambiguous notation is decided by ───────


_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")


def _words(text: str) -> list[str]:
    """Lowercase words, splitting camelCase and snake_case (``dnaHelix_01``)."""
    return [w.casefold() for w in _SPLIT.findall(text)]


@dataclass
class LabContext:
    cues: Counter                 # every word anywhere in the lab, for clue counting
    lowercase_words: set[str]     # words written in lowercase in the narration (emphasis check)

    @classmethod
    def from_lab(cls, lab: dict, topic: str | None = None) -> "LabContext":
        cues, lower = Counter(_words(topic or "")), set()

        def walk(v, in_narration=False):
            if isinstance(v, str):
                cues.update(_words(v))
                if in_narration:
                    lower.update(w for w in re.findall(r"\b[a-z]+\b", v))
            elif isinstance(v, dict):
                for k, x in v.items():
                    walk(x, in_narration or k == "narration")
            elif isinstance(v, list):
                for x in v:
                    walk(x, in_narration)
        walk(lab)
        return cls(cues, lower)

    @classmethod
    def from_texts(cls, texts: list[str], topic: str | None = None) -> "LabContext":
        return cls.from_lab({"modules": [{"clips": [{"narration": t} for t in texts]}]}, topic)


def choose_sense(notation: str, glossary: Glossary, ctx: LabContext) -> str | None:
    """The meaning of ambiguous notation in this lab: the pinned one, else the one
    with strictly the most clue-word hits in the lab; None if the clues don't decide."""
    if notation in glossary.pins:
        return glossary.pins[notation]
    scores = sorted(((sum(ctx.cues[c] for c in spec["cues"]), meaning)
                     for meaning, spec in glossary.senses.get(notation, {}).items()), reverse=True)
    if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        return None
    return scores[0][1]


# ── The rules ─────────────────────────────────────────────────────────


@dataclass
class RulesResult:
    spoken_text: str
    changes: list[dict] = field(default_factory=list)   # {rule, original, spoken}
    flags: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # terms to review (glossary candidates)


_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
# Curly quotes stay as written (voices read them fine); after a digit they are primes.
_CHARS = str.maketrans({
    "′": "'", "ʹ": "'", "″": '"',
    "‑": "-", "‐": "-", "−": "-", " ": " ", " ": " ",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
})
_SUPER = {"²": "^2", "³": "^3", "¹": "^1", "⁴": "^4"}
# Characters that need no handling: voices read them as punctuation or pauses.
_PLAIN = set(" .,;:!?'\"()[]-—–…/\n\t’‘“”") | set("abcdefghijklmnopqrstuvwxyz"
                                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
# Two-letter words that look like an axis variable (b + y = "by").
_NOT_VARIABLES = {"by", "my", "ax", "ox", "ex", "oy", "ay", "oz"}


_ARTICLE = re.compile(r"(?:^|\W)(?:a|an|the|this|that|each|every|one)\s+$", re.IGNORECASE)


def _one(n: str, before: str) -> bool:
    """Singular unit word: for 1, and for an adjective ("a 180 degree turn")."""
    return n.lstrip("-") == "1" or bool(_ARTICLE.search(before))


class _Rules:
    def __init__(self, glossary: Glossary, ctx: LabContext):
        self.g, self.ctx = glossary, ctx
        self.changes: list[dict] = []
        self.flags: list[str] = []
        self.unresolved: list[str] = []
        self.prime = choose_sense("number_prime", glossary, ctx)
        self.math = choose_sense("axis_variable", glossary, ctx) is not None
        self.units = {sym: spec for sym, spec in glossary.units.items()
                      if not spec["cues"] or any(ctx.cues[c] for c in spec["cues"])}

    def sub(self, rule: str, pattern, repl, text: str, flags=0) -> str:
        """``re.sub`` that records each replacement that changes the text."""
        rx = pattern if isinstance(pattern, re.Pattern) else re.compile(pattern, flags)

        def one(m):
            out = repl(m) if callable(repl) else m.expand(repl)
            if out is None:
                return m.group(0)
            if out != m.group(0):
                self.changes.append({"rule": rule, "original": m.group(0).strip(),
                                     "spoken": " ".join(out.split())})
            return out
        return rx.sub(one, text)

    def flag(self, term: str, why: str) -> None:
        if term not in self.unresolved:
            self.unresolved.append(term)
            self.flags.append(f"'{term}': {why}")

    # Each step works on the output of the one before; order matters.

    def apply(self, text: str) -> str:
        t = text.strip().translate(_CHARS)
        for sup, rep in _SUPER.items():
            t = t.replace(sup, rep)
        t = re.sub(r"(?<=\d)[’‘]", "'", t)               # 5’ -> 5'
        t = re.sub(r"(?<=\d)[”“]", '"', t)
        t = re.sub(r"(?<=\d)–(?=\d)", "-", t)          # en dash in a number range
        t = self.glossary_terms(t)
        t = self.strands(t)
        t = self.letter_chains(t)
        t = self.sequences(t)
        t = self.primes(t)
        t = self.degrees(t)
        t = self.unit_rules(t)
        t = self.formulas(t)
        t = self.math_rules(t)
        t = self.ranges_and_signs(t)
        t = self.decimals(t)
        t = self.slashes(t)
        t = self.symbol_rules(t)
        t = self.acronyms(t)
        self.leftovers(t)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r" ([,.;:!?)])", r"\1", t)
        return re.sub(r"\( ", "(", t).strip()

    def glossary_terms(self, t: str) -> str:
        said = {k: v for k, v in self.g.terms.items() if v is not None}
        if not said:
            return t
        rx = re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in
                                                  sorted(said, key=len, reverse=True))
                        + r")(?=(?:'s|s)?(?![\w]))")
        return self.sub("glossary", rx, lambda m: said[m.group(1)], t)

    def _prime_word(self, meaning: str, n: str) -> str:
        return self.g.senses["number_prime"][meaning]["say"][0 if n in ("1", "-1") else 1]

    def strands(self, t: str) -> str:
        # 5'-ATGC-3': end labels hyphenated to a sequence are primes; the hyphens are pauses.
        if "prime" not in self.g.senses.get("number_prime", {}):
            return t
        p = lambda m: f"{m.group(1)} {self._prime_word('prime', m.group(1))}"  # noqa: E731
        w = lambda n: f"{n} {self._prime_word('prime', n)}"  # noqa: E731
        # Between two end labels a run of capitals is a sequence, however short.
        t = self.sub("strand", r"(?<![\w.])(\d+)'-([A-Z]+)-(\d+)'(?![\w'])",
                     lambda m: f"{w(m.group(1))}, {', '.join(m.group(2))}, {w(m.group(3))}", t)
        t = self.sub("strand", r"(?<![\w.])(\d+)'-(?=[A-Z])", lambda m: p(m) + ", ", t)
        return self.sub("strand", r"(?<=[A-Z])-(\d+)'(?![\w'])", lambda m: ", " + p(m), t)

    def letter_chains(self, t: str) -> str:
        # A-T-G-C, H-O-H, A-T pairs: single capitals joined by hyphens, read as letters.
        def chain(m):
            letters = m.group(0).split("-")
            return (", " if len(letters) >= 4 else " ").join(letters)
        return self.sub("letter chain", r"(?<![\w-])[A-Z](?:-[A-Z])+(?![\w-])", chain, t)

    def sequences(self, t: str) -> str:
        # TACGGTCA: a long run of a few capital letters is a sequence, read letter by letter.
        def seq(m):
            s = m.group(0)
            if s in self.g.terms or s.casefold() in self.ctx.lowercase_words or len(set(s)) > 4:
                return None
            return ", ".join(s)
        return self.sub("sequence", r"(?<![\w'])[A-Z]{6,}(?![\w'])", seq, t)

    def primes(self, t: str) -> str:
        # Unambiguous local patterns first: 5'10", 23°26', 5'→3', "the 3' end".
        t = self.sub("feet and inches", r"\b(\d+)' ?(\d+)\"",
                     lambda m: f"{m.group(1)} {'foot' if m.group(1) == '1' else 'feet'} "
                               f"{m.group(2)} {'inch' if m.group(2) == '1' else 'inches'}", t)
        t = self.sub("degrees and arcminutes", r"(\d+) ?° ?(\d+)'(?: ?(" + _NUM + ")\")?",
                     lambda m: f"{m.group(1)} {'degree' if m.group(1) == '1' else 'degrees'} "
                               f"{m.group(2)} {'arcminute' if m.group(2) == '1' else 'arcminutes'}"
                               + (f" {m.group(3)} arcseconds" if m.group(3) else ""), t)
        if "prime" in self.g.senses.get("number_prime", {}):
            w = self._prime_word
            t = self.sub("prime", r"(?<![\w.])(\d+)' ?(?:→|->|-to-|-|–|to\b) ?(\d+)'(?![\w'])",
                         lambda m: f"{m.group(1)} {w('prime', m.group(1))} to "
                                   f"{m.group(2)} {w('prime', m.group(2))}", t)
            t = self.sub("prime", r"(?<![\w.])(\d+)'(?= ?-? ?(?:end|ends|cap|carbon|terminus|UTR|"
                                  r"hydroxyl|phosphate)\b)",
                         lambda m: f"{m.group(1)} {w('prime', m.group(1))}", t)
        # The rest by the lab's clues. "1990's" is a plural, not a prime.
        rx = re.compile(r"(?<![\w.'])(\d+)'(?![\w'\"])")
        if self.prime is None:
            for m in rx.finditer(t):
                self.flag(m.group(0), "prime, feet or arcminutes? The lab has no clues; "
                                      "left as written (pin a meaning in the glossary)")
            return t
        return self.sub(f"number prime ({self.prime})", rx,
                        lambda m: f"{m.group(1)} {self._prime_word(self.prime, m.group(1))}", t)

    def degrees(self, t: str) -> str:
        def deg(m):
            n, scale = m.group(1), m.group(2)
            word = "degree" if _one(n, t[:m.start()]) else "degrees"
            return f"{n} {word}" + ({"C": " Celsius", "F": " Fahrenheit"}.get(scale or "", ""))
        t = self.sub("degrees", r"(" + _NUM + r") ?°(?: ?([CF])\b)?", deg, t)
        return self.sub("degrees", r"°(?: ?([CF])\b)?",
                        lambda m: " degrees" + {"C": " Celsius", "F": " Fahrenheit"}
                        .get(m.group(1) or "", ""), t)

    def _unit_words(self, symbol: str, plural: bool) -> str | None:
        """'m/s^2' -> 'meters per second squared' (None if any part is unknown)."""
        parts = symbol.split("/")
        out = []
        for i, part in enumerate(parts):
            m = re.fullmatch(r"(.+?)(?:\^([23]))?", part)
            spec = self.units.get(m.group(1))
            if spec is None:
                return None
            out.append(spec["say"][1 if plural and i == 0 else 0]
                       + {"2": " squared", "3": " cubed"}.get(m.group(2) or "", ""))
        return " per ".join(out)

    def unit_rules(self, t: str) -> str:
        if self.units:
            syms = "|".join(re.escape(s) for s in sorted(self.units, key=len, reverse=True))
            one = rf"(?:{syms})(?:\^[23])?"
            unit = rf"{one}(?:/{one})?"

            def after_number(m):
                n, sym, glued = m.group(1), m.group(3), m.group(2) == ""
                if glued and "/" not in sym and len(sym.split("^")[0]) < 2:
                    return None          # "1990s", "3D": a single letter glued on is not a unit
                words = self._unit_words(sym, not _one(n, t[:m.start()]))
                return None if words is None else f"{n} {words}"
            t = self.sub("unit", rf"(?<![\w.])(-?{_NUM})( ?)({unit})(?![\w/^])", after_number, t)
            # A compound unit on its own ("in m/s") is still a unit.
            t = self.sub("unit", rf"(?<![\w/])({one}/{one})(?![\w/^])",
                         lambda m: self._unit_words(m.group(1), True), t)
        return self.sub("percent", r"(?<=\d) ?%", " percent", t)

    def formulas(self, t: str) -> str:
        # XeF4 -> "X e F 4": element symbols said as letters, counts as numbers.
        def formula(m):
            coef, body = m.group(1) or "", m.group(2)
            if body in self.g.terms:
                return None
            groups = re.findall(r"([A-Z][a-z]?)(\d*)", body)
            if "".join(e + n for e, n in groups) != body or not all(e in ELEMENTS for e, _ in groups):
                return None
            if not any(n for _, n in groups):
                return None              # no counts: may be a word or acronym (HeLa, CNS)
            said = " ".join(" ".join(e) + (f" {n}" if n else "") for e, n in groups)
            return f"{coef} {said}".strip()
        return self.sub("formula", r"(?<![\w'])(\d+)?((?:[A-Z][a-z]?\d*){1,})(?=(?:'s)?(?![\w']))",
                        formula, t)

    def math_rules(self, t: str) -> str:
        t = self.sub("power", r"\^2\b", " squared", t)
        t = self.sub("power", r"\^3\b", " cubed", t)
        t = self.sub("power", r"\^(-?\w+)", r" to the power of \1", t)
        for sym, word in self.g.symbols.items():
            if any(ch.isalpha() for ch in sym):     # Greek letters: pad when glued to a word
                t = self.sub("greek letter", re.escape(sym), f" {word} ", t)
        # v0 / v₀ -> "v zero"; Vx -> "V x" (only in a lab with math clues).
        t = self.sub("subscript", r"(?<![\w'])([A-Za-z])0(?![\w'])", r"\1 zero", t)
        if self.math:
            def variable(m):
                # "by", "ax" are words; "By" too at a sentence start. "Ax" mid-sentence is not.
                if m.group(0).casefold() in _NOT_VARIABLES and (
                        m.group(0).islower() or re.search(r"(?:^|[.!?:]\s+)$", t[:m.start()])):
                    return None
                return f"{m.group(1)} {m.group(2)}"
            t = self.sub("axis variable", r"(?<![\w'-])([A-Za-z])([xyz])(?![\w'-])", variable, t)
        if "=" in t:
            t = self.sub("operator", r" ?= ?", " equals ", t)
            t = self.sub("operator", r"(?<=[\w)]) - (?=[\w(])", " minus ", t)
        t = self.sub("operator", r"(?<=[\w)]) ?\+ ?(?=[\w(])", " plus ", t)
        return t

    def ranges_and_signs(self, t: str) -> str:
        t = self.sub("range", rf"(?<![\w.-])({_NUM})-({_NUM})(?![\w.])", r"\1 to \2", t)
        return self.sub("sign", rf"(?<![\w.)-])-({_NUM})", r"minus \1", t)

    def decimals(self, t: str) -> str:
        # The voice may read "104.5" as "104. 5"; digits after the point one by one.
        return self.sub("decimal", r"(?<![\w.])(\d+(?:,\d{3})*)\.(\d+)(?!\w|\.\d)",
                        lambda m: f"{m.group(1)} point {' '.join(m.group(2))}", t)

    def slashes(self, t: str) -> str:
        t = self.sub("slash", r"\band/or\b", "and or", t)
        return self.sub("slash", r"(?<![\w/])([A-Za-z]{2,})/([A-Za-z]{2,})(?![\w/])", r"\1 or \2", t)

    def symbol_rules(self, t: str) -> str:
        for sym, word in self.g.symbols.items():
            if not any(ch.isalpha() for ch in sym) and sym in t:
                t = self.sub("symbol", rf" ?{re.escape(sym)} ?", f" {word} ", t)
        return t

    def acronyms(self, t: str) -> str:
        def acronym(m):
            word, base = m.group(0), m.group(1)
            if base in self.g.terms or word in self.g.terms:
                return None                          # known: read as written
            if len(base) >= 3 and base.casefold() in self.ctx.lowercase_words:
                return word.casefold()               # emphasis: "press NEXT"
            self.flag(word, "acronym not in the glossary; left for the voice")
            return None
        return self.sub("emphasis", r"(?<![\w'])([A-Z][A-Z]+)(?:'s|s)?(?![\w'])", acronym, t)

    def leftovers(self, t: str) -> None:
        for ch in sorted({c for c in t if c not in _PLAIN and not c.isalnum()}):
            self.flag(ch, "symbol with no rule; left for the voice")


def apply_rules(text: str, glossary: Glossary, ctx: LabContext) -> RulesResult:
    r = _Rules(glossary, ctx)
    spoken = r.apply(text)
    return RulesResult(spoken, r.changes, r.flags, r.unresolved)


# ── CLI: preview a lab, list terms to review ─────────────────────────


def _labs(paths: list[Path]):
    for p in paths:
        files = sorted(p.rglob("*_output.json")) if p.is_dir() else [p]
        for f in files:
            try:
                lab = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(lab, dict) and isinstance(lab.get("modules"), list):
                yield f, lab


def _topic_of(path: Path) -> str | None:
    name = path.name.removesuffix("_output.json") + "_metrics.json"
    for cand in (path.with_name(name), path.parent.parent / "Metrics" / name):
        try:
            return json.loads(cand.read_text(encoding="utf-8")).get("lab_name")
        except (OSError, ValueError):
            continue
    return None


def _clips(lab: dict):
    for m in lab["modules"]:
        for c in m.get("clips") or []:
            if isinstance(c, dict) and isinstance(c.get("narration"), str) and c["narration"].strip():
                yield c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spoken-form rules (#94): preview and review terms.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("preview", help="show each clip's narration and spoken text")
    pv.add_argument("lab", type=Path)
    pv.add_argument("--topic")
    pv.add_argument("--all", action="store_true", help="also show clips the rules leave unchanged")
    tm = sub.add_parser("terms", help="terms the rules and glossary don't cover, with counts")
    tm.add_argument("paths", nargs="+", type=Path, help="lab output files or folders")
    tm.add_argument("--json", action="store_true")
    for p in (pv, tm):
        p.add_argument("--glossary", type=Path, action="append", help="an extra glossary file")
    args = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")

    try:
        if args.cmd == "preview":
            labs = list(_labs([args.lab]))
            if not labs:
                print(f"error: {args.lab} is not a json_lab Lab", file=sys.stderr)
                return 2
            path, lab = labs[0]
            topic = args.topic or _topic_of(path)
            g = load_glossary(topic, args.glossary)
            ctx = LabContext.from_lab(lab, topic)
            print(f"{path.name}  topic: {topic or '(none)'}  glossary {g.version}: {', '.join(g.files)}")
            print(f"number prime -> {_Rules(g, ctx).prime or 'undecided'}\n")
            for c in _clips(lab):
                r = apply_rules(c["narration"], g, ctx)
                if r.changes or r.flags or args.all:
                    print(f"[{c.get('audioClip') or c.get('clipName')}]")
                    print(f"  written: {c['narration'].strip()}")
                    print(f"  spoken:  {r.spoken_text}")
                    for f in r.flags:
                        print(f"  flag:    {f}")
                    print()
            return 0

        found: dict[str, Counter] = {}
        examples: dict[str, str] = {}
        rules = Counter()
        n_labs = n_clips = 0
        for path, lab in _labs(args.paths):
            topic = _topic_of(path)
            g = load_glossary(topic, args.glossary)
            ctx = LabContext.from_lab(lab, topic)
            n_labs += 1
            for c in _clips(lab):
                n_clips += 1
                r = apply_rules(c["narration"], g, ctx)
                rules.update(ch["rule"] for ch in r.changes)
                for term in r.unresolved:
                    found.setdefault(term, Counter())[topic or "(no topic)"] += 1
                    if term not in examples:
                        i = c["narration"].find(term)
                        examples[term] = c["narration"][max(0, i - 40): i + len(term) + 40].strip()
        if args.json:
            print(json.dumps({"labs": n_labs, "clips": n_clips, "rules": dict(rules),
                              "terms": {t: {"count": sum(c.values()), "topics": dict(c),
                                            "example": examples[t]} for t, c in found.items()}},
                             indent=2, ensure_ascii=False))
            return 0
        print(f"{n_labs} labs, {n_clips} narrated clips")
        print("Rules applied: " + ", ".join(f"{k} {v}" for k, v in rules.most_common()))
        print(f"\n{len(found)} term(s) to review (not in the glossary, or undecided):")
        for term, c in sorted(found.items(), key=lambda kv: -sum(kv[1].values())):
            topics = ", ".join(f"{k} {v}" for k, v in c.most_common(3))
            print(f"  {term:<14} {sum(c.values()):>5}  {topics}\n      …{examples[term]}…")
        return 0
    except GlossaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""
Audit a lab: every model and texture it references, with the context needed to
describe it (#106; spec: PROVISIONING.md, "Steps per lab", step 1). No LLM.

An asset is a distinct ``SceneObject.prefab`` or ``SceneObject.texture`` value,
compared ignoring case and surrounding spaces (clip changes can't set either, so
only initial objects count). For each one the audit collects how the lab uses it:
the objects, their modules, scale, color, parent and texture or prefab pairing, any
text on them, narration lines that mention them, and the ``purpose`` a hierarchical
run's asset plan gave it. ``demoPrefab`` is the module's system object, not an asset.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
SCHEMAS_DIR = HERE.parent / "Schemas"

from library import name_key  # noqa: E402

SYSTEM_PREFABS = {"demoprefab"}
MAX_USES = 8            # uses kept per asset (all are counted)
MAX_NARRATION = 3       # narration lines kept per asset
MAX_LINE = 240          # characters per narration line or text
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class InputError(Exception):
    """The input isn't a lab this tool can audit."""


@dataclass
class AssetUse:
    module: int                 # 1-based
    module_name: str
    object: str
    parent: Optional[str] = None
    scale: Optional[list] = None
    color: Optional[list] = None
    paired_with: Optional[str] = None   # a model's texture, or a texture's prefab
    text: list[str] = field(default_factory=list)


@dataclass
class AssetRequest:
    name: str                   # first spelling seen
    kind: str                   # "model" or "texture"
    spellings: list[str] = field(default_factory=list)
    uses: list[AssetUse] = field(default_factory=list)
    use_count: int = 0
    modules: list[int] = field(default_factory=list)
    narration: list[str] = field(default_factory=list)
    purpose: Optional[str] = None

    @property
    def key(self) -> str:
        return name_key(self.name)

    def to_json(self) -> dict:
        return asdict(self)


def load_lab(path: Path) -> dict:
    """A json_lab ``Lab`` from ``path``, validated (InputError otherwise)."""
    try:
        lab = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(lab, dict) or "modules" not in lab:
        raise InputError(f"{Path(path).name} is not a full json_lab Lab (no 'modules'). "
                         "L1 outlines, module-only outputs and ARLEM specs aren't supported.")
    if str(SCHEMAS_DIR) not in sys.path:
        sys.path.append(str(SCHEMAS_DIR))
    from json_lab import Lab
    try:
        Lab.model_validate(lab)
    except Exception as exc:  # pydantic.ValidationError
        raise InputError(f"{Path(path).name} is not a valid json_lab Lab: "
                         + " ".join(str(exc).splitlines()[:3])) from exc
    return lab


def paired_metrics(path: Path) -> Optional[dict]:
    """The benchmark or hierarchical metrics record for an ``*_output.json``, if any:
    beside it, or in the sibling Metrics/ folder (as narration finds it)."""
    path = Path(path)
    if not path.name.endswith("_output.json"):
        return None
    name = path.name[: -len("_output.json")] + "_metrics.json"
    for cand in (path.with_name(name), path.parent.parent / "Metrics" / name):
        if cand.is_file():
            try:
                return {"_path": str(cand), **json.loads(cand.read_text(encoding="utf-8"))}
            except (OSError, ValueError):
                return None
    return None


def purposes_from_metrics(metrics: Optional[dict]) -> dict[str, str]:
    """Asset name key -> purpose, from a hierarchical run's registry (if present)."""
    reg = (metrics or {}).get("registry") or {}
    out = {}
    for group in ("invented_prefabs", "invented_textures"):
        for name, info in (reg.get(group) or {}).items():
            if isinstance(info, dict) and info.get("purpose"):
                out[name_key(name)] = info["purpose"]
    return out


def name_words(name: str) -> str:
    """The words in an object or asset name, for finding it in narration:
    'dnaHelixPrefab' -> 'dna helix', 'heart_cutaway_texture' -> 'heart cutaway'."""
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name.strip())
    words = re.sub(r"[_\-\s]+", " ", words).lower().split()
    while words and words[-1] in {"prefab", "texture", "tex", "model", "obj", "mesh"}:
        words.pop()
    return " ".join(words)


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_LINE else text[: MAX_LINE - 1].rstrip() + "…"


def _is_text(c) -> bool:
    """A text component, as saved (componentType) or as the schema names it (type)."""
    return isinstance(c, dict) and (c.get("componentType") == "textMeshPro"
                                    or c.get("type") == "TextMeshProComponent")


def _texts(components) -> list[str]:
    return [_clip(c["text"]) for c in components or [] if _is_text(c) and c.get("text")]


def audit_lab(lab: dict, purposes: Optional[dict[str, str]] = None) -> list[AssetRequest]:
    """Every model and texture the lab references, in order of first use."""
    purposes = purposes or {}
    assets: dict[tuple[str, str], AssetRequest] = {}
    mentions: dict[tuple[str, str], set[str]] = {}   # asset -> phrases to look for

    for m_index, module in enumerate(lab.get("modules", []), start=1):
        m_name = module.get("moduleName", "")
        changes_text: dict[str, list[str]] = {}
        for clip in module.get("clips") or []:
            for ch in clip.get("changes") or []:
                changes_text.setdefault(ch.get("target", ""), []).extend(_texts(ch.get("components")))
        for obj in module.get("objects", []):
            prefab, texture = obj.get("prefab"), obj.get("texture")
            text = _texts(obj.get("components")) + changes_text.get(obj.get("name", ""), [])
            for kind, value, paired in (("model", prefab, texture), ("texture", texture, prefab)):
                if not value or not str(value).strip():
                    continue
                if kind == "model" and name_key(value) in SYSTEM_PREFABS:
                    continue
                k = (kind, name_key(value))
                req = assets.get(k)
                if req is None:
                    req = assets[k] = AssetRequest(name=value.strip(), kind=kind,
                                                   purpose=purposes.get(name_key(value)))
                if value.strip() not in req.spellings:
                    req.spellings.append(value.strip())
                req.use_count += 1
                if m_index not in req.modules:
                    req.modules.append(m_index)
                if len(req.uses) < MAX_USES:
                    req.uses.append(AssetUse(
                        module=m_index, module_name=m_name, object=obj.get("name", ""),
                        parent=obj.get("parent"), scale=obj.get("scale"), color=obj.get("color"),
                        paired_with=paired.strip() if paired and str(paired).strip() else None,
                        text=list(dict.fromkeys(text))[:3]))
                phrases = mentions.setdefault(k, set())
                for p in (name_words(obj.get("name", "")), name_words(value)):
                    if p:
                        phrases.add(p)

    for k, req in assets.items():
        patterns = [re.compile(r"\b" + re.escape(p) + r"\b") for p in sorted(mentions.get(k, ()))]
        found: list[str] = []
        for m_index in req.modules:
            module = lab["modules"][m_index - 1]
            for clip in module.get("clips") or []:
                for sentence in _SENTENCE.split(clip.get("narration") or ""):
                    if any(p.search(sentence.lower()) for p in patterns):
                        line = _clip(sentence)
                        if line not in found:
                            found.append(line)
                    if len(found) >= MAX_NARRATION:
                        break
                if len(found) >= MAX_NARRATION:
                    break
            if len(found) >= MAX_NARRATION:
                break
        req.narration = found
    return list(assets.values())

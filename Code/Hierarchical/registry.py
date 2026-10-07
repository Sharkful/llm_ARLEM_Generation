"""
Running asset registry and component catalog for hierarchical generation.

Carries what earlier modules created forward into later steps, so a later module
can reuse a prefab, texture, or new component instead of inventing a near-duplicate:

- Component catalog: the built-in json_lab components (derived from the schema
  classes so it can't drift) plus every NewComponent an accepted module used.
- Invented prefabs / textures: assets outside the moon-lab library that earlier
  modules introduced.
- Module summaries: one compact entry per accepted module for later prompts.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import _paths  # noqa: F401  (sys.path for Schemas / Benchmark)

from json_lab import Component, DemoModule, NewComponent
from lab_metrics import KNOWN_PREFABS, KNOWN_TEXTURES


# Display spelling of the moon-lab library. lab_metrics keeps the same sets
# lowercased for case-insensitive novelty checks; the assert keeps them in sync.
BUILTIN_PREFABS = (
    "SunPrefab", "moveableSphere", "clickableSphere", "tinySphere",
    "textPrefab", "robotIdle",
)
BUILTIN_TEXTURES = ("2k_earth_daymap", "2k_moon", "2k_sun", "balldimpled")
assert {p.lower() for p in BUILTIN_PREFABS} == KNOWN_PREFABS - {"demoprefab"}
assert set(BUILTIN_TEXTURES) == KNOWN_TEXTURES

# Fields that are routing tags, not configuration the model sets.
_TAG_FIELDS = {"type", "componentType"}


@dataclass
class CatalogEntry:
    name: str
    description: str
    builtin: bool
    origin_module: int | None = None  # 1-based module index that invented it


def _builtin_catalog() -> list[CatalogEntry]:
    """One entry per json_lab component class except NewComponent (the mechanism
    for new ones), described by its docstring plus a field summary."""
    entries = []
    for cls in typing.get_args(Component):
        if cls is NewComponent:
            continue
        fields = []
        for fname, finfo in cls.model_fields.items():
            if fname in _TAG_FIELDS:
                continue
            desc = (finfo.description or "").split(". ")[0].strip()
            fields.append(f"{fname} ({desc})" if desc else fname)
        doc = " ".join((cls.__doc__ or "").split())
        entries.append(CatalogEntry(
            name=cls.__name__,
            description=f"{doc} Fields: {', '.join(fields)}." if fields else doc,
            builtin=True,
        ))
    return entries


def _iter_module_components(module: DemoModule):
    for obj in module.objects:
        yield from obj.components or []
    for clip in module.clips:
        for change in clip.changes or []:
            yield from change.components or []


@dataclass
class ModuleSummary:
    index: int  # 1-based
    module_name: str
    description: str
    object_names: list[str]
    num_clips: int

    def render(self) -> str:
        return (
            f"Module {self.index} \"{self.module_name}\": {self.description} "
            f"(objects: {', '.join(self.object_names)}; {self.num_clips} clips)"
        )


@dataclass
class AssetRegistry:
    components: list[CatalogEntry] = field(default_factory=_builtin_catalog)
    invented_prefabs: dict[str, int] = field(default_factory=dict)   # name -> module index
    invented_textures: dict[str, int] = field(default_factory=dict)
    modules: list[ModuleSummary] = field(default_factory=list)

    # ── queries ──────────────────────────────────────────────────────
    def catalog_names(self) -> list[str]:
        return [c.name for c in self.components]

    def new_component_names(self) -> list[str]:
        return [c.name for c in self.components if not c.builtin]

    def find_component(self, name: str) -> CatalogEntry | None:
        key = name.strip().lower()
        return next((c for c in self.components if c.name.lower() == key), None)

    # ── updates ──────────────────────────────────────────────────────
    def register_module(self, index: int, module: DemoModule) -> None:
        """Record an accepted module: its new components, novel assets, summary."""
        for comp in _iter_module_components(module):
            if isinstance(comp, NewComponent) and self.find_component(comp.scriptName) is None:
                self.components.append(CatalogEntry(
                    name=comp.scriptName,
                    description=" ".join(comp.scriptDescription.split()),
                    builtin=False,
                    origin_module=index,
                ))
        known_prefabs = {p.lower() for p in BUILTIN_PREFABS} | {p.lower() for p in self.invented_prefabs}
        known_textures = {t.lower() for t in BUILTIN_TEXTURES} | {t.lower() for t in self.invented_textures}
        for obj in module.objects:
            if obj.prefab and obj.prefab.lower() not in known_prefabs:
                self.invented_prefabs[obj.prefab] = index
                known_prefabs.add(obj.prefab.lower())
            if obj.texture and obj.texture.lower() not in known_textures:
                self.invented_textures[obj.texture] = index
                known_textures.add(obj.texture.lower())
        self.modules.append(ModuleSummary(
            index=index,
            module_name=module.moduleName,
            description=module.description,
            object_names=[o.name for o in module.objects],
            num_clips=len(module.clips),
        ))

    # ── prompt rendering ─────────────────────────────────────────────
    def render_component_catalog(self) -> str:
        lines = ["Built-in components:"]
        lines += [f"- {c.name}: {c.description}" for c in self.components if c.builtin]
        new = [c for c in self.components if not c.builtin]
        if new:
            lines.append("")
            lines.append(
                "New components already invented by earlier modules (reuse these by "
                "their exact name when they fit):"
            )
            lines += [
                f"- {c.name} (from module {c.origin_module}): {c.description}" for c in new
            ]
        return "\n".join(lines)

    def render_assets(self) -> str:
        def _fmt(items: dict[str, int]) -> str:
            return ", ".join(f"{n} (module {i})" for n, i in items.items()) or "none yet"
        return "\n".join([
            f"Built-in prefabs: {', '.join(BUILTIN_PREFABS)}",
            f"Built-in textures: {', '.join(BUILTIN_TEXTURES)}",
            f"Prefabs invented by earlier modules: {_fmt(self.invented_prefabs)}",
            f"Textures invented by earlier modules: {_fmt(self.invented_textures)}",
        ])

    def render_previous_modules(self) -> str:
        if not self.modules:
            return "None yet; this is the first module."
        return "\n".join(m.render() for m in self.modules)

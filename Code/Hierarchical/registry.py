"""
Running asset registry and component catalog for hierarchical generation.

Carries what earlier modules created forward into later steps, so a later module
can reuse a prefab, texture, or new component instead of inventing a near-duplicate:

- Component catalog: the built-in json_lab components (derived from the schema
  classes so it can't drift) plus every NewComponent an accepted module used.
- Invented prefabs / textures: assets outside the moon-lab library that earlier
  modules introduced, with the purpose the asset plan gave them.
- Module summaries: one compact entry per accepted module for later prompts,
  including what it covered (built in code, no LLM call).
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import _paths  # noqa: F401  (sys.path for Schemas / Benchmark)

from json_lab import Component, DemoModule, NewComponent
from lab_metrics import KNOWN_PREFABS, KNOWN_TEXTURES
from module_asset_plan import ModuleAssetPlan, component_key


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
    # New components only: inspector fields as {name, type, description} (#74),
    # so later modules reuse the same field names.
    inputs: list[dict] = field(default_factory=list)

    def render(self) -> str:
        text = f"{self.name} (from module {self.origin_module}): {self.description}"
        if self.inputs:
            text += " Inputs: " + "; ".join(
                f"{i['name']} ({i['type']}): {i['description']}" for i in self.inputs
            ) + "."
        return text


def _builtin_catalog() -> list[CatalogEntry]:
    """One entry per json_lab component class except NewComponent (the mechanism
    for new ones), described by its docstring plus every field's full description."""
    entries = []
    for cls in typing.get_args(Component):
        if cls is NewComponent:
            continue
        fields = []
        for fname, finfo in cls.model_fields.items():
            if fname in _TAG_FIELDS:
                continue
            desc = " ".join((finfo.description or "").split())
            fields.append(f"{fname} ({desc})" if desc else fname)
        doc = " ".join((cls.__doc__ or "").split())
        entries.append(CatalogEntry(
            name=cls.__name__,
            description=f"{doc} Fields: {', '.join(fields)}." if fields else doc,
            builtin=True,
        ))
    return entries


def iter_module_components(module: DemoModule):
    """Every component in a module: on its objects and in its clips' changes.
    The single definition of where components can live (pipeline.py reuses it)."""
    for obj in module.objects:
        yield from obj.components or []
    for clip in module.clips:
        for change in clip.changes or []:
            yield from change.components or []


@dataclass
class InventedAsset:
    module: int                   # 1-based module index that introduced it
    purpose: str | None = None    # the planned object's purpose, when known

    def render(self, name: str) -> str:
        return f"{name} (module {self.module}: {self.purpose})" if self.purpose else \
            f"{name} (module {self.module})"


@dataclass
class ModuleSummary:
    index: int  # 1-based
    module_name: str
    description: str
    object_names: list[str]
    num_clips: int
    covered: str = ""

    def render(self) -> str:
        line = (
            f"Module {self.index} \"{self.module_name}\": {self.description} "
            f"(objects: {', '.join(self.object_names)}; {self.num_clips} clips)"
        )
        return f"{line}\n  covered: {self.covered}" if self.covered else line


@dataclass
class AssetRegistry:
    components: list[CatalogEntry] = field(default_factory=_builtin_catalog)
    invented_prefabs: dict[str, InventedAsset] = field(default_factory=dict)
    invented_textures: dict[str, InventedAsset] = field(default_factory=dict)
    modules: list[ModuleSummary] = field(default_factory=list)
    # Lowercase membership sets for "known" assets: lab_metrics' moon-lab library
    # (the same sets analyze_assets uses, incl. the system demoPrefab) plus
    # everything earlier modules invented. Kept in sync by register_module.
    _known_prefabs: set[str] = field(default_factory=lambda: set(KNOWN_PREFABS), repr=False)
    _known_textures: set[str] = field(default_factory=lambda: set(KNOWN_TEXTURES), repr=False)

    # ── queries ──────────────────────────────────────────────────────
    def catalog_names(self) -> list[str]:
        return [c.name for c in self.components]

    def new_component_names(self) -> list[str]:
        return [c.name for c in self.components if not c.builtin]

    def find_component(self, name: str) -> CatalogEntry | None:
        key = component_key(name)
        return next((c for c in self.components if component_key(c.name) == key), None)

    # ── updates ──────────────────────────────────────────────────────
    def register_module(
        self,
        index: int,
        module: DemoModule,
        asset_plan: ModuleAssetPlan | None = None,
        scene_purpose: str | None = None,
    ) -> None:
        """Record an accepted module: its new components, novel assets (with the
        purpose its asset plan gave them), and a summary for later prompts."""
        purposes = {o.name: o.purpose for o in asset_plan.objects} if asset_plan else {}
        for comp in iter_module_components(module):
            if isinstance(comp, NewComponent) and self.find_component(comp.scriptName) is None:
                self.components.append(CatalogEntry(
                    name=comp.scriptName,
                    description=" ".join(comp.scriptDescription.split()),
                    builtin=False,
                    origin_module=index,
                    inputs=[i.model_dump(exclude={"value"}) for i in comp.inputs or []],
                ))
        for obj in module.objects:
            purpose = purposes.get(obj.name)
            if obj.prefab and not self.is_known_asset(prefab=obj.prefab):
                self.invented_prefabs[obj.prefab] = InventedAsset(index, purpose)
                self._known_prefabs.add(obj.prefab.strip().lower())
            if obj.texture and not self.is_known_asset(texture=obj.texture):
                self.invented_textures[obj.texture] = InventedAsset(index, purpose)
                self._known_textures.add(obj.texture.strip().lower())
        covered = "; ".join(filter(None, [scene_purpose, *module.educationalObjectives]))
        self.modules.append(ModuleSummary(
            index=index,
            module_name=module.moduleName,
            description=module.description,
            object_names=[o.name for o in module.objects],
            num_clips=len(module.clips),
            covered=covered,
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
                "their exact PascalCase name when they fit):"
            )
            lines += [f"- {c.render()}" for c in new]
        return "\n".join(lines)

    def is_known_asset(self, prefab: str | None = None, texture: str | None = None) -> bool:
        """True if the prefab/texture is in the moon-lab library or was invented by
        an earlier module (case-insensitive, same rule as lab_metrics)."""
        if prefab is not None:
            return prefab.strip().lower() in self._known_prefabs
        if texture is not None:
            return texture.strip().lower() in self._known_textures
        return True

    def render_assets(self) -> str:
        def _fmt(items: dict[str, InventedAsset]) -> str:
            return "; ".join(a.render(n) for n, a in items.items()) or "none yet"
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

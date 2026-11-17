"""
Builder plug-ins for provisioning (spec: PROVISIONING.md, "Builders").

A builder makes an asset exactly from data (a molecule from PubChem, a protein from
the Protein Data Bank). Each one describes, in plain words, what it can make, what it
needs and which options it takes. The describe-and-route step (``routing.py``, #106)
offers the LLM the registered builders by those descriptions, as routes named
``builder:<name>``, and asks for the builder's input (``builder_input``) and options
(``builder_options``); adding a builder never needs a routing change. Builders are
general by data source, never written for one lab topic.

The contract (``BuilderInfo.build``): ``build(request, out_dir) -> BuildResult``.
It writes a GLB (and a thumbnail) into ``out_dir`` and returns them with the library
entry to store them under (provenance included), or ``ok=False`` with a reason, so
the asset falls through to search (#110) or the composite builder (#112). The same
input and data give the same files.

Registered builders (``registry()``): ``molecule`` (#111, ``molecule.py``). Builders
load lazily, so routing works without their dependencies (trimesh) installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class BuilderInfo:
    name: str          # route name is "builder:<name>"
    kind: str          # "model" or "texture"
    description: str   # what it can make, for the route prompt
    needs: str         # what the route step must supply as builder_input
    # (option, what it does): what the route step may ask for in builder_options
    options: tuple[tuple[str, str], ...] = ()
    build: Optional[Callable[["BuildRequest", Path], "BuildResult"]] = None
    # Bumped whenever the builder's output changes; recorded in the entries it builds
    # (Provenance.builder_version), so build_assets --save replaces an entry an older
    # version made (never a newer or a reviewed one).
    version: int = 1

    def option_names(self) -> tuple[str, ...]:
        return tuple(o for o, _ in self.options)

    def route_text(self) -> str:
        """How the route prompt offers this builder."""
        text = f"{self.description} Needs: {self.needs}."
        if self.options:
            text += " Options: " + "; ".join(f"{o}: {d}" for o, d in self.options) + "."
        return text


@dataclass(frozen=True)
class BuildRequest:
    name: str                          # the asset's name in the lab
    kind: str                          # "model" or "texture"
    input: str                         # builder_input from the plan
    options: tuple[str, ...] = ()      # builder_options from the plan
    description: str = ""              # the plan's description
    must_be_right: tuple[str, ...] = ()


@dataclass
class BuildResult:
    ok: bool
    reason: str = ""                                     # why not, when not ok
    files: dict[str, Path] = field(default_factory=dict)  # library role -> file in out_dir
    entry: Optional[dict] = None                          # library.LibraryEntry fields
    details: dict = field(default_factory=dict)           # builder facts, for the manifest


REGISTRY: list[BuilderInfo] = []
# Modules whose ``INFO`` is registered on first use of ``registry()``.
DEFAULT_BUILDERS = ("molecule",)
_defaults_loaded = False


def register(info: BuilderInfo) -> BuilderInfo:
    if any(b.name == info.name for b in REGISTRY):
        raise ValueError(f"builder {info.name!r} is already registered")
    REGISTRY.append(info)
    return info


def registry() -> list[BuilderInfo]:
    """Every registered builder, the default ones included."""
    global _defaults_loaded
    if not _defaults_loaded:
        for module in DEFAULT_BUILDERS:   # an import error propagates; the next call retries
            info = importlib.import_module(module).INFO
            if not any(b.name == info.name for b in REGISTRY):
                REGISTRY.append(info)
        _defaults_loaded = True
    return REGISTRY


def get(name: str) -> Optional[BuilderInfo]:
    """The builder for a route name (``molecule`` or ``builder:molecule``)."""
    name = name.removeprefix("builder:")
    return next((b for b in registry() if b.name == name), None)

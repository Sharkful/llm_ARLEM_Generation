"""
Builder plug-ins for provisioning (spec: PROVISIONING.md, "Builders").

A builder makes an asset exactly from data (a molecule from PubChem, a protein from
the Protein Data Bank). Each one describes, in plain words, what it can make and
what it needs. The describe-and-route step (``routing.py``, #106) offers the LLM
the registered builders by those descriptions, as routes named ``builder:<name>``,
so adding a builder never needs a routing change. Builders are general by data
source, never written for one lab topic.

The registry is empty until the first builder lands (#111); until then structures
route to search or composite.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuilderInfo:
    name: str          # route name is "builder:<name>"
    kind: str          # "model" or "texture"
    description: str   # what it can make, for the route prompt
    needs: str         # what the route step must supply (e.g. "the compound's name")


REGISTRY: list[BuilderInfo] = []


def register(info: BuilderInfo) -> BuilderInfo:
    if any(b.name == info.name for b in REGISTRY):
        raise ValueError(f"builder {info.name!r} is already registered")
    REGISTRY.append(info)
    return info

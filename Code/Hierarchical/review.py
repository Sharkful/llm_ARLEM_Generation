"""
Checkpoint review interface for hierarchical generation.

After every step (the lab plan, each module's asset plan, each module) the
pipeline hands a ``Checkpoint`` to a ``Reviewer`` and acts on its decision:

- ``Approve()``           accept the output and move on
- ``Revise(feedback)``    regenerate this step with the previous output and the
                          feedback appended to the conversation
- ``Replace(output)``     accept an edited output instead (dict or model); it is
                          re-validated against the step's response model

``AutoApproveReviewer`` approves everything, so a run goes end-to-end unattended.
Human-in-the-loop and LLM self-critique reviewers implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union

from pydantic import BaseModel


@dataclass
class Checkpoint:
    step: str                      # "plan" | "asset_plan" | "module"
    index: Optional[int]           # 1-based module index; None for the lab plan
    output: BaseModel              # the step's validated output
    revision: int = 0              # how many revisions this step has had so far
    context: dict[str, Any] = field(default_factory=dict)  # e.g. scene, asset plan


@dataclass
class Approve:
    note: Optional[str] = None


@dataclass
class Revise:
    feedback: str


@dataclass
class Replace:
    output: Union[BaseModel, dict]
    note: Optional[str] = None


ReviewDecision = Union[Approve, Revise, Replace]


class Reviewer(Protocol):
    name: str

    def review(self, checkpoint: Checkpoint) -> ReviewDecision: ...


class AutoApproveReviewer:
    """Approves every checkpoint; the default for unattended test runs."""

    name = "auto-approve"

    def review(self, checkpoint: Checkpoint) -> ReviewDecision:
        return Approve()

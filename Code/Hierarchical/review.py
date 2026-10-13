"""
Checkpoint review interface for hierarchical generation.

After every step (the lab plan, each module's asset plan, each module) the
pipeline hands a ``Checkpoint`` to a ``Reviewer`` and acts on its decision:

- ``Approve()``           accept the output and move on
- ``Revise(feedback)``    regenerate this step with the previous output and the
                          feedback appended to the conversation
- ``Replace(output)``     accept an edited output instead (dict or model); it is
                          re-validated against the step's response model. An edit
                          that fails validation comes back to the reviewer with
                          ``last_error`` (every validation error at once) and
                          ``rejected_output``, and counts as a revision round.

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
    output: BaseModel              # the step's current validated output
    revision: int = 0              # how many revision rounds this step has had so far
    # e.g. scene, asset plan, approved plan, earlier-module summaries, run_dir
    context: dict[str, Any] = field(default_factory=dict)
    feedback_history: list[str] = field(default_factory=list)  # every Revise so far
    # Set when the reviewer's last Replace failed validation: all of its errors,
    # and the edit itself, so it can be fixed rather than redone.
    last_error: Optional[str] = None
    rejected_output: Optional[dict] = None


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

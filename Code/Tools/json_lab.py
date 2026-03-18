"""
AR Lab Pydantic Data Models (v2)

Designed for use with Instructor to enable LLM-based lab generation.
Matches the v2 JSON schema produced by convert_lab.py.

Usage with Instructor:
    import instructor
    from openai import OpenAI  # or anthropic
    from lab_models import Lab

    client = instructor.from_openai(OpenAI())
    lab = client.chat.completions.create(
        model="gpt-4o",
        response_model=Lab,
        messages=[{"role": "user", "content": "Create an AR lab about..."}],
    )
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema


# ── Primitive Types ───────────────────────────────────────────────────

Vec3 = Annotated[
    list[Union[int, float]],
    Field(
        min_length=3,
        max_length=3,
        description="3D vector as [x, y, z]",
    ),
]

Color4 = Annotated[
    list[Union[int, float]],
    Field(
        min_length=4,
        max_length=4,
        description="RGBA color as [r, g, b, a], each 0-255",
    ),
]


# ── Enums ─────────────────────────────────────────────────────────────

class ModuleType(str, Enum):
    """Types of activity modules."""

    DEMO = "demo"


# ── Components ────────────────────────────────────────────────────────
# Each component is a separate model with a Literal discriminator so
# that Pydantic (and Instructor) can resolve the correct type from
# the "componentType" field. Add new component types here as needed.


class TextMeshProComponent(BaseModel):
    """Renders 3D text via TextMeshPro."""

    componentType: Literal["textMeshPro"] = "textMeshPro"
    text: str = Field(description="The displayed text content")
    color: Optional[Color4] = Field(
        default=None, description="Text color in RGBA"
    )
    fontSize: Optional[Union[int, float]] = Field(
        default=None, description="Font size in world-space units"
    )
    wrapText: Optional[bool] = Field(
        default=None, description="Whether to wrap text"
    )


class SimpleRotationComponent(BaseModel):
    """Continuously rotates an object around its Y axis."""

    componentType: Literal["simpleRotation"] = "simpleRotation"
    timeRate: float = Field(
        default=1.0, description="Speed multiplier for the rotation"
    )
    rotationTime: Union[int, float] = Field(
        description="Duration of one full rotation in seconds"
    )


class SimpleOrbitComponent(BaseModel):
    """Orbits an object around a center point."""

    componentType: Literal["simpleOrbit"] = "simpleOrbit"
    initialPosition: Vec3 = Field(
        description="Initial orbital position relative to the orbit center as [x, y, z]"
    )
    orbitalPeriod: float = Field(
        description="Time for one full orbit in seconds"
    )
    timeRate: float = Field(
        default=1.0, description="Speed multiplier for the orbit"
    )
    orbitalScale: float = Field(
        default=1.0, description="Scale factor for the orbital radius"
    )
    synchronousRotation: bool = Field(
        default=True,
        description="Whether the object keeps the same face toward the center (tidally locked)",
    )


class CheckAngleComponent(BaseModel):
    """Checks if an object is oriented at a target angle and triggers a callback on success."""

    componentType: Literal["checkAngle"] = "checkAngle"
    azmTarget: int = Field(
        description="Target azimuth angle in degrees (0-360)"
    )
    azmTolerance: int = Field(
        default=30, description="Acceptable deviation from target azimuth"
    )
    altTarget: int = Field(
        description="Target altitude angle in degrees"
    )
    altTolerance: int = Field(
        default=30, description="Acceptable deviation from target altitude"
    )
    audioClipSuccess: Optional[str] = Field(
        default=None, description="Audio clip to play on success"
    )
    callBackObjects: Optional[list[str]] = Field(
        default=None,
        description="Names of objects to notify on success",
    )


class RigidBodyComponent(BaseModel):
    """Configures Unity physics on the object."""

    componentType: Literal["rigidBody"] = "rigidBody"
    mass: float = Field(default=1.0)
    drag: float = Field(default=0.0)
    angularDrag: float = Field(default=0.0)
    isKinematic: bool = Field(
        default=True,
        description="If true, object is not driven by the physics engine",
    )
    useGravity: bool = Field(default=False)
    xConstraint: bool = Field(default=False)
    yConstraint: bool = Field(default=False)
    zConstraint: bool = Field(default=False)
    xRotationConstraint: bool = Field(default=False)
    yRotationConstraint: bool = Field(default=False)
    zRotationConstraint: bool = Field(default=False)


class PointerReceiverComponent(BaseModel):
    """Makes an object interactable via VR pointer/controller."""

    componentType: Literal["pointerReceiver"] = "pointerReceiver"
    draggable: bool = Field(
        default=False, description="Whether the user can grab and move this object"
    )
    kinematicWhileIdle: bool = Field(
        default=True,
        description="Keep object kinematic when not being dragged",
    )
    faceWhileDragging: bool = Field(default=False)
    matchWallWhileDragging: bool = Field(default=False)
    invertForward: bool = Field(default=False)


# Discriminated union of all component types.
# Instructor uses this to resolve the correct subtype from componentType.
Component = Annotated[
    Union[
        TextMeshProComponent,
        SimpleRotationComponent,
        SimpleOrbitComponent,
        CheckAngleComponent,
        RigidBodyComponent,
        PointerReceiverComponent,
    ],
    Field(discriminator="componentType"),
]


# ── Scene Objects ─────────────────────────────────────────────────────


class SceneObject(BaseModel):
    """An object instantiated in the 3D scene from a prefab.

    Every object has a prefab type, a unique name, and a position/scale.
    Optional fields like texture, color, and components extend behavior.
    """

    prefab: str = Field(
        description=(
            "Name of the prefab to instantiate. Known built-in prefabs: "
            "'SunPrefab', 'moveableSphere', 'clickableSphere', 'tinySphere', "
            "'textPrefab', 'robotIdle'. For new topics, invent a descriptive "
            "prefab name that matches the subject matter (e.g. 'earthPrefab', "
            "'dnaStrandPrefab', 'atomPrefab')."
        )
    )
    name: str = Field(
        description="Unique identifier for this object within the module"
    )
    parent: Optional[str] = Field(
        default=None,
        description='Parent object name. Use "[CURRENT_LAB]" for the lab root.',
    )
    position: Vec3 = Field(
        description="Local position relative to parent as [x, y, z]"
    )
    eulerAngles: Optional[Vec3] = Field(
        default=None,
        description="Local rotation in euler angles [x, y, z] degrees. Defaults to [0, 0, 0].",
    )
    scale: Vec3 = Field(
        description="Local scale as [x, y, z]"
    )
    enabled: Optional[bool] = Field(
        default=None,
        description="Whether the object starts enabled. Defaults to true.",
    )
    texture: Optional[str] = Field(
        default=None,
        description="Texture resource name to apply (e.g. '2k_earth_daymap', '2k_moon')",
    )
    color: Optional[Color4] = Field(
        default=None, description="Object color in RGBA"
    )
    components: Optional[list[Component]] = Field(
        default=None,
        description="Behavioral components attached to this object",
    )


# ── Clip Changes (Sparse Deltas) ─────────────────────────────────────


class ObjectChange(BaseModel):
    """A sparse delta update to an existing scene object within a clip.

    Only include fields that are actually changing. The target field
    identifies which object to update by name.
    """

    target: str = Field(
        description="Name of the object to modify (must match a SceneObject.name)"
    )
    position: Optional[Vec3] = Field(
        default=None, description="New local position [x, y, z]"
    )
    eulerAngles: Optional[Vec3] = Field(
        default=None, description="New rotation [x, y, z] degrees"
    )
    scale: Optional[Vec3] = Field(
        default=None, description="New scale [x, y, z]"
    )
    enabled: Optional[bool] = Field(
        default=None, description="Enable or disable the object"
    )
    color: Optional[Color4] = Field(
        default=None, description="New object color in RGBA"
    )
    components: Optional[list[Component]] = Field(
        default=None,
        description="Components to add or update on this object",
    )


# ── Clips ─────────────────────────────────────────────────────────────


class Clip(BaseModel):
    """A single step in the module's timeline.

    Each clip can play audio, auto-advance to the next clip, and apply
    a set of sparse changes to scene objects.
    """

    clipName: str = Field(
        description="Unique clip identifier (e.g. 'clip0', 'clip1')"
    )
    audioClip: Optional[str] = Field(
        default=None,
        description="Audio resource name to play during this clip",
    )
    autoAdvance: bool = Field(
        default=False,
        description="If true, automatically advance to the next clip when audio/timer finishes",
    )
    changes: Optional[list[ObjectChange]] = Field(
        default=None,
        description="Sparse updates to apply to scene objects at the start of this clip",
    )


# ── Activity Modules ──────────────────────────────────────────────────


class DemoModule(BaseModel):
    """A demonstration/guided walkthrough module.

    Presents a sequence of clips that narrate content while manipulating
    3D objects in the scene. The student navigates via Next/Previous buttons
    (injected automatically at runtime).
    """

    moduleType: SkipJsonSchema[Literal["demo"]] = "demo"
    prefab: SkipJsonSchema[Literal["demoPrefab"]] = "demoPrefab"
    moduleName: str = Field(
        description="Display name of this module"
    )
    description: str = Field(
        description="Short summary of what this module covers"
    )
    author: str = Field(description="Module author name")
    institution: Optional[str] = Field(
        default=None, description="Author's institution"
    )
    dateCreated: Optional[str] = Field(
        default=None, description="Creation date"
    )
    educationalObjectives: list[str] = Field(
        description="Learning objectives for this module"
    )
    instructions: Optional[list[str]] = Field(
        default=None,
        description="Instructions shown to the student at the start of the module",
    )
    objects: list[SceneObject] = Field(
        description="Initial objects to spawn when the module starts"
    )
    clips: list[Clip] = Field(
        description="Ordered sequence of clips forming the module timeline"
    )

    @model_validator(mode="after")
    def validate_object_references(self) -> "DemoModule":
        """Validate that all ObjectChange targets reference a defined SceneObject name,
        and that SceneObject names are unique within the module."""
        names = [obj.name for obj in self.objects]
        seen: set[str] = set()
        for name in names:
            if name in seen:
                raise ValueError(f"Duplicate SceneObject name '{name}' in module '{self.moduleName}'")
            seen.add(name)

        for clip in self.clips:
            if clip.changes:
                for change in clip.changes:
                    if change.target not in seen:
                        raise ValueError(
                            f"Clip '{clip.clipName}' references unknown object '{change.target}'. "
                            f"Valid names: {sorted(seen)}"
                        )
        return self


# Currently only DemoModule exists. If more module types are added,
# restore the discriminated union on "moduleType".
Module = DemoModule


# ── Top-Level Lab ─────────────────────────────────────────────────────


class Lab(BaseModel):
    """A complete AR lab experience.

    Contains metadata about the course and a sequence of activity modules
    that the student progresses through.
    """

    version: SkipJsonSchema[Literal["2.0"]] = "2.0"
    labId: str = Field(description="Unique identifier for this lab")
    author: str = Field(description="Lab author name")
    courseName: str = Field(description="Name of the course this lab belongs to")
    estimatedLength: Optional[str] = Field(
        default=None, description="Estimated time to complete (e.g. '30 minutes')"
    )
    objectives: list[str] = Field(
        description="High-level learning objectives for the entire lab"
    )
    modules: list[Module] = Field(
        description="Ordered sequence of activity modules"
    )

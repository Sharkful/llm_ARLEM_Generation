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

from datetime import date
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema


# ── Primitive Types ───────────────────────────────────────────────────

def _coerce_number_list(v):
    """Normalize an LLM-emitted numeric vector before list parsing.

    Handles two failure modes seen in the sweeps:
      * a whole vector crammed into one comma-joined string
        (``"-1.05,2.0,-1.4"`` -> ``[-1.05, 2.0, -1.4]``), and
      * individual coordinates emitted as numeric strings
        (``"-0.82"`` -> ``-0.82``), which Pydantic's smart ``Union[int, float]``
        rejects rather than coerces.

    Anything it can't parse is passed through untouched so the normal list
    validator still raises a clean error."""
    if isinstance(v, str):
        v = v.split(",")
    elif (
        isinstance(v, (list, tuple))
        and len(v) == 1
        and isinstance(v[0], str)
        and "," in v[0]
    ):
        v = v[0].split(",")
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, str):
                try:
                    x = float(x.strip())
                except ValueError:
                    pass
            out.append(x)
        return out
    return v


Vec3 = Annotated[
    list[Union[int, float]],
    BeforeValidator(_coerce_number_list),
    Field(
        min_length=3,
        max_length=3,
        description="3D vector as [x, y, z]",
    ),
]

Color4 = Annotated[
    list[Union[int, float]],
    BeforeValidator(_coerce_number_list),
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
# Each component is a separate model tagged by a Literal `type` field. Two
# tag fields are kept in lockstep, one per consumer:
#
#   type           - LLM-facing. Literal matches the Python class name exactly
#                    (e.g. "TextMeshProComponent"), the naming the model expects.
#                    Pydantic smart-union resolution selects the subtype on THIS
#                    field (see the `Component` union below). Excluded from
#                    serialization (exclude=True), so it never reaches the headset.
#   componentType  - Headset-facing. Literal is the Unity component identifier
#                    (e.g. "textMeshPro") that the device deserializer and the
#                    original-lab reference data use. Hidden from the LLM schema
#                    (SkipJsonSchema) and auto-filled from its default, so the
#                    model never sees or sets it.
#
# Net effect: the model reads/writes `type` with class-name values; serialized
# output carries only `componentType` with the canonical Unity value.
# Each component subclasses ConstToEnumSchemaMixin (below) so its single-value
# `type` Literal is emitted as a JSON-Schema enum, which Gemini requires.
# Add new component types here as needed (keep both literals in sync).


class ConstToEnumSchemaMixin(BaseModel):
    """Emit single-value ``Literal`` fields as JSON-Schema ``enum`` instead of ``const``.

    This conversion exists to satisfy Gemini. Pydantic v2 renders a one-value
    ``Literal`` (e.g. our ``type`` tag) as ``{"const": "X"}``. Google's
    google-genai SDK builds function-calling tool schemas through a strict
    ``types.Schema`` model that forbids the ``const`` keyword, so a ``const``
    field makes Gemini's ``GENAI_TOOLS`` path reject the whole schema with
    "Extra inputs are not permitted". The one-element form ``{"enum": ["X"]}``
    is semantically identical and is what Gemini accepts.

    Because OpenAI and Anthropic accept ``enum`` and ``const`` interchangeably,
    doing the swap on the model itself (rather than per provider) is safe and
    lets a single shared schema go to every provider unchanged. The hook below
    is provider-agnostic: it runs whenever this model's JSON schema is built,
    so all providers receive the ``enum`` form.
    """

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        # 1. Run Pydantic's normal schema generation. Returns this model's JSON
        #    Schema dict exactly as it would be emitted by default, with the
        #    single-value `type` Literal rendered as {"const": "<ClassName>"}.
        schema = handler(core_schema)
        # 2. Rewrite any single-value const on this model's own fields into the
        #    equivalent one-element enum. For our components the only matching
        #    field is the `type` tag; every other field is left untouched. The
        #    value is read out of the existing `const`, never restated, so it
        #    cannot drift from the `Literal`.
        for prop in schema.get("properties", {}).values():
            if "const" in prop:
                prop["enum"] = [prop.pop("const")]
        # 3. Return the modified schema. This fires once per model that inherits
        #    the mixin, so each component is converted as `Lab` recurses into it.
        return schema


class TextMeshProComponent(ConstToEnumSchemaMixin):
    """Renders 3D text via TextMeshPro."""

    type: Literal["TextMeshProComponent"] = Field("TextMeshProComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["textMeshPro"]] = "textMeshPro"
    text: str = Field(description="The displayed text content")
    color: Optional[Color4] = Field(
        default=None, description="Text color in RGBA"
    )
    fontSize: Optional[int] = Field(
        ge=20,
        le=80,
        default=30,
        description="72pt font is 1m tall in the lab. The default value is 30. The value chosen must be an integer greater than 20 and less than 80"
    )
    wrapText: Optional[bool] = Field(
        default=None, description="Whether to wrap text"
    )

    @field_validator("fontSize", mode="before")
    @classmethod
    def _coerce_fontsize(cls, v):
        """Cosmetic field: silently round float/numeric input to int and clamp
        into [20, 80] rather than forcing an instructor retry. Non-numeric input
        is passed through untouched so the standard int validator emits a clean
        error. See sweep 2-3 error analysis: fontSize was the #2 retry driver,
        and the range is arbitrary enough that a quiet clamp beats a paid retry."""
        if v is None:
            return v
        try:
            v = round(float(v))
        except (TypeError, ValueError):
            return v
        return max(20, min(80, v))


class SimpleRotationComponent(ConstToEnumSchemaMixin):
    """Continuously rotates an object around its Y axis."""

    type: Literal["SimpleRotationComponent"] = Field("SimpleRotationComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["simpleRotation"]] = "simpleRotation"
    timeRate: float = Field(
        default=1.0, description="Speed multiplier for the rotation"
    )
    rotationTime: Union[int, float] = Field(
        description=(
            "Duration of one full 360 rotation, in SECONDS (a time, not a "
            "rotational speed). Larger = slower. e.g. 8 means the object "
            "completes one rotation every 8 seconds. Do not provide a speed "
            "in degrees/second or an angular-velocity vector here."
        )
    )


class SimpleOrbitComponent(ConstToEnumSchemaMixin):
    """Orbits an object around a center point."""

    type: Literal["SimpleOrbitComponent"] = Field("SimpleOrbitComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["simpleOrbit"]] = "simpleOrbit"
    initialPosition: Vec3 = Field(
        description="Initial orbital position relative to the orbit center as [x, y, z]"
    )
    orbitalPeriod: float = Field(
        # Models name this `period` (values [20, 65, ...]) and `periodSeconds`
        # ([12, 30, 90, ...]) as often as `orbitalPeriod` — same concept, same
        # seconds unit, so accept them. (NB: `startAngle`/`speed` are NOT aliased
        # — those are a different, angle/speed-based mental model, not a period.)
        validation_alias=AliasChoices("orbitalPeriod", "period", "periodSeconds"),
        description="Time for one full orbit in seconds",
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


class CheckAngleComponent(ConstToEnumSchemaMixin):
    """Checks if an object is oriented at a target angle and triggers a callback on success."""

    type: Literal["CheckAngleComponent"] = Field("CheckAngleComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["checkAngle"]] = "checkAngle"
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


class RigidBodyComponent(ConstToEnumSchemaMixin):
    """Configures Unity physics on the object."""

    type: Literal["RigidBodyComponent"] = Field("RigidBodyComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["rigidBody"]] = "rigidBody"
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


class PointerReceiverComponent(ConstToEnumSchemaMixin):
    """Makes an object interactable via VR pointer/controller."""

    type: Literal["PointerReceiverComponent"] = Field("PointerReceiverComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["pointerReceiver"]] = "pointerReceiver"
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

class NewComponent(ConstToEnumSchemaMixin):
    """A name and description for a new C# Monobehavior component that should implement functionality not covered in the other components"""

    type: Literal["NewComponent"] = Field("NewComponent", exclude=True)
    componentType: SkipJsonSchema[Literal["newscript"]] = "newscript"
    scriptName: str = Field(
        min_length=5,
        pattern=r'^\w+$',
        description="string name of the script file and the C# class it contains. should be all lowercase, alphanumeric, and without leading digits, minimum length 5"
    )
    scriptDescription: str = Field(
        min_length=20,
        description="""A description of what the component script does. It should note all required inputs, either references to other objects, or numeric values that can be set in the inspector.
        The description should also note the outputs and effects of the script."""
    )


# Plain (smart) union of all component types — no `discriminator=`.
# Pydantic v2 smart-union resolution selects the subtype on the LLM-facing
# `type` Literal (only the matching member validates), so this behaves like a
# discriminated union for well-formed output without emitting the JSON-Schema
# `oneOf` + `discriminator` that Gemini's schema translator rejects. Keeping it
# discriminator-free lets one shared model run on OpenAI, Anthropic, and Gemini.
# `componentType` stays hidden from the model.
Component = Union[
    TextMeshProComponent,
    SimpleRotationComponent,
    SimpleOrbitComponent,
    CheckAngleComponent,
    RigidBodyComponent,
    PointerReceiverComponent,
    NewComponent,
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
        description=
            """Texture resource filename to apply to this prefab.
            currently available textures: "2k_earth_daymap", "2k_moon", "2k_sun", "balldimpled"
            If you need a new texture that is not yet available, create a descriptive name for the file (e.g. 'Italian_Loaf_texture', 'Siamese_Cat_Texture')""",
    )
    color: Optional[Color4] = Field(
        default=None, description="Object color in RGBA, can set this instead of a texture, or to tint a texture."
    )
    components: Optional[list[Component]] = Field(
        default=None,
        description=
            """C# monobehavior components that alter behavior. Currently available components are in this schema, and include:
            TextMeshProComponent, SimpleRotationComponent, SimpleOrbitComponent, CheckAngleComponent, RigidBodyComponent, PointerReceiverComponent, NewComponent
            NewComponent is only to be used if you need new functionality that the other components cannot provide. You can add multiple components if needed.
            When creating a NewComponent, keep the scope simple. When possible split complex behavior into multiple smaller NewComponents that can be reused"""
    )


# ── Clip Changes (Sparse Deltas) ─────────────────────────────────────


class ObjectChange(BaseModel):
    """A sparse delta update to an existing scene object within a clip.

    Only include fields that are actually changing. The target field
    identifies which object to update by name.
    """

    target: str = Field(
        # Models reliably name this key differently — across the sweeps the
        # identifier showed up as `object` (101×) and `objectName` (99×) as
        # often as `target`/`name`. Accept all four rather than burn a retry
        # on key-naming drift (the deltas themselves are already correct).
        validation_alias=AliasChoices("target", "name", "object", "objectName"),
        description="Name of the object to modify (must match the name of an existing SceneObject)",
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
        description=
            """Audio resource name to play during this clip, typically for narrating information or instruction.
            Should be a descriptive name of what the content of the clip is, ie: "lab_introduction", or "topic_5_recap".
            Note there should be no file extension. If there is no need for audio this clip, leave as None"""
    )
    autoAdvance: bool = Field(
        default=False,
        description="If true, automatically advance to the next clip when audio/timer finishes",
    )
    changes: Optional[list[ObjectChange]] = Field(
        default=None,
        description="Sparse updates to apply to scene objects at the start of this clip",
    )
    changeMeaning: str = Field(
        min_length=30,
        description="A brief description of what changed in the scene from the last clip, and what its purpose was"
    )
    narration: Optional[str] = Field(
        default=None,
        description="plain text of what information is narrated during this clip. Can be None if there is no narration."
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
    author: SkipJsonSchema[str] = "A Robot"
    institution: SkipJsonSchema[str] = "MTSU"
    dateCreated: SkipJsonSchema[str] = str(date.today())
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
    author: SkipJsonSchema[str] = "A Robot"
    institution: SkipJsonSchema[str] = "MTSU"
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

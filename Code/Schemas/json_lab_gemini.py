"""
AR Lab Pydantic Data Models (v2) — Gemini-Compatible Flat Variant

Eliminates all Union types and discriminated unions so the schema
can be consumed by Google Gemini via Instructor.

Changes from json_lab.py:
  - Vec3 / Color4: list[float] instead of list[Union[int, float]]
    (the _coerce_number_list BeforeValidator is ported as-is)
  - Union[int, float] scalar fields → float
  - 7 separate component models → single flat Component with componentType enum
  - Module discriminated union → DemoModule used directly
  - Component fields from all types merged as Optional with componentType validator

Intentionally NOT ported from json_lab.py (Gemini-specific reasons):
  - The dual `type`/`componentType` discriminator split (json_lab.py): it exists
    only to fix discriminated-union resolution on OpenAI/Anthropic. This flat
    model has no union — the componentType enum already constrains the value and
    serializes the Unity id the headset wants.
  - validation_alias / AliasChoices (orbitalPeriod, ObjectChange.target): aliases
    interact unpredictably with GENAI_STRUCTURED_OUTPUTS strict schema generation,
    and Gemini's enum-constrained output is far less prone to key-naming drift.
  - the word-character regex pattern on scriptName: Gemini's controlled-generation
    schema subset does not support `pattern`; the rule is kept in the description.

Usage with Instructor + Gemini:
    import instructor, google.generativeai as genai
    from json_lab_gemini import Lab

    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name="gemini-2.5-flash"),
        use_async=False,
    )
    lab = client.create(response_model=Lab, messages=[...])
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


# ── Primitive Types ───────────────────────────────────────────────────
# Gemini cannot handle Union[int, float]; use float only.

def _coerce_number_list(v):
    """Normalize an LLM-emitted numeric vector before list parsing.

    Handles two failure modes seen in the sweeps:
      * a whole vector crammed into one comma-joined string
        (``"-1.05,2.0,-1.4"`` -> ``[-1.05, 2.0, -1.4]``), and
      * individual coordinates emitted as numeric strings
        (``"-0.82"`` -> ``-0.82``).

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
    list[float],
    BeforeValidator(_coerce_number_list),
    Field(
        min_length=3,
        max_length=3,
        description="3D vector as [x, y, z]",
    ),
]

Color4 = Annotated[
    list[float],
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


class ComponentType(str, Enum):
    """All supported component types (flat enum replaces discriminated union)."""
    TEXT_MESH_PRO = "textMeshPro"
    SIMPLE_ROTATION = "simpleRotation"
    SIMPLE_ORBIT = "simpleOrbit"
    CHECK_ANGLE = "checkAngle"
    RIGID_BODY = "rigidBody"
    POINTER_RECEIVER = "pointerReceiver"
    NEWSCRIPT = "newscript"


# ── Flat Component ───────────────────────────────────────────────────
# Merges all component types into a single model.
# Only the fields relevant to the chosen componentType should be set;
# all others should be left as None.


class Component(BaseModel):
    """A behavioral component attached to a scene object.

    This is a flat representation — set componentType, then fill in only
    the fields that belong to that type. Leave unrelated fields as None.
    """

    componentType: ComponentType = Field(
        description="Which component type this represents"
    )

    # ── TextMeshPro fields ───────────────────────────────────────────
    text: Optional[str] = Field(
        default=None,
        description="[textMeshPro] The displayed text content",
    )
    color: Optional[Color4] = Field(
        default=None,
        description="[textMeshPro] Text color in RGBA",
    )
    fontSize: Optional[int] = Field(
        default=None,
        ge=20,
        le=80,
        description="[textMeshPro] 72pt font is 1m tall in the lab. The default value is 30. The value chosen must be an integer greater than 20 and less than 80",
    )
    wrapText: Optional[bool] = Field(
        default=None,
        description="[textMeshPro] Whether to wrap text",
    )

    # ── SimpleRotation fields ────────────────────────────────────────
    timeRate: Optional[float] = Field(
        default=None,
        description="[simpleRotation, simpleOrbit] Speed multiplier",
    )
    rotationTime: Optional[float] = Field(
        default=None,
        description=(
            "[simpleRotation] Duration of one full 360 rotation, in SECONDS (a "
            "time, not a rotational speed). Larger = slower. e.g. 8 means the "
            "object completes one rotation every 8 seconds. Do not provide a "
            "speed in degrees/second or an angular-velocity vector here."
        ),
    )

    # ── SimpleOrbit fields ───────────────────────────────────────────
    initialPosition: Optional[Vec3] = Field(
        default=None,
        description="[simpleOrbit] Initial orbital position relative to center [x, y, z]",
    )
    orbitalPeriod: Optional[float] = Field(
        default=None,
        description="[simpleOrbit] Time for one full orbit in seconds",
    )
    orbitalScale: Optional[float] = Field(
        default=None,
        description="[simpleOrbit] Scale factor for the orbital radius",
    )
    synchronousRotation: Optional[bool] = Field(
        default=None,
        description="[simpleOrbit] Whether the object keeps the same face toward center (tidally locked)",
    )

    # ── CheckAngle fields ────────────────────────────────────────────
    azmTarget: Optional[int] = Field(
        default=None,
        description="[checkAngle] Target azimuth angle in degrees (0-360)",
    )
    azmTolerance: Optional[int] = Field(
        default=None,
        description="[checkAngle] Acceptable deviation from target azimuth",
    )
    altTarget: Optional[int] = Field(
        default=None,
        description="[checkAngle] Target altitude angle in degrees",
    )
    altTolerance: Optional[int] = Field(
        default=None,
        description="[checkAngle] Acceptable deviation from target altitude",
    )
    audioClipSuccess: Optional[str] = Field(
        default=None,
        description="[checkAngle] Audio clip to play on success",
    )
    callBackObjects: Optional[list[str]] = Field(
        default=None,
        description="[checkAngle] Names of objects to notify on success",
    )

    # ── RigidBody fields ─────────────────────────────────────────────
    mass: Optional[float] = Field(
        default=None, description="[rigidBody] Mass"
    )
    drag: Optional[float] = Field(
        default=None, description="[rigidBody] Drag"
    )
    angularDrag: Optional[float] = Field(
        default=None, description="[rigidBody] Angular drag"
    )
    isKinematic: Optional[bool] = Field(
        default=None,
        description="[rigidBody] If true, not driven by physics engine",
    )
    useGravity: Optional[bool] = Field(
        default=None, description="[rigidBody] Use gravity"
    )
    xConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")
    yConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")
    zConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")
    xRotationConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")
    yRotationConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")
    zRotationConstraint: Optional[bool] = Field(default=None, description="[rigidBody]")

    # ── PointerReceiver fields ───────────────────────────────────────
    draggable: Optional[bool] = Field(
        default=None,
        description="[pointerReceiver] Whether the user can grab and move this object",
    )
    kinematicWhileIdle: Optional[bool] = Field(
        default=None,
        description="[pointerReceiver] Keep object kinematic when not being dragged",
    )
    faceWhileDragging: Optional[bool] = Field(default=None, description="[pointerReceiver]")
    matchWallWhileDragging: Optional[bool] = Field(default=None, description="[pointerReceiver]")
    invertForward: Optional[bool] = Field(default=None, description="[pointerReceiver]")

    # ── NewComponent (newscript) fields ──────────────────────────────
    # newscript requests a brand-new C# MonoBehaviour for functionality the
    # other components cannot provide.
    scriptName: Optional[str] = Field(
        default=None,
        min_length=5,
        description="[newscript] Name of the script file and the C# class it contains. Should be all lowercase, alphanumeric (no spaces or symbols), without leading digits, minimum length 5",
    )
    scriptDescription: Optional[str] = Field(
        default=None,
        min_length=20,
        description="[newscript] A description of what the component script does. It should note all required inputs, either references to other objects, or numeric values that can be set in the inspector. The description should also note the outputs and effects of the script.",
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
            """C# monobehavior components that alter behavior. For each component set componentType to one of:
            textMeshPro, simpleRotation, simpleOrbit, checkAngle, rigidBody, pointerReceiver, newscript.
            Use 'newscript' only if you need new functionality the other components cannot provide (fill in scriptName and scriptDescription).
            You can add multiple components if needed. When creating a newscript, keep the scope simple. When possible split complex behavior into multiple smaller newscripts that can be reused""",
    )


# ── Clip Changes (Sparse Deltas) ─────────────────────────────────────


class ObjectChange(BaseModel):
    """A sparse delta update to an existing scene object within a clip.

    Only include fields that are actually changing. The target field
    identifies which object to update by name.
    """

    target: str = Field(
        description="Name of the object to modify (must match the name of an existing SceneObject)"
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
            Note there should be no file extension. If there is no need for audio this clip, leave as None""",
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
# No Module union needed — only DemoModule exists, used directly.


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
    modules: list[DemoModule] = Field(
        description="Ordered sequence of activity modules (demo type)"
    )

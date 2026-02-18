from typing import List, Optional, Literal, Union, Any
from pydantic import BaseModel, Field

# --- Basic Types ---

# Represents a 3D vector (x, y, z)
Vector3 = List[float]

# Represents a Color (R, G, B, A). 
# Note: The JSON shows inconsistent ranges (0-1.0 for objects, 0-255 for text).
# The model accepts float to accommodate both.
Color = List[float]

# --- Component Models ---

class SimpleRotationComponent(BaseModel):
    componentType: Literal["simpleRotation"]
    timeRate: float = Field(..., description="Multiplier for the rotation speed.")
    rotationTime: float = Field(..., description="Time in seconds for a full rotation.")

class SimpleOrbitComponent(BaseModel):
    componentType: Literal["simpleOrbit"]
    moonPosition: Vector3 = Field(..., description="The initial position of the orbiting body.")
    orbitalPeriod: float = Field(..., description="Time required for one full orbit.")
    timeRate: float = Field(..., description="Time simulation multiplier.")
    orbitalScale: float = Field(..., description="Scale factor for the orbit radius.")
    synchronousRotation: bool = Field(False, description="If true, the object rotates to face the center of the orbit (tidal locking).")

class TextMeshProComponent(BaseModel):
    componentType: Literal["textMeshPro"]
    text: str = Field(..., description="The content of the text label.")
    color: Color = Field(..., description="RGBA color values (0-255 scale observed in text components).")
    fontSize: float
    wrapText: bool

class RigidBodyComponent(BaseModel):
    componentType: Literal["rigidBody"]
    mass: float
    drag: float
    angularDrag: float
    isKinematic: bool
    useGravity: bool
    # Constraints
    xConstraint: bool = False
    yConstraint: bool = False
    zConstraint: bool = False
    xRotationConstraint: bool = False
    yRotationConstraint: bool = False
    zRotationConstraint: bool = False

class PointerReceiverComponent(BaseModel):
    componentType: Literal["pointerReceiver"]
    draggable: bool
    kinematicWhileIdle: bool
    faceWhileDragging: bool
    matchWallWhileDragging: bool
    invertForward: bool

class CheckAngleComponent(BaseModel):
    componentType: Literal["checkAngle"]
    audioClipSuccess: Optional[str] = None
    azmTarget: float = Field(..., description="Target Azimuth angle.")
    azmTolerance: float = Field(..., description="Allowed deviation from target azimuth.")
    altTarget: float = Field(..., description="Target Altitude angle.")
    altTolerance: float = Field(..., description="Allowed deviation from target altitude.")
    callBackObjects: Optional[List[str]] = Field(None, description="Names of objects to notify upon success.")

# Union of all possible components
Component = Union[
    SimpleRotationComponent,
    SimpleOrbitComponent,
    TextMeshProComponent,
    RigidBodyComponent,
    PointerReceiverComponent,
    CheckAngleComponent
]

# --- Scene Objects ---

class SceneObject(BaseModel):
    prefab: Literal[
        "SunPrefab", 
        "moveableSphere", 
        "clickableSphere", 
        "textPrefab", 
        "demoPrefab", 
        "robotIdle", 
        "tinySphere"
    ] = Field(..., description="The template identifier for the 3D object.")
    name: str = Field(..., description="Unique identifier for this object instance.")
    parent: str = Field(..., description="Name of the parent object or '[CURRENT_LAB]' for root.")
    position: Vector3
    scale: Optional[Vector3] = None
    eulerAngles: Optional[Vector3] = Field(None, description="Rotation in degrees (x, y, z).")
    texture: Optional[str] = Field(None, description="Texture filename (without extension).")
    color: Optional[Color] = Field(None, description="RGBA color override.")
    enabled: Optional[bool] = True
    components: Optional[List[Component]] = Field(None, description="List of behaviors attached to this object.")

# --- Animation/Logic Clips ---

class ObjectChange(BaseModel):
    """Defines a modification to an existing object during a clip step."""
    target: str = Field(..., description="The 'name' of the SceneObject to modify.")
    # Optional fields representing the partial update
    position: Optional[Vector3] = None
    eulerAngles: Optional[Vector3] = None
    scale: Optional[Vector3] = None
    color: Optional[Color] = None
    components: Optional[List[Component]] = Field(None, description="Updates or replaces specific components on the target.")

class Clip(BaseModel):
    clipName: str
    audioClip: Optional[str] = Field(None, description="Filename of the audio narration.")
    autoAdvance: bool = Field(False, description="If true, the lab proceeds to the next clip automatically after audio finishes.")
    changes: Optional[List[ObjectChange]] = Field(None, description="List of state changes to apply when this clip starts.")

# --- Modules ---

class Module(BaseModel):
    moduleType: Literal["demo"] = Field("demo", description="The classification of the module logic.")
    prefab: Literal["demoPrefab"] = "demoPrefab"
    moduleName: str
    description: str
    author: Optional[str] = None
    institution: Optional[str] = None
    dateCreated: Optional[str] = None
    educationalObjectives: List[str]
    instructions: List[str]
    objects: List[SceneObject]
    clips: List[Clip]

# --- Root Lab Structure ---

class ARLab(BaseModel):
    version: str = Field(..., description="Schema version (e.g., '2.0').")
    labId: str
    author: str
    courseName: str
    estimatedLength: str
    objectives: List[str]
    modules: List[Module]
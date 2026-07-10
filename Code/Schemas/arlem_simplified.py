
from typing import List, Optional, Literal, Set, Dict, Annotated
from pydantic import Field, field_validator, model_validator, BeforeValidator, AliasChoices

from _schema_helpers import ConstToEnumSchemaMixin, clamp_number

# Clamped scalar types: coerce numeric strings / int<->float and silently clamp
# *subjective* bounded values (positions, rotations, sizes, scale, volume) rather than
# forcing an instructor retry. See _schema_helpers.clamp_number.
_OffsetCm = Annotated[float, BeforeValidator(clamp_number(-1000.0, 1000.0))]
_AngleDeg = Annotated[float, BeforeValidator(clamp_number(0.0, 360.0))]
_SizeCm = Annotated[float, BeforeValidator(clamp_number(0.01, 1000.0))]
_Scale = Annotated[float, BeforeValidator(clamp_number(0.01, 1000.0))]
_Volume = Annotated[int, BeforeValidator(clamp_number(0.0, 100.0, as_int=True))]

# ==========================================
# PART 1: SIMPLIFIED WORKPLACE (Environment)
# ==========================================

class POI(ConstToEnumSchemaMixin):
    """
    Defines a specific point of interest on a physical object relative to its origin.
    """
    # Constraint: ID is Alphanumeric, Character String (100) 
    id: str = Field(..., max_length=100, description="Unique identifier for this specific point.")
    # Constraint: magnitude < 1000.00
    x_offset: _OffsetCm = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the x-axis in cm.")
    y_offset: _OffsetCm = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the y-axis in cm.")
    z_offset: _OffsetCm = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the z-axis in cm.")
    # Constraint: angles 0-360.0
    x_rotation: _AngleDeg = Field(0.0, ge=0.0, le=360.0, description="Pitch in Euler angles (degrees).")
    y_rotation: _AngleDeg = Field(0.0, ge=0.0, le=360.0, description="Yaw in Euler angles (degrees).")
    z_rotation: _AngleDeg = Field(0.0, ge=0.0, le=360.0, description="Roll in Euler angles (degrees).")

    @model_validator(mode='after')
    def validate_default_origin(self):
        """
        Enforce zero offset if Id is "default"
        """
        if self.id == "default":
            # Check for nonzero offsets
            if any([self.x_offset != 0, self.y_offset != 0, self.z_offset != 0]):
                raise ValueError(
                    "If POI id is 'default', it must be placed at the origin (0,0,0), rotation is allowed"
                )
        return self

class Detectable(ConstToEnumSchemaMixin):
    """
    Configuration for the computer vision system to recognize objects.
    """
    id: str = Field(..., max_length=100, description="Unique identifier referenced by Things, Places, or Persons.")
    type: Literal["marker", "anchor"] = Field(..., description="tracking method, 'marker' is an image target, 'anchor' spatial anchor relative to workplace origin")
    url: Optional[str] = Field(None, max_length=2000, description="URL to an asset bundle defining the image marker or anchor data. When generating new content, use a descriptive placeholder URL (e.g. 'https://assets.example.com/marker.bundle').")

class Tangible(ConstToEnumSchemaMixin):
    """
    place or thing in the workspace
    """
    id: str = Field(..., max_length=100, description="Unique identifier for the tangible")
    name: str = Field(..., max_length=1000, description="human-readable short description of place or thing; or name of the person")
    detectable: str = Field(None, max_length=100, description="ID of the Detectable for this tangible")
    pois: Optional[List[POI]] = Field(default_factory=list, description="List of Points of Interest for this tangible")

class Primitive(ConstToEnumSchemaMixin):
    """
    Definition of what augmentations are supported in this workspace: audio, video, models, animations, or text labels.
    Note, this does not correspond with specific instances, like an image, but defines if any kind of image display is supported
    """
    id: Literal["animation", "image", "video", "audio", "label"] = Field(..., description="Unique media type of the augmentation.")
    # Optional size fields for bounding boxes
    x_size: Optional[_SizeCm] = Field(None, ge=0.01, le=1000.00, description="image or object size on X axis (cm)")
    y_size: Optional[_SizeCm] = Field(None, ge=0.01, le=1000.00, description="image or object size on Y axis (cm)")
    z_size: Optional[_SizeCm] = Field(None, ge=0.01, le=1000.00, description="image or object size on Z axis (cm)")
    volume: Optional[_Volume] = Field(None, ge=0, le=100, description="Volume for audio primatives")

class Predicate(ConstToEnumSchemaMixin):
    """
    reusable instructional augmentations, like an image or animation, to be used in an activity
    """
    id: str = Field(..., max_length=100, description="Unique id for this instructional augmentation")
    type: Literal["animation", "image", "video", "audio", "label"] = Field(..., description="what type of primitive, instance of Primitive.id, cannot be a value not defined in the workplace primitives list")
    scale: _Scale = Field(1.0, ge=0.01, le=1000.00, description="normalized scale factor applied to primitive on all axes")
    url: Optional[str] = Field(None, max_length=1000, description="Source URL for the file (GLB, fbx, MP4, PNG, etc.). When generating new content, use a descriptive placeholder URL (e.g. 'https://assets.example.com/my_model.glb').")

class Workplace(ConstToEnumSchemaMixin):
    """
    The static environment definition containing all available resources.
    """
    id: str = Field(..., max_length=100, description="Unique identifier for this workplace")
    name: str = Field(..., max_length=1000, description="Descriptive name of the workplace")
    origin: str = Field(..., max_length=100, description="ID of the Detectable that serves as the World Origin")
    things: List[Tangible] = Field(default_factory=list, description="Physical tools, machines, or materials")
    places: List[Tangible] = Field(default_factory=list, description="Locations or zones in the workplace")
    detectables: List[Detectable] = Field(default_factory=list, min_length=1, description="Markers or Anchors used for tracking")
    primitives: List[Primitive] = Field(default_factory=list, min_length=1, description="Supported media types for predicates")
    predicates: List[Predicate] = Field(default_factory=list, min_length=1, description="A specific media primitive augmentation that corresponds to a verb or action")

    @model_validator(mode='after')
    def validate_workplace(self):
        """
        Checks that the origin has a valid detectable,
        Places, things, persons reference valid detectables,
        and predicates reference valid primitive types
        """
        # Get valid ids
        valid_detectable_ids = {d.id for d in self.detectables}
        valid_primitive_ids = {p.id for p in self.primitives}

        # Origin validation
        if self.origin not in valid_detectable_ids:
            raise ValueError(f"Workplace origin '{self.origin}' is not defined in the 'detectables' list")
        
        # Tangible validation
        # Helper function to validate list of tangibles
        def validate_tangible_list(items: list, tangible_type: str):
            # no two tangibles of  the same type should have the same detectable
            used_detectables_in_kind: Set[str] = set()

            for item in items:
                # Check existence
                if item.detectable not in valid_detectable_ids:
                    raise ValueError(
                        f"The {tangible_type} '{item.id}' references a missing detectable '{item.detectable}'."
                    )
                
                # Check uniqueness within type
                if item.detectable in used_detectables_in_kind:
                    raise ValueError(
                            f"The {tangible_type} {item.id} is trying to use the same detectable '{item.detectable}' as another {tangible_type}. "
                            f"Detectable IDs must be unique within the {tangible_type} list."
                        )
        
        validate_tangible_list(self.things, "Thing")
        validate_tangible_list(self.places, "Place")

        # Predicate Validation
        for predicate in self.predicates:
            if predicate.type not in valid_primitive_ids:
                raise ValueError(
                    f"Predicate '{predicate.id}' has type '{predicate.type}', which is not a defined Primitive ID."
                )
        return self

# ==========================================
# PART 2: LINEAR ACTIVITY (Logic)
# ==========================================

class Instruction(ConstToEnumSchemaMixin):
    """
    Text/Audio instructions displayed to the user.
    """
    title: str = Field(..., max_length=100, description="Short headline of the instruction")
    description: str = Field(..., max_length=5000, description="Narrative text describing what to do")

class Activate(ConstToEnumSchemaMixin):
    """
    Command to spawn an object or effect.
    """
    target: str = Field(..., max_length=100, description="ID of the Thing, Place, or Person to apply this to.")
    type: Literal["primitive", "predicate", "warning", "action"] = Field(..., description="Category of what is being activated: 'primitive' (a workplace primitive media type), 'predicate' (a reusable workplace augmentation), 'warning' (a hazard sign), or 'action' (launch another action).")
    augmentation: str = Field(..., max_length=100, description="ID of the predicate, primitive, or warning from the workspace, or another action ID, of what should be played / displayed")
    poi: Optional[str] = Field(None, max_length=100, description="ID of the POI on the tangible target where the augmentation appears.")
    url: Optional[str] = Field(None, max_length=2000, description="for when type='primitive' and augmentation='image', 'video', 'audio', or 'animation' the url of the resource to display. Use a descriptive placeholder URL when generating new content.")
    text: Optional[str] = Field(None, description="Text to display when augmentation is type 'label'")
    state: Optional[str] = Field(None, max_length=100, description="Keyframe or state ID used only for augments that are animations.")

class Deactivate(ConstToEnumSchemaMixin):
    """
    Command to remove an object or effect.
    """
    target: str = Field(..., max_length=100, description="ID of the target tangible to clear augmentations from, or '*' for all tangibles.")
    type: Optional[Literal["primitive", "predicate", "warning", "action", "*"]] = Field(None, description="Type of element to remove: 'primitive', 'predicate', 'warning', 'action', or '*' for all types.")
    augmentation: Optional[str] = Field(None, max_length=100, description="Specific augmentation ID to remove, or '*' for all augmentations on tangible")
    poi: Optional[str] = Field(None, max_length=100, description="POI from which to remove augmentations")

class Trigger(ConstToEnumSchemaMixin):
    """
    Specifies events that move the activity from enter to exit state.
    """
    mode: Literal["click", "detect"] = Field(..., description="Input method (UI click or object detection).")
    id: str = Field(..., max_length=100, description="ID of the entity to listen to (Action ID, Tangible ID, or Sensor ID).")
    duration: Optional[int] = Field(None, description="Time in ms required for gaze detection to trigger in ms")

class ActionFlow(ConstToEnumSchemaMixin):
    """
    The set of commands executed when entering or exiting a step.
    """
    remove_self: bool = Field(False, alias="removeSelf", description="For Enter: True means Exit is immidiately executed, For Exit: True deactivates instructions and augmentations from current step")
    activates: Optional[List[Activate]] = Field(default_factory=list, description="List of items to display.")
    deactivate: Optional[List[Deactivate]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("deactivate", "deactivates"),
        description="List of items to hide.",
    )

class Action(ConstToEnumSchemaMixin):
    """
    A single step in the activity workflow.
    """
    id: str = Field(..., max_length=100, description="Unique ID for this step.")
    instruction: Instruction = Field(..., description="Text displayed to the user during this step.")
    enter: Optional[ActionFlow] = Field(None, description="Commands executed immediately when this step starts.")
    exit: Optional[ActionFlow] = Field(None, description="Commands executed when the Trigger conditions are met.")
    triggers: List[Trigger] = Field(default_factory=list,
                                    description="Conditions that cause the step to complete.",
                                    examples=[[
                                        {
                                            "mode": "click",
                                            "type": "action",
                                            "viewport": "actions",
                                            "id": "start"
                                        }, {
                                            "mode": "voice",
                                            "type": "action",
                                            "viewport": "actions",
                                            "id": "start"
                                        }
                                    ], [
                                        {
                                            "id": "start",
                                            "mode": "click",
                                            "type": "action",
                                            "viewport": "actions"
                                        }, {
                                            "mode": "detect",
                                            "id": "board1",
                                            "type": "tangible",
                                            "duration": 3
                                        }
                                    ]

                                    ])

class Activity(ConstToEnumSchemaMixin):
    """
    The root object defining the logic flow of the AR experience.
    """
    id: str = Field(..., max_length=100, description="Unique ID for the activity.")
    name: str = Field(..., max_length=1000, description="Human-readable title.")
    description: str = Field(..., max_length=10000, description="Overview of the activity.")
    workplace: str = Field(..., max_length=100, description="id of the workplace definition to use for these activities")
    start: str = Field(..., max_length=100, description="ID of the first Action to execute.")
    actions: List[Action] = Field(default_factory=list, description="All available steps/states in this activity.")

    # Field Validator for actions, all ids should be unique
    @field_validator('actions')
    @classmethod
    def validate_actions(cls, actions):
        # Generate list of all ids
        list_ids = [action.id for action in actions]
        # remove duplicates
        set_ids = set()
        for id in list_ids:
            # Check if id is already used
            if id in set_ids:
                raise ValueError(
                    f"Activity id : '{id}' is not unique, more than one activity have the same id"
                )
            # If not, add to set of used ids
            set_ids.add(id)
        return actions 

    # Model validator for start field, should reference an existing action.id.
    @model_validator(mode='after')
    def validate_activity(self):
        # Get all valid ids
        valid_action_ids = {action.id for action in self.actions}

        # check that the start action matches
        if self.start not in valid_action_ids:
            raise ValueError(
                f"The start field of the activity references an invalid action.id : {self.start}"
            )
        
        # Check that each Action sub field is referencing other actions correctly
        for act_num, action in enumerate(self.actions):
            en = action.enter
            if en is None:
                continue
            # check activate augmentation of type 'action'
            for idx, act in enumerate(en.activates or []):
                if act.type == 'action' and act.augmentation not in valid_action_ids:
                    raise ValueError(
                        f"Action {act_num} referenced an invalid action id in activate {idx}"
                    )
            # Check deactivate augmentations of type 'action'. augmentation is
            # Optional (default None) and may be the documented '*' wildcard
            # ("all augmentations on tangible"); only a concrete, non-wildcard
            # value has to name a real action. This mirrors the scenario-level
            # validate_deactivate_list, which already encodes the same rule but
            # was unreachable behind this stricter check (review F3).
            for idx, deact in enumerate(en.deactivate or []):
                if (deact.type == 'action'
                        and deact.augmentation not in (None, '*')
                        and deact.augmentation not in valid_action_ids):
                    raise ValueError(
                        f"Action {act_num} referenced an invalid action id in deactivate {idx}"
                    )
        return self


# ==========================================
# PART 3: Container for Entire Scenario
# ==========================================

class ARLEMScenario(ConstToEnumSchemaMixin):
    """
    Container to hold both workplace and activity definitions
    allows cross validation of the activity with the workplace
    """
    workplace: Workplace
    activity: Activity

    # Validate Action fields that reference elements of the workplace
    @model_validator(mode='after')
    def validate_scenario(self):
        # (Simplified Action has no location or predicate fields; no per-action checks needed here)

        # Validate workplace id
        if self.activity.workplace != self.workplace.id:
            # Silently fix
            self.activity.workplace = self.workplace.id
        return self
        

    @model_validator(mode='after')
    def validate_activity_flows(self) -> 'ARLEMScenario':
        """
        Validates both 'Activate', 'Deactivate', and 'Message' commands in all Action flows.
        Handles strict ID checking and wildcard '*' permissions.
        """
        # Deeper validation
        # =========================================================
        # 1. INDEXING (O(1) Lookups)
        # =========================================================
        
        # Tangible Map: IDs -> Object Objects (for POI lookup)
        # Unioning Tangible, Place, and Person as they are all physical targets
        tangible_map: Dict[str, Tangible] = {}
        
        for t in self.workplace.things: tangible_map[t.id] = t
        for p in self.workplace.places: tangible_map[p.id] = p
        # ID Sets for Resource Categories
        ids_primitives = {p.id for p in self.workplace.primitives}
        ids_predicates = {p.id for p in self.workplace.predicates}
        ids_actions    = {a.id for a in self.activity.actions}

        # =========================================================
        # 3. HELPER: Validate 'Activate' (Strict - No Wildcards)
        # =========================================================
        def validate_activate_list(activates: List['Activate'], action_id: str):
            for i, act in enumerate(activates):
                err_ctx = f"Action '{action_id}' (Activate[{i}])"

                # Check Target — always a tangible (Thing/Place/Person), for every
                # type including 'action'. When type='action' the launched action id
                # lives in `augmentation`, not `target`; this aligns with the
                # per-Activity check and the Deactivate convention from PR #51
                # (review F4). Mirrors the arlem_full fix.
                if act.target not in tangible_map:
                    raise ValueError(f"{err_ctx}: Target '{act.target}' not found in Workplace.")

                # Check POI (target is a tangible)
                if act.poi and act.poi != "default":
                    target_obj = tangible_map[act.target]
                    # pois is Optional; explicit null must fail the POI check, not crash (#49)
                    valid_pois = {p.id for p in target_obj.pois or []}
                    if act.poi not in valid_pois:
                        raise ValueError(f"{err_ctx}: POI '{act.poi}' not found on target '{act.target}'.")

                # Check Augmentation ID against the workspace resource of the matching
                # type, or — for type='action' — another action in this Activity.
                if act.augmentation:
                    if act.type == 'primitive' and act.augmentation not in ids_primitives:
                        raise ValueError(f"{err_ctx}: Primitive '{act.augmentation}' not found.")
                    elif act.type == 'predicate' and act.augmentation not in ids_predicates:
                        raise ValueError(f"{err_ctx}: Predicate '{act.augmentation}' not found.")
                    elif act.type == 'action' and act.augmentation not in ids_actions:
                        raise ValueError(f"{err_ctx}: Action '{act.augmentation}' not found.")

        # =========================================================
        # 4. HELPER: Validate 'Deactivate' (Handles Wildcards '*')
        # =========================================================
        def validate_deactivate_list(deactivates: List['Deactivate'], action_id: str):
            for i, deact in enumerate(deactivates):
                err_ctx = f"Action '{action_id}' (Deactivate[{i}])"

                # --- A. Validate Target ---
                target_is_wildcard = (deact.target == '*')
                
                if not target_is_wildcard:
                    # Target must be an existing Tangible OR an Action
                    is_tangible = deact.target in tangible_map
                    is_action = deact.target in ids_actions
                    
                    if not (is_tangible or is_action):
                        raise ValueError(
                            f"{err_ctx}: Target '{deact.target}' is not a known Tangible or Action ID."
                        )
                
                # --- B. Validate POI (If your Deactivate model has POI) ---
                # Logic: You cannot specify a POI if you are wildcarding the target.
                if getattr(deact, 'poi', None):
                    if target_is_wildcard:
                        raise ValueError(f"{err_ctx}: Cannot specify POI with wildcard target '*'.")
                    
                    # If specific target, validate POI existence
                    if deact.target in tangible_map:
                        target_obj = tangible_map[deact.target]
                        valid_pois = {p.id for p in target_obj.pois or []}
                        if deact.poi != "default" and deact.poi not in valid_pois:
                            raise ValueError(f"{err_ctx}: POI '{deact.poi}' not found on '{deact.target}'.")

                # --- C. Validate Augmentation ID ---
                # Augmentation is optional. If present, check strictness unless it is '*'.
                aug_id = deact.augmentation
                aug_is_wildcard = (aug_id == '*')
                
                if aug_id and not aug_is_wildcard:
                    # We use 'type' to narrow down the check if possible
                    d_type = deact.type if (deact.type and deact.type != '*') else None
                    
                    found = False
                    if d_type == 'primitive':
                        found = aug_id in ids_primitives
                    elif d_type == 'predicate':
                        found = aug_id in ids_predicates
                    elif d_type == 'action':
                        found = aug_id in ids_actions
                    else:
                        # If type is '*' or None, the ID must exist in ANY valid list
                        found = (aug_id in ids_primitives or 
                                 aug_id in ids_predicates or 
                                 aug_id in ids_actions)
                    
                    if not found:
                        type_msg = f" (Type: {d_type})" if d_type else " (Any Type)"
                        raise ValueError(
                            f"{err_ctx}: Augmentation '{aug_id}'{type_msg} not found in Workplace."
                        )

        # =========================================================
        # 5. TRAVERSAL LOOP
        # =========================================================
        for action in self.activity.actions:
            # Validate 'Enter' Flow
            if action.enter:
                if action.enter.activates:
                    validate_activate_list(action.enter.activates, action.id)
                if action.enter.deactivate:
                    validate_deactivate_list(action.enter.deactivate, action.id)

            # Validate 'Exit' Flow
            if action.exit:
                if action.exit.activates:
                    validate_activate_list(action.exit.activates, action.id)
                if action.exit.deactivate:
                    validate_deactivate_list(action.exit.deactivate, action.id)

        return self
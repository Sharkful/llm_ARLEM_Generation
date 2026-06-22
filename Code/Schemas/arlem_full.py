from typing import List, Optional, Literal, Union, Set, Dict
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# ==========================================
# PART 1: WORKPLACE MODEL (The Environment)
# ==========================================

class POI(BaseModel):
    """
    Defines a specific point of interest on a physical object relative to its origin.
    """
    # Constraint: ID is Alphanumeric, Character String (100) 
    id: str = Field(..., max_length=100, description="Unique identifier for this specific point.")
    # Constraint: magnitude < 1000.00
    x_offset: float = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the x-axis in cm.")
    y_offset: float = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the y-axis in cm.")
    z_offset: float = Field(0.0, ge=-1000.00, le=1000.00, description="Offset on the z-axis in cm.")
    # Constraint: angles 0-360.0
    x_rotation: float = Field(0.0, ge=0.0, le=360.0, description="Pitch in Euler angles (degrees).")
    y_rotation: float = Field(0.0, ge=0.0, le=360.0, description="Yaw in Euler angles (degrees).")
    z_rotation: float = Field(0.0, ge=0.0, le=360.0, description="Roll in Euler angles (degrees).")

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

class Detectable(BaseModel):
    """
    Configuration for the computer vision system to recognize objects.
    """
    id: str = Field(..., max_length=100, description="Unique identifier referenced by Things, Places, or Persons.")
    type: Literal["marker", "anchor"] = Field(..., description="tracking method, 'marker' is an image target, 'anchor' spatial anchor relative to workplace origin")
    sensor: Optional[Literal["tracking", "mapping"]] = Field("tracking", description="sensor subsystem to use, image tracking or spatial mapping).")
    # Not sure if HttpUrl will work with the LLM filling it in
    url: Optional[HttpUrl] = Field(None, max_length=2000, description="URL to an asset bundle defining the defining image marker or anchor data.")

class Tangible(BaseModel):
    """
    place or thing in the workspace
    """
    id: str = Field(..., max_length=100, description="Unique identifier for the tangible")
    name: str = Field(..., max_length=1000, description="human-readable short description of place or thing; or name of the person")
    detectable: str = Field(None, max_length=100, description="ID of the Detectable for this tangible")
    pois: Optional[List[POI]] = Field(default_factory=list, description="List of Points of Interest for this tangible")

class Person(Tangible):
    """
    A human user or instructor in the workplace.
    """
    twitter: Optional[str] = Field(None, max_length=15, description="Twitter handle for the person.")
    mbox: Optional[str] = Field(None, max_length=254, description="Email or ID used for xAPI logging identification.")
    persona: Optional[str] = Field(None, max_length=1000, description="Role or group membership (e.g., 'learner', 'instructor').")

class SensorData(BaseModel):
    """
    Definition of a data stream variable.
    """
    key: str = Field(..., max_length=100, description="The variable identifier/topic in the data stream.")
    type: Optional[Literal["string", "float", "integer", "boolean"]] = Field("string", description="The data type of the variable")

class Sensor(BaseModel):
    """
    IoT sensor configuration for receiving external data.
    """
    id: str = Field(..., max_length=100, description="Unique identifier for the sensor.")
    url: str = Field(...,
                     pattern=r"^[a-zA-Z0-9+.-]+:\/\/[a-zA-Z0-9.-]+(:[0-9]{1,5})?$",
                     max_length=1000,
                     description="Connection URL, of the form 'protocol://host.domain[:port]'",
                     examples=["mqtt://test.mosquitto.org:1883"])
    username: Optional[str] = Field(None, max_length=100, description="Auth username for the sensor stream.")
    password: Optional[str] = Field(None, max_length=100, description="Auth password for the sensor stream.")
    data: List[SensorData] = Field(default_factory=list, min_items=1, description="List of data variables to listen for.")

class Device(BaseModel):
    """
    Hardware available for the experience (e.g., HoloLens, Tablet).
    """
    id: str = Field(..., max_length=100, description="Unique identifier for the device.")
    type: str = Field(..., max_length=100, description="Type of device (e.g., 'hololens', 'tablet').")
    name: str = Field(..., max_length=1000, description="Human-readable name of the device.")
    owner: Optional[str] = Field(None, max_length=100, description="Person ID of the device owner.")
    url: Optional[str] = Field(...,
                     pattern=r"^[a-zA-Z0-9+.-]+:\/\/[a-zA-Z0-9.-]+(:[0-9]{1,5})?$",
                     max_length=1000,
                     description="Connection URL, of the form 'protocol://host.domain[:port]'",
                     examples=["mqtt://test.mosquitto.org:1883"])
    topic: Optional[str] = Field(None, max_length=1000, description="MQTT channel name this device listens to.")
    username: Optional[str] = Field(None, max_length=100, description="Auth username")
    password: Optional[str] = Field(None, max_length=100, description="Auth password")

class App(BaseModel):
    """
    External web apps or widgets.
    """
    id: str = Field(..., max_length=100, description="Unique identifier for the app")
    type: Literal["widget", "app", "prefab"] = Field(..., description="Type of application: html widget, launch command, or app prefab")
    name: str = Field(..., max_length=100, description="Human-readable description")
    url: str = Field(...,
                     pattern=r"^[a-zA-Z0-9+.-]+:\/\/[a-zA-Z0-9.-]+$",
                     max_length=1000,
                     description="URL of the manifest file, launch command, or download link of app prefab",
                     examples=["mqtt://test.mosquitto.org:1883"])

class Primitive(BaseModel):
    """
    Definition of what augmentations are supported in this workspace: audio, video, models, animations, or text labels.
    Note, this does not correspond with specific instances, like an image, but defines if any kind of image display is supported
    """
    id: Literal["animation", "image", "video", "audio", "label"] = Field(..., description="Unique media type of the augmentation.")
    # Optional size fields for bounding boxes
    x_size: Optional[float] = Field(None, gte=0.01, lte=1000.00, description="image or object size on X axis (cm)")
    y_size: Optional[float] = Field(None, gte=0.01, lte=1000.00, description="image or object size on Y axis (cm)")
    z_size: Optional[float] = Field(None, gte=0.01, lte=1000.00, description="image or object size on Z axis (cm)")
    volume: Optional[int] = Field(None, gte=0, lte=100, description="Volume for audio primatives")

class Predicate(BaseModel):
    """
    reusable instructional augmentations, like an image or animation, to be used in an activity
    """
    id: str = Field(..., max_length=100, description="Unique id for this instructional augmentation")
    type: Literal["animation", "image", "video", "audio", "label"] = Field(..., description="what type of primitive, instance of Primitive.id, cannot be a value not defined in the workplace primitives list")
    scale: float = Field(1.0, gte=0.01, lte=1000.00, description="normalized scale factor applied to primitive on all axes")
    # Don't know if max_length str validator works with HttpUrl class
    url: HttpUrl = Field(None, max_length=1000, description="Source URL for the file (GLB, fbx, MP4, PNG, etc.). When generating new content, use a descriptive placeholder URL (e.g. 'https://assets.example.com/my_model.glb').")

class Warning(BaseModel):
    """
    ISO 7010 Hazard signs.
    """
    id: str = Field(..., max_length=100, description="Unique id for this instructional augmentation")
    type: Literal["animation", "image", "video", "audio", "label"] = Field(..., description="what type of primitive, instance of Primitive.id, cannot be a value not defined in the workplace primitives list")
    scale: float = Field(1.0, gte=0.01, lte=1000.00, description="normalized scale factor applied to primitive on all axes")
    symbol: str = Field(..., max_length=100, description="Name of the symbol prefab")

class Workplace(BaseModel):
    """
    The static environment definition containing all available resources.
    """
    id: str = Field(..., max_length=100, description="Unique identifier for this workplace")
    name: str = Field(..., max_length=1000, description="Descriptive name of the workplace")
    origin: str = Field(..., max_length=100, description="ID of the Detectable that serves as the World Origin")
    things: List[Tangible] = Field(default_factory=list, description="Physical tools, machines, or materials")
    places: List[Tangible] = Field(default_factory=list, description="Locations or zones in the workplace")
    persons: List[Person] = Field(default_factory=list, min_items=1, description="Users involved in the scenario")
    sensors: Optional[List[Sensor]] = Field(default_factory=list, description="Connected IoT devices")
    devices: List[Device] = Field(default_factory=list, min_items=1, description="AR hardware used for delivery")
    apps: Optional[List[App]] = Field(default_factory=list, description="External widgets or apps")
    detectables: List[Detectable] = Field(default_factory=list, min_items=1, description="Markers or Anchors used for tracking")
    primitives: List[Primitive] = Field(default_factory=list, min_items=1, description="Supported media types for predicates")
    predicates: List[Predicate] = Field(default_factory=list, min_items=1, description="A specific media primitive augmentation that corresponds to a verb or action")
    warnings: List[Warning] = Field(default_factory=list, description="Supported Warnings")

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
        validate_tangible_list(self.persons, "Person")

        # Predicate Validation
        for predicate in self.predicates:
            if predicate.type not in valid_primitive_ids:
                raise ValueError(
                    f"Predicate '{predicate.id}' has type '{predicate.type}', which is not a defined Primitive ID."
                )
        return self


# ==========================================
# PART 2: ACTIVITY MODEL (The Logic)
# ==========================================

class Instruction(BaseModel):
    """
    Text/Audio instructions displayed to the user.
    """
    title: str = Field(..., max_length=100, description="Short headline of the instruction")
    description: str = Field(..., max_length=5000, description="Narrative text describing what to do")

class Activate(BaseModel):
    """
    Command to spawn an object or effect.
    """
    target: str = Field(..., max_length=100, description="ID of the Thing, Place, or Person to apply this to.")
    type: Literal["primitive", "predicate", "warning", "action"] = Field(..., max_length=1000, description="Category of the object being activated.")
    augmentation: str = Field(..., max_length=100, description="ID of the predicate, primitive, or warning from the workspace, or another action ID, of what should be played / displayed")
    poi: Optional[str] = Field(None, max_length=100, description="ID of the POI on the tangible target where the augmentation appears.")
    url: Optional[HttpUrl] = Field(None, max_length=2000, description="for when type='primitive' and augmentation='image', 'video', 'audio', or 'animation' the url of the resource to display")
    text: Optional[str] = Field(None, description="Text to display when augmentation is type 'label'")
    state: Optional[str] = Field(None, max_length=100, description="Keyframe or state ID used only for augments that are animations.")
    viewport: Optional[Literal["actions", "reactions", "warnings"]] = Field(None, description="If set, attaches object to the screen/HUD instead of the world.")
    option: Optional[str] = Field(None, max_length=100, description="Configuration option (e.g. direction 'up' for a pointer).")

    # Need to validate that the target field matches a valid tangible
    # that augmentation matches a valid predicate, primitive, or warning from the workspace, or another action from the Activity
    # # # did activity id validation on the Activity model
    # that the poi points to a valid poi on the referenced target tangible
    # Should likely be done in the outermost container class

    # not including sensor key label that shows IoT sensor data variable over stream. that is too much

class Deactivate(BaseModel):
    """
    Command to remove an object or effect.
    """
    target: str = Field(..., max_length=100, description="ID of the target tangible to clear augmentations from, or '*' for all tangibles.")
    type: Optional[Literal["primitive", "predicate", "warning", "action", "*"]] = Field(None, max_length=100, description="Type of element to remove, '*' for all")
    augmentation: Optional[str] = Field(None, max_length=100, description="Specific augmentation ID to remove, or '*' for all augmentations on tangible")
    # These may not be needed
    poi: Optional[str] = Field(None, max_length=100, description="POI from which to remove augmentations")
    viewport: Optional[Literal["actions", "reactions", "warnings"]] = Field(None, max_length=100, description="which viewport to remove augmentations from")

    # Need to validate target tangible ids (thing, place, person, or another action.id)
    # Need to validate augmetation ids (predicate, primitive, warning, or another action.id)
    # # # Partly done, did action id validation in action model
    # Need to validate POI, if there is a poi, there must be a corresponding poi on the target tangible

    # the '*' to clear all tangibles relies on runtime information, to know what tangibles are available. 
    # To truly validate that expression, you'd have to run through all actions leading up to this one
    # to see what tangibles are there, and then if you had a poi set, you had to check those tangibles for the poi
    # for now, the simple check is to simply review all tangibles available in the workspace.

class Message(BaseModel):
    """
    Communication between devices, users, or sensors.
    """
    target: str = Field(..., max_length=100, description="ID of the Person, Device, or Sensor receiving the message.")
    type: Literal["person", "device", "sensor"] = Field(..., description="Type of recipient.")
    text: str = Field(..., max_length=1000, description="Body of the message.")
    viewport: Optional[Literal["alerts", "actions", "reactions"]] = Field(None, description="specify a viewport to display the message")
    key: Optional[str] = Field(None, max_length=100, description="Topic key for sensor/device broadcasting.")
    launch: Optional[str] = Field(None, max_length=100, description="ID of an action to trigger on the recipient device.")

    # Need to validate the target against person, device, and sensor ids available in the workplace depending on what type is set
    # Validate launch if there to see if it matches a valid action id, done at root level

### STILL NEEDS REVIEW
class IfLogic(BaseModel):
    """
    Conditional logic based on xAPI statements (analytics).
    """
    url: HttpUrl = Field(...,
                         description="xAPI Query URL to check for statements matching the verb and action conditions",
                         examples=["http://example.com/xAPI/statements?verb=passed&activity=id"])
    then_action: str = Field(..., alias="then", max_length=100, description="Action ID to trigger if condition is met (query has one or more results)")
    else_action: str = Field(..., alias="else", max_length=100, description="Action ID to trigger if condition is NOT met (query has zero results)")
    min_results: Optional[int] = Field(None, alias="min", gte=0, lte=65535, description="Minimum number of statements required.")
    max_results: Optional[int] = Field(None, alias="max", gte=0, lte=65535, description="Maximum number of statements allowed.")
    # Need to validate then and else for valid action ids at the activity level

### STILL NEEDS REVIEW
class Trigger(BaseModel):
    """
    Specifies events that move the activity from enter to exit state.
    """
    mode: Literal["click", "voice", "detect", "sensor"] = Field(..., description="Input method (UI click, voice command, object detection, or IoT data).")
    id: str = Field(..., max_length=100, description="ID of the entity to listen to (Action ID, Tangible ID, or Sensor ID).")
    duration: Optional[int] = Field(None, description="Time in ms required for gaze detection to trigger in ms")
    type: Optional[Literal["tangible", "action"]] = Field(None, description="Type of entity it is sensitive to")
    viewport: Optional[Literal["actions", "reactions", "warnings"]] = Field(None, description="Viewport to monitor")
    key: Optional[str] = Field(None, max_length=100, description="Variable key to check (for sensor mode).")
    value: Optional[str] = Field(None, max_length=100, description="Threshold value to check (for sensor mode). if operator ir 'between' use a;b as the format")
    operator: Optional[Literal["equal", "exceed", "below", "between"]] = Field("equal", description="Comparison operator (for sensor mode).")
    # Need to validate that id matches a valid tangible, action, or sensor

class ActionFlow(BaseModel):
    """
    The set of commands executed when entering or exiting a step.
    """
    remove_self: bool = Field(False, alias="removeSelf", description="For Enter: True means Exit is immidiately executed, For Exit: True deactivates instructions and augmentations from current step")
    activates: Optional[List[Activate]] = Field(default_factory=list, description="List of items to display.")
    deactivate: Optional[List[Deactivate]] = Field(default_factory=list, description="List of items to hide.")
    messages: Optional[List[Message]] = Field(default_factory=list, description="Messages to send.")
    if_logic: Optional[List[IfLogic]] = Field(default_factory=list, alias="if", description="Conditional logic checks.")

class Action(BaseModel):
    """
    A single step in the activity workflow.
    """
    id: str = Field(..., max_length=100, description="Unique ID for this step.")
    viewport: Literal["actions", "reactions", "warnings"] = Field(..., description="Visual area to display instructions for this action")
    type: Literal["actions", "reactions", "warnings"] = Field(..., description="Category of the action, used to style visual appearance")
    device: Optional[str] = Field(..., max_length=100, description="Specify on which device to execute the action")
    location: Optional[str] = Field(..., max_length=1000, description="place id where the action happens")
    predicate: Optional[str] = Field(..., max_length=1000, description="specify verb to log user interaction via xapi, if none is specified uses 'launched', xapi support is optional")
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

    # Need to verify that device.id, place.id, and predicate.id are valid by checking the workplace model.
    # Done in outermost container

ISO_639_1_CODES: Set[str] = {
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs", "ca", "ce",
    "ch", "co", "cr", "cs", "cu", "cv", "cy", "da", "de", "dv", "dz", "ee",
    "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fj", "fo", "fr",
    "fy", "ga", "gd", "gl", "gn", "gu", "gv", "ha", "he", "hi", "ho", "hr",
    "ht", "hu", "hy", "hz", "ia", "id", "ie", "ig", "ii", "ik", "io", "is",
    "it", "iu", "ja", "jv", "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn",
    "ko", "kr", "ks", "ku", "kv", "kw", "ky", "la", "lb", "lg", "li", "ln",
    "lo", "lt", "lu", "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms",
    "mt", "my", "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv",
    "ny", "oc", "oj", "om", "or", "os", "pa", "pi", "pl", "ps", "pt", "qu",
    "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk",
    "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", "ta",
    "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw",
    "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo", "wa", "wo", "xh", "yi",
    "yo", "za", "zh", "zu"
}

class Activity(BaseModel):
    """
    The root object defining the logic flow of the AR experience.
    """
    id: str = Field(..., max_length=100, description="Unique ID for the activity.")
    name: str = Field(..., max_length=1000, description="Human-readable title.")
    description: str = Field(..., max_length=10000, description="Overview of the activity.")
    language: str = Field("en", min_length=2, max_length=2, description="2 character ISO 639-1 language code, 'en' for english")
    workplace: str = Field(..., max_length=100, description="id of the workplace definition to use for these activities")
    start: str = Field(..., max_length=100, description="ID of the first Action to execute.")
    actions: List[Action] = Field(default_factory=list, description="All available steps/states in this activity.")

    # Add field validator for language iso 639-1 code
    @field_validator('language')
    @classmethod
    def validate_language(cls, lg):
        code = lg.lower()

        if code not in ISO_639_1_CODES:
            raise ValueError(
                f"'{lg}' is not a valid ISO 639-1 language code"
            )

    # Need some way to validate the workplace reference. ARLEM says it should be a url, but could also be a local file
    # ??
    # Consider putting the workplace directly into the Activity, so we can reference it in later model validation checks.
    # validate in larger container below

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
            # check activate augmentation of type 'action'
            en = action.enter
            for idx, act in enumerate(en.activates):
                if act.type == 'action' and act.augmentation not in valid_action_ids:
                    raise ValueError(
                        f"Action {act_num} referenced an invalid action id in activate {idx}"
                    )
            # Check deactivate augmentations of type 'action'
            for idx, deact in enumerate(en.deactivate):
                if deact.type == 'action' and deact.augmentation not in valid_action_ids:
                    raise ValueError(
                        f"Action {act_num} referenced an invalid action id in deactivate {idx}"
                    )
            # Check messages with launch ids
            for idx, msg in enumerate(en.messages):
                if msg.launch and msg.launch not in valid_action_ids:
                    raise ValueError(
                        f"Action {act_num} referenced an invalid action id in message {idx}"
                    )
        return self


# ==========================================
# PART 3: Container for Entire Scenario
# ==========================================

class ARLEMScenario(BaseModel):
    """
    Container to hold both workplace and activity definitions
    allows cross validation of the activity with the workplace
    """
    workplace: Workplace
    activity: Activity

    # Validate Action fields that reference elements of the workplace
    @model_validator(mode='after')
    def validate_scenario(self):
        # get valid ids fro mworkplace
        valid_place_ids = {p.id for p in self.workplace.places}
        valid_device_ids = {t.id for t in self.workplace.devices}
        valid_predicate_ids = {p.id for p in self.workplace.predicates}

        # Iterate through actions list
        for action in self.activity.actions:
            # Check values
            if action.location and action.location not in valid_place_ids:
                raise ValueError(
                    f"Action '{action.id}' refers to location '{action.location}', "
                    f"but that ID does not exist in the Workplace places."
                )
            if action.device and action.device not in valid_device_ids:
                raise ValueError(
                    f"Action '{action.id}' refers to device '{action.device}', "
                    f"but that ID does not exist in the Workplace devices."
                )
            if action.predicate and action.predicate not in valid_predicate_ids:
                raise ValueError(
                    f"Action '{action.id}' uses predicate '{action.predicate}', "
                    f"which is not defined in the Workplace."
                )
        
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
        tangible_map: Dict[str, Union[Tangible, Person]] = {}
        
        for t in self.workplace.things: tangible_map[t.id] = t
        for p in self.workplace.places: tangible_map[p.id] = p
        for p in self.workplace.persons: tangible_map[p.id] = p
        # ID Sets for Resource Categories
        ids_primitives = {p.id for p in self.workplace.primitives}
        ids_predicates = {p.id for p in self.workplace.predicates}
        ids_actions    = {a.id for a in self.activity.actions}
        ids_warnings   = {w.id for w in self.workplace.warnings}
        ids_persons    = {p.id for p in self.workplace.persons}
        ids_devices    = {d.id for d in self.workplace.devices}
        ids_sensors    = {s.id for s in self.workplace.sensors}

        # =========================================================
        # 2. HELPER: Validate 'Messages'
        # =========================================================
        def validate_message_list(messages: List['Message'], action_id: str):
            for i, msg in enumerate(messages):
                err_ctx = f"Action '{action_id}' (Message[{i}])"

                # --- A. Validate Target against specific Type lists ---
                target_found = False
                
                # Check the target against the appropriate list based on its type
                # ********* this aggregates the checks together, but it may be useful to have them 
                # seperate for more specific error messages sent back to the llm.
                if msg.type == 'person':
                    target_found = msg.target in ids_persons
                elif msg.type == 'device':
                    target_found = msg.target in ids_devices
                elif msg.type == 'sensor':
                    target_found = msg.target in ids_sensors
                
                if not target_found:
                    raise ValueError(
                        f"{err_ctx}: Target '{msg.target}' of type '{msg.type}' "
                        f"was not found in the Workplace lists."
                    )

                # --- B. Validate Launch (Optional) ---
                if msg.launch:
                    if msg.launch not in ids_actions:
                        raise ValueError(
                            f"{err_ctx}: Launch trigger '{msg.launch}' is not a valid Action ID."
                        )

        # =========================================================
        # 3. HELPER: Validate 'Activate' (Strict - No Wildcards)
        # =========================================================
        def validate_activate_list(activates: List['Activate'], action_id: str):
            for i, act in enumerate(activates):
                err_ctx = f"Action '{action_id}' (Activate[{i}])"

                # Check Target (Must be Action ID or Tangible ID)
                if act.type == 'action':
                    if act.target not in ids_actions:
                        raise ValueError(f"{err_ctx}: Target '{act.target}' is not a valid Action ID.")
                else:
                    if act.target not in tangible_map:
                        raise ValueError(f"{err_ctx}: Target '{act.target}' not found in Workplace.")

                    # Check POI (Only if target is tangible)
                    if act.poi and act.poi != "default":
                        target_obj = tangible_map[act.target]
                        valid_pois = {p.id for p in target_obj.pois}
                        if act.poi not in valid_pois:
                            raise ValueError(f"{err_ctx}: POI '{act.poi}' not found on target '{act.target}'.")

                # Check Augmentation ID
                if act.augmentation:
                    if act.type == 'primitive' and act.augmentation not in ids_primitives:
                        raise ValueError(f"{err_ctx}: Primitive '{act.augmentation}' not found.")
                    elif act.type == 'predicate' and act.augmentation not in ids_predicates:
                        raise ValueError(f"{err_ctx}: Predicate '{act.augmentation}' not found.")
                    elif act.type == 'warning' and act.augmentation not in ids_warnings:
                        raise ValueError(f"{err_ctx}: Warning '{act.augmentation}' not found.")

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
                        valid_pois = {p.id for p in target_obj.pois}
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
                    elif d_type == 'warning':
                        found = aug_id in ids_warnings
                    elif d_type == 'action':
                        found = aug_id in ids_actions
                    else:
                        # If type is '*' or None, the ID must exist in ANY valid list
                        found = (aug_id in ids_primitives or 
                                 aug_id in ids_predicates or 
                                 aug_id in ids_warnings or 
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
                if action.enter.messages:
                    validate_message_list(action.enter.messages, action.id)

            # Validate 'Exit' Flow
            if action.exit:
                if action.exit.activates:
                    validate_activate_list(action.exit.activates, action.id)
                if action.exit.deactivate:
                    validate_deactivate_list(action.exit.deactivate, action.id)
                if action.exit.messages:
                    validate_message_list(action.exit.messages, action.id)

        return self
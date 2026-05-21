from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------
# Optional runtime configuration
# ---------------------------------------------------------------------

class TTSRuntimeConfig(BaseModel):
    """
    Runtime configuration used by the generation script.

    This is NOT necessarily part of the authored narration JSON.
    It allows the code to validate voices dynamically without hard-coding
    the voice list into the narration schema.

    Example:
        runtime_config = TTSRuntimeConfig(
            allowed_voices=["alloy", "verse", "nova", "onyx"],
            default_voice="alloy"
        )
    """

    model_config = ConfigDict(extra="forbid")

    allowed_voices: list[str] = Field(
        default_factory=list,
        description=(
            "Voices currently available from the TTS provider. "
            "This should be loaded from the provider, a config file, "
            "or a periodically updated local cache."
        ),
    )

    default_voice: Optional[str] = Field(
        default=None,
        description="Default voice used when the narration file omits a voice.",
    )

    default_style_preset: Optional[str] = Field(
        default=None,
        description="Default style preset used when the narration file omits one.",
    )

    @field_validator("allowed_voices")
    @classmethod
    def normalize_allowed_voices(cls, voices: list[str]) -> list[str]:
        return [v.strip() for v in voices if v.strip()]


# ---------------------------------------------------------------------
# Style model
# ---------------------------------------------------------------------

class ClipStyle(BaseModel):
    """
    Optional natural-language style guidance for a clip.

    These are suggestions, not enumerated values. The LLM or author can
    describe tone, pace, and emphasis naturally. The generation layer can
    translate these into provider-specific TTS instructions.

    Examples:
        tone: "friendly but professional"
        pace: "slow enough for first-year students"
        emphasis: "lightly emphasize the key equation"
        instructions: "sound like an instructor explaining this at a whiteboard"
    """

    model_config = ConfigDict(extra="forbid")

    tone: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional tone suggestion, such as 'friendly', "
            "'calm and precise', or 'curious and encouraging'."
        ),
    )

    pace: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional pacing suggestion, such as 'moderate', "
            "'slow for emphasis', or 'brisk but clear'. "
            "This is not a timing guarantee."
        ),
    )

    emphasis: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=300,
        description=(
            "Optional clip-level emphasis guidance. This is a suggestion, "
            "not a controlled vocabulary."
        ),
    )

    instructions: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description=(
            "Optional free-form narration instructions for the clip."
        ),
    )

    @field_validator("tone", "pace", "emphasis", "instructions")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


# ---------------------------------------------------------------------
# Segment models
# ---------------------------------------------------------------------

class SaySegment(BaseModel):
    """
    Spoken narration segment.

    The TTS system should speak this text. Do not include explicit timing
    here; spoken duration is determined by the text, voice, style, pacing,
    and TTS model.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["say"] = Field(
        default="say",
        description="Segment type. Use 'say' for spoken narration.",
    )

    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Text to be spoken by the TTS system.",
    )

    emphasis: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=300,
        description=(
            "Optional natural-language emphasis suggestion for this segment. "
            "Examples: 'light emphasis on the phrase lower error', "
            "'make this sound like a key takeaway', or 'no special emphasis'."
        ),
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("SaySegment.text cannot be blank.")
        return value

    @field_validator("emphasis")
    @classmethod
    def strip_emphasis(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class PauseSegment(BaseModel):
    """
    Intentional silence in the narration.

    Pause timing is the only explicit timing field in the authored JSON.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["pause"] = Field(
        default="pause",
        description="Segment type. Use 'pause' for intentional silence.",
    )

    seconds: float = Field(
        ...,
        gt=0,
        le=10,
        description=(
            "Length of the pause in seconds. Typical values are between "
            "0.3 and 1.5 seconds. Longer pauses should be used sparingly."
        ),
    )


NarrationSegment = Annotated[
    Union[SaySegment, PauseSegment],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------
# Clip, module, and full script models
# ---------------------------------------------------------------------

class NarrationClip(BaseModel):
    """
    One generated audio clip.

    Clips are ordered by list position. The generation pipeline should
    number clips automatically within each module.

    Recommended filename convention:

        m{module_number:02d}_c{clip_number:03d}_{safe_clip_title}.mp3

    If title is omitted:

        m{module_number:02d}_c{clip_number:03d}.mp3

    All generated audio files for a lab should go into one output
    directory, not module subdirectories.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=120,
        description=(
            "Optional human-readable clip title. May be used for filenames "
            "after being converted to a safe slug."
        ),
    )

    voice: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Optional TTS voice name. This is intentionally a string, not an "
            "enum, because available voices should be loaded dynamically from "
            "the provider or local configuration."
        ),
    )

    style: Optional[ClipStyle] = Field(
        default=None,
        description=(
            "Optional natural-language style guidance for this clip."
        ),
    )

    segments: list[NarrationSegment] = Field(
        ...,
        min_length=1,
        description="Ordered list of spoken and pause segments for this clip.",
    )

    @field_validator("title", "voice")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def clip_must_include_speech(self) -> "NarrationClip":
        if not any(segment.type == "say" for segment in self.segments):
            raise ValueError("Each clip must contain at least one 'say' segment.")
        return self


class NarrationModule(BaseModel):
    """
    One module within the AR lab.

    Modules are ordered by list position. No explicit module_order field
    is needed.
    """

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="Optional human-readable module title.",
    )

    clips: list[NarrationClip] = Field(
        ...,
        min_length=1,
        description="Ordered list of narration clips in this module.",
    )

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ARLabNarrationScript(BaseModel):
    """
    Authored JSON input for AR-lab TTS narration.

    Design choices:
    - lab_id and lab_title are optional.
    - course information is excluded.
    - AR anchors are excluded.
    - module order comes from list order.
    - clip order comes from list order.
    - clip_key is excluded.
    - filenames are generated by convention.
    - all audio files for a lab go into one directory.
    - voice names are strings, not enums.
    - voices should be validated dynamically by the generation layer.
    - style, emphasis, tone, and pace are suggestions, not enumerations.
    - explicit timing appears only in pause segments.

    Minimal valid structure:

        {
          "modules": [
            {
              "clips": [
                {
                  "segments": [
                    {
                      "type": "say",
                      "text": "Welcome to the lab."
                    }
                  ]
                }
              ]
            }
          ]
        }
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="0.1",
        description="Schema version for the narration JSON format.",
    )

    lab_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description=(
            "Optional lab identifier. May be used for logging or output "
            "directory naming, but is not required."
        ),
    )

    lab_title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=160,
        description="Optional human-readable lab title.",
    )

    default_voice: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Optional default TTS voice name. This is intentionally a string, "
            "not an enum. Validate it dynamically against currently available "
            "voices at generation time."
        ),
    )

    style_preset: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional global style suggestion, such as 'clear lecture', "
            "'warm tutor', or 'calm first-year explanation'. This is not "
            "an enumeration."
        ),
    )

    modules: list[NarrationModule] = Field(
        ...,
        min_length=1,
        description="Ordered list of modules in the AR lab narration.",
    )

    @field_validator("schema_version", "lab_id", "lab_title", "default_voice", "style_preset")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("String fields cannot be blank if provided.")
        return value

    def validate_voices_against_runtime_config(
        self,
        runtime_config: TTSRuntimeConfig,
    ) -> None:
        """
        Validate voice names against a dynamically supplied list.

        This is deliberately separate from the Pydantic schema so the
        available voice list can be updated without editing this model.

        Raises:
            ValueError if a specified voice is not in runtime_config.allowed_voices.
        """

        allowed = set(runtime_config.allowed_voices)

        if not allowed:
            # No runtime voice list available; skip validation.
            return

        voices_to_check: list[tuple[str, str]] = []

        if self.default_voice:
            voices_to_check.append(("default_voice", self.default_voice))

        for module_index, module in enumerate(self.modules, start=1):
            for clip_index, clip in enumerate(module.clips, start=1):
                if clip.voice:
                    location = f"modules[{module_index}].clips[{clip_index}].voice"
                    voices_to_check.append((location, clip.voice))

        invalid = [
            (location, voice)
            for location, voice in voices_to_check
            if voice not in allowed
        ]

        if invalid:
            details = ", ".join(
                f"{location}={voice!r}" for location, voice in invalid
            )
            allowed_list = ", ".join(sorted(allowed))
            raise ValueError(
                f"Invalid TTS voice(s): {details}. "
                f"Allowed voices are: {allowed_list}"
            )
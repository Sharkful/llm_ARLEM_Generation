"""
Benchmark Configuration

Defines model configurations, prompts, and settings for
benchmarking LLM-based AR lab generation across providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from prompt_builder import Level, Structure


# ── Enums ────────────────────────────────────────────────────────────

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelSize(str, Enum):
    """Rough capability/cost tier, used by the --small/medium/large-models flags."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class SpecType(str, Enum):
    """Which Pydantic specification to generate against."""
    JSON_LAB = "json_lab"              # Lab (json_lab.py) - single demo module
    ARLEM = "arlem"                    # ARLEMScenario (arlem_full.py) - workplace + activity
    ARLEM_SIMPLIFIED = "arlem_simple"  # ARLEMScenario (arlem_simplified.py) - stripped-down ARLEM


# ── Model Registry ───────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a single model to benchmark."""
    model_id: str
    provider: Provider
    display_name: str
    size: ModelSize = ModelSize.MEDIUM
    max_retries: int = 3
    # Per-1M-token pricing (overrides DEFAULT_PRICING if set)
    input_price: Optional[float] = None
    output_price: Optional[float] = None

    @property
    def pricing_config(self) -> Optional[dict]:
        if self.input_price is not None and self.output_price is not None:
            return {"input": self.input_price, "output": self.output_price}
        return None


# Final generation-test roster (locked in issue #32, re-verified 2026-07).
# Every model_id here must have a matching DEFAULT_PRICING key or cost computes
# as $0 silently. Dropped this round: gpt-5-mini/gpt-5-nano (superseded/retired),
# gpt-4o-mini (legacy), claude-sonnet-4.6 (→ Sonnet 5), gemini-2.5-pro/2.5-flash
# (kept 2.5-flash-lite as the cheapest data point).
MODELS = {
    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-5.5":      ModelConfig("gpt-5.5",      Provider.OPENAI, "GPT-5.5",      ModelSize.LARGE),
    "gpt-5.4":      ModelConfig("gpt-5.4",      Provider.OPENAI, "GPT-5.4",      ModelSize.MEDIUM),
    "gpt-5.4-mini": ModelConfig("gpt-5.4-mini", Provider.OPENAI, "GPT-5.4 Mini", ModelSize.SMALL),
    "gpt-5.4-nano": ModelConfig("gpt-5.4-nano", Provider.OPENAI, "GPT-5.4 Nano", ModelSize.SMALL),

    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-opus-4.8":  ModelConfig("claude-opus-4-8",           Provider.ANTHROPIC, "Claude Opus 4.8", ModelSize.LARGE),
    "claude-sonnet-5":  ModelConfig("claude-sonnet-5",           Provider.ANTHROPIC, "Claude Sonnet 5", ModelSize.MEDIUM),
    "claude-haiku-4.5": ModelConfig("claude-haiku-4-5-20251001", Provider.ANTHROPIC, "Claude Haiku 4.5", ModelSize.SMALL),

    # ── Google Gemini ─────────────────────────────────────────────────────
    # Registry key (CLI alias) stays "gemini-3.1-pro"; model_id is the -preview id
    # because generativelanguage v1beta only serves gemini-3.1-pro-preview, not the
    # GA-renamed plain id (issue #42). Precedent: claude-opus-4.8 → claude-opus-4-8.
    "gemini-3.1-pro":        ModelConfig("gemini-3.1-pro-preview", Provider.GOOGLE, "Gemini 3.1 Pro",        ModelSize.LARGE),
    "gemini-3.5-flash":      ModelConfig("gemini-3.5-flash",       Provider.GOOGLE, "Gemini 3.5 Flash",      ModelSize.MEDIUM),
    "gemini-3.1-flash-lite": ModelConfig("gemini-3.1-flash-lite",  Provider.GOOGLE, "Gemini 3.1 Flash Lite", ModelSize.SMALL),
    "gemini-2.5-flash-lite": ModelConfig("gemini-2.5-flash-lite",  Provider.GOOGLE, "Gemini 2.5 Flash Lite", ModelSize.SMALL),
}


def models_by_size(*sizes: ModelSize) -> list[str]:
    """Return model_ids whose tier is in `sizes`, preserving MODELS order."""
    wanted = set(sizes)
    return [mid for mid, cfg in MODELS.items() if cfg.size in wanted]


# ── Prompt Templates ─────────────────────────────────────────────────

# System prompt shared across all spec types
SYSTEM_PROMPT = """\
You are an expert educational content designer specializing in Augmented Reality \
(AR) learning experiences. You create structured, pedagogically sound AR lab \
specifications that follow strict schemas.

Your designs should:
- Be scientifically accurate and age-appropriate for university students
- Include clear learning objectives tied to each module/action
- Use realistic 3D object placements with sensible positions, scales, and rotations
- Create a logical progression of steps that builds understanding incrementally
- Include descriptive audio clip names and meaningful text labels
- Reference plausible prefab names and textures for the subject matter
"""

# ── JSON Lab Prompt (multi-module: one demo module per sub-topic) ────

JSON_LAB_PROMPT_TEMPLATE = """\
Create a complete AR lab experience about: {topic}

Requirements:
- Create a Lab containing MULTIPLE "demo" type modules — one per major sub-topic
  or conceptual stage of the topic (aim for one module per distinct idea the
  learner must grasp, typically 3-6 modules).
- Modules should appear in a logical teaching order that builds understanding
  incrementally.
- Each module owns its own scene objects and runs through its own sequence of
  clips covering that sub-topic.
- Across the whole lab, include at least {min_objects} total 3D scene objects and
  at least {min_clips} total clips, distributed across the modules.
- Each clip should use sparse object changes to animate/update the scene
- Objects should have realistic positions (within a ~3m workspace centered on the user)
- Scales should be appropriate for a tabletop AR experience (objects typically 0.02-0.5 units)
- Include educational objectives for the lab as a whole and for each module
- Use descriptive prefab names (e.g., 'heartPrefab', 'moleculePrefab') for the topic
- Reference plausible texture names where appropriate
- Add behavioral components (rotation, orbit, text labels) where they enhance learning

The lab should feel like a sequence of guided scenes, each its own module, where a \
narrator explains the topic while 3D objects appear, move, and change to illustrate \
key concepts.
"""

# ── ARLEM Prompt (Workplace + Activity) ──────────────────────────────

ARLEM_PROMPT_TEMPLATE = """\
Create a complete ARLEM AR training scenario about: {topic}

Requirements for the Workplace:
- Define a realistic workplace with at least {min_things} things (tools/materials)
- Include at least {min_places} places (workstations/zones)
- Add 1 person (the learner) and 1 device (AR headset)
- Define detectables (anchors) for spatial tracking of things and places
- Include at least 3 primitives (supported media types: label, image, audio, video, animation)
- Create predicates (instructional augmentations) that will be used in the activity steps
- All POIs with id "default" must have zero offsets

Requirements for the Activity:
- Create at least {min_actions} sequential action steps
- Each action must have enter/exit flows with activates and deactivates
- Use triggers (click or voice) to advance between steps
- Include clear instructions for each step with title and description
- Actions should reference valid workplace resources (things, places, predicates)
- The workflow should be a logical progression for learning the topic
- Language should be "en" (English)

The scenario should represent a realistic hands-on training experience \
where AR augmentations guide the learner through a physical task.
"""


# ── Benchmark Run Configuration ──────────────────────────────────────

@dataclass
class BenchmarkRunConfig:
    """Configuration for a single benchmark run."""
    # What to generate
    spec_type: SpecType = SpecType.JSON_LAB
    topic: str = "The Solar System - Relative sizes and orbits of planets"

    # YAML-driven prompt construction (L1-L4). When both are set, the
    # legacy `topic`-based template is bypassed in favor of the prompt builder.
    lab_name: Optional[str] = None
    level: Optional[Level] = None
    structure: Structure = Structure.MULTI_MODULE

    # Generation parameters
    min_objects: int = 4       # JSON Lab: minimum scene objects
    min_clips: int = 5         # JSON Lab: minimum clips
    min_things: int = 3        # ARLEM: minimum workplace things
    min_places: int = 2        # ARLEM: minimum workplace places
    min_actions: int = 5       # ARLEM: minimum activity actions

    # Instructor settings
    max_retries: int = 3

    # Output
    save_output: bool = True
    output_dir: str = "Artifacts/Data/Benchmark"

    def get_prompt(self) -> str:
        """Build the user prompt from the template and config values.

        For YAML-driven runs (lab_name + level set), callers should use
        `prompt_builder.build_prompt` directly — it also returns the
        response model and dispatches the Gemini variant. This method
        only handles the legacy free-form --topic path.
        """
        if self.spec_type == SpecType.JSON_LAB:
            return JSON_LAB_PROMPT_TEMPLATE.format(
                topic=self.topic,
                min_objects=self.min_objects,
                min_clips=self.min_clips,
            )
        else:
            return ARLEM_PROMPT_TEMPLATE.format(
                topic=self.topic,
                min_things=self.min_things,
                min_places=self.min_places,
                min_actions=self.min_actions,
            )


# ── Default Benchmark Suites ────────────────────────────────────────

# Topics to test across models
DEFAULT_TOPICS = [
    "The Solar System - Relative sizes and orbits of planets",
    "Human Heart Anatomy - Chambers, valves, and blood flow",
    "Basic Circuit Components - Resistors, capacitors, and LEDs",
]

# Quick test: cheapest current model per provider (all roster keys)
QUICK_BENCHMARK_MODELS = ["gpt-5.4-nano", "claude-haiku-4.5", "gemini-2.5-flash-lite"]

# Full benchmark: comprehensive model comparison
FULL_BENCHMARK_MODELS = list(MODELS.keys())

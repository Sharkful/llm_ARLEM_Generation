"""
Benchmark Configuration

Defines model configurations, prompts, and settings for
benchmarking LLM-based AR lab generation across providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


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
    max_retries: int = 3
    # Per-1M-token pricing (overrides DEFAULT_PRICING if set)
    input_price: Optional[float] = None
    output_price: Optional[float] = None

    @property
    def pricing_config(self) -> Optional[dict]:
        if self.input_price is not None and self.output_price is not None:
            return {"input": self.input_price, "output": self.output_price}
        return None


# Pre-defined model configs for quick benchmarking
MODELS = {
    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-5.4":      ModelConfig("gpt-5.4",      Provider.OPENAI, "GPT-5.4"),
    "gpt-5.4-mini": ModelConfig("gpt-5.4-mini", Provider.OPENAI, "GPT-5.4 Mini"),
    "gpt-5.4-nano": ModelConfig("gpt-5.4-nano", Provider.OPENAI, "GPT-5.4 Nano"),
    "gpt-5.1":      ModelConfig("gpt-5.1",      Provider.OPENAI, "GPT-5.1"),
    "gpt-5-mini":   ModelConfig("gpt-5-mini",   Provider.OPENAI, "GPT-5 Mini"),
    "gpt-5-nano":   ModelConfig("gpt-5-nano",   Provider.OPENAI, "GPT-5 Nano"),
    "gpt-4o-mini":  ModelConfig("gpt-4o-mini",  Provider.OPENAI, "GPT-4o Mini"),

    # ── Anthropic ─────────────────────────────────────────────────────────
    "claude-opus-4.6":   ModelConfig("claude-opus-4-6",          Provider.ANTHROPIC, "Claude Opus 4.6"),
    "claude-sonnet-4.6": ModelConfig("claude-sonnet-4-6",        Provider.ANTHROPIC, "Claude Sonnet 4.6"),
    "claude-haiku-4.5":  ModelConfig("claude-haiku-4-5-20251001", Provider.ANTHROPIC, "Claude Haiku 4.5"),
    "claude-3-haiku":    ModelConfig("claude-3-haiku-20240307",   Provider.ANTHROPIC, "Claude 3 Haiku (dep. Apr 2026)"),

    # ── Google Gemini ─────────────────────────────────────────────────────
    "gemini-3.1-pro":        ModelConfig("gemini-3.1-pro-preview",        Provider.GOOGLE, "Gemini 3.1 Pro"),
    "gemini-3.1-flash-lite": ModelConfig("gemini-3.1-flash-lite-preview", Provider.GOOGLE, "Gemini 3.1 Flash Lite"),
    "gemini-3-flash":        ModelConfig("gemini-3.1-flash-image-preview", Provider.GOOGLE, "Gemini 3 Flash"),
    "gemini-2.5-pro":        ModelConfig("gemini-2.5-pro",        Provider.GOOGLE, "Gemini 2.5 Pro"),
    "gemini-2.5-flash":      ModelConfig("gemini-2.5-flash",      Provider.GOOGLE, "Gemini 2.5 Flash"),
    "gemini-2.5-flash-lite": ModelConfig("gemini-2.5-flash-lite", Provider.GOOGLE, "Gemini 2.5 Flash Lite"),
}


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

# ── JSON Lab Prompt (single DemoModule with clips) ───────────────────

JSON_LAB_PROMPT_TEMPLATE = """\
Create a complete AR lab experience about: {topic}

Requirements:
- The lab MUST contain exactly 1 module: a "demo" type module (bridge/demonstration)
- The module should guide the student through the topic using a sequence of clips
- Include at least {min_objects} 3D scene objects that are relevant to the topic
- Create at least {min_clips} clips that form a logical educational narrative
- Each clip should use sparse object changes to animate/update the scene
- Objects should have realistic positions (within a ~3m workspace centered on the user)
- Scales should be appropriate for a tabletop AR experience (objects typically 0.02-0.5 units)
- Include educational objectives for both the lab and the module
- Use descriptive prefab names (e.g., 'heartPrefab', 'moleculePrefab') for the topic
- Reference plausible texture names where appropriate
- Add behavioral components (rotation, orbit, text labels) where they enhance learning

The lab should feel like a guided walkthrough where a narrator explains the topic \
while 3D objects appear, move, and change to illustrate key concepts.
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

    # Generation parameters
    min_objects: int = 5       # JSON Lab: minimum scene objects
    min_clips: int = 6         # JSON Lab: minimum clips
    min_things: int = 3        # ARLEM: minimum workplace things
    min_places: int = 2        # ARLEM: minimum workplace places
    min_actions: int = 5       # ARLEM: minimum activity actions

    # Instructor settings
    max_retries: int = 3

    # Output
    save_output: bool = True
    output_dir: str = "Artifacts/Data/Benchmark"

    def get_prompt(self) -> str:
        """Build the user prompt from the template and config values."""
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

# Quick test: small set of fast/cheap models
QUICK_BENCHMARK_MODELS = ["gpt-4o-mini", "claude-haiku-4.5", "gemini-2.5-flash-lite"]

# Full benchmark: comprehensive model comparison
FULL_BENCHMARK_MODELS = list(MODELS.keys())

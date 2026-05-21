import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str | None = (os.getenv("ANTHROPIC_API_KEY") or "").strip() or None
OPENAI_API_KEY: str | None = (os.getenv("OPENAI_API_KEY") or "").strip() or None

PROJECTS_DIR = Path(
    os.getenv(
        "NARRATION_PROJECTS_DIR",
        str(Path(__file__).parent / "narration_projects"),
    )
).resolve()
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_TTS_PROVIDER = "openai"
DEFAULT_TTS_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"

OPENAI_TTS_VOICES = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "shimmer"]

# Cost per 1K tokens (USD)
CLAUDE_PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015, "cached": 0.0003},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075, "cached": 0.0015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004, "cached": 0.00008},
}

# Cost per 1K characters (USD)
OPENAI_TTS_PRICES: dict[str, dict[str, float]] = {
    "tts-1": {"per_1k_chars": 0.015},
    "tts-1-hd": {"per_1k_chars": 0.030},
}

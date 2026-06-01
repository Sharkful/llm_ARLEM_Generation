"""
AR Lab Generator

Generates a single JSON Lab and saves it to Artifacts/Data/Generated Labs/.
Intended for producing labs that can be patched into the AR headset prototype.

Usage:
    # Generate with a specific model and topic
    python "Code/Testing/generate_lab.py" --model claude-sonnet-4.6 --topic "Photosynthesis"

    # Custom output filename
    python "Code/Testing/generate_lab.py" --model gpt-4o-mini --topic "DNA Replication" --name dna_lab

    # Adjust retries
    python "Code/Testing/generate_lab.py" --model claude-haiku-4.5 --topic "Cell Mitosis" --max-retries 5

    # List available models
    python "Code/Testing/generate_lab.py" --list-models
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Code" / "Tools"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from benchmark_config import (
    MODELS,
    Provider,
    SYSTEM_PROMPT,
    JSON_LAB_PROMPT_TEMPLATE,
)

OUTPUT_DIR = PROJECT_ROOT / "Artifacts" / "Data" / "Generated Labs"


def create_instructor_client(model_config):
    import instructor

    provider_prefix = {
        Provider.OPENAI: "openai",
        Provider.ANTHROPIC: "anthropic",
        Provider.GOOGLE: "google",
    }
    prefix = provider_prefix[model_config.provider]

    kwargs = {}
    if model_config.provider == Provider.GOOGLE:
        kwargs["api_key"] = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        kwargs["mode"] = instructor.Mode.GENAI_STRUCTURED_OUTPUTS

    return instructor.from_provider(f"{prefix}/{model_config.model_id}", **kwargs)


def generate_lab(
    model_id: str,
    topic: str,
    max_retries: int = 3,
    min_objects: int = 5,
    min_clips: int = 6,
) -> dict:
    if model_id not in MODELS:
        raise ValueError(f"Unknown model '{model_id}'. Run --list-models to see options.")

    model_config = MODELS[model_id]
    is_gemini = model_config.provider == Provider.GOOGLE

    if is_gemini:
        from json_lab_gemini import Lab
    else:
        from json_lab import Lab

    client = create_instructor_client(model_config)

    user_prompt = JSON_LAB_PROMPT_TEMPLATE.format(
        topic=topic,
        min_objects=min_objects,
        min_clips=min_clips,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    create_kwargs = dict(
        response_model=Lab,
        messages=messages,
        max_retries=max_retries,
    )
    if model_config.provider == Provider.ANTHROPIC:
        create_kwargs["max_tokens"] = 8192

    print(f"Generating lab with {model_config.display_name}...")
    print(f"Topic: {topic}")
    start = time.time()
    result = client.create(**create_kwargs)
    elapsed = round(time.time() - start, 2)
    print(f"Done in {elapsed}s")

    return result.model_dump(mode="json", exclude_none=True)


def save_lab(output_json: dict, name: str | None, model_id: str, topic: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if name:
        # Sanitize the provided name
        safe_name = name.replace(" ", "_").replace("/", "-")
        filename = f"{safe_name}.json"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model_id.replace("/", "_").replace(".", "-")
        safe_topic = topic[:30].replace(" ", "_").replace("/", "-")
        filename = f"{safe_model}_{safe_topic}_{ts}.json"

    out_path = OUTPUT_DIR / filename
    out_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a single AR JSON Lab and save it for headset use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", "-m", type=str, help="Model ID (e.g., claude-sonnet-4.6)")
    parser.add_argument("--topic", "-t", type=str, default="The Solar System - Relative sizes and orbits of planets",
                        help="Topic for the AR lab")
    parser.add_argument("--name", "-n", type=str, default=None,
                        help="Output filename (without .json). Defaults to auto-generated name.")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max instructor retries on validation failure (default: 3)")
    parser.add_argument("--min-objects", type=int, default=5,
                        help="Minimum scene objects to request (default: 5)")
    parser.add_argument("--min-clips", type=int, default=6,
                        help="Minimum clips to request (default: 6)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available model IDs and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for model_id, config in MODELS.items():
            print(f"  {model_id:<30} {config.provider.value:<12} {config.display_name}")
        return

    if not args.model:
        print("Error: --model is required. Use --list-models to see options.")
        sys.exit(1)

    try:
        output_json = generate_lab(
            model_id=args.model,
            topic=args.topic,
            max_retries=args.max_retries,
            min_objects=args.min_objects,
            min_clips=args.min_clips,
        )
        out_path = save_lab(output_json, args.name, args.model, args.topic)
        print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# AR Lab Narration Generator

Converts a plain-text description of an augmented reality lab into a folder of MP3 audio files, ready for use in an AR experience. The pipeline uses two LLM calls (Anthropic Claude) to plan and write the narration, then synthesizes audio via the OpenAI TTS API.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [API Keys](#api-keys)
4. [CLI Reference](#cli-reference)
5. [Pipeline Design](#pipeline-design)
6. [Output Artifacts](#output-artifacts)
7. [Narration Script Format](#narration-script-format)
8. [Scene Brief Format](#scene-brief-format)
9. [Project Structure](#project-structure)
10. [Configuration](#configuration)
11. [Running Tests](#running-tests)
12. [Extending the System](#extending-the-system)

---

## Quick Start

```powershell
# Activate the conda environment
conda activate ar-experiments

# Navigate to the narration_generator directory
cd llm_ARLEM_Generation/narration_generator

# Dry run — generates narration script only, no audio, no OpenAI cost
python cli.py run --file tests/fixtures/sample_description.txt --title "My Lab" --dry-run

# Full run — generates narration script + MP3 audio files
python cli.py run --file my_lab_description.txt --title "My Lab"

# List all generated projects
python cli.py list

# Check status of a specific project
python cli.py status my_lab_20260521_143022
```

Output lands in `narration_projects/{project_id}/` inside this directory.

---

## Requirements

### Python environment

All dependencies live in the shared `ar-experiments` conda environment.

New packages added for this subproject (run from `llm_ARLEM_Generation/`):

```
pip install openai pydub mutagen tenacity pytest pytest-mock
```

Or install everything at once:

```
pip install -r requirements.txt
```

### ffmpeg (required for audio generation)

`pydub` requires `ffmpeg` on the system PATH to encode MP3 files.

**Windows (Chocolatey):**
```
choco install ffmpeg
```

**Windows (manual):** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract, and add the `bin/` folder to your PATH.

**Verify:**
```
ffmpeg -version
```

ffmpeg is only needed for the TTS stage. Dry runs (`--dry-run`) do not require it.

---

## API Keys

Two API keys are required. Store them in a `.env` file in this directory (`narration_generator/.env`) — never commit this file or paste keys into the terminal.

```
# narration_generator/.env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The `.env` file is loaded automatically at startup. The `config.py` strips whitespace from keys, but it is still best practice to paste them cleanly.

### Which key is used for what

| Key | Used for | Cost basis |
|-----|---------|------------|
| `ANTHROPIC_API_KEY` | LLM pipeline (scene brief + narration script) | Per token (input/output/cached) |
| `OPENAI_API_KEY` | TTS audio synthesis | Per 1,000 characters |

A typical lab (3 modules, ~10 clips) costs approximately **$0.05–$0.10 total** — roughly $0.02 for LLM calls and $0.05 for TTS. A cost report is saved to every project folder.

`--dry-run` skips TTS entirely and only requires the Anthropic key.

---

## CLI Reference

All commands are run from the `narration_generator/` directory.

### `run` — generate a project

```
python cli.py run [description] [options]
```

| Argument | Description |
|----------|-------------|
| `description` | Lab description as a positional string (or use `--file`) |
| `--file`, `-f` | Read description from a text file |
| `--title`, `-t` | Human-readable project title (used in the project folder name) |
| `--model` | Override the LLM model (default: `claude-sonnet-4-6`) |
| `--voice` | Override the default TTS voice (default: `alloy`) |
| `--tts-model` | Override the TTS model (default: `tts-1`) |
| `--dry-run` | Run LLM stages only — print the narration script, skip TTS |

**Examples:**

```powershell
# From a file, full run
python cli.py run --file description.txt --title "Gradient Descent Lab"

# Inline description, dry run
python cli.py run "A lab showing how a lens focuses light" --title "Optics Lab" --dry-run

# Use a higher-quality TTS model and a different voice
python cli.py run --file desc.txt --title "My Lab" --tts-model tts-1-hd --voice nova
```

**Available voices:** `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### `list` — show all projects

```
python cli.py list
```

Prints a table of all projects in `narration_projects/` with their status and creation time.

### `status` — inspect a project

```
python cli.py status <project_id>
```

Prints manifest details and audio generation summary for the named project.

---

## Pipeline Design

The generator runs in three sequential stages, each producing a JSON artifact.

```
User description (plain text)
        │
        ▼
┌─────────────────────┐
│  Stage 1: LLM       │  Claude generates a lightweight "scene brief"
│  Scene Brief        │  describing what happens in each clip and why
│  (ARSceneNarration  │  it matters instructionally — NOT narration text
│   Brief)            │
└─────────────────────┘
        │  scene_brief.json
        ▼
┌─────────────────────┐
│  Stage 2: LLM       │  Claude reads the brief and writes natural
│  Narration Script   │  spoken narration with say/pause segments
│  (ARLabNarration    │  per clip, preserving module/clip structure
│   Script)           │
└─────────────────────┘
        │  narration_script.json
        ▼
  Validation pass
  (voice checks, warnings)
        │
        ▼
┌─────────────────────┐
│  Stage 3: TTS       │  Each say segment → OpenAI TTS → MP3 bytes
│  Audio Generation   │  Each pause segment → pydub silence
│                     │  Segments concatenated → one MP3 per clip
└─────────────────────┘
        │  audio/*.mp3 + audio_manifest.json
        ▼
  Cost report saved
```

### Why two LLM stages?

The scene brief stage separates **instructional intent** (what should be communicated) from **narration craft** (how to say it). This produces more focused, pedagogically sound narration than asking a single prompt to do both jobs at once. The brief is also a useful checkpoint for reviewing what the system understood about the lab before committing to a full script.

### TTS approach: segment-based

Each clip's audio is built segment by segment:
- `say` segments are sent to the TTS API individually
- `pause` segments are generated as digital silence (exact duration)
- All segments are concatenated with `pydub` into one MP3 file per clip

This gives precise control over timing and allows pauses to be edited in the narration JSON without re-generating any speech.

### Staleness detection

Each `AudioClipRecord` in the audio manifest stores a SHA-256 hash of the clip's text, voice, TTS model, and style. If the narration script is edited and audio is re-generated, only clips whose hash has changed need to be re-synthesized. This is not yet wired into the CLI but the `pipeline/staleness.py` module is ready to use.

### Retry and error handling

Both LLM stages use `tenacity` for exponential backoff on rate-limit and connection errors (up to 5 attempts, 4–60 second delays). The `instructor` library handles structural retries when the LLM output fails Pydantic validation (up to 3 attempts per call).

---

## Output Artifacts

Every project creates a folder at `narration_projects/{project_id}/` containing:

| File | Description |
|------|-------------|
| `description.txt` | Original input description |
| `scene_brief.json` | LLM-generated instructional brief (Stage 1 output) |
| `narration_script.json` | LLM-generated narration script (Stage 2 output, TTS input) |
| `project_manifest.json` | Project config and status |
| `audio_manifest.json` | Per-clip records: filename, voice, hash, duration, status |
| `generation_log.jsonl` | Append-only log of every LLM call, TTS call, and validation event |
| `cost_report.json` | Token usage and cost breakdown for the run |
| `audio/` | MP3 files, one per clip |

### Audio filenames

Files are named by convention — never authored manually:

```
m{module:02d}_c{clip:03d}_{safe_title}.mp3
```

Examples:
```
m01_c001_introducing_the_loss_surface.mp3
m01_c002_marking_the_starting_position.mp3
m02_c001_the_gradient_arrow.mp3
```

Module and clip numbers are 1-based and derived from list order in the narration script.

---

## Narration Script Format

The narration script (`narration_script.json`) is the primary human-reviewable artifact. It is also the direct input to the TTS stage and can be hand-edited before re-running audio generation.

Full specification: [`ar_tts_narration_json_standard.md`](ar_tts_narration_json_standard.md)

### Minimal example

```json
{
  "schema_version": "0.1",
  "lab_title": "Gradient Descent",
  "default_voice": "alloy",
  "modules": [
    {
      "title": "Introduction",
      "clips": [
        {
          "title": "What Is a Loss Surface",
          "segments": [
            {
              "type": "say",
              "text": "What you're looking at is called a loss surface."
            },
            {
              "type": "pause",
              "seconds": 0.5
            },
            {
              "type": "say",
              "text": "The lower the point, the better the model performs."
            }
          ]
        }
      ]
    }
  ]
}
```

### Segment types

| Type | Required fields | Notes |
|------|----------------|-------|
| `say` | `type`, `text` | Text sent to TTS. Optional `emphasis` (natural language hint). |
| `pause` | `type`, `seconds` | Silence inserted between say segments. `0 < seconds <= 10`. |

### Voice and style

- `default_voice` at the top level applies to all clips unless overridden
- Each clip can have its own `voice` and `style` object
- `style` fields (`tone`, `pace`, `emphasis`, `instructions`) are free-text suggestions passed to the TTS provider
- Voice names are validated against the runtime voice list at generation time

### Editing the script

You can edit `narration_script.json` directly in any text editor and then re-run audio generation. The pipeline validates the edited script before calling the TTS API. Common edits:

- Change narration text in `say` segments
- Adjust pause durations
- Add or remove pause segments
- Change voice for a specific clip
- Modify `style` guidance

---

## Scene Brief Format

The scene brief (`scene_brief.json`) is an intermediate planning artifact generated by Stage 1. It captures **instructional intent** — what changes in the scene and why it matters — without any narration text.

Full specification: [`ar_scene_narration_brief_standard.md`](ar_scene_narration_brief_standard.md)

```json
{
  "scene": "A 3D bowl-shaped loss surface floating in AR space.",
  "objectives": [
    "Explain what a loss surface represents",
    "Show how gradient descent reduces loss iteratively"
  ],
  "modules": [
    {
      "title": "Introduction",
      "clips": [
        {
          "change": "The loss surface appears in the AR environment",
          "meaning": "Students need visual context before the concept of descent makes sense"
        }
      ]
    }
  ]
}
```

The brief is not intended for human review in normal use — it is an intermediate step. However, reviewing it is useful when the final narration seems off, since it reveals what the LLM understood about the lab's instructional structure.

---

## Project Structure

```
narration_generator/
│
├── cli.py                      Entry point — run, list, status commands
├── config.py                   API keys, paths, model names, price tables
├── conftest.py                 pytest sys.path setup
├── pytest.ini                  pytest config (testpaths = tests)
│
├── narration_definitions.py    Pydantic models: ARLabNarrationScript, NarrationClip,
│                               SaySegment, PauseSegment, ClipStyle, TTSRuntimeConfig
├── script_definitions.py       Pydantic models: ARSceneNarrationBrief, SceneModuleBrief,
│                               SceneClipBrief
│
├── pipeline/
│   ├── brief_generator.py      Stage 1: description → ARSceneNarrationBrief (via instructor)
│   ├── script_generator.py     Stage 2: ARSceneNarrationBrief → ARLabNarrationScript (via instructor)
│   ├── tts_generator.py        Stage 3: ARLabNarrationScript → MP3 (OpenAI TTS + pydub)
│   ├── filename_utils.py       Filename convention: make_safe_slug, make_clip_filename, enumerate_clips
│   ├── staleness.py            Hash-based stale audio detection: compute_clip_hash, is_stale
│   ├── cost_tracker.py         Token and character cost recording: CostTracker
│   └── validation.py           Pre-TTS validation: validate_narration_script, ValidationResult
│
├── models/
│   ├── manifest_models.py      ProjectManifest, AudioManifest, AudioClipRecord
│   ├── log_models.py           LLMCallEntry, TTSCallEntry, ValidationEntry, SystemEntry
│   └── cost_models.py          CostReport, LLMCostRecord, TTSCostRecord
│
├── storage/
│   └── project_storage.py      ProjectStorage — all disk I/O for a project folder
│
├── tests/
│   ├── fixtures/
│   │   ├── sample_description.txt
│   │   ├── sample_brief.json
│   │   └── sample_script.json
│   ├── test_filename_utils.py
│   ├── test_staleness.py
│   ├── test_validation.py
│   ├── test_brief_generator.py
│   ├── test_script_generator.py
│   └── test_tts_generator.py
│
├── narration_projects/         Generated output (git-ignored)
│   └── {project_id}/
│       ├── description.txt
│       ├── scene_brief.json
│       ├── narration_script.json
│       ├── project_manifest.json
│       ├── audio_manifest.json
│       ├── generation_log.jsonl
│       ├── cost_report.json
│       └── audio/
│           └── m01_c001_*.mp3 ...
│
└── *.md                        Design documents (see below)
```

### Design documents

| File | Contents |
|------|----------|
| [`ar_tts_generation_flask_design_plan.md`](ar_tts_generation_flask_design_plan.md) | Full system architecture, phased implementation plan, open design questions |
| [`ar_tts_narration_json_standard.md`](ar_tts_narration_json_standard.md) | Narration script JSON specification (schema v0.1) |
| [`ar_scene_narration_brief_standard.md`](ar_scene_narration_brief_standard.md) | Scene brief JSON specification |

---

## Configuration

`config.py` reads from environment variables (via `.env`) with sensible defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for LLM stages |
| `OPENAI_API_KEY` | — | Required for TTS (not needed with `--dry-run`) |
| `NARRATION_PROJECTS_DIR` | `./narration_projects` | Output directory for generated projects |

### Overriding defaults at runtime

LLM model, TTS model, and voice can all be overridden per-run via CLI flags. See [CLI Reference](#cli-reference).

### Price tables

`config.py` contains `CLAUDE_PRICES` and `OPENAI_TTS_PRICES` dictionaries used to estimate costs in the cost report. These are estimates based on published pricing and should be updated if pricing changes.

---

## Running Tests

All tests are unit tests with no API calls. Run from the `narration_generator/` directory:

```powershell
python -m pytest -v
```

Expected: **57 tests, all passing**.

Tests cover:
- `filename_utils` — slug generation, filename convention, clip enumeration
- `staleness` — hash determinism, stale detection on text/voice/model/pause changes
- `validation` — voice errors, title warnings, length warnings
- `brief_generator` — LLM call structure, log entry fields (mocked)
- `script_generator` — LLM call structure, log entry fields (mocked)
- `tts_generator` — segment routing, provider calls, file saving (mocked)

---

## Extending the System

### Adding a TTS provider

Implement the `TTSProvider` protocol in `pipeline/tts_generator.py`:

```python
class MyTTSProvider:
    @property
    def provider_name(self) -> str: return "myprovider"

    @property
    def model_name(self) -> str: return "my-model"

    def synthesize(self, text: str, voice: str, **kwargs) -> bytes:
        # Return raw MP3 bytes
        ...
```

Pass an instance to `TTSGenerator(provider=MyTTSProvider(...), default_voice="...")` in `cli.py`. No other changes needed — the rest of the pipeline is provider-agnostic.

### Editing and re-running audio

1. Open `narration_projects/{project_id}/narration_script.json`
2. Edit any `say` text or `pause` seconds
3. Re-run with the same project directory (a selective regeneration mode using staleness detection is planned but not yet wired into the CLI)

### Adding a review UI

The design document [`ar_tts_generation_flask_design_plan.md`](ar_tts_generation_flask_design_plan.md) contains a full plan for a Flask-based review interface with:
- Clip-by-clip narration editing
- Audio playback per clip
- Selective regeneration of stale clips
- Validation display

All supporting data structures (`ProjectManifest`, `AudioManifest`, `AudioClipRecord`, `generation_log.jsonl`) are already in place. The Flask layer would read/write these files through `ProjectStorage` and call the same pipeline functions used by the CLI.

### Using the Pydantic models directly

The two core schema files can be imported independently of the CLI:

```python
import sys
sys.path.insert(0, "path/to/narration_generator")

from narration_definitions import ARLabNarrationScript
from script_definitions import ARSceneNarrationBrief

# Load and validate an existing script
script = ARLabNarrationScript.model_validate_json(
    open("narration_script.json").read()
)

# Access clips
for module in script.modules:
    for clip in module.clips:
        text = " ".join(s.text for s in clip.segments if s.type == "say")
        print(clip.title, len(text))
```

# AR Lab TTS Narration JSON Standard

**Status:** Draft standard 0.1  
**Primary use:** Structured text-to-speech narration for AR lab generation  
**Audience:** Coding models, human developers, instructional designers, and reviewers  

---

## 1. Purpose

This JSON format describes narration scripts for augmented reality lab experiences. The file is intended to provide structured input to a text-to-speech generation pipeline, especially one using OpenAI TTS voices or a similar provider.

The format organizes narration by **modules** and **clips**. Each clip contains an ordered sequence of spoken text segments and optional pause segments. The JSON should be simple enough for an LLM to generate reliably, but structured enough for validation, audio generation, filename generation, and later expansion.

The authored JSON is not intended to describe AR object placement, scene anchors, spatial audio positions, animation timing, or detailed synchronization. Those should be handled by separate AR scene metadata or downstream generation code.

---

## 2. Design Goals

The standard is guided by the following principles:

1. **Keep the authored JSON lightweight.**  
   The narration file should mainly describe what is said and how the narration is organized.

2. **Use list order for ordering.**  
   Modules and clips are ordered by their position in the JSON arrays. No separate `module_order`, `clip_order`, or `clip_key` fields are required.

3. **Avoid over-constraining narration.**  
   Speech duration depends on the text, voice, TTS model, pacing, and style instructions. The authored file should not specify speech durations.

4. **Use explicit timing only for pauses.**  
   Pause segments may specify `seconds`. This is the only timing field in the authored JSON.

5. **Treat voice names dynamically.**  
   Voice names should be strings, not hard-coded enumerations. Available voices should be loaded from the TTS provider, a configuration file, or a cached provider capability list.

6. **Treat style fields as suggestions.**  
   Style, tone, pace, and emphasis are free-text suggestions, not enumerated values.

7. **Generate filenames by convention.**  
   Authored JSON should not need to include filenames. The generation pipeline should create filenames based on module number, clip number, and optional clip title.

8. **Keep all lab audio files in one directory.**  
   The file naming convention should distinguish modules and clips without requiring subdirectories per module.

---

## 3. Conceptual Structure

The JSON structure is:

```text
script
  modules[]
    clips[]
      segments[]
```

A segment is one of two types:

```text
say    spoken narration
pause  intentional silence
```

---

## 4. Minimal Valid JSON

The smallest valid narration file is:

```json
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
```

---

## 5. Recommended Authored JSON Example

```json
{
  "schema_version": "0.1",
  "lab_title": "Gradient Descent",
  "default_voice": "alloy",
  "style_preset": "clear lecture for first-year college students",
  "modules": [
    {
      "title": "The Update Rule",
      "clips": [
        {
          "title": "Introduction",
          "style": {
            "tone": "friendly and clear",
            "pace": "moderate, with extra care around the equation",
            "instructions": "Sound like an instructor explaining this at a whiteboard."
          },
          "segments": [
            {
              "type": "say",
              "text": "Welcome to Lab Two. In this module, we will explore the update rule that powers gradient descent."
            },
            {
              "type": "pause",
              "seconds": 0.6
            },
            {
              "type": "say",
              "text": "The central idea is simple: start with a value, measure how the error changes, and then take a step in the direction that lowers the error.",
              "emphasis": "Make this sound like the main takeaway."
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 6. Top-Level Fields

### Required

| Field | Type | Description |
|---|---|---|
| `modules` | array | Ordered list of narration modules. Must contain at least one module. |

### Optional

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema version. Default is `0.1` if omitted. |
| `lab_id` | string | Optional lab identifier. Useful for logging or output directory naming. |
| `lab_title` | string | Optional human-readable lab title. |
| `default_voice` | string | Optional default TTS voice. This should be validated dynamically by the generation pipeline. |
| `style_preset` | string | Optional global narration style suggestion. This is free text, not an enum. |

### Excluded Top-Level Fields

The following fields are intentionally excluded from the standard:

| Excluded Field | Reason |
|---|---|
| `course` | Course metadata is not needed for audio generation. |
| `output` | Output directories and filenames should be controlled by the generation pipeline. |
| `global_warnings` | Warnings should be generated by validation/preflight tools, not authored manually. |

---

## 7. Module Fields

A module groups a set of clips. Modules are ordered by their position in the `modules` array.

### Required

| Field | Type | Description |
|---|---|---|
| `clips` | array | Ordered list of narration clips. Must contain at least one clip. |

### Optional

| Field | Type | Description |
|---|---|---|
| `title` | string | Human-readable module title. Useful for review, but not required for generation. |

### Excluded Module Fields

| Excluded Field | Reason |
|---|---|
| `module_id` | Not required; module number is inferred from list position. |
| `module_order` | Not required; order is given by array position. |

---

## 8. Clip Fields

A clip corresponds to one generated audio file. Clips are ordered by their position within the module's `clips` array.

### Required

| Field | Type | Description |
|---|---|---|
| `segments` | array | Ordered list of spoken and pause segments. Must contain at least one `say` segment. |

### Optional

| Field | Type | Description |
|---|---|---|
| `title` | string | Optional human-readable clip title. May be used for filename slug generation. |
| `voice` | string | Optional voice override for this clip. Should be validated dynamically. |
| `style` | object | Optional clip-level style guidance. |

### Excluded Clip Fields

| Excluded Field | Reason |
|---|---|
| `clip_id` | Not required; clip number is inferred from list position. |
| `clip_key` | Not required; filenames are generated from position and optional title. |
| `filename` | Generated by the pipeline, not authored manually. |
| `ar_anchor` | AR placement and interaction metadata should be stored separately. |
| `target_duration_s` | Over-constrains narration. |
| `duration_hint_s` | Removed to keep timing out of authored speech text. |
| `max_duration_s` | Removed from authored JSON; strict synchronization belongs elsewhere. |
| `estimated_duration_s` | Generated by preflight or post-processing, not authored. |
| `actual_duration_s` | Generated after audio creation, not authored. |

---

## 9. Style Object

The `style` object is optional and applies to a single clip. These fields are suggestions to the TTS generation layer. They are not controlled vocabularies.

```json
{
  "tone": "friendly and clear",
  "pace": "moderate, with extra care around the equation",
  "emphasis": "Lightly emphasize the key idea.",
  "instructions": "Sound like an instructor explaining this at a whiteboard."
}
```

### Optional Style Fields

| Field | Type | Description |
|---|---|---|
| `tone` | string | Broad tone suggestion. Example: `friendly and clear`. |
| `pace` | string | Pacing suggestion. Example: `slow enough for first-year students`. Not a timing guarantee. |
| `emphasis` | string | Clip-level emphasis guidance. |
| `instructions` | string | Free-form narration instructions. |

---

## 10. Segment Types

Segments are ordered. The generated audio should concatenate the rendered segments in order.

### 10.1 `say` Segment

A `say` segment contains text to be spoken.

```json
{
  "type": "say",
  "text": "The learning rate controls the size of each step."
}
```

Optional segment-level emphasis:

```json
{
  "type": "say",
  "text": "The learning rate controls the size of each step.",
  "emphasis": "Make this sound like an important takeaway."
}
```

#### Required Fields

| Field | Type | Description |
|---|---|---|
| `type` | string | Must be `say`. |
| `text` | string | Text to be spoken. Must not be blank. |

#### Optional Fields

| Field | Type | Description |
|---|---|---|
| `emphasis` | string | Natural-language emphasis suggestion for this segment. |

### 10.2 `pause` Segment

A `pause` segment specifies intentional silence.

```json
{
  "type": "pause",
  "seconds": 0.6
}
```

#### Required Fields

| Field | Type | Description |
|---|---|---|
| `type` | string | Must be `pause`. |
| `seconds` | number | Length of the pause in seconds. Must be greater than 0. |

#### Pause Length Guidance

Typical pauses should be between `0.3` and `1.5` seconds.

Long pauses are allowed but should be used sparingly. A recommended validation rule is to reject pauses longer than `10` seconds unless the pipeline has a specific reason to allow them.

---

## 11. Timing Policy

The authored JSON should not specify speech duration.

Allowed timing field:

```text
pause.seconds
```

Fields intentionally excluded from authored JSON:

```text
target_duration_s
duration_hint_s
max_duration_s
estimated_duration_s
actual_duration_s
```

Rationale:

- The length of spoken narration depends on text, voice, style, model behavior, and generation settings.
- Hard timing constraints can produce awkward narration.
- Duration measurement should happen after generation.
- Preflight estimates and actual audio durations can be stored in a generated report or separate metadata file.

---

## 12. Voice Policy

Voice fields should be strings, not enums.

Examples:

```json
{
  "default_voice": "alloy"
}
```

```json
{
  "voice": "nova"
}
```

The schema should not hard-code the provider's voice list. The generation pipeline should validate voice names against a runtime source, such as:

1. A provider capability query.
2. A local configuration file.
3. A periodically updated cache.

Example runtime voice configuration:

```json
{
  "allowed_voices": [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse"
  ],
  "default_voice": "alloy"
}
```

If no valid voice list is available at runtime, the generation code may skip voice validation and use the configured provider default.

---

## 13. Filename Convention

The authored JSON should not include filenames.

The generation pipeline should create one audio file per clip. All generated files for a lab should be placed in the same output directory.

Recommended filename pattern:

```text
m{module_number:02d}_c{clip_number:03d}_{safe_clip_title}.mp3
```

If the clip has no title:

```text
m{module_number:02d}_c{clip_number:03d}.mp3
```

Examples:

```text
m01_c001_introduction.mp3
m01_c002_formula_explanation.mp3
m02_c001_learning_rate_intro.mp3
```

Filename generation rules:

1. Module numbers start at 1.
2. Clip numbers start at 1 within each module.
3. Clip titles should be converted to lowercase slug form.
4. Spaces and punctuation should be replaced with underscores or removed.
5. Filenames should be deterministic.
6. Duplicate slugs should be disambiguated by the module and clip numbers.

---

## 14. Validation Expectations

A validator should enforce the following:

1. The root object must contain `modules`.
2. `modules` must contain at least one module.
3. Each module must contain `clips`.
4. Each `clips` array must contain at least one clip.
5. Each clip must contain `segments`.
6. Each `segments` array must contain at least one segment.
7. Each clip must contain at least one `say` segment.
8. A `say` segment must contain non-blank `text`.
9. A `pause` segment must contain `seconds > 0`.
10. A recommended upper bound for `pause.seconds` is `10` seconds.
11. Unknown fields should be rejected unless the schema is explicitly revised.
12. Voice names should be validated separately against the runtime voice list.

---

## 15. Generated Metadata

The generation pipeline may produce additional metadata after validation or audio creation. This should not be added to the authored JSON unless there is a deliberate post-processing format.

Possible generated metadata includes:

```json
{
  "filename": "m01_c001_introduction.mp3",
  "estimated_duration_s": 18.5,
  "actual_duration_s": 19.1,
  "warnings": []
}
```

Recommended approach:

- Keep the authored narration JSON clean.
- Save generated metadata in a separate manifest file, such as:

```text
audio_manifest.json
```

The manifest can map modules and clips to generated filenames, durations, warnings, and provider metadata.

---

## 16. Relationship to AR Metadata

This file does not specify:

- AR anchors
- 3D object placement
- scene triggers
- spatial audio positions
- animation timing
- Unity object names
- interaction logic

Those should be handled by separate AR scene files or a separate integration manifest.

This separation keeps the narration standard reusable across different AR platforms and rendering systems.

---

## 17. Recommended Pydantic Model Behavior

The Pydantic model used for this standard should:

1. Use discriminated unions for segment types.
2. Forbid unknown fields by default.
3. Treat voices as strings.
4. Treat style fields as free-text strings.
5. Strip whitespace from optional strings.
6. Reject blank strings when provided.
7. Require at least one `say` segment per clip.
8. Validate voice names through a separate runtime configuration step, not through enums.

---

## 18. Guidance for LLM Generation

When asking an LLM to generate this format, the prompt should emphasize:

- Produce valid JSON only.
- Use the module and clip arrays to express order.
- Do not invent filenames.
- Do not include AR anchors.
- Do not include course metadata.
- Do not include speech duration fields.
- Use pauses only when pedagogically useful.
- Use style and emphasis as natural-language suggestions.
- Keep clips reasonably short and coherent.
- Prefer several short clips over one long clip when the narration naturally changes topic.

---

## 19. Future Expansion Possibilities

Future versions may add related but separate standards for:

1. **Audio generation manifest**  
   Generated filenames, durations, warnings, provider information, and checksums.

2. **Localization**  
   Alternate languages or translated narration while preserving module and clip structure.

3. **Caption/subtitle export**  
   WebVTT, SRT, or Unity-compatible caption timing generated after TTS creation.

4. **Voice profiles**  
   Named voice/style bundles stored outside the authored narration JSON.

5. **AR integration manifest**  
   Separate mapping from generated audio files to AR triggers, objects, or scene events.

6. **Accessibility metadata**  
   Captions, transcripts, reading level estimates, and alternate narration versions.

These should be added carefully without making the core authored narration file too heavy.

---

## 20. Summary

This standard defines a lightweight JSON format for AR lab narration. The core file describes:

- modules,
- clips,
- spoken text,
- explicit pauses,
- optional voice choices,
- optional natural-language style guidance.

It intentionally excludes:

- course metadata,
- AR anchors,
- filenames,
- clip keys,
- explicit speech timing,
- generated warnings,
- generated duration metadata.

The result is a simple, robust input format that can be generated by LLMs, reviewed by humans, validated with Pydantic, and used by a TTS generation pipeline.

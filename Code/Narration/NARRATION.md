# Narration generation: goals spec

Status: **agreed** (#91). This spec decides what the narration tool produces.
Built so far: the zip layout (#92, `bundle.py`) and the narration CLI (#93,
`narrate_lab.py` + `tts.py`). The spoken-form pass (#94) is next.

## Why

Generated labs already carry their narration. In the wave-2 json_lab L3/L4
outputs (104 labs, 1,824 clips):

- `Clip.narration` is filled on 98% of clips (median 29 words, longest
  966 characters). That is about 3,500 characters, or roughly 4 minutes of
  audio, per lab.
- `Clip.audioClip` is filled on 95% of clips. Within a lab the names are
  unique and safe to use as file names.

The headset plays a clip's audio by resource name (`Resources.Load<AudioClip>`
/ `MediaCatalogue.GetAudioClip`). So narration generation means turning each
clip's existing `narration` into one audio file named after its `audioClip`, and
shipping those files inside the lab resources zip. The lab's content is not
rewritten.

## Goals

1. **Narrate an existing lab, faithfully.** The input is a finished lab file.
   The audio says what `Clip.narration` says. No new text, structure or clips
   are invented.
2. **One audio file per narrated clip,** named `<audioClip>.<ext>`, so it drops
   into the headset with no changes there.
3. **Headset-ready output.** The final product is the lab resources zip (#92),
   built from a working folder that can always rebuild it.
4. **Usable as paper data.** Each run records the provider, requested and
   served model, voice, spoken-form mode, cost, time and per-clip details, in
   the same spirit as the benchmark metrics files.
5. **Cheap to rerun.** A rerun redoes only the clips whose inputs changed.
   Later review and regeneration tools (#96, #97) build on this.
6. **Provider-neutral.** Two cloud providers first (OpenAI `gpt-4o-mini-tts`,
   Gemini 3.8 Flash TTS), local models later (#95), all behind one small
   interface.

Non-goals for now: rewording narration (polish pass, #98), voice cloning
(#98), review UIs (#96), per-module hierarchical narration (#97), ARLEM, and
languages other than English.

## Input

- **A complete json_lab `Lab` file:** a one-shot output
  (`*_json_lab_L3_*_output.json`, `*_L4_*`, either `--structure`) or a
  hierarchical run's assembled lab. It must validate as `Code/Schemas/json_lab.Lab`.
- **Not supported:** L1 `LabOutline`, module-only `DemoModule` outputs, and
  ARLEM specs. The tool exits with a clear message for these.
- **Topic:** the spoken-form pass and the glossary are keyed by topic (the lab
  YAML name, e.g. `dna_structure_and_replication`). A `Lab` does not record its
  topic, so the tool resolves it in this order:
  1. `--topic`
  2. `lab_name` in the paired `*_metrics.json` beside the input, if present
  3. none, which means the shared glossary only

  The LLM spoken-form mode also gets `courseName`, `labId` and `objectives`
  as context.

## Clip handling (edge cases)

Clips are visited in order: module by module, then clip by clip. In the table,
`m` is the 1-based module number. "The bundled lab" is the copy of the lab
JSON that goes into the zip; **the input file is never modified.**

| Case | Seen in wave 2 | Handling |
|---|---|---|
| `narration` and `audioClip` both set | 95% of clips | Narrate as `<audioClip>.<ext>`. |
| `narration` set, `audioClip` empty | Gemini 3.1 Pro: only 20–49% of clips named | Generate the name from `clipName`, used alone when no other clip in the lab has that `clipName` and no clip uses it as an audio name, otherwise prefixed `m{m:02d}_` (e.g. `m03_clip4`; `clipName` often repeats across modules). Add `_2`, `_3`… on collision. Names are compared ignoring case throughout, because `Intro.wav` and `intro.wav` are one file on Windows and macOS. Write the name into the bundled lab only, and record it as `name_change: generated` in the manifest. |
| `audioClip` set, `narration` empty or blank | GPT-5.4 nano L4: 25% of clips | No audio. Remove `audioClip` from the bundled lab so the headset doesn't ask for a missing file. Report it as `dropped_name`. |
| Neither set | rare | Nothing to do; counted as `silent`. |
| Duplicate `audioClip` within a lab, including names that differ only in case | 0 labs | Same narration: one shared file, used under the first clip's exact name. Different narration: keep the first and rename the later ones with `_2`, `_3`… in the bundled lab. Reported. |
| `audioClip` has unsafe characters or an extension | 0 clips | Remove the extension and make the name file-safe. If that changes the name, write the new name into the bundled lab. Reported. |
| `CheckAngleComponent.audioClipSuccess` | 10 references | A sound effect, not narration. Left as is and listed in the report only. |
| `autoAdvance: true` on a clip with no audio | 34 of 623 auto-advance clips | Left as is and reported. How the headset advances a clip with no audio is unverified. |

Every rename and drop is listed in the manifest, so the bundled lab can be
compared with the original.

## Text sent to the voice service

- **Display text stays put.** `Clip.narration` is never changed; it stays the
  subtitle and display text.
- **Spoken text** is produced by the spoken-form pass (#94):
  `--spoken-form {llm,rules,none}`. It may only expand symbols, numbers, units,
  formulas and acronyms in context (DNA `5'` → "five-prime"). It never rewords.
  The default is chosen by measured cost (#94).
- **One request per clip.** The whole clip's spoken text goes in one request,
  for natural flow across sentences. There is no segment splitting and no
  inserted silences. The longest wave-2 clip (966 characters) fits every
  candidate provider's per-request limit; a longer clip is an error, never
  split silently.
- **One voice and one style for the whole lab.** Per-clip voices or styles
  are out of scope; a fixed setup keeps paper runs comparable.

### Defaults (frozen for paper runs)

| Provider | Model | Voice |
|---|---|---|
| OpenAI | `gpt-4o-mini-tts` (dated snapshot pinned in #93) | `marin`, one of the two voices OpenAI recommends for best quality |
| Gemini | Gemini 3.8 Flash TTS | `Charon`, a stock voice commonly picked for academic narration |

Style instruction, the same for both providers:

> Speak as a clear, friendly science instructor guiding a student through a
> lab, at a calm, moderate pace.

These defaults are set once in code and recorded in every manifest. Changing
any of them means a new run group; old and new runs are never mixed. #93 does
one short listening check before the first paper run, and that is the last
chance to swap a voice.

## Output

The **working folder** goes under
`Artifacts/Data/Narration/[<run group>/]<run name>/`. Like `OneShot/` and
`Hierarchical/`, it is never pooled with `Benchmark/`. The run name is
`<input stem>_<provider>_<timestamp>`.

```
<run name>/
  narration_manifest.json     run record + per-clip records (below)
  <Name>Lab/                  exactly what the zip contains (see "Resources zip")
    Audios/<audioClip>.wav
    Textures/
    Videos/
    <Lab_Name>.json           the bundled lab (renames/drops applied)
  <Name>Lab.zip               the resources zip, built from <Name>Lab/
```

- The manifest is rewritten after every clip. An interrupted run can be
  resumed, and finished clips are not paid for twice.
- The zip is built from the `<Name>Lab/` folder at the end of a run, and is
  rebuilt on any later run. It is never edited by hand.
- Audio is never lossy-encoded: request raw PCM from the provider and write it
  as WAV in the bundle format below. Both providers return that format
  natively, so nothing is resampled.

**What is committed:**
- Committed: `narration_manifest.json` and the bundled `<Lab_Name>.json`.
- Not committed: the `.wav` files and the zip, which contains them. Both are
  covered by `.gitignore` rules.
- Hand-picked example audio can be committed later, under a path with an
  explicit `.gitignore` exception, decided when we pick the examples.

### Resources zip (#92)

The headset downloads `<lab id>.zip`, unpacks it into `lab_resources/`, opens
`lab_resources/<lab id>/` and takes the first `.json` it finds there
(`LoginManager.cs` / `LabManager.cs` on `jw-expermiments`). So the zip holds
one top-level folder with the zip's own name:

```
<Name>Lab.zip
  <Name>Lab/
    Audios/<audioClip>.wav     one per narrated clip; flat, .wav only
    Textures/                  present, empty for now
    Videos/                    present, empty for now
    <Lab_Name>.json            the lab, json_lab v2.0 format
```

- **Names come from `labId`.** Split on any non-alphanumeric run, drop
  standalone "lab" words so names never end in `LabLab`, and capitalize each
  word's first letter only, so acronyms survive.
  - `dna_structure_replication` → `DnaStructureReplicationLab.zip`, folder
    `DnaStructureReplicationLab/`, and `Dna_Structure_Replication.json`.
  - `GP1-AR-Vectors-Projectile-01` → `GP1ARVectorsProjectile01Lab`.
  - `--bundle-name` replaces the zip and folder name, e.g. to match the lab's
    id on the server. The zip, its folder and the server id must all match.
  - A labId with no usable words needs `--bundle-name`.
- **Audio files** are named after each clip's `audioClip` value, which is how
  the headset looks them up, not after `clipName`. The name is used exactly
  as written, never trimmed, and the extension is a lowercase `.wav`, because
  the headset's filesystem is case-sensitive. Format: **WAV, 24 kHz, 16-bit
  PCM, mono**.
- **The lab JSON ships in our v2.0 format as is,** with the clip renames and
  drops above applied. The headset currently reads the older Transmission
  format, so it needs a matching update to read v2.0; that work is outside this
  repo. `prototype_json_patcher.revert_lab()` converts v2.0 to Transmission if
  that is ever needed.
- **Zips are reproducible:** entries are sorted with a fixed timestamp, so the
  same contents give the same bytes. Every folder has its own entry, so empty
  `Textures/` and `Videos/` survive unpacking. The zip is readable by everyone
  (0644), so a web server can serve it.
- **`Code/Narration/bundle.py`** implements the names, the layout, zipping and
  a validator:

  ```
  python "Code/Narration/bundle.py" names    <lab.json> [--bundle-name NAME]
  python "Code/Narration/bundle.py" zip      <Name>Lab/
  python "Code/Narration/bundle.py" validate <Name>Lab.zip | <Name>Lab/
  ```

  The validator returns errors for:
  - a damaged zip: corrupted contents, or an entry the headset can't unpack
    (unsupported compression, encryption). A download cut off partway has no
    zip index, so it is reported as "not a zip file" instead
  - a file in a bundle folder that can't be read (e.g. a broken symlink)
  - a top-level folder that doesn't match the zip name, or more than one
    top-level entry (e.g. `__MACOSX/`)
  - a missing `Audios/`, `Textures/` or `Videos/`, or anything else at the top
  - not exactly one lab JSON (the extension must be a lowercase `.json`), a
    JSON that isn't a valid json_lab `Lab`, or modules, clips or `audioClip`
    values of the wrong type
  - files other than lowercase `.wav`, or subfolders, in `Audios/`, or a WAV
    that isn't 24 kHz, 16-bit, mono PCM with some audio in it
  - a clip `audioClip` that isn't a safe file name (spaces included) or has no
    matching `Audios/<name>.wav`

  It returns warnings for audio files no clip uses, and for sound effects
  (`audioClipSuccess`) with no file. `validate` exits 1 if there are any
  errors. `Code/Narration/smoke_test.py` checks all of this offline,
  including all 104 wave-2 labs.

### `narration_manifest.json`

**Run record:**

| Group | Fields |
|---|---|
| Source | input path; `labId`; source model and level; `lab_name` (from the paired metrics file, if any) |
| Voice service | provider; requested model; served model(s); voice; style instruction; audio format and sample rate |
| Spoken form | mode; LLM model, tokens and cost (llm mode); glossary version (rules mode) |
| Counts | clips narrated, skipped (unchanged), failed, silent; generated, dropped and renamed names |
| Totals | characters; audio seconds; TTS cost; spoken-form cost; total cost; wall time; transient retries |
| Provenance | tool git commit; package versions; timestamps |

**Per clip:** module and clip index; `clipName`; final audio name (and why it
changed, if it did); display text; spoken text; characters; audio seconds;
cost; wall time; time to first audio and (OpenAI) the longest gap between
audio pieces; served model; the provider's request id (to match billing
records); status (`generated` / `skipped` / `failed` / `silent` / `dropped`,
plus `shared` for a clip that reuses another clip's file); `transient_errors`,
one entry per retried attempt with its HTTP status, exception type, a short
readable `reason` (e.g. "the provider was temporarily unavailable (HTTP 503)")
and the trimmed message; `error` and `error_reason` for a clip that failed; and
a **source hash**. The run record also keeps a `runs` list (one
entry per run of the folder) and an overall `status`: `complete`, `incomplete`
(some clips failed; resume it), `failed` (stopped early) or `invalid` (the zip
failed validation). Totals cover every clip with audio, so a resumed lab's cost
stays whole; `cost_usd_this_run` is the part this run paid for.

The source hash covers the spoken text, provider, model, voice, style
instruction, format and sample rate. A rerun skips a clip when the hash
matches and the file exists, unless `--overwrite` is passed.

### Paper metrics

These come straight from the manifest: cost per lab and per audio minute,
seconds of audio per lab, wall time and real-time factor, and failure and
retry rates. Each is reported per provider, per spoken-form mode, and per
source model and level.

Quality is judged by listening checks; there is no automatic transcription
metric.

## Narration CLI (#93)

```
python "Code/Narration/narrate_lab.py" <lab_output.json> --provider {openai,gemini} [--dry-run]
    [--voice NAME] [--topic NAME] [--bundle-name NAME] [--run-group NAME] [--overwrite]
python "Code/Narration/narrate_lab.py" --resume <run folder> [--overwrite]
```

- `--dry-run` prints the clip plan (names and any name changes) and a character
  count, and sends nothing. It needs no API key.
- A run writes the working folder, narrates clip by clip (rewriting the manifest
  after each), then zips and validates the bundle. Exit 0 means complete and
  valid; 1 means incomplete, failed or invalid (resume it); 2 means bad input or
  environment.
- `--resume` reruns a folder with the settings in its manifest. Only clips whose
  audio is missing or whose source hash changed are sent; `--overwrite` sends
  all of them. Valid audio is never paid for twice: unchanged clips are skipped
  even after a stop-the-run error, and a run killed part-way keeps the records
  of the clips it never reached.
- `--spoken-form` only offers `none` until #94 adds `rules` and `llm`.
- **Failures:**
  - A clip that fails is recorded and the run continues. No zip is built until
    every narrated clip has audio, and a stale zip is removed.
  - A reply with no audio (an empty stream, or Gemini with a finish reason such
    as OTHER) is retried like a transient error. It only fails its own clip,
    never the run.
  - HTTP 401/402/403/404 (key, billing such as used-up prepaid credits,
    permission, unknown model), or audio in an unexpected format, stops the run
    at once, since every clip would fail and unusable audio is still billed.
  - Transient errors (429/5xx/connection) are retried up to 3 attempts, with
    the same policy as `benchmark._transient_retryer()`, and counted. The SDK
    clients never retry on their own.
  - **Every request has a 90 s limit** (`tts.REQUEST_TIME_LIMIT`; a clip
    normally takes under 12 s). At the limit the request is abandoned and retried
    like a transient error, recorded as "no complete reply within the
    per-request time limit".
    - OpenAI streams its audio. A watchdog closes a stream still open at 90 s,
      whether it has gone silent or is trickling, and the client timeout covers
      the wait for the stream to start. This is a hard limit. A reply that
      finishes in the instant the watchdog fires is kept.
    - Gemini answers in one piece. Its limit is the SDK timeout, which applies
      to each wait (connecting, each read) rather than to the whole request,
      plus the same 90 s sent to the server as a deadline (`X-Server-Timeout`).
      So it isn't strictly a hard total: a reply trickling in could run longer,
      though one-piece replies make that unlikely.
    - Each clip records `first_audio_seconds` and, for OpenAI,
      `max_gap_seconds`, which show whether a slow request waited before
      starting or stalled partway.
  - A clip longer than the provider's limit (OpenAI 4,096 characters, our Gemini
    cap 4,000) fails without a request.
- **OpenAI:** `/v1/audio/speech` with `gpt-4o-mini-tts-2025-12-15`, voice
  `marin`, the style sentence as `instructions`, `response_format="pcm"` and
  `stream_format="sse"`. The SSE stream is used because its closing
  `speech.audio.done` event reports token usage, which the plain audio response
  doesn't. The speech endpoint doesn't say which model answered: the
  2026-10-27 live run got no `openai-model` header, so OpenAI's served model is
  recorded as unknown. Each clip records the request id (`x-request-id`)
  instead.
- **Gemini:** `generateContent` on `gemini-3.8-flash-tts` with audio output,
  `speech_config.voice_config.voice = "Charon"`, and the style sentence in the
  part's `speech_metadata.style`, so it is never read aloud. Raw PCM
  (`audio/L16;rate=24000`) and WAV responses are both accepted. Anything not
  24 kHz, 16-bit, mono is a format error, never resampled.
- **Live run, 2026-10-27** (`Artifacts/Data/Narration/live-check/`):
  - Both providers narrated the DNA lab into valid zips, and Gemini 3.8 Flash
    TTS accepted the flat `voice` field and `speech_metadata.style`.
  - Gemini returned 25.0 audio tokens per second of audio and OpenAI 24.6.
  - Still open: whether Gemini follows the style without ever reading it aloud
    (the #93 listening check), and whether the prices in `tracking/pricing.py`
    match what was billed (#101).
  - Some OpenAI requests stalled for 2–4 minutes before succeeding (wall time
    was 1.85× the audio length, against Gemini's 0.34×). That led to the 90 s
    per-request limit above. The offline smoke test
  (`smoke_test.py` parts 2 and 3: a fake service, then mock OpenAI and Gemini
  servers) covers everything else.

## Engineering rules (from the repo, not from JW's code)

- **Pricing** lives in `Code/Benchmark/tracking/pricing.py`.
  `gpt-4o-mini-tts` and Gemini TTS bill by tokens (text in, audio out), so use
  the reported usage, not a per-character estimate.
- **Retries:** transient errors only (429 / 5xx / connection), as in
  `benchmark._transient_retryer()`, counted separately. Authentication and
  config errors fail at once.
- **Served model:** record what the provider says it actually served, as
  `tracking.metrics.served_model_of` does for LLMs, and warn when it differs from the
  requested model.
- **LLM calls** (spoken-form `llm` mode) go through `create_instructor_client`
  plus `provider_create_kwargs` and are tracked by `tracking/`, like every LLM
  entry point (CLAUDE.md). TTS calls are not instructor calls.
- **Environment check** before any call, in the style of
  `benchmark.require_environment()`: provider SDK installed and API key set.
- **Dependencies** go in the root `requirements.txt` and keys in the root
  `.env`; no separate environment. Gemini TTS needs `google-genai>=2.25.0`
  (the first release with `speech_metadata` and `VoiceConfig.voice`). The CLI
  checks this and stops with the upgrade command.
- **Offline smoke test** with a fake TTS provider and a fake LLM, in the style
  of `Code/Hierarchical/smoke_test.py`, so it runs with no API keys.

## Review of JW's `narration_generator/`

Branch `jw-expermiments`, commit 093a906. It turns a plain-text description
into a Claude scene brief, then a Claude narration script (spoken lines and
pauses), then OpenAI `tts-1` MP3s joined with pydub. Its 57 unit tests pass.
Reviewed against the goals above:

| Idea | Verdict | Why |
|---|---|---|
| Plain-text description as input | **Drop** | Goal 1: we narrate an existing lab. Its output can't be matched to our clips. |
| Two LLM stages (brief → script) | **Drop** | The brief repeats our `Clip.changeMeaning`, and the script replaces our `Clip.narration`. Regenerating the text cuts the audio loose from the lab and roughly doubles LLM cost. Rewording is the deferred polish pass (#98). |
| Say/pause segment script format | **Drop** | Our narration is one string per clip with no pause data. One request per clip flows more naturally. If pauses are ever needed, use the provider's own pause markup inside the spoken text. |
| Per-segment TTS joined with pydub | **Drop** | Leaves seams in intonation between segments (its own design doc says so). Not needed at our clip lengths. |
| MP3 from TTS → decode → re-encode | **Drop** | Two lossy encodes per clip. We request uncompressed audio and encode once. |
| `m01_c001_<title>.mp3` file names | **Drop** | The headset needs `<audioClip>`. Our fallback is `m{m:02d}_{clipName}`. |
| Hash-based staleness | **Keep idea, rewrite** | The right idea, but it was never connected to the CLI, and its hash leaves out provider and format. Ours is used on every rerun. |
| `TTSProvider` protocol | **Keep idea, rewrite** | Small and pluggable is right. Ours also returns usage, served model and audio format, and passes the style instruction explicitly: JW's collects style instructions and then drops them (`OpenAITTSProvider.synthesize` ignores them, and `tts-1` doesn't support them). |
| Voice check against a fixed list | **Rewrite** | The list is OpenAI's only. Each provider needs its own. |
| Script warnings (very short clip, long segment, long pause) | **Keep in part** | Very short and very long narration go in our per-clip report as warnings, never blocking. Pause checks don't apply. |
| `CostTracker` and `config.py` price tables | **Drop** | **Bug, reproduced:** LLM cost always comes out $0. The generators build log entries without a cost and `cli.py` appends them to `cost_tracker._llm_entries` directly, skipping `record_llm_call`. A Sonnet 4.6 call of 2,000 input + 1,500 output tokens shows $0.00 (expected ~$0.03). Pricing is also per-character only, which doesn't fit token-billed TTS. We use `tracking/`. |
| `tenacity` retry around the LLM call | **Drop** | **Reproduced:** it retries every exception, including a bad API key (5 attempts before failing). Each attempt also includes instructor's 3 reasks, so up to 15 requests. Usage is read only from the final response, so reasks go uncounted. We retry transient errors only. |
| `instructor.from_anthropic(...)` called directly | **Drop** | Bypasses `create_instructor_client`, the Anthropic `tool_choice` guard and `tracking/` (CLAUDE.md rule for every entry point). Anthropic-only. |
| Model defaults (`claude-sonnet-4-6`, `tts-1`) | **Drop** | Older than the current roster. Our TTS models are pinned per provider and recorded. |
| Project manifest, audio manifest, `generation_log.jsonl` | **Keep idea, simplify** | Recording everything is right. We merge it into one `narration_manifest.json`, rewritten after each clip, as hierarchical metrics are. |
| `ProjectStorage` class | **Rewrite** | A few file helpers are enough. Writes should be atomic so a crash can't leave a half-written manifest. |
| Status flow and review mode (`workflow_mode`, `is_accepted`) | **Defer** | Belongs to the review tools (#96). |
| Flask review UI plan | **Defer** | Evaluate under #96. |
| Mocked offline tests | **Keep approach** | Ours is a smoke test with a fake provider and fake LLM, plus unit tests for name handling and spoken-form rules. |
| Separate `requirements`, conda env, `.env` in a subfolder | **Drop** | Use the repo's own. |
| Per-clip voice/style, `emphasis` hints | **Drop** | One voice and style per lab (see "Text sent to the voice service"). `emphasis` was never sent to the provider. |
| `datetime.utcnow()` | **Drop** | Deprecated; use timezone-aware times. |

Nothing from `narration_generator/` is copied. The ideas marked "keep" are
reimplemented to this spec.

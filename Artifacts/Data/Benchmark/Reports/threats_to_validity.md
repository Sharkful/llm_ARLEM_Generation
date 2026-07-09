# Benchmark — Threats to Validity

*Standalone methodology note. Applies to all benchmark findings docs in this
folder (the formative summary, the statistics report, and any future lab-findings
docs). Last updated 2026-07-09.*

> **Scope note (2026-07-09):** sections 1–6 describe the June 2026 json_lab runs.
> The decode-mode/schema asymmetry they document was resolved before the July
> formative sweep: every provider now runs the same free-decode tool-call path
> (`GENAI_TOOLS` for Gemini) on one unified schema per spec (Gemini twins
> retired, issues #26/#29). Section 7 covers the July sweep's own confound.

This note records known confounds in the cross-provider / cross-model comparison
so that individual findings docs can link here instead of restating it. It is
descriptive of limitations, not a results doc.

---

## 1. Structured-output enforcement is not uniform across providers

Each provider reaches a valid `Lab` by a different mechanism. See
`create_instructor_client` in `Code/Testing/benchmark.py`:

| Provider | Instructor mode | How the schema is enforced |
|---|---|---|
| OpenAI | `Mode.TOOLS` (default) | JSON generated into a tool/function call; Instructor validates against the Pydantic model and **re-prompts on failure**. The model decodes freely; invalid output is caught after the fact and retried. |
| Anthropic | `Mode.ANTHROPIC_TOOLS` | Same: tool call → validate → retry. |
| Gemini | **`Mode.GENAI_STRUCTURED_OUTPUTS`** | Schema passed as a `response_schema`; Gemini **constrains generation at decode time** to conform. |

The Gemini mode was adopted because the freer Gemini modes failed enum validation
in early runs (commit `7940d09`, "Use GENAI_STRUCTURED_OUTPUTS mode for Gemini to
fix enum validation"). It was a reliability fix to make the runs complete at all,
**not** a deliberate, neutral comparison choice.

## 2. Two confounds, not one

The Gemini column differs from the others in two ways:

1. **Decode mode.** Constrained decoding masks schema-invalid tokens *as* the
   model generates. The OpenAI/Anthropic path lets the model generate freely and
   validates afterward (hence the retry loop exists for them and not for Gemini).
   These are different sampling processes, not just different libraries.
2. **Schema shape.** Gemini is fed a *flat, union-free* twin of the model
   (`Code/Schemas/json_lab_gemini.py`) because its schema translator cannot express
   discriminated unions; OpenAI/Anthropic receive the full discriminated-union
   model (`Code/Schemas/json_lab.py`). Different structure, field optionality, and
   discriminator strategy. Field descriptions and constraints are now kept in sync
   between the two files (see commit `66be908`), but the structural difference
   remains.

## 3. Likely direction of bias

Both factors plausibly **inflate Gemini's apparent reliability**:

- Constrained decoding *cannot* emit a schema-invalid component, so Gemini's
  near-zero schema-validation failure rate is partly an artifact of the mode.
- The flat model removes the discriminated-union resolution that was the single
  largest failure driver for the other providers (the `union_tag_not_found`
  errors fixed between Run 1 and Run 2).

So "Gemini was perfect throughout" should be read as "perfect *under constrained
decoding on a simpler schema*," not as evidence of stronger underlying generation.
Reliability/success-rate and retry-count comparisons that include Gemini are the
most affected; raw structural-output metrics (object/clip counts) are affected
less directly but are still produced under a different decode regime.

## 4. Effect on content quality is unsettled

Whether decode-time constraint also changes the *quality* of the output (not just
its validity) is genuinely contested in the literature, so we make no claim
either way — we only flag that the asymmetry exists and is not currently measured
for this task:

- Tam et al., ["Let Me Speak Freely? A Study on the Impact of Format Restrictions
  on Performance of Large Language Models"](https://arxiv.org/abs/2408.02442)
  (EMNLP 2024 Industry) — report that format restriction can degrade
  reasoning-heavy performance, with the effect more pronounced on weaker models.
- The structured-generation community (e.g. the dottxt / `outlines` response,
  "Say What You Mean") — argue the observed degradation stems from suboptimal
  prompting under constraint rather than from constraining itself, and that
  constrained generation does not inherently hurt and improves parse-validity.

The mechanism both sides agree on: masking invalid tokens pushes the model off
its preferred trajectory; smaller models have less probability mass to spare and
are hit harder.

## 5. Implication for the planned local / small-model phase

This is where the asymmetry bites hardest. Small local models are exactly the
regime where constrained decoding is reported to hurt most, and they can be run
with or without grammar constraints (e.g. `outlines` / `xgrammar` / llama.cpp
GBNF). **Decode strategy should be treated as an explicit, recorded experimental
factor** (constrained vs. free-generate-and-retry), not an incidental per-provider
default — otherwise a "small models do worse" result is confounded by "small
models were constrained while large Gemini effectively was too."

## 6. Planned mitigations

- **(a) Re-test a freer Gemini mode.** Check whether `Mode.GEMINI_JSON` (schema in
  the prompt, then validate-and-retry) validates reliably now. If it does, it puts
  Gemini on the same path as the other providers — and may even let it run the
  full (non-flat) model, collapsing both confounds at once.
- **(b) Measure the effect on this task.** Run a small Gemini A/B
  (`GENAI_STRUCTURED_OUTPUTS` vs `GEMINI_JSON`) on the same labs/levels and compare
  validity rate, retry count, and structural output. This quantifies the gap for
  ARLEM generation specifically instead of relying on general literature.
- **(c) Make it traceable.** Record the Instructor mode and the model variant
  (flat vs. union) in every metrics record so the enforcement path is recoverable
  per run.

## 7. Retry fairness in the 2026-07-05/08 formative sweep (issue #44)

The benchmark's design gives every run up to 3 attempts (validate → feed the
error back → retry). In the July sweep, two instructor 1.14.4 bugs made the
*retry request itself* malformed, so several models effectively ran with
**retries=0** while the rest of the field got 3 attempts:

- **Anthropic Haiku 4.5** (14 cells): parallel `tool_use` blocks in the failed
  attempt were replayed without matching `tool_result` blocks → API 400 on
  every retry.
- **Gemini 3.x — all of gemini-3.1-pro-preview (22), gemini-3.1-flash-lite (17),
  gemini-3.5-flash (9)**: replayed functionCall turns dropped the required
  `thought_signature` → 400 INVALID_ARGUMENT on every retry. Gemini 2.5 was
  unaffected.
- Additionally, ~10 `arlem` cells (gpt-5.5, claude-sonnet-5, gpt-5.4-mini) got
  useless retry feedback (`'NoneType' object is not iterable`) from a None-safety
  bug in our own ARLEM cross-validators (issue #49), degrading their
  self-correction odds without blocking the retries themselves.

**Consequence:** cross-provider schema-failure and retry-count comparisons that
include Haiku 4.5 or Gemini 3.x models are not valid for records produced on
instructor 1.14.4 — those models never saw their correction prompts.

**Resolution (2026-07-09):** instructor pinned to 1.15.4, which round-trips the
Gemini thought signatures on reask (verified by forced-failure probes on
gemini-3.1-flash-lite and gemini-3.1-pro-preview). The Anthropic replay bug is
*not* fixed upstream; instead runs now pass
`tool_choice={..., "disable_parallel_tool_use": true}` so parallel tool calls —
which always fail parsing under instructor and were pure wasted attempts —
cannot occur in the first place. Note this means Anthropic models are decoding
under a slightly tighter tool-choice constraint than other providers; the
freedom it removes could only ever produce a failed attempt, but flag it when
comparing attempt-1 failure rates. A third fallback restores retryability of
Gemini parallel-function-call responses (instructor 1.15.4 made that shape a
non-retryable hard fail; `benchmark.py::_patch_genai_parallel_call_retry`).

**Caveat status:** the retries=0 caveat applies until the 79 affected cells are
re-run on 1.15.4 (list in issue #44). Once the re-runs supersede the dead
records (`load_runs(dedupe_latest=True)`), drop this caveat for those cells;
records are distinguishable by their `instructor` behavior era via run
timestamps (re-runs are ≥ 2026-07-09).

## Status

| Item | State |
|---|---|
| Asymmetry documented (this note) | ✅ 2026-06-22 |
| (a) `GEMINI_JSON` re-test | ☐ obsolete — uniform free-decode landed instead (#26/#29) |
| (b) Gemini decode-mode A/B | ☐ not started |
| (c) Record mode + variant in metrics | ✅ `decode_mode` recorded per run |
| Retry-fairness caveat (§7) | ⏳ active until #44 re-runs land |

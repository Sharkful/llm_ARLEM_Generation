# Handoff — Gemini Union Investigation (2026-06-22)

> Working note to jumpstart a fresh Claude conversation on another machine.
> Branch: **`Formative_Runs`**. Delete this file once the investigation closes.
> To resume: `git pull`, then point Claude at this file.

---

## 1. The question on the table

The benchmark compares LLM providers on structured AR-lab generation. A
methodology concern surfaced: **Gemini is benchmarked under a different regime
than OpenAI/Anthropic**, which confounds the comparison. Two confounds:

1. **Decode mode** — Gemini uses `GENAI_STRUCTURED_OUTPUTS` (constrained decoding);
   OpenAI/Anthropic use tool-calling + validate-and-retry.
2. **Schema shape** — Gemini gets a *flat, union-free* twin (`json_lab_gemini.py`)
   because it historically couldn't handle Pydantic unions; the others get the
   full discriminated-union model (`json_lab.py`).

Full write-up: [`Artifacts/Data/Benchmark/Reports/threats_to_validity.md`](Artifacts/Data/Benchmark/Reports/threats_to_validity.md).

**The live thread:** Google added `anyOf`/full-JSON-Schema support for Gemini 2.5+
(Nov 2025), so the "no unions" limitation may be obsolete. If Gemini can take a
union model now, we could **delete the flat twin and run one schema across all
three providers**, erasing confound #2 entirely.

---

## 2. WHERE TO RESUME (the decisive next experiment)

**Build a discriminator-free union variant of the real model and test it on all
three providers.** This determines whether the flat `json_lab_gemini.py` can be
retired.

- Take `json_lab.py`'s `Component` (and any other discriminated union) and replace
  `Annotated[Union[...], Field(discriminator="type")]` with a **plain `Union[...]`**
  (no `discriminator`). Keep the `Literal` `type`/`componentType` fields — Pydantic
  smart-union resolves on them (the toy proved this works on Gemini).
- Test it on: **Gemini** (does `GENAI_STRUCTURED_OUTPUTS` accept + generate?) AND
  **OpenAI + Anthropic** (does smart-union still resolve correctly *without* the
  discriminator, i.e. no regression vs. today's discriminated union?).
- Decision it unblocks: one shared union model for everyone, or keep the flat twin.

> The user wants to **review the toy result (§4) a bit more before** committing to
> this. Don't start §2 until they've signed off.

---

## 3. Environment notes for the new machine

- **`git pull` on `Formative_Runs`** gets everything below.
- Needs `.env` with `GEMINI_API_KEY` (and `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
  for the cross-provider test). `from_provider` reads `GOOGLE_API_KEY`; the runner
  falls back to `GEMINI_API_KEY`.
- venv: **instructor 1.14.4**, **google-genai 1.68.0**.
- ⚠️ **The Run 1–3 sweep data (`suite_results_20260608_*.json`) is NOT in git** —
  it lives on the *other* (original) machine. So `build_summary_report.py` /
  `build_statistics_report.py` **cannot regenerate the report** from the new
  machine. (This is why the threats note was written as a standalone `.md` instead
  of being baked into the report notebook.) Only `suite_results_20260529_101607.json`
  is present in-repo.
- Reproduce the union probe: `python "Code/Testing/probe_gemini_unions.py"`.

---

## 4. The toy experiment result (UNDER REVIEW)

Goal: isolate "can Gemini do unions?" from "is our giant Lab schema the problem?"
using a tiny toy schema — a `Literal`-tagged `Union` inside a `list` (the exact
`json_lab.Component` pattern). Script: [`Code/Testing/probe_gemini_unions.py`](Code/Testing/probe_gemini_unions.py).

**Schema shapes Pydantic emits:**
- Discriminated union (`Field(discriminator=...)`) → `oneOf` + `discriminator`
- Plain union (`Union[Cat, Dog]`) → `anyOf`

**Results (model `gemini-2.5-flash-lite`, `max_retries=0`):**

| Mode | Union shape | Result |
|---|---|---|
| `GENAI_STRUCTURED_OUTPUTS` | discriminated (`oneOf`+`discriminator`) | ❌ FAIL |
| `GENAI_STRUCTURED_OUTPUTS` | **plain (`anyOf`)** | ✅ **OK — `['cat','dog','dog']`** |
| `GENAI_TOOLS` | discriminated | ❌ FAIL (output didn't validate) |
| `GENAI_TOOLS` | plain (`anyOf`) | ❌ FAIL |

**Why each failure happens (this is the important part):**
- Discriminated union fails **locally in the google-genai SDK**, before hitting the
  API: its `Schema` pydantic model rejects the `discriminator` keyword —
  `properties.animals.items.discriminator: Extra inputs are not permitted`. So it's
  not "Gemini can't do unions"; it's "the SDK won't translate `oneOf`+`discriminator`."
- `GENAI_TOOLS` + plain union fails because that path's stricter schema subset
  rejects `const` (which `Literal` defaults emit): `...pet_type.const: Extra inputs
  are not permitted`. So `GENAI_STRUCTURED_OUTPUTS` is actually the *more*
  JSON-Schema-complete path.

**Conclusions:**
1. **Gemini 2.5+ CAN do unions now** — but only as **plain `anyOf` unions**, and only
   under `GENAI_STRUCTURED_OUTPUTS`. Plain-union + `Literal` fields → Pydantic
   smart-union resolved the types correctly on Gemini.
2. **Only `GENAI_STRUCTURED_OUTPUTS` and `GENAI_TOOLS` are reachable** via
   `from_provider("google/...")`. `GEMINI_JSON` / `JSON` are legacy-SDK only and are
   rejected — so the "GEMINI_JSON free-retry" mitigation in the threats doc §6 is
   **wrong and needs correcting**.

---

## 5. Open items / decisions

1. **(MAIN)** Run the discriminator-free real-schema experiment across all 3
   providers (§2). Gated on user reviewing §4.
2. **Update `threats_to_validity.md` §6 + status table** with §4 findings:
   correct the unreachable-`GEMINI_JSON` item; record that plain-union works on
   Gemini; note the decode-mode comparison would be `GENAI_TOOLS` vs
   `GENAI_STRUCTURED_OUTPUTS` (but `GENAI_TOOLS` rejects `const`, so it needs a
   schema it can accept). **Not yet done.**
3. **`_gemini` → `_flat` rename: do NOT do it.** The likely better outcome is
   *deleting* the flat twin (one shared union model), not generalizing it. Decide
   after §2.
4. **Probe script fate** — currently committed (`0508ddb`). Keep as a reproducible
   experiment, or remove once the investigation closes.

---

## 6. What changed this session (all on `Formative_Runs`, pushed)

| Commit | What |
|---|---|
| `66be908` | Synced `json_lab_gemini.py` with base `json_lab.py` (fontSize clamp + `_coerce_number_list`, `newscript`/NewComponent, `changeMeaning`/`narration`, `SkipJsonSchema` author/institution, expanded descriptions). Aliases + the `type`/`componentType` discriminator split deliberately NOT ported. |
| `159076c` | Renamed `Code/Tools` → **`Code/Schemas`** (the dir holds only Pydantic schemas; "Models" would collide with the LLM `MODELS` registry). Updated both `sys.path` inserts + all doc refs. Imports unchanged (dir is on `sys.path`). Runner verified via `--list-labs`. |
| `98b5dcd` | Added `threats_to_validity.md` (standalone methodology note) + a pointer from `BENCHMARK.md`. |
| `0508ddb` | Added `probe_gemini_unions.py` + `BENCHMARK.md` "under revalidation" caveat on the Gemini-union claim. |

(This `HANDOFF.md` is the only thing committed after `0508ddb`.)

---

## 7. Key files

- [`Code/Schemas/json_lab.py`](Code/Schemas/json_lab.py) — full model, discriminated unions (OpenAI/Anthropic).
- [`Code/Schemas/json_lab_gemini.py`](Code/Schemas/json_lab_gemini.py) — flat union-free twin (Gemini). Candidate for deletion.
- [`Code/Testing/benchmark.py`](Code/Testing/benchmark.py) — runner; `create_instructor_client` (~L90–120) sets per-provider modes; `estimate_gemini_schema_tokens` (~L132) uses genai's `t_schema` (OpenAPI path).
- [`Code/Testing/probe_gemini_unions.py`](Code/Testing/probe_gemini_unions.py) — the union probe.
- [`Artifacts/Data/Benchmark/Reports/threats_to_validity.md`](Artifacts/Data/Benchmark/Reports/threats_to_validity.md) — methodology note (needs the §5.2 update).
- [`Code/Testing/BENCHMARK.md`](Code/Testing/BENCHMARK.md) — CLI ref + Gemini-handling note.

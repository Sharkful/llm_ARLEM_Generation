# How cost is computed

Every cost figure in a run record is an **estimate**: the token counts each
response reports, priced at the published per-token rates in `pricing.py`.
To check it against what a provider actually billed, see
`Code/Analysis/BILLING_RECONCILE.md` (#75).

## 1. Tokens, per response (`metrics.py::TokenUsage.from_completion`)

Every provider's usage is mapped to one meaning:
- `prompt_tokens` is **all input**. `cached_tokens` (cache reads) and
  `cache_write_tokens` are parts of it.
- `completion_tokens` is **all output, reasoning included**. `reasoning_tokens` is
  the part the provider reports as reasoning.

| Provider | Input | Output |
|---|---|---|
| OpenAI Chat Completions | `prompt_tokens`, which includes `prompt_tokens_details.cached_tokens` / `cache_write_tokens` | `completion_tokens`, which includes `completion_tokens_details.reasoning_tokens` |
| OpenAI Responses (GPT-6) | `input_tokens`, which includes `input_tokens_details.cached_tokens` / `cache_write_tokens` | `output_tokens`, which includes `output_tokens_details.reasoning_tokens` |
| Anthropic | `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`; Anthropic reports the cache counts outside `input_tokens`, so they are folded in | `output_tokens`, which includes thinking; Anthropic reports no separate thinking count |
| Gemini | `prompt_token_count`, which includes `cached_content_token_count` | `candidates_token_count` + `thoughts_token_count`; Gemini reports thinking apart from the candidates but bills it as output (#84) |

**Gemini records made before #84** left thinking out of `completion_tokens`, so
their cost understates output. In those records `total − prompt − completion` is
exactly the unbilled thinking. Each branch is checked against the real SDK
response types in `Code/Hierarchical/smoke_test.py` (section 25b).

## 2. Price, per response (`pricing.py`, `tracker.pricing_for`)

Each response is priced at the **model that served it**: the `model` /
`model_version` it reports, recorded per attempt.
- **Lookup:** first the exact `DEFAULT_PRICING` key, then the id without a dated
  suffix (`gpt-6-luna-2026-09-01` → `gpt-6-luna`).
- **Fallback:** when neither has an entry, the requested model's price is used,
  with a one-time warning, and the id is listed in `cost.unpriced_served_models`.
  Add the price to `pricing.py`.
- **Example:** Google answers `gemini-3.5-flash` requests with `gemini-3.6-flash`,
  so those runs are priced as 3.6 Flash.
- **Override:** a `ModelConfig.input_price` / `output_price` override applies to
  the requested model's entry only.

Within an entry (`PricingCalculator.cost_breakdown`):

```
uncached = prompt − cached − cache_write        (clamped at 0; see warnings)
cost     = uncached × input + cached × cached_input + cache_write × cache_write
         + completion × output                  (per 1M tokens)
```

- **Rate defaults:** an entry without `cached_input` / `cache_write` bills those
  tokens at `input`. That is how every wave-2 run was costed.
- **Anthropic:** cache writes are the 5-minute TTL rate (1.25× input), and reads
  are 0.1× (0.05× on Opus 5.5).
- **OpenAI:** GPT-5.6 and later bill writes at 1.25× input.
- **Gemini:** implicit caching has no write charge.

## 3. Where cost is computed and summed

| Level | How |
|---|---|
| Call | `tracker.cost_of(call)`: each attempt (request) priced at its served model, summed, reasks included |
| Run (one-shot) | `tracking.cost.total_usd` = the sum over its calls |
| Step (hierarchical) | `StepRecord.cost_usd` = the sum over the calls made during the step |
| Run (hierarchical) | `totals.cost_usd` = the sum over steps, resumed ones included; `tracking` covers the last session only |
| Analysis | `load_runs()` reads `totals` when present, else `tracking`. `reprice_runs.py` re-costs a CSV through the same `PricingCalculator`, and adds pre-#84 Gemini thinking from the token gap (`thinking_cost_usd`). |

## 4. What the estimate doesn't price (warned about and counted in `cost`)

- **Long-context tiers.** Some models bill a higher rate above a prompt size
  (`long_context_above` in the entry). Our prompts stay far below these, so the
  tiers aren't priced; any request above one is counted in
  `cost.long_context_requests`. Thresholds:
  - Claude Haiku 5.5: 100K
  - GPT-5.5 and GPT-6.1 Sol: 272K
  - Gemini 3.1 Pro and 2.5 Pro: 200K
  - Gemini 4 Argon: 200K, to be confirmed
- **Introductory prices.** Entries store the list (post-discount) rate, so runs
  compare across time. `intro_until` marks the discount's last day, and a run
  before it gets a `cost.notes` entry. Billed cost in that window is lower.
- **Cache split larger than the input.** When a response reports more cached
  plus written tokens than input tokens, uncached input is priced as 0. This is
  counted in `cost.cache_split_overflow`. A forum report says OpenAI can do this;
  no run so far has.
- **Not modelled at all:** batch / priority tiers, Gemini cache storage (implicit
  caching has none), and Anthropic's 1-hour cache TTL (writes at 2×; the pipeline
  only uses 5 minutes).

The record's `cost.priced_as` lists the entries the estimate used.

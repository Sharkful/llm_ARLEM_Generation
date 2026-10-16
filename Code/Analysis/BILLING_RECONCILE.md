# Checking cost estimates against provider billing (#75)

Run cost in every metrics file is an **estimate**: recorded tokens × the rates in
`tracking/pricing.py`. `reconcile_billing.py` compares those numbers with what each
provider reports, so a few runs show how close the estimate is.

## What the providers offer

| | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Tokens | Usage report, 1m / 1h / 1d buckets: uncached input, cache reads, cache writes (5m / 1h), output | Usage API (completions), 1m / 1h / 1d buckets: input, cached input, output, request count | No usage API |
| Cost | Cost report, **daily only**, by workspace | Costs API, **daily only**, by project | Cloud Billing export to BigQuery: usage and cost per SKU per project |
| Credential | Admin API key (`sk-ant-admin…`, organizations only) | Admin key (org owner) | Billing access, BigQuery export enabled |
| Delay | Usually within 5 minutes | Minutes | Usually within a day, sometimes over 24 hours |
| Per request | No | No | No |

None of them reports cost per request, so our tracker stays the only per-call
record. What the reports can confirm:
- **Tokens per run**, if nothing else ran in the same bucket during the run's window.
- **Dollars per day.** The cost reports are daily, so the comparison is a day's
  billed cost against the sum of that day's estimates.
- **Gemini per SKU.** The billing export lists input, output and cached tokens
  separately, so any billed output beyond what we record shows up. That is the
  thinking-token gap in #84.

The OpenAI Usage API reports no cache-write count. Cache writes are counted
inside its input, so the script folds ours into uncached input before comparing.

## One-time setup

Benchmark traffic needs buckets that nothing else uses. Otherwise other traffic
counts as ours, and on Anthropic the cost report can't be filtered by API key.

1. **Anthropic.**
   - In the Console, create a workspace (e.g. `benchmark`) and an API key in it.
     Use that key as `ANTHROPIC_API_KEY` for runs.
   - Create an Admin API key (Settings → Admin keys).
   - Set `ANTHROPIC_ADMIN_KEY` and `ANTHROPIC_BENCH_WORKSPACE_ID` (`wrkspc_…`).
2. **OpenAI.**
   - Create a project (e.g. `benchmark`) and an API key in it. Use that key as
     `OPENAI_API_KEY` for runs.
   - Create an Admin key (organization settings).
   - Set `OPENAI_ADMIN_KEY` and `OPENAI_BENCH_PROJECT_ID` (`proj_…`).
3. **Gemini.**
   - Use a GCP project only for benchmark runs, and make its API key the
     `GEMINI_API_KEY`.
   - Enable **detailed** Cloud Billing export to BigQuery on its billing account.
     The first data can take hours to appear.
   - Set `GEMINI_BENCH_PROJECT_ID` (the GCP project id).
   - Optional: `pip install google-cloud-bigquery` and run
     `gcloud auth application-default login`, so the script can run the query
     itself.

The keys go in `.env` (git-ignored); the script loads it.

## Running it

```bash
# ~10 minutes after the runs: token check (Anthropic, OpenAI) + today's cost so far
python "Code/Analysis/reconcile_billing.py" --runs "Artifacts/Data/Hierarchical/testing/Metrics"

# next day: the complete daily cost, and the Gemini billing export
python "Code/Analysis/reconcile_billing.py" --runs "Artifacts/Data/Hierarchical/testing/Metrics" \
    --bq-table "<project>.<dataset>.gcp_billing_export_resource_v1_<BILLING_ACCOUNT_ID>" \
    --csv reconcile.csv
```

Options:
- `--provider` limits the check to one provider.
- `--no-cost` skips the daily cost check.

How it works:
- **Window:** only the record's `tracking` block (this session's API calls) and
  its UTC `started_at` / `finished_at` window are compared. For a resumed
  hierarchical run, that means only the steps generated in that session.
- **Records from before #75** have no UTC fields. The window falls back to
  `timestamp − wall_time_seconds`, read as this machine's local time.
- **Overlapping runs:** runs of the same model whose windows overlap are
  compared as one group, because minute buckets can't split them.
- **Daily cost:** the comparison counts only the runs passed in, so pass every
  run of that UTC day.

## What to expect

- **Anthropic and OpenAI tokens:** within about 1%. A larger gap means the
  tracker misses something, for example a request it never saw.
- **Gemini:** billed output is above our recorded output by about the thinking
  tokens, which the script prints separately. Our cost leaves those out until
  #84 is fixed.

Not verified until the first live run:
- OpenAI's usage field names (`input_tokens`, `input_cached_tokens`,
  `output_tokens`, `num_model_requests`), and whether Responses API traffic
  appears under the completions usage endpoint.
- The Gemini API's `service.description` in the billing export. The query
  accepts both `Gemini API` and `Generative Language API`.

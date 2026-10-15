"""
Check recorded tokens and estimated cost against what the provider reports (#75).

Every run's cost is an estimate: recorded tokens x the published rates in
``tracking/pricing.py``. This script reads the providers' own usage and billing
data for the same window and prints ours vs theirs, so a few runs can show how
far the estimate is off (cache pricing, Gemini thinking tokens, #84).

What each provider offers (nothing is per request; see the #75 comment):
- Anthropic: Usage report (1-minute buckets: uncached input, cache reads, 5m/1h
  cache writes, output) and Cost report (daily only, USD). Admin API key.
- OpenAI: Usage API, completions (1-minute buckets: input, cached input,
  output, request count) and Costs API (daily only). Admin key.
- Gemini: no usage API. The Cloud Billing export to BigQuery has usage and cost
  per SKU per project, usually within a day. This script prints the query (and
  runs it when ``google-cloud-bigquery`` is installed and ``--bq-table`` is set).

Dedicated buckets: usage and cost cover everything in the workspace (Anthropic),
project (OpenAI) or GCP project (Gemini). Run benchmark traffic through keys that
belong to a bucket used for nothing else, or other traffic is counted as ours.

Token check: runs are grouped by provider and model; runs whose windows overlap
are compared together, since minute buckets can't split them. Cost check: per
UTC day, the bucket's billed cost against the summed estimates of the runs
passed in, so pass every run of that day.

Only this session's API calls are compared: the record's ``tracking`` block and
its ``started_at`` / ``finished_at`` window (UTC). Records written before those
fields existed fall back to ``timestamp - wall_time_seconds`` read as this
machine's local time.

Environment (or the matching flags):
    ANTHROPIC_ADMIN_KEY, ANTHROPIC_BENCH_WORKSPACE_ID
    OPENAI_ADMIN_KEY,    OPENAI_BENCH_PROJECT_ID
    GEMINI_BENCH_PROJECT_ID (GCP project id), --bq-table project.dataset.table

Usage:
    python "Code/Analysis/reconcile_billing.py" --runs "Artifacts/Data/Hierarchical/testing/Metrics"
    python "Code/Analysis/reconcile_billing.py" --runs a_metrics.json b_metrics.json --provider anthropic
    python "Code/Analysis/reconcile_billing.py" --runs <dir> --no-cost --csv out.csv

Read-only: the billing endpoints are free to call, and nothing is written except
``--csv``. No LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

try:  # .env holds the admin keys on the user's machine; optional
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ANTHROPIC_BASE = os.environ.get("ANTHROPIC_ADMIN_BASE_URL", "https://api.anthropic.com")
OPENAI_BASE = os.environ.get("OPENAI_ADMIN_BASE_URL", "https://api.openai.com")

# Usage lands within minutes but not instantly, and a bucket is stamped with its
# start: widen the window so the first and last minute are fully inside.
WINDOW_BEFORE = timedelta(minutes=1)
WINDOW_AFTER = timedelta(minutes=2)

TOKEN_KINDS = ("uncached_input", "cache_read", "cache_write", "output", "requests")


# ── runs ─────────────────────────────────────────────────────────────

@dataclass
class Run:
    path: Path
    provider: str
    model: str
    served: list[str]
    start: datetime          # UTC
    end: datetime            # UTC
    approx_window: bool      # True when derived from the local end-time timestamp
    tokens: dict[str, int]
    thinking: int            # tokens beyond prompt + completion (Gemini thinking, #84)
    cost_usd: float


def _utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.astimezone().astimezone(timezone.utc)


def load_run(path: Path) -> Run:
    rec = json.loads(path.read_text(encoding="utf-8"))
    tr = rec["tracking"]
    tok = tr["tokens"]
    prompt, cached = tok.get("prompt", 0), tok.get("cached", 0)
    write, completion = tok.get("cache_write", 0), tok.get("completion", 0)
    if rec.get("started_at") and rec.get("finished_at"):
        start, end, approx = _utc(rec["started_at"]), _utc(rec["finished_at"]), False
    else:
        end = _utc(rec["timestamp"])   # naive local time -> this machine's zone
        start = end - timedelta(seconds=rec.get("wall_time_seconds") or 0)
        approx = True
    return Run(
        path=path, provider=rec["provider"], model=rec["model"],
        served=list(rec.get("served_models") or tr.get("served_models") or []),
        start=start, end=end, approx_window=approx,
        tokens={
            "uncached_input": prompt - cached - write,
            "cache_read": cached,
            "cache_write": write,
            "output": completion,
            # requests = instructor calls + their reasks (transient re-sends are calls)
            "requests": tr.get("total_calls", 0) + (tr.get("retries") or {}).get("total", 0),
        },
        thinking=max(0, tok.get("total", 0) - prompt - completion),
        cost_usd=float((tr.get("cost") or {}).get("total_usd") or 0.0),
    )


def find_runs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(p.glob("*_metrics.json")) if p.is_dir() else [p]
    return out


def same_model(run: Run, reported: Optional[str]) -> bool:
    """A report row belongs to the run's model: the requested id, a dated snapshot
    of it, or a model the run was actually served by."""
    if not reported:
        return False
    reported = reported.removeprefix("models/")
    pat = re.compile(re.escape(run.model) + r"(-\d{4}-\d{2}-\d{2}|-\d{8})?")
    return bool(pat.fullmatch(reported)) or reported in run.served


@dataclass
class Group:
    """Runs on one provider and model whose windows overlap (compared together)."""
    runs: list[Run] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return min(r.start for r in self.runs) - WINDOW_BEFORE

    @property
    def end(self) -> datetime:
        return max(r.end for r in self.runs) + WINDOW_AFTER

    def ours(self) -> dict[str, int]:
        return {k: sum(r.tokens[k] for r in self.runs) for k in TOKEN_KINDS}


def group_runs(runs: list[Run]) -> list[Group]:
    """Group by (provider, model), merging runs whose padded windows overlap. Runs
    of different models are kept apart even when concurrent: reports split by model."""
    groups: list[Group] = []
    by_key: dict[tuple[str, str], list[Run]] = {}
    for r in runs:
        by_key.setdefault((r.provider, r.model), []).append(r)
    for _, rs in sorted(by_key.items()):
        rs.sort(key=lambda r: r.start)
        for r in rs:
            if groups and groups[-1].runs[0].model == r.model \
                    and groups[-1].runs[0].provider == r.provider \
                    and r.start - WINDOW_BEFORE <= groups[-1].end:
                groups[-1].runs.append(r)
            else:
                groups.append(Group([r]))
    return groups


# ── provider reports ─────────────────────────────────────────────────

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paged(client: httpx.Client, url: str, params: list[tuple[str, str]]) -> list[dict]:
    """All buckets of a paginated report (both providers use has_more / next_page)."""
    buckets, page = [], None
    while True:
        resp = client.get(url, params=params + ([("page", page)] if page else []))
        resp.raise_for_status()
        body = resp.json()
        buckets += body.get("data", [])
        if not body.get("has_more"):
            return buckets
        page = body["next_page"]


def anthropic_usage(client: httpx.Client, group: Group, workspace: Optional[str]) -> dict[str, int]:
    params = [("starting_at", _iso(group.start)), ("ending_at", _iso(group.end)),
              ("bucket_width", "1m"), ("limit", "1440"), ("group_by[]", "model")]
    if workspace:
        params.append(("workspace_ids[]", workspace))
    theirs = dict.fromkeys(TOKEN_KINDS[:-1], 0)
    for bucket in _paged(client, f"{ANTHROPIC_BASE}/v1/organizations/usage_report/messages", params):
        for row in bucket.get("results", []):
            if not same_model(group.runs[0], row.get("model")):
                continue
            creation = row.get("cache_creation") or {}
            theirs["uncached_input"] += row.get("uncached_input_tokens", 0)
            theirs["cache_read"] += row.get("cache_read_input_tokens", 0)
            theirs["cache_write"] += (creation.get("ephemeral_5m_input_tokens", 0)
                                      + creation.get("ephemeral_1h_input_tokens", 0))
            theirs["output"] += row.get("output_tokens", 0)
    return theirs


def openai_usage(client: httpx.Client, group: Group, project: Optional[str]) -> dict[str, int]:
    params = [("start_time", str(int(group.start.timestamp()))),
              ("end_time", str(int(group.end.timestamp()))),
              ("bucket_width", "1m"), ("limit", "1440"), ("group_by", "model")]
    if project:
        params.append(("project_ids", project))
    theirs = {"uncached_input": 0, "cache_read": 0, "output": 0, "requests": 0}
    for bucket in _paged(client, f"{OPENAI_BASE}/v1/organization/usage/completions", params):
        for row in bucket.get("results", []):
            if not same_model(group.runs[0], row.get("model")):
                continue
            # input_tokens includes the cached part, as our prompt_tokens does. The
            # Usage API reports no cache-write count, so writes stay in uncached
            # input (print_group folds ours the same way).
            cached = row.get("input_cached_tokens", 0)
            theirs["uncached_input"] += row.get("input_tokens", 0) - cached
            theirs["cache_read"] += cached
            theirs["output"] += row.get("output_tokens", 0)
            theirs["requests"] += row.get("num_model_requests", 0)
    return theirs


def anthropic_daily_cost(client: httpx.Client, day: datetime, workspace: Optional[str]) -> float:
    params = [("starting_at", _iso(day)), ("ending_at", _iso(day + timedelta(days=1))),
              ("bucket_width", "1d"), ("group_by[]", "workspace_id")]
    cents = 0.0
    for bucket in _paged(client, f"{ANTHROPIC_BASE}/v1/organizations/cost_report", params):
        for row in bucket.get("results", []):
            if workspace is None or row.get("workspace_id") == workspace:
                cents += float(row.get("amount") or 0)   # decimal string, in cents
    return cents / 100


def openai_daily_cost(client: httpx.Client, day: datetime, project: Optional[str]) -> float:
    params = [("start_time", str(int(day.timestamp()))),
              ("end_time", str(int((day + timedelta(days=1)).timestamp()))),
              ("bucket_width", "1d"), ("limit", "1"), ("group_by", "project_id")]
    usd = 0.0
    for bucket in _paged(client, f"{OPENAI_BASE}/v1/organization/costs", params):
        for row in bucket.get("results", []):
            if project is None or row.get("project_id") == project:
                usd += float((row.get("amount") or {}).get("value") or 0)
    return usd


def gemini_sql(table: str, project: str, start: datetime, end: datetime) -> str:
    """Usage and cost per SKU for the Gemini API in one project and window. Billing
    rows are hourly, so the window widens to whole hours."""
    lo = start.replace(minute=0, second=0, microsecond=0)
    hi = end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return f"""SELECT sku.description AS sku, usage.unit AS unit,
       SUM(usage.amount) AS amount, SUM(cost) AS cost
FROM `{table}`
WHERE project.id = '{project}'
  AND service.description IN ('Gemini API', 'Generative Language API')
  AND usage_start_time >= TIMESTAMP('{_iso(lo)}')
  AND usage_start_time < TIMESTAMP('{_iso(hi)}')
GROUP BY sku, unit
ORDER BY cost DESC"""


# ── reporting ────────────────────────────────────────────────────────

def _diff(ours: int, theirs: Optional[int]) -> str:
    if theirs is None:
        return "n/a"
    if theirs == 0:
        return "—" if ours == 0 else "theirs 0"
    return f"{(ours - theirs) / theirs:+.1%}"


def print_group(group: Group, theirs: dict[str, int], rows: list[dict]) -> None:
    ours = group.ours()
    if "cache_write" not in theirs:
        # OpenAI reports no write count: cache writes are inside its input tokens
        ours["uncached_input"] += ours["cache_write"]
    names = ", ".join(r.path.name for r in group.runs)
    approx = " (window approximate: old record)" if any(r.approx_window for r in group.runs) else ""
    print(f"\n{group.runs[0].provider} / {group.runs[0].model}: {len(group.runs)} run(s), "
          f"{_iso(group.start)} -> {_iso(group.end)}{approx}")
    if len(group.runs) > 1:
        print(f"  windows overlap, compared together: {names}")
    print(f"  {'tokens':<16}{'ours':>14}{'theirs':>14}{'diff':>10}")
    for k in TOKEN_KINDS:
        t = theirs.get(k)
        print(f"  {k:<16}{ours[k]:>14,}{'' if t is None else f'{t:,}':>14}{_diff(ours[k], t):>10}")
        rows.append({"kind": "tokens", "provider": group.runs[0].provider,
                     "model": group.runs[0].model, "window_start": _iso(group.start),
                     "window_end": _iso(group.end), "runs": names, "metric": k,
                     "ours": ours[k], "theirs": t})


def utc_days(runs: list[Run]) -> list[datetime]:
    return sorted({r.start.replace(hour=0, minute=0, second=0, microsecond=0) for r in runs})


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="metrics files or folders")
    ap.add_argument("--provider", choices=["anthropic", "openai", "google"])
    ap.add_argument("--no-cost", action="store_true", help="skip the daily cost check")
    ap.add_argument("--anthropic-workspace", default=os.environ.get("ANTHROPIC_BENCH_WORKSPACE_ID"))
    ap.add_argument("--openai-project", default=os.environ.get("OPENAI_BENCH_PROJECT_ID"))
    ap.add_argument("--gemini-project", default=os.environ.get("GEMINI_BENCH_PROJECT_ID"))
    ap.add_argument("--bq-table", help="Cloud Billing detailed export table (project.dataset.table)")
    ap.add_argument("--csv", type=Path, help="also write the comparison rows here")
    args = ap.parse_args(argv)

    runs = [load_run(p) for p in find_runs(args.runs)]
    if args.provider:
        runs = [r for r in runs if r.provider == args.provider]
    if not runs:
        print("No runs found.")
        return 1
    providers = sorted({r.provider for r in runs})
    keys = {"anthropic": os.environ.get("ANTHROPIC_ADMIN_KEY"),
            "openai": os.environ.get("OPENAI_ADMIN_KEY")}
    missing = [p for p in providers if p in keys and not keys[p]]
    if missing:
        names = {"anthropic": "ANTHROPIC_ADMIN_KEY", "openai": "OPENAI_ADMIN_KEY"}
        print("Missing admin key: " + ", ".join(names[p] for p in missing)
              + ". Usage and cost reports need an Admin API key (Anthropic: an "
              "organization's sk-ant-admin key; OpenAI: an org-owner admin key), "
              "not the inference key. Set it in .env or the environment.")
        return 2
    buckets = {"anthropic": args.anthropic_workspace, "openai": args.openai_project}
    for p in providers:
        if p in buckets and not buckets[p]:
            print(f"WARNING: no dedicated {p} bucket set; reports cover the whole "
                  "organization, so other traffic will count as ours.")

    clients = {
        "anthropic": httpx.Client(timeout=60, headers={
            "x-api-key": keys["anthropic"] or "", "anthropic-version": "2023-06-01"}),
        "openai": httpx.Client(timeout=60, headers={
            "Authorization": f"Bearer {keys['openai'] or ''}"}),
    }
    rows: list[dict] = []
    for group in group_runs(runs):
        prov = group.runs[0].provider
        if prov == "anthropic":
            theirs = anthropic_usage(clients[prov], group, args.anthropic_workspace)
        elif prov == "openai":
            theirs = openai_usage(clients[prov], group, args.openai_project)
        else:
            continue
        print_group(group, theirs, rows)

    if not args.no_cost:
        for prov, fetch in (("anthropic", anthropic_daily_cost), ("openai", openai_daily_cost)):
            prov_runs = [r for r in runs if r.provider == prov]
            for day in utc_days(prov_runs):
                ours = sum(r.cost_usd for r in prov_runs if r.start.date() == day.date())
                theirs = fetch(clients[prov], day, buckets[prov])
                print(f"\n{prov} cost {day.date()} (UTC): ours ${ours:.4f}, billed ${theirs:.4f} "
                      f"({_diff(round(ours * 1e6), round(theirs * 1e6))}); counts only the runs "
                      "passed in, so pass every run of that day")
                rows.append({"kind": "cost", "provider": prov, "model": "",
                             "window_start": day.date().isoformat(), "window_end": "",
                             "runs": "", "metric": "usd", "ours": round(ours, 6),
                             "theirs": round(theirs, 6)})

    gemini = [r for r in runs if r.provider == "google"]
    if gemini:
        project = args.gemini_project or "<GEMINI_BENCH_PROJECT_ID>"
        start, end = min(r.start for r in gemini), max(r.end for r in gemini)
        print("\ngoogle: no usage API; compare with the Cloud Billing export "
              "(usually complete within a day). Ours over the window:")
        for model in sorted({r.model for r in gemini}):
            ms = [r for r in gemini if r.model == model]
            t = {k: sum(r.tokens[k] for r in ms) for k in TOKEN_KINDS}
            print(f"  {model}: input {t['uncached_input']:,} (+ cached {t['cache_read']:,}), "
                  f"output {t['output']:,}, thinking {sum(r.thinking for r in ms):,} "
                  f"(not in our cost, #84), estimated ${sum(r.cost_usd for r in ms):.4f}")
        sql = gemini_sql(args.bq_table or "<billing_export_table>", project, start, end)
        try:
            from google.cloud import bigquery
        except ImportError:
            bigquery = None
        if args.bq_table and bigquery is not None and args.gemini_project:
            print("\n  Billed per SKU:")
            for row in bigquery.Client().query(sql).result():
                print(f"  {row.sku:<70}{row.amount:>16,.0f} {row.unit:<10} ${row.cost:.4f}")
        else:
            print("\n  Run in BigQuery (or pass --bq-table and --gemini-project with "
                  "google-cloud-bigquery installed):\n")
            print("\n".join("    " + line for line in sql.splitlines()))

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "provider", "model", "window_start",
                                              "window_end", "runs", "metric", "ours", "theirs"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

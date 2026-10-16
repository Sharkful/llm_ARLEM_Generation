"""
Offline check for reconcile_billing.py (#75): no API keys, no network.

A mock HTTP server stands in for the Anthropic and OpenAI admin endpoints
(usage report with pagination, cost report, OpenAI usage and costs); fixture
metrics files stand in for runs. Checks the token sums and grouping, the cost
check, the filters sent, the old-record fallback, the Gemini SQL, and the
missing-key exit.

    python "Code/Analysis/reconcile_smoke_test.py"
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FAILED = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILED.append(msg)


REQUESTS: list[tuple[str, dict, dict]] = []


def _anthropic_usage(q):
    rows_p1 = [{"model": "claude-haiku-5-5", "uncached_input_tokens": 600,
                "cache_read_input_tokens": 300,
                "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 200},
               {"model": "claude-sonnet-5-5", "uncached_input_tokens": 99999,   # other model
                "cache_read_input_tokens": 0, "cache_creation": {}, "output_tokens": 99999}]
    rows_p2 = [{"model": "claude-haiku-5-5", "uncached_input_tokens": 400,
                "cache_read_input_tokens": 200,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50},
                "output_tokens": 100}]
    if "page" not in q:
        return {"data": [{"starting_at": "x", "results": rows_p1}], "has_more": True,
                "next_page": "page_2"}
    return {"data": [{"starting_at": "y", "results": rows_p2}], "has_more": False}


ROUTES = {
    "/v1/organizations/usage_report/messages": _anthropic_usage,
    "/v1/organizations/cost_report": lambda q: {"data": [{"results": [
        {"workspace_id": "wrkspc_bench", "amount": "1.50"},     # cents
        {"workspace_id": "wrkspc_other", "amount": "900.00"}]}], "has_more": False},
    "/v1/organization/usage/completions": lambda q: {"data": [{"results": [
        {"model": "gpt-6-luna-2026-09-01", "input_tokens": 1100, "input_cached_tokens": 400,
         "output_tokens": 500, "num_model_requests": 3},
        {"model": "gpt-6.1-sol", "input_tokens": 77777, "input_cached_tokens": 0,
         "output_tokens": 1, "num_model_requests": 9}]}], "has_more": False},
    "/v1/organization/costs": lambda q: {"data": [{"results": [
        {"project_id": "proj_bench", "amount": {"value": 0.02, "currency": "usd"}},
        {"project_id": "proj_other", "amount": {"value": 5.0, "currency": "usd"}}]}],
        "has_more": False},
}


class Mock(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        REQUESTS.append((u.path, q, dict(self.headers)))
        body = json.dumps(ROUTES[u.path](q)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


srv = HTTPServer(("127.0.0.1", 0), Mock)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
os.environ.update({"ANTHROPIC_ADMIN_BASE_URL": base, "OPENAI_ADMIN_BASE_URL": base,
                   "ANTHROPIC_ADMIN_KEY": "sk-ant-admin-test", "OPENAI_ADMIN_KEY": "sk-admin-test"})
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconcile_billing as R  # noqa: E402  (reads the base URLs at import)

TMP = Path(tempfile.mkdtemp(prefix="reconcile_smoke_"))


def record(name, *, provider, model, start, end, prompt, cached, write, completion,
           total=None, calls=1, retries=0, cost=0.0, served=None, utc=True):
    rec = {"timestamp": "2026-10-15T10:00:00", "provider": provider, "model": model,
           "served_models": served or [model], "wall_time_seconds": 120.0,
           "tracking": {"total_calls": calls, "retries": {"total": retries},
                        "tokens": {"prompt": prompt, "completion": completion,
                                   "total": total if total is not None else prompt + completion,
                                   "cached": cached, "cache_write": write},
                        "cost": {"total_usd": cost}}}
    if utc:
        rec["started_at"], rec["finished_at"] = start, end
    p = TMP / f"{name}_metrics.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


try:
    # Two overlapping Haiku runs (compared together) whose sums match the mock exactly:
    # uncached 1000, cache reads 500, writes 150, output 300.
    record("haiku_a", provider="anthropic", model="claude-haiku-5-5",
           start="2026-10-15T12:00:00+00:00", end="2026-10-15T12:05:00+00:00",
           prompt=1000, cached=300, write=100, completion=200, cost=0.010)
    record("haiku_b", provider="anthropic", model="claude-haiku-5-5",
           start="2026-10-15T12:06:00+00:00", end="2026-10-15T12:08:00+00:00",
           prompt=650, cached=200, write=50, completion=100, cost=0.005)
    # A GPT-6 run served by a dated snapshot; its 50 cache writes sit inside OpenAI's input.
    record("luna", provider="openai", model="gpt-6-luna",
           start="2026-10-15T13:00:00+00:00", end="2026-10-15T13:02:00+00:00",
           prompt=1100, cached=400, write=50, completion=450, calls=2, retries=1, cost=0.03,
           served=["gpt-6-luna-2026-09-01"])
    # An old Gemini record without the UTC fields, with 300 thinking tokens.
    record("flash", provider="google", model="gemini-3.5-flash", start=None, end=None,
           prompt=800, cached=100, write=0, completion=200, total=1300, cost=0.002, utc=False)

    out_csv = TMP / "out.csv"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = R.main(["--runs", str(TMP), "--anthropic-workspace", "wrkspc_bench",
                       "--openai-project", "proj_bench", "--gemini-project", "gcp-bench",
                       "--bq-table", "billing.ds.gcp_billing_export_resource_v1_X",
                       "--csv", str(out_csv)])
    text = buf.getvalue()
    import csv  # noqa: E402
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    tok = {(r["provider"], r["metric"]): (int(r["ours"]), r["theirs"]) for r in rows if r["kind"] == "tokens"}
    cost = {r["provider"]: (float(r["ours"]), float(r["theirs"])) for r in rows if r["kind"] == "cost"}

    check(code == 0, "runs to completion")
    check("2 run(s)" in text and "windows overlap, compared together" in text,
          "overlapping runs of one model are compared as one group")
    check(all(tok[("anthropic", k)] == (v, str(v)) for k, v in
              {"uncached_input": 1000, "cache_read": 500, "cache_write": 150, "output": 300}.items()),
          "Anthropic: uncached, cache reads, 5m+1h writes and output summed over both pages, "
          "other models ignored, and ours matches")
    check(tok[("anthropic", "requests")][1] == "", "Anthropic reports no request count (n/a)")
    check(tok[("openai", "uncached_input")] == (700, "700")
          and tok[("openai", "cache_read")] == (400, "400")
          and tok[("openai", "output")] == (450, "500")
          and tok[("openai", "requests")] == (3, "3"),
          "OpenAI: dated snapshot matched, cache writes folded into input, differences kept")
    check(cost.get("anthropic") == (0.015, 0.015) and cost.get("openai") == (0.03, 0.02),
          "daily cost: only the dedicated workspace / project, cents converted to dollars")
    usage_reqs = [q for p, q, _ in REQUESTS if p.endswith("usage_report/messages")]
    check(len(usage_reqs) == 2 and usage_reqs[1].get("page") == ["page_2"]
          and all(q.get("workspace_ids[]") == ["wrkspc_bench"] and q.get("bucket_width") == ["1m"]
                  for q in usage_reqs),
          "Anthropic usage: 1-minute buckets, workspace filter, follows next_page")
    a_hdr = next(h for p, _, h in REQUESTS if p.startswith("/v1/organizations"))
    o_req = next((q, h) for p, q, h in REQUESTS if p.endswith("usage/completions"))
    check(a_hdr.get("x-api-key") == "sk-ant-admin-test"
          and o_req[1].get("Authorization") == "Bearer sk-admin-test"
          and o_req[0].get("project_ids") == ["proj_bench"],
          "admin keys sent, OpenAI project filter sent")
    window = next(q for p, q, _ in REQUESTS if p.endswith("usage_report/messages"))
    check(window["starting_at"] == ["2026-10-15T11:59:00Z"]
          and window["ending_at"] == ["2026-10-15T12:10:00Z"],
          "window padded by 1 minute before and 2 after the group's runs")
    check("thinking 300" in text and "`billing.ds.gcp_billing_export_resource_v1_X`" in text
          and "project.id = 'gcp-bench'" in text,
          "Gemini: thinking tokens shown, SQL names the export table and project")
    check(R.load_run(TMP / "flash_metrics.json").approx_window,
          "an old record without UTC fields falls back to timestamp - wall time")

    # Grouping: a run outside the padded window starts its own group.
    far = R.load_run(record("haiku_c", provider="anthropic", model="claude-haiku-5-5",
                            start="2026-10-15T15:00:00+00:00", end="2026-10-15T15:01:00+00:00",
                            prompt=1, cached=0, write=0, completion=1))
    near = [R.load_run(TMP / "haiku_a_metrics.json"), R.load_run(TMP / "haiku_b_metrics.json")]
    groups = R.group_runs(near + [far])
    check([len(g.runs) for g in groups] == [2, 1], "a run hours later is compared on its own")

    # No admin key: clear exit before any request.
    os.environ.pop("ANTHROPIC_ADMIN_KEY")
    n_before = len(REQUESTS)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = R.main(["--runs", str(TMP), "--provider", "anthropic"])
    check(code == 2 and "ANTHROPIC_ADMIN_KEY" in buf.getvalue() and len(REQUESTS) == n_before,
          "a missing admin key exits with a fix-it message and sends nothing")
finally:
    srv.shutdown()
    shutil.rmtree(TMP, ignore_errors=True)

print("\nALL PASSED" if not FAILED else f"\n{len(FAILED)} FAILED")
sys.exit(1 if FAILED else 0)

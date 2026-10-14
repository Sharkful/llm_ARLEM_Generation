"""
Capture the exact tool / function schemas each provider would be sent (issue #70).

Points the OpenAI, Anthropic and google-genai SDKs at a local server that records
each request body and answers 400, then makes one benchmark-style call per
(provider, response model). Nothing reaches a real API and no keys are needed.

The point is gate 3 of #44 for an instructor upgrade: run it on the old pin and on
the new one, then diff. A change in what a provider receives for the same response
model (e.g. instructor starting to emit ``const`` or a discriminator again, which
Gemini rejects) shows up here before it costs a sweep.

Usage:
    python "Code/Benchmark/probes/probe_tool_schemas.py" before.json
    pip install -r requirements.txt            # the new pin
    python "Code/Benchmark/probes/probe_tool_schemas.py" after.json --diff before.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for sub in ("Code/Benchmark", "Code/Schemas", "Code/Hierarchical", ""):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

# Representative models: the schema path depends on the provider and decode mode,
# not on the model id. One wave-2 model per provider, plus one current-roster model
# per provider (#72: GPT-6 on the Responses API, pinned reasoning effort).
PROBE_MODELS = ("gpt-5.4-mini", "claude-haiku-4.5", "gemini-3.5-flash",
                "gpt-6-luna", "claude-haiku-5.5", "gemini-3.8-flash")

# Request keys that carry the conversation, not the schema; left out of the record.
# The system text is kept separately (``system_text``): the Anthropic request's
# "call the tool" line lives there and is part of what the upgrade gate checks.
CONVERSATION_KEYS = {"messages", "contents", "input", "system", "systemInstruction"}


def system_text(body: dict) -> str:
    """Every system instruction in a request, in order, as one string (Anthropic
    ``system``, Gemini ``systemInstruction``, OpenAI ``role: system`` messages)."""
    def text(x):
        if isinstance(x, str):
            return [x]
        if isinstance(x, list):
            return [t for item in x for t in text(item)]
        if isinstance(x, dict):
            if "text" in x:
                return [x["text"]]
            return text(x.get("parts") or x.get("content") or [])
        return []

    parts = text(body.get("system")) + text(body.get("systemInstruction"))
    parts += [t for m in body.get("messages", []) if m.get("role") == "system"
              for t in text(m.get("content"))]
    return "\n".join(parts)

captured: list[dict] = []


class Capture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        captured.append({"path": self.path, "body": body})
        data = json.dumps({"error": {"type": "invalid_request_error", "code": 400,
                                     "status": "INVALID_ARGUMENT",
                                     "message": "probe: request captured"}}).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def response_models() -> dict:
    import arlem_full
    import arlem_simplified
    from json_lab import DemoModule, Lab
    from lab_outline import LabOutline
    from module_asset_plan import ModuleAssetPlan
    from pipeline import LabPlan, PlannedDemoModule

    return {
        "json_lab.Lab": Lab,
        "json_lab.DemoModule": DemoModule,
        "lab_outline.LabOutline": LabOutline,
        "arlem_full.ARLEMScenario": arlem_full.ARLEMScenario,
        "arlem_simplified.ARLEMScenario": arlem_simplified.ARLEMScenario,
        "hierarchical.LabPlan": LabPlan,
        "hierarchical.ModuleAssetPlan": ModuleAssetPlan,
        "hierarchical.PlannedDemoModule": PlannedDemoModule,
    }


def capture_all() -> dict:
    srv = HTTPServer(("127.0.0.1", 0), Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    os.environ.update({
        "OPENAI_BASE_URL": f"{base}/v1", "OPENAI_API_KEY": "probe",
        "ANTHROPIC_BASE_URL": base, "ANTHROPIC_API_KEY": "probe",
        "GOOGLE_GEMINI_BASE_URL": base, "GEMINI_API_KEY": "probe",
        "NO_PROXY": ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1"])),
    })
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    import instructor
    from benchmark import create_instructor_client, provider_create_kwargs
    from benchmark_config import MODELS, SpecType

    result = {"instructor": instructor.__version__, "requests": {}}
    for model_key in PROBE_MODELS:
        mc = MODELS[model_key]
        client = create_instructor_client(mc, SpecType.JSON_LAB)
        for name, rm in response_models().items():
            before = len(captured)
            kwargs = dict(response_model=rm, max_retries=0,
                          messages=[{"role": "system", "content": "probe"},
                                    {"role": "user", "content": "probe"}])
            kwargs.update(provider_create_kwargs(mc, rm))
            try:
                client.create(**kwargs)
            except Exception:  # noqa: BLE001 — the 400 is the point
                pass
            reqs = captured[before:]
            if not reqs:
                print(f"  {model_key:<18} {name:<34} NO REQUEST SENT")
                continue
            body = reqs[0]["body"]
            result["requests"][f"{model_key}/{name}"] = {
                **{k: v for k, v in body.items() if k not in CONVERSATION_KEYS},
                "system_text": system_text(body),
            }
            print(f"  {model_key:<18} {name:<34} captured "
                  f"({len(json.dumps(body.get('tools', '')))} chars of tools)")
    srv.shutdown()
    return result


def diff(a, b, path="") -> list[str]:
    if type(a) is not type(b):
        return [f"{path}: {a!r:.80} -> {b!r:.80}"]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: added {json.dumps(b[k])[:80]}")
            elif k not in b:
                out.append(f"{path}/{k}: removed {json.dumps(a[k])[:80]}")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f"{path}: list length {len(a)} -> {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: {a!r:.80} -> {b!r:.80}"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path, help="Where to write the captured schemas (JSON)")
    p.add_argument("--diff", type=Path, help="Earlier capture to compare against")
    args = p.parse_args()
    logging.getLogger("instructor").setLevel(logging.CRITICAL)  # every probe call 400s

    # A capture under the wrong instructor makes a misleading gate baseline: stop if
    # a provider SDK is missing, warn if instructor isn't the requirements.txt pin.
    from benchmark import require_environment
    from benchmark_config import MODELS
    require_environment([MODELS[m] for m in PROBE_MODELS])

    result = capture_all()
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\ninstructor {result['instructor']}: {len(result['requests'])} requests -> {args.out}")

    for key, req in result["requests"].items():
        text = json.dumps(req.get("tools", req))
        if key.startswith("gemini") and ('"const"' in text or '"discriminator"' in text):
            print(f"  WARNING {key}: Gemini tool schema contains const/discriminator")

    if args.diff:
        old = json.loads(args.diff.read_text(encoding="utf-8"))
        print(f"\nDiff against instructor {old['instructor']} ({args.diff}):")
        changes = diff(old["requests"], result["requests"])
        print("\n".join(f"  {c}" for c in changes) if changes else "  no differences")


if __name__ == "__main__":
    main()

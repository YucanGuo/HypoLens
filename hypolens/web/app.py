"""
GraphHT Web UI  —  Flask backend
================================
Provides these endpoints:

  POST /api/build_run   — validate + parse a hand-built (Visual Builder) pattern
  POST /api/execute     — Server-Sent Events stream running the *actual* GHT
                          engine (Opt_PHASE sampling + hypothesis testing) for
                          a parsed_pattern produced by /api/build_run
  GET  /api/schema/<dataset> — schema summary for the Visual Builder
  GET  /               — redirects to the homepage

SSE event types (for /api/execute):
  status      — human-readable progress messages (loading graph, sampling…)
  result      — final GHTExecutor report (ground truth + sampling results)
  error       — translation/execution failure
  done        — stream finished marker

Usage
-----
  python -m hypolens.web.app [--host 0.0.0.0] [--port 7860]
    [--dataset Citation|MovieLens|Yelp]
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

# ── make project root importable ──────────────────────────────────────────────
_WEB_DIR     = Path(__file__).parent
_HYPOLENS    = _WEB_DIR.parent
_PROJECT_ROOT = _HYPOLENS.parent
for p in [str(_PROJECT_ROOT), str(_HYPOLENS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, Response, jsonify, redirect, render_template, request, stream_with_context, url_for

from hypolens.tools.report_summary import build_report_summary

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False

# ── helpers ───────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Convert numpy scalar types (bool_, float64, int64, …) to native Python
    so json.dumps can serialise them.  numpy.bool_ / numpy.float64 report
    __class__.__name__ == 'bool' / 'float', which is why the default TypeError
    message is misleading."""
    try:
        import numpy as np
    except ImportError:
        raise TypeError(repr(obj))
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(repr(obj))


def _sse(event: str, data: Any) -> str:
    """Format a single SSE message."""
    payload = json.dumps(data, ensure_ascii=False, default=_json_default)
    return f"event: {event}\ndata: {payload}\n\n"


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/app")
def index():
    datasets = ["MovieLens", "Citation", "Yelp"]
    default  = app.config.get("DEFAULT_DATASET", "MovieLens")
    import os as _os
    return render_template(
        "index.html",
        datasets=datasets,
        default_dataset=default,
        # Lets the Follow-up Hypotheses panel offer an LLM/rule-based choice
        # only when an LLM is actually usable.
        llm_configured=bool(_os.getenv("LLM_API_KEY", "")),
    )


@app.route("/api/schema/<dataset>")
def api_schema(dataset: str):
    """Return a trimmed schema summary for the given dataset."""
    _SCHEMA_DIR = _PROJECT_ROOT / "ghtql" / "schema"
    path = _SCHEMA_DIR / f"{dataset.lower()}.json"
    if not path.exists():
        return jsonify({"error": f"Schema not found for dataset: {dataset}"}), 404
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Optional companion file produced by dataprep/value_stats_extractor.py —
    # top-N most frequent values per attribute, used to power autocomplete
    # candidates in the Visual Builder. Missing gracefully (older datasets
    # that haven't been (re)processed yet just get no candidates).
    values_path = _SCHEMA_DIR / f"{dataset.lower()}_values.json"
    node_value_stats: dict = {}
    edge_value_stats: dict = {}
    if values_path.exists():
        with open(values_path, encoding="utf-8") as f:
            value_stats = json.load(f)
        node_value_stats = value_stats.get("node_types", {})
        edge_value_stats = value_stats.get("edge_types", {})

    # Build a lightweight summary
    node_types = {}
    for nt, info in raw.get("node_types", {}).items():
        attrs = {}
        nt_values = node_value_stats.get(nt, {})
        for attr_name, attr_info in info.get("attributes", {}).items():
            top_values = [v["value"] for v in nt_values.get(attr_name, [])]
            attrs[attr_name] = {
                "dtype":      attr_info.get("dtype"),
                "coverage":   attr_info.get("coverage"),
                "samples":    attr_info.get("samples", [])[:3],
                "top_values": top_values,
            }
        # binary_flags summary (Yelp) — sorted by coverage desc
        bf = info.get("binary_flags")
        if bf:
            bf_sorted = sorted(bf.items(), key=lambda x: x[1].get("coverage", 0), reverse=True)
            bf_sample = [
                {"name": name, "coverage": info_bf.get("coverage", 0)}
                for name, info_bf in bf_sorted[:12]
            ]
        else:
            bf_sample = []
        node_types[nt] = {
            "count":         info.get("count"),
            "attributes":    attrs,
            "binary_flags_count": len(bf) if bf else 0,
            "binary_flags_sample": bf_sample,
        }

    edge_types = {}
    for et, info in raw.get("edge_types", {}).items():
        attrs = {}
        et_values = edge_value_stats.get(et, {})
        for attr_name, attr_info in info.get("attributes", {}).items():
            top_values = [v["value"] for v in et_values.get(attr_name, [])]
            attrs[attr_name] = {
                "dtype":      attr_info.get("dtype"),
                "samples":    attr_info.get("samples", [])[:3],
                "top_values": top_values,
            }
        edge_types[et] = {"count": info.get("count"), "attributes": attrs}

    # Supplement edge_types with any edge names that appear only in connectivity
    # (some schema JSONs have incomplete edge_types sections)
    for conn in raw.get("connectivity", []):
        et = conn.get("edge")
        if et and et not in edge_types:
            edge_types[et] = {"count": conn.get("count"), "attributes": {}}

    payload = json.dumps({
        "dataset":      raw.get("dataset"),
        "graph_type":   raw.get("graph_type"),
        "node_count":   raw.get("node_count"),
        "edge_count":   raw.get("edge_count"),
        "node_types":   node_types,
        "edge_types":   edge_types,
        "connectivity": raw.get("connectivity", []),
    }, ensure_ascii=False)
    return Response(payload, mimetype="application/json")


@app.route("/api/build_run", methods=["POST"])
def api_build_run():
    """
    Validate + parse a pattern built on the Visual Builder canvas.

    Request JSON:
      {
        "dataset": str,
        "pattern_id": str (optional, default "visual-pattern"),
        "pattern": { "nodes": [...], "edges": [...] },
        "target":  { "node": {...} } | { "node_diff": {...} },
        "test":    { "agg": "mean", "op": str, "rhs": ... }
      }

    Response JSON:
      { "success": bool, "ghtql": str?, "parsed_pattern": dict?,
        "errors": [str], "warnings": [str] }
    """
    from ghtql.unparser import unparse_entry
    from ghtql.parser import parse_string, GHTQLError
    from hypolens.tools.validator import GHTQLValidator

    body    = request.get_json(force=True, silent=True) or {}
    dataset = (body.get("dataset") or "Citation").strip()
    pid     = (body.get("pattern_id") or "visual-pattern").strip() or "visual-pattern"
    pattern = body.get("pattern") or {}
    target  = body.get("target")
    test    = body.get("test")

    nodes = pattern.get("nodes") or []
    edges = pattern.get("edges") or []

    if not nodes:
        return jsonify({"success": False, "errors": ["Pattern must contain at least one node."], "warnings": []})
    if not target:
        return jsonify({"success": False, "errors": ["A TEST target (node or node_diff) is required."], "warnings": []})
    if not test or not test.get("op") or test.get("rhs") is None:
        return jsonify({"success": False, "errors": ["TEST clause requires an operator and a right-hand side."], "warnings": []})

    entry = {
        "dataset": dataset,
        "pattern": {"nodes": nodes, "edges": edges},
        "target":  target,
        "test":    {"agg": test.get("agg", "mean"), "op": test["op"], "rhs": test["rhs"]},
    }

    # ── build GHT-QL source from the canvas state ─────────────────────────────
    try:
        ghtql_src = unparse_entry(pid, entry)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"success": False, "errors": [f"Failed to build GHT-QL: {exc}"], "warnings": []})

    # ── validate (syntax + schema) ────────────────────────────────────────────
    validator = GHTQLValidator()
    v_result  = validator.validate(ghtql_src)
    if not v_result["valid"]:
        return jsonify({
            "success":  False,
            "ghtql":    ghtql_src,
            "errors":   v_result["errors"],
            "warnings": v_result["warnings"],
        })

    # ── parse back into the canonical pattern dict ────────────────────────────
    try:
        parsed_patterns = parse_string(ghtql_src)
    except GHTQLError as exc:
        return jsonify({"success": False, "ghtql": ghtql_src, "errors": [f"Parse error: {exc}"], "warnings": v_result["warnings"]})

    return jsonify({
        "success":        True,
        "ghtql":          ghtql_src.strip(),
        "parsed_pattern": parsed_patterns,
        "errors":         [],
        "warnings":       v_result["warnings"],
    })


@app.route("/api/execute", methods=["POST"])
def api_execute():
    """
    Run the *real* GHT engine (Opt_PHASE sampling + one-sample hypothesis
    testing, see ght_engine/executor.py) for a parsed GHT-QL pattern,
    streamed as Server-Sent Events.

    Request JSON:
      {
        "parsed_pattern": {<pattern_id>: {...}}  — output of /api/run or /api/build_run
                          | {...}                — a single pattern entry directly
        "sampling_percent": [5]   (optional, % of total nodes; defaults to a single ratio)
        "num_samples":      10            (optional, number of repeated samples)
        "sampling_method":  "Opt_PHASE"   (optional, "Opt_PHASE" | "PHASE")
        "mode":             "sample"      (optional, "sample" | "full" | "compare")
      }

    SSE events emitted:
      status  – { "message": str }
      result  – GHTExecutor.run() report dict
      error   – { "message": str }
      done    – {}
    """
    from ght_engine.executor import GHTExecutor, GHTExecutorError

    body = request.get_json(force=True, silent=True) or {}
    parsed_pattern = body.get("parsed_pattern")
    if not parsed_pattern or not isinstance(parsed_pattern, dict):
        return jsonify({"error": "parsed_pattern (dict) is required"}), 400

    # Accept either {pattern_id: entry} (agent/build_run output) or a bare entry.
    if "pattern" in parsed_pattern and "dataset" in parsed_pattern:
        entry = parsed_pattern
    else:
        pattern_id = body.get("pattern_id") or next(iter(parsed_pattern))
        entry = parsed_pattern.get(pattern_id)
        if entry is None:
            return jsonify({"error": f"pattern_id '{pattern_id}' not found in parsed_pattern"}), 400

    sampling_percent = body.get("sampling_percent") or [5]
    num_samples      = int(body.get("num_samples") or 10)
    sampling_method  = body.get("sampling_method") or "Opt_PHASE"
    mode             = body.get("mode") or "sample"

    def generate():
        q: queue.Queue = queue.Queue()

        def worker():
            try:
                def progress(msg: str) -> None:
                    q.put(("status", {"message": msg}))

                executor = GHTExecutor()
                report = executor.run(
                    entry,
                    sampling_percent=sampling_percent,
                    num_samples=num_samples,
                    sampling_method=sampling_method,
                    mode=mode,
                    progress_cb=progress,
                )
                q.put(("result", report))
            except GHTExecutorError as exc:
                q.put(("error", {"message": str(exc)}))
            except Exception:  # noqa: BLE001
                q.put(("error", {"message": traceback.format_exc()}))
            finally:
                q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                yield _sse("done", {})
                break
            ev, data = item
            yield _sse(ev, data)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":           "no-cache",
            "X-Accel-Buffering":       "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    Call an LLM (DeepSeek by default) to write a plain-language summary of the
    GHT execution report so non-expert users can understand the hypothesis-test
    result.

    Request JSON:
      { "report": <GHTExecutor.run() report dict> }

    Response JSON:
      { "summary": str }   on success
      { "error": str }     if the LLM is not configured or the call fails
    """
    body = request.get_json(force=True, silent=True) or {}
    report = body.get("report")
    if not report or not isinstance(report, dict):
        return jsonify({"error": "report (dict) is required"}), 400

    # ── build a compact text snapshot of the report for the prompt ──────────
    report_text = build_report_summary(report)

    # ── call the LLM ───────────────────────────────────────────────────────
    import os as _os
    api_key  = _os.getenv("LLM_API_KEY", "")
    base_url = _os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model    = _os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        return jsonify({"error": "LLM not configured (set LLM_API_KEY environment variable)"}), 503

    try:
        from hypolens.llm import LLMClient, system_msg, user_msg
        client = LLMClient(model=model, api_key=api_key, base_url=base_url)
        prompt = (
            "You are a statistical analyst. Given the hypothesis-test result "
            "below, write a concise INSIGHTFUL conclusion (2-4 sentences, "
            "max ~80 words) that goes beyond restating the result. The "
            "audience is a non-expert who wants to understand what the result "
            "MEANS in practice.\n\n"
            "Your conclusion should:\n"
            "  1. State the practical takeaway (is the hypothesis supported?).\n"
            "  2. Comment on the STRENGTH of the evidence — is the sample mean "
            "comfortably above/below the threshold, or is it a close call?\n"
            "  3. If sampling was used, note whether the sampling-based "
            "result is trustworthy (did most subgraphs agree? was the "
            "sampling ratio adequate for the graph size?).\n"
            "  4. If the full-graph ground truth is available (compare mode), "
            "note whether sampling agreed with the exact result — this tells "
            "the user whether the cheaper sampling method is reliable for "
            "similar queries in future.\n"
            "  5. If execution-time information is present, you may add ONE "
            "short clause about cost: in compare mode, you can mention that "
            "sampling was faster than the full-graph exact pass (citing the "
            "speed-up factor if given); in full-graph mode, you may note the "
            "exact run's cost. Do not dwell on timing.\n"
            "Do NOT use jargon like 'H0', 'p-value', 'null hypothesis'. "
            "Do NOT just repeat numbers — interpret what they mean.\n\n"
            "IMPORTANT — p-value semantics in this system are INVERTED "
            "relative to the usual convention, because of how the test is "
            "constructed:\n"
            "  * The user's hypothesis (e.g. 'mean > 25') is formulated as "
            "the NULL hypothesis H0, NOT as the alternative H1.\n"
            "  * So scipy's t-test asks: 'is the data consistent with H0 "
            "(the user's claim)?'\n"
            "  * p >= 0.05  → fail to reject H0 → the hypothesis is "
            "ACCEPTED / SUPPORTED by the data.\n"
            "  * p <  0.05  → reject H0 → the hypothesis is REJECTED.\n"
            "  * A HIGHER p-value means STRONGER support (p = 1 is the "
            "strongest possible support).\n"
            "This is the opposite of the usual 'small p = significant = "
            "effect exists' convention, because here the user's claim IS "
            "the null, not the alternative. Do NOT re-interpret the p-value "
            "yourself — follow the result line below. Your conclusion MUST "
            "be consistent with it: if result = accepted, the data "
            "supports the claim; if result = rejected, it does not.\n\n"
            f"--- Test Result ---\n{report_text}\n--- End ---"
        )
        summary = client.chat([system_msg("You are a helpful statistics assistant who explains hypothesis-test results with practical insight."),
                               user_msg(prompt)],
                              max_tokens=256, temperature=0.4)
        client.close()
        return jsonify({"summary": summary.strip()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"LLM call failed: {exc}"}), 502


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Dispatch follow-up requests to the LLM or rule-based recommendation tool.

    Accepts a bare pattern or ``{id: entry}``, an optional report, provider,
    and rule-selection/refinement/refresh options. Returns recommendations,
    the selected provider, and an optional fallback note.
    """
    from hypolens.tools.llm_recommender import LLMRecommendationError, LLMRecommender

    body = request.get_json(force=True, silent=True) or {}
    parsed_pattern = body.get("parsed_pattern")
    if not parsed_pattern or not isinstance(parsed_pattern, dict):
        return jsonify({"error": "parsed_pattern (dict) is required"}), 400

    # Adapt either supported request shape to the tools' pattern-entry API.
    if "pattern" in parsed_pattern and "dataset" in parsed_pattern:
        entry = parsed_pattern
    else:
        entry = parsed_pattern.get(next(iter(parsed_pattern)))
        if entry is None:
            return jsonify({"error": "could not extract pattern entry"}), 400

    import os as _os
    api_key = _os.getenv("LLM_API_KEY", "")
    provider = (body.get("provider") or "").strip().lower()
    if provider not in ("llm", "rules"):
        provider = "llm" if api_key else "rules"

    def _rules_payload(note: str | None = None):
        from hypolens.tools.rule_recommender import get_recommender
        recommender = get_recommender("rules")
        recommendations = recommender.recommend(entry, {
            "rules": body.get("rules"),
            "mode": body.get("refinement_mode") or "single",
            "exclude": body.get("exclude") or [],
        }) if recommender is not None else []
        payload = {"recommendations": recommendations, "provider": "rules"}
        if note:
            payload["note"] = note
        return jsonify(payload)

    if provider == "rules":
        return _rules_payload()
    if not api_key:
        return _rules_payload("LLM not configured; showing rule-based recommendations.")

    recommender = LLMRecommender(
        model=_os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=_os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    )
    try:
        recommendations = recommender.recommend(entry, {"report": body.get("report") or {}})
    except LLMRecommendationError:
        return _rules_payload("LLM call failed; showing rule-based recommendations.")
    if not recommendations:
        return _rules_payload("No valid LLM suggestions; showing rule-based recommendations.")
    return jsonify({"recommendations": recommendations, "provider": "llm"})


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(prog="python -m hypolens.web.app",
                                 description="Start the GraphHT web UI.")
    ap.add_argument("--host",        default="127.0.0.1")
    ap.add_argument("--port",        type=int, default=7860)
    ap.add_argument("--dataset",     default="MovieLens",
                    choices=["Citation", "MovieLens", "Yelp"])
    ap.add_argument("--debug",       action="store_true")
    # LLM (DeepSeek) — CLI args override env vars for /api/summarize
    ap.add_argument("--api-key",     default=None,
                    help="LLM API key (e.g. DeepSeek sk-...). Falls back to LLM_API_KEY env var.")
    ap.add_argument("--base-url",    default=None,
                    help="LLM API base URL. Default: https://api.deepseek.com/v1")
    ap.add_argument("--model",       default=None,
                    help="LLM model name. Default: deepseek-chat")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Promote CLI LLM args into env vars so /api/summarize (which reads os.getenv)
    # picks them up without any code change to the route.
    import os as _os
    if args.api_key:  _os.environ["LLM_API_KEY"]  = args.api_key
    if args.base_url: _os.environ["LLM_BASE_URL"] = args.base_url
    if args.model:    _os.environ["LLM_MODEL"]    = args.model
    app.config.update(
        DEFAULT_DATASET = args.dataset,
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

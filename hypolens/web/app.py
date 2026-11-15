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
import re
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


def _build_report_summary(report: dict) -> str:
    """Build a compact, human-readable text snapshot of a GHT execution report.

    Shared by /api/summarize (LLM conclusion) and /api/recommend (LLM follow-up
    question generation) so both prompts see the same view of the result.
    """
    v = report.get("verdict") or {}
    gt = report.get("ground_truth") or {}
    sr = (report.get("sampling_results") or [{}])[0] or {}
    gs = report.get("graph_stats") or {}
    is_mu = report.get("is_mu_test") is True
    test_scalar = report.get("test_scalar")
    comp = report.get("comparison") or "=="

    verdict_str = v.get("accept_h0")
    verdict_txt = ("accepted" if verdict_str is True
                   else "rejected" if verdict_str is False
                   else "inconclusive")
    rhs_label = ("the population mean (μ"
                 + (f" = {test_scalar}" if test_scalar is not None else "")
                 + ")") if is_mu else str(test_scalar)

    lines = [
        f"Dataset:          {report.get('dataset')}",
        f"Pattern type:     {report.get('pattern_type')}",
        f"Hypothesis (H0):  {report.get('H0') or '—'}",
        f"Attribute tested:  {report.get('attribute')}",
        f"Test description:  {report.get('test_description') or '—'}",
        f"Comparison:        sample_mean {comp} {rhs_label}",
        f"Test mode:         {report.get('mode')}",
        f"Graph size:        {gs.get('num_nodes')} nodes / {gs.get('num_edges')} edges",
        f"Total time:        {report.get('total_time_seconds')}s",
        "",
        f"Result:            H0 {verdict_txt}",
    ]
    if v.get("p_value") is not None:
        lines.append(f"p-value:           {v['p_value']}")
    if v.get("vote"):
        lines.append(f"Vote:              {v['vote']}")
    if gt:
        lines.append(f"Ground truth mean: {gt.get('value')}")
        lines.append(f"Ground truth matches: {gt.get('n_matches')}")
    full_graph_ref = v.get("full_graph")
    if full_graph_ref and full_graph_ref.get("accept_h0") is not None:
        fg_verdict = "accepted" if full_graph_ref["accept_h0"] else "rejected"
        lines.append(f"Full-graph result: H0 {fg_verdict}")
        if verdict_str is not None:
            agreement = ("consistent" if full_graph_ref["accept_h0"] == verdict_str
                         else "inconsistent")
            lines.append(f"Sampling vs full-graph: {agreement}")
    sr_reps = sr.get("repeats") or []
    if sr_reps:
        means = [r.get("sample_mean") for r in sr_reps if r.get("sample_mean") is not None]
        if means:
            lines.append(f"Sample means:      {len(means)} repeats, "
                         f"min={min(means):.4f}, max={max(means):.4f}, "
                         f"avg={sum(means)/len(means):.4f}")
        lines.append(f"Sampling ratio:    {sr.get('sampling_percent')}%")
        lines.append(f"Accuracy:          {sr.get('accuracy')}")
    return "\n".join(str(x) for x in lines)


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
    return render_template("index.html", datasets=datasets, default_dataset=default)


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
    report_text = _build_report_summary(report)

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


# Range-implying adjective + noun. The engine only filters by EXACT attribute
# values, so "older users" / "high-rated movies" / "popular papers" describe
# ranges the engine cannot express — reject them.
_RANGE_ADJ_RE = re.compile(
    r"\b(?:older|younger|elder|senior|junior|high-rated|low-rated|"
    r"higher-rated|lower-rated|popular|unpopular|longer|shorter|"
    r"longer-running|recent|older|newer|classic|mainstream|niche)"
    r"\s+(?:users|user|movies|movie|papers|paper|citations|citation|"
    r"reviews|review|ratings|rating|nodes|edges|items|item|"
    r"products|product|businesses|business|restaurants|restaurant)\b",
    re.IGNORECASE,
)


def _looks_like_range_filter(desc: str) -> bool:
    """
    Heuristic: detect descriptions that imply a range filter the engine
    cannot express (e.g. "older users", "high-rated movies", "popular
    papers"). The engine only supports exact attribute values, not ranges.
    """
    return bool(desc) and bool(_RANGE_ADJ_RE.search(desc))


# (node_label, attribute) pairs excluded from the schema fed to the LLM
# recommendation prompt because they're categorical integer codes, not
# meaningful measurements — their mean is meaningless (e.g. movie.genre =
# 1..5 genre IDs). Stripping them at the schema level means the LLM never
# sees them as candidate attributes, so it can't propose hypotheses about
# them. Kept in sync with the frontend vbIsNumericAttr special case in
# index.html.
_CATEGORICAL_INT_ATTRS = {("movie", "genre")}


def _schema_value_summary(dataset: str, max_attr: int = 8, max_vals: int = 6) -> str:
    """
    Compact text summary of node/edge attribute values for the LLM prompt.
    Returns lines like:
      movie.year : ["2000's", "1990's", "2010's", "1980's", "1970's"]
      user.age   : ["1", "2", "3", "4"]
    Uses the <dataset>_values.json (top-N most frequent values) when
    available; falls back to the raw schema samples otherwise. Returns
    "" when the dataset schema can't be loaded.
    """
    schema_dir = _PROJECT_ROOT / "ghtql" / "schema"
    base       = schema_dir / f"{dataset.lower()}"
    values_path = base.with_name(base.name + "_values.json")
    schema_path = base.with_suffix(".json")
    if not schema_path.exists():
        return ""
    lines: list[str] = []
    # load top-N most frequent values (preferred source)
    node_vals: dict = {}
    edge_vals: dict = {}
    if values_path.exists():
        with open(values_path, encoding="utf-8") as f:
            vs = json.load(f)
        node_vals = vs.get("node_types", {}) or {}
        edge_vals = vs.get("edge_types", {}) or {}
    # load raw schema (for samples fallback + attribute ordering)
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    for kind, types, vals_map in (
        ("node", schema.get("node_types", {}) or {}, node_vals),
        ("edge", schema.get("edge_types", {}) or {}, edge_vals),
    ):
        for tname, tinfo in types.items():
            attrs = (tinfo.get("attributes") or {}).items()
            # cap attributes: prefer those with top_values > 0
            attr_items = sorted(attrs, key=lambda kv: -len(vals_map.get(kv[0]) or []))
            for aname, ainfo in attr_items[:max_attr]:
                # skip categorical-int codes (e.g. movie.genre) — mean is
                # meaningless, and we don't want the LLM proposing hypotheses
                # about them
                if (tname, aname) in _CATEGORICAL_INT_ATTRS:
                    continue
                top = [str(v["value"]) for v in (vals_map.get(aname) or [])][:max_vals]
                if not top:
                    # fall back to schema samples (3 raw values)
                    top = [str(s) for s in (ainfo.get("samples") or [])][:max_vals]
                if not top:
                    continue
                lines.append(f"  {tname}.{aname} : {top}")
    if not lines:
        return ""
    return "Attribute value examples (use EXACT values from here, including any quoting/punctuation):\n" + "\n".join(lines)


def _reflect_recommendations(
    candidates: list[dict],
    current_ghtql:   str,
    report_summary:  str,
    model:           str,
    api_key:         str,
    base_url:        str,
) -> tuple[list[dict], bool]:
    """
    Reflection step — a single LLM call that (a) FILTERS unreasonable
    candidates and (b) RANKS the reasonable ones by quality.

    Returns ``(ranked, ok)``:
      * ``ok=True,  ranked non-empty`` → reflection succeeded with survivors;
        ``ranked`` is the filtered+ranked subset in LLM (best-first) order.
      * ``ok=True,  ranked == []``      → reflection succeeded but FILTERED
        EVERYTHING OUT; caller should retry generation.
      * ``ok=False``                    → TECHNICAL failure (exception, no
        JSON, no parseable indices); caller should fall back to the
        unreflected candidate list rather than retrying.
    """
    cand_lines = []
    for i, rec in enumerate(candidates):
        cand_lines.append(
            f"[{i}] description: {rec.get('description', '')}\n"
            f"    ghtql:\n{rec.get('ghtql', '')}"
        )
    candidates_block = "\n\n".join(cand_lines)

    prompt = (
        "You are a graph hypothesis-test analyst reviewing candidate follow-up "
        "questions. Below are the current pattern, its test result, and a "
        "list of candidate follow-up questions (each labeled with an index).\n\n"
        f"Current pattern:\n{current_ghtql}\n\n"
        f"Test result:\n{report_summary}\n\n"
        f"Candidate follow-ups:\n{candidates_block}\n\n"
        "Engine constraints (recap — reflection has no memory of the "
        "generation prompt):\n"
        "  - Supported TEST forms: a mean against μ or a numeric threshold "
        "(one-sample); |nA.attr - nB.attr| against μ or a threshold "
        "(absolute node difference); edgeA.attr - edgeB.attr against a "
        "numeric threshold, typically 0 (signed paired difference between "
        "two edges sharing a common node — e.g. how two user roles rate "
        "the same movie).\n"
        "  - NOT supported: an aggregate on the rhs (TEST mean(...) > "
        "mean(...)), or comparing two separate populations/patterns.\n"
        "  - Filters use EXACT attribute values only (e.g. age:1, "
        'genre:"Sci-Fi"). NOT ranges (no age>18, no "older users").\n'
        "  - Pattern topology is a linear PATH (n0->n1->n2->...). No "
        "cycles, squares, stars, or branching.\n\n"
        "For each candidate, judge whether it is REASONABLE:\n"
        "  1. Description faithfully matches its GHT-QL (attribute values, "
        "what is tested, threshold).\n"
        "  2. Tests a genuinely DIFFERENT hypothesis from the current "
        "pattern (not a trivial operator flip or same-attribute rerun).\n"
        "  3. Answerable by the engine (supported TEST form, no range "
        "filter, path topology only).\n"
        "  4. Meaningful and non-redundant given the current test result.\n"
        "  5. The description is UNAMBIGUOUS: a comparison like \"Do A rate "
        "X higher than B?\" is ambiguous (\"than B\" could be misread as the "
        "object being rated, not the group). Descriptions must disambiguate "
        "with a clarifier, e.g. \"Do A rate X higher than B DO?\" or "
        "\"A versus B: who rates X higher?\". Ambiguous comparisons rank "
        "LOWER than unambiguous ones; if a description cannot be read "
        "either way correctly, treat it as unreasonable.\n"
        "Unreasonable candidates are EXCLUDED from ranking.\n\n"
        "Rank the REASONABLE candidates by QUALITY (insightfulness + "
        "diversity + faithfulness), best first.\n\n"
        "Output ONLY a JSON array, each element:\n"
        '  {"index": <int>, "reason": "<one short sentence>"}\n'
        "Ordered BEST-FIRST, including ONLY reasonable candidates. Output "
        "the JSON array and nothing else."
    )

    try:
        from hypolens.llm import LLMClient, system_msg, user_msg
        with LLMClient(model=model, api_key=api_key, base_url=base_url) as client:
            raw = client.chat(
                [system_msg("You are a GHT-QL question reviewer."),
                 user_msg(prompt)],
                max_tokens=800, temperature=0.2,
            )
    except Exception:  # noqa: BLE001
        return [], False

    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return [], False
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], False
    if not isinstance(arr, list):
        return [], False

    seen_idx: set[int] = set()
    ordered_idx: list[int] = []
    for el in arr:
        if not isinstance(el, dict):
            continue
        try:
            idx = int(el.get("index"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(candidates)) or idx in seen_idx:
            continue
        seen_idx.add(idx)
        ordered_idx.append(idx)
    # ok=True even if empty — that means "filtered all out", which the caller
    # distinguishes from a technical failure and handles by retrying.
    return [candidates[i] for i in ordered_idx], True


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Generate **validated** follow-up GHT-QL questions based on the current
    pattern and its test result.  The LLM proposes candidates; each is
    syntax-validated and parsed before being returned, so the frontend only
    receives questions it can safely load onto the Visual Builder canvas.

    Request JSON:
      { "parsed_pattern": <entry dict | {id: entry}>,
        "report":        <GHTExecutor.run() report dict> }

    Response JSON:
      { "recommendations": [
          { "description": str,
            "ghtql":         str,
            "parsed_pattern": <entry dict>
          }, ...
        ] }
    """
    from ghtql.unparser import unparse_entry
    from ghtql.parser import parse_string, GHTQLError
    from hypolens.tools.validator import GHTQLValidator

    body          = request.get_json(force=True, silent=True) or {}
    parsed_pattern = body.get("parsed_pattern")
    report         = body.get("report") or {}
    if not parsed_pattern or not isinstance(parsed_pattern, dict):
        return jsonify({"error": "parsed_pattern (dict) is required"}), 400

    # ── extract the entry (accept {id: entry} or bare entry) ────────────────
    if "pattern" in parsed_pattern and "dataset" in parsed_pattern:
        entry = parsed_pattern
    else:
        pid   = next(iter(parsed_pattern))
        entry = parsed_pattern.get(pid)
        if entry is None:
            return jsonify({"error": "could not extract pattern entry"}), 400

    # ── unparse current pattern for LLM context ─────────────────────────────
    pattern_id = entry.get("id") or "current"
    try:
        current_ghtql = unparse_entry(pattern_id, entry)
    except Exception:  # noqa: BLE001
        current_ghtql = "(unparse failed)"

    report_summary = _build_report_summary(report) if report else ""

    # load the dataset's attribute value examples so the LLM uses EXACT
    # values (e.g. "2000's" rather than 2000) when filtering attributes
    schema_values = _schema_value_summary(entry.get("dataset") or "")

    # ── call the LLM to generate candidate follow-ups ───────────────────────
    import os as _os
    api_key  = _os.getenv("LLM_API_KEY", "")
    base_url = _os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model    = _os.getenv("LLM_MODEL", "deepseek-chat")
    if not api_key:
        return jsonify({"error": "LLM not configured (set LLM_API_KEY environment variable)"}), 503

    prompt = (
        "You are a graph hypothesis-test analyst. Based on the GHT-QL pattern "
        "and its test result below, generate 5-8 follow-up questions that "
        "explore genuinely DIFFERENT hypotheses — not trivial variations whose "
        "answers can be directly inferred from the current result.\n\n"
        "GHT-QL syntax (the current pattern below is a working example):\n"
        "  -- <pattern_id>\n"
        "  MATCH (n0:label {attr:\"value\"})->(n1:label),\n"
        "        (n1)->(n2:label {attr:val})\n"
        "  ON <Dataset>\n"
        "  TEST mean(<target>) <op> <rhs>\n"
        "Where:\n"
        "  <target> = node_id.attr  |  node_id->node_id.attr  |  |nA.attr - nB.attr|  |\n"
        "            edgeA.attr - edgeB.attr   (edgeA/edgeB are var->var refs)\n"
        "  <op>     = <  >              (only < and > are supported)\n"
        "  <rhs>    = mu  |  <number>      (use ONLY mu or a number, NOT mean(...))\n\n"
        "The PAIRED EDGE-DIFFERENCE form (an important tool):\n"
        "  TEST mean(n0->n1.rating - n2->n1.rating) > 0\n"
        "This means: for every matched path, take the difference of the SAME "
        "attribute on two DIFFERENT pattern edges, then test the mean of those "
        "per-path differences against a scalar threshold (almost always 0). "
        "Use it for paired comparisons where two roles interact with a shared "
        "middle node — e.g. \"Do programmers rate a movie higher than another "
        "user rates the SAME movie?\" (the shared movie controls for item "
        "effects). Requirements:\n"
        "  - Both edges must be declared in the MATCH clause and connect to a "
        "COMMON node (e.g. n0->n1 and n2->n1 share n1).\n"
        "  - Both edges must use the SAME attribute name.\n"
        "  - The rhs must be a NUMBER (typically 0), never mu.\n\n"
        f"Current pattern:\n{current_ghtql}\n\n"
        f"Test result:\n{report_summary}\n\n"
        "CRITICAL RULES:\n"
        "  1. DO NOT generate a question that is identical or equivalent to "
        "the current pattern. Before outputting each candidate, compare it "
        "against the Current pattern above — if it tests the SAME attribute "
        "with the SAME operator and rhs, REJECT it.\n"
        "  2. The pattern topology MUST be a linear PATH (a chain of nodes "
        "connected by edges: n0->n1->n2->...). Do NOT create cycles, "
        "squares, stars, or branching structures — the execution engine "
        "(PHASE sampler) only supports path-based patterns.\n"
        "  3. DO NOT flip the operator direction (e.g. '> mu' → '< mu') — "
        "the answer is directly inferable from the current result.\n"
        "  4. DO NOT set the threshold to the observed sample mean or any "
        "value trivially close to it.\n"
        "  5. DO explore genuinely different questions:\n"
        "     - Test a DIFFERENT attribute on a DIFFERENT node in the "
        "pattern (not just a synonym for the current attribute).\n"
        "     - Add or remove an attribute filter to change which subgraphs "
        "are matched.\n"
        "     - Test an EDGE attribute (var->var.attr) instead of a node "
        "attribute, or vice versa.\n"
        "     - Test the difference between two nodes' attributes "
        "(|nA.attr - nB.attr|), where both nodes are on the path.\n"
        "     - Test a PAIRED EDGE comparison (edgeA.attr - edgeB.attr "
        "around a shared node) — e.g. comparing how two different user "
        "roles rate the same movie, or the same user role rating two "
        "different kinds of items.\n"
        "     - Extend the pattern path with an additional node/edge hop.\n"
        "  6. Each question must be a complete GHT-QL statement with --, "
        "MATCH, ON, TEST clauses.\n"
        "  7. Use ONLY node labels and attributes that appear in the current "
        "pattern's MATCH clause or in the dataset schema. Do NOT invent "
        "attributes that don't exist on the node types in the pattern.\n"
        "  8. When filtering an attribute, use EXACT values from the value "
        "examples list below — including any quoting, punctuation, or "
        "apostrophe-s suffix (e.g. if the dataset stores year as \"2000's\", "
        "you MUST write {year:\"2000's\"}, NOT {year:2000}). If the value "
        "is a string in the dataset, quote it; if it is a number, write it "
        "bare.\n"
        "  9. Each recommendation must be meaningfully different from the "
        "others AND from the original question.\n"
        " 10. The description MUST faithfully describe what the pattern "
        "actually tests. Concretely:\n"
        "     - Every attribute value you set in the MATCH clause (e.g. "
        "{genre:\"Sci-Fi\"} or {age:1}) MUST be reflected in the description. "
        "If the pattern sets age:1, the description must say \"users with "
        "age=1\" — NEVER say \"older users\" or \"young users\".\n"
        "     - The description must mention the SAME attribute value "
        "strings that appear in the MATCH clause. If the pattern filters "
        "genre:\"Sci-Fi\", the description must say \"Sci-Fi\" — not "
        "\"Action\" or \"Drama\".\n"
        "     - The description must state what is being tested (which "
        "attribute/target, against what threshold).\n"
        " 11. Comparison rules — what the engine supports:\n"
        "     - One-sample tests against mu or a numeric threshold: "
        "always OK.\n"
        "     - PAIRED comparisons inside ONE matched path: OK in two "
        "forms — |nA.attr - nB.attr| (absolute node difference, rhs can "
        "be mu or a number) and edgeA.attr - edgeB.attr (signed edge "
        "difference around a shared node, rhs must be a number, "
        "typically 0).\n"
        "     - NOT supported: comparing two aggregates on the rhs "
        "(e.g. TEST mean(n1.age) > mean(n3.age) — the rhs must be mu or "
        "a number, NEVER mean(...)); comparing two separate populations "
        "or two separate patterns (e.g. \"Do users rate Sci-Fi movies "
        "higher than Action movies?\" would need two patterns and is NOT "
        "supported).\n"
        "     If your question requires comparing two groups that "
        "cannot be expressed as a paired comparison inside one path "
        "pattern, REJECT it and pick a different question.\n"
        " 12. The system only filters by EXACT attribute values (e.g. "
        "age:1, genre:\"Sci-Fi\"). It does NOT support range filters "
        "(age>18, rating>=4). Therefore:\n"
        "     - DO NOT describe filters with range-implying adjectives: "
        "\"older users\", \"younger users\", \"high-rated movies\", "
        "\"low-rated movies\", \"popular papers\", \"long movies\", "
        "\"recent movies\", etc. These imply ranges the engine cannot "
        "express.\n"
        "     - INSTEAD, describe attribute filters by their literal "
        "value: \"users with age=1\", \"movies with genre=Sci-Fi\", "
        "\"papers with doc_type=Journal\".\n"
        "     - If you are tempted to write \"older/younger/higher/lower/"
        "popular/long/short/recent/old X\", REJECT that question and "
        "pick a different one — the engine cannot answer it.\n"
        " 13. AMBIGUOUS comparison phrasing: a sentence like \"Do students "
        "rate Sci-Fi movies higher than K-12 students?\" is grammatically "
        "ambiguous — \"than K-12 students\" could be read as the object "
        "being rated (\"rate X higher than [they rate] Y\") instead of the "
        "group doing the rating. When a description compares two groups/roles, "
        "disambiguate it explicitly:\n"
        "     - Add the auxiliary \"do\": \"Do students rate Sci-Fi movies "
        "higher than K-12 students DO?\"\n"
        "     - Or use versus phrasing: \"Students versus K-12 students: who "
        "rates Sci-Fi movies higher?\"\n"
        "     - When both sides are within ONE subject, keep the object "
        "explicit: \"students rate Sci-Fi movies higher than they rate "
        "Documentaries\".\n"
        "     - NEVER write \"<group A> rate <X> higher than <group B>\" "
        "without a clarifier — it invites the object-comparison reading.\n\n"
        + (schema_values + "\n\n" if schema_values else "")
        + "Output ONLY a JSON array, each element:\n"
        '  {"description": "<short human-readable question, max 12 words>", '
        '"ghtql": "<full GHT-QL string>"}\n'
        "Output the JSON array and nothing else."
    )

    # ── generate + reflect, up to 5 attempts ───────────────────────────────
    # Each attempt: LLM generates candidates → syntactic filter → reflection
    # (filter+rank). Reflection is INSIDE the loop so that if it filters
    # EVERYTHING out we retry generation, rather than returning an empty list.
    #   ok=True  ranked non-empty → success, break
    #   ok=True  ranked empty      → reflection filtered all → retry generation
    #   ok=False                   → technical failure → fall back to THIS
    #                               attempt's unreflected candidates and break
    #                               (don't waste another generation call on a
    #                               transient reflection error).
    # After the loop: if every attempt filtered all out, fall back to the last
    # non-empty pre-reflection list (unreflected) — guarantees ≥1 result
    # whenever generation ever produced a valid candidate.
    from hypolens.llm import LLMClient, system_msg, user_msg
    validator             = GHTQLValidator()
    seen_ghql             = set()
    current_norm          = " ".join(current_ghtql.split())
    final_recs: list      = []
    last_pre_reflection: list = []

    for _ in range(5):
        if final_recs:
            break
        try:
            client = LLMClient(model=model, api_key=api_key, base_url=base_url)
            raw    = client.chat([system_msg("You are a GHT-QL query generator."),
                                  user_msg(prompt)],
                                 # NOTE: edge_diff patterns are verbose (3 nodes,
                                 # two MATCH clauses); 600 tokens truncated the
                                 # JSON array mid-candidate and killed parsing.
                                 max_tokens=2000, temperature=0.7)
            client.close()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"LLM call failed: {exc}"}), 502

        # ── extract JSON array from LLM output ──────────────────────────────
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            continue
        try:
            candidates = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

        # ── validate + parse each candidate; keep only valid ones ───────────
        attempt_recs: list = []
        for cand in candidates:
            ghtql_src = (cand.get("ghtql") or "").strip()
            desc      = (cand.get("description") or "").strip()
            if not ghtql_src or ghtql_src in seen_ghql:
                continue
            # reject near-duplicates of the current pattern (normalised comparison)
            if " ".join(ghtql_src.split()) == current_norm:
                continue
            # reject range-implying descriptions (older users, high-rated movies…)
            if _looks_like_range_filter(desc):
                continue
            # syntax + schema validation
            v_result = validator.validate(ghtql_src)
            if not v_result["valid"]:
                continue
            # parse into canonical entry dict
            try:
                parsed = parse_string(ghtql_src)
            except GHTQLError:
                continue
            # take only single-pattern results
            if not parsed or len(parsed) != 1:
                continue
            pid      = next(iter(parsed))
            entry2   = parsed[pid]
            # reject non-path topologies — PHASE sampler only supports path-based patterns
            if entry2.get("type") not in ("node", "edge", "path"):
                continue
            # reject the (still unexecutable) aggregate-rhs two-sample form:
            # TEST mean(...) > mean(...). Paired comparisons must instead use
            # the edge_diff target (a->b.attr - c->d.attr), which IS supported.
            if isinstance(entry2.get("test", {}).get("rhs"), dict):
                continue
            seen_ghql.add(ghtql_src)
            attempt_recs.append({
                "description":   desc or ghtql_src[:60],
                "ghtql":         ghtql_src.strip(),
                "parsed_pattern": entry2,
            })
            if len(attempt_recs) >= 8:
                break

        if not attempt_recs:
            continue  # this attempt produced nothing valid → retry generation

        last_pre_reflection = attempt_recs  # remember for final fallback

        # reflection needs ≥2 candidates to be meaningful; with exactly 1 we
        # just keep it (no filter/rank) — better an unjudged question than none.
        if len(attempt_recs) == 1:
            final_recs = attempt_recs
            break

        ranked, ok = _reflect_recommendations(
            attempt_recs, current_ghtql, report_summary, model, api_key, base_url,
        )
        if ok and ranked:
            final_recs = ranked          # reflection succeeded with survivors
            break
        if ok and not ranked:
            continue                    # reflection filtered all → retry
        # ok=False → technical reflection failure: use unreflected, don't retry
        final_recs = attempt_recs
        break

    # ── guarantee ≥1 recommendation whenever generation ever produced one ──
    # Every attempt's reflection filtered all out → fall back to the most
    # recent pre-reflection list (unreflected, syntactically valid).
    if not final_recs and last_pre_reflection:
        final_recs = last_pre_reflection

    # truncate to top 3 (reflection-ranked, or pre-reflection order on fallback)
    if len(final_recs) > 3:
        final_recs = final_recs[:3]

    return jsonify({"recommendations": final_recs})


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

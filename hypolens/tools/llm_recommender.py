"""LLM-based follow-up generation, validation, reflection, and ranking.

This module accepts pattern/report dictionaries and explicit model settings.
It has no dependency on Flask, HTTP request objects, or the Web application.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ghtql.parser import GHTQLError, parse_string
from ghtql.unparser import unparse_entry
from hypolens.llm import LLMClient, system_msg, user_msg
from hypolens.tools.report_summary import build_report_summary
from hypolens.tools.validator import GHTQLValidator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMRecommendationError(RuntimeError):
    """The LLM could not generate candidates; callers may use another provider."""


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


class LLMRecommender:
    """Generate up to three validated follow-up hypotheses.

    ``options["report"]`` supplies the current statistical result. Empty output
    means generation produced no usable hypotheses. Service failures raise
    ``LLMRecommendationError``; choosing a fallback belongs to the caller.
    """

    name = "llm"

    def __init__(self, *, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def recommend(self, entry: dict, options: dict[str, Any] | None = None) -> list[dict]:
        options = options or {}
        pattern_id = entry.get("id") or "current"
        try:
            current_ghtql = unparse_entry(pattern_id, entry)
        except Exception:  # noqa: BLE001
            current_ghtql = "(unparse failed)"
        report = options.get("report") or {}
        report_summary = build_report_summary(report) if report else ""
        schema_values = _schema_value_summary(entry.get("dataset") or "")
        model, api_key, base_url = self.model, self.api_key, self.base_url

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
        validator             = GHTQLValidator()
        seen_ghql             = set()
        current_norm          = " ".join(current_ghtql.split())
        final_recs: list      = []
        last_pre_reflection: list = []

        for _ in range(5):
            if final_recs:
                break
            try:
                with LLMClient(model=model, api_key=api_key, base_url=base_url) as client:
                    raw = client.chat(
                        [system_msg("You are a GHT-QL query generator."), user_msg(prompt)],
                        # Paired edge-difference candidates need room for full patterns.
                        max_tokens=2000, temperature=0.7,
                    )
            except Exception as exc:  # noqa: BLE001
                raise LLMRecommendationError("LLM generation failed") from exc

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

        return [{k: v for k, v in rec.items() if k != "ghtql"}
                for rec in final_recs[:3]]

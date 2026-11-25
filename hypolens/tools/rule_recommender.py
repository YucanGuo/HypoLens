"""
Rule-based follow-up hypothesis recommender.

Generates follow-up hypotheses by applying predefined, explainable edits to
the current hypothesis — no LLM call and no GHT execution.  Every candidate is
re-validated and re-parsed, so it can be loaded onto the Visual Builder and
executed directly.  Strategies are registered by name so the web layer can
select one (``get_recommender("rules")``).

Rules (each independently switchable):

  * attribute_substitution        n1.Sci-Fi=1        -> n1.Comedy=1
  * attribute_value_substitution  n0.occupation=a    -> n0.occupation=b
  * constraint_addition           (none)             -> n1.year="1990's"
  * constraint_relaxation         drop an existing constraint (more general
                                  hypothesis — no value substitution)
  * test_target_modification      change the test threshold, or replace the
                                  tested attribute with another numeric one

Refinement modes:

  * single: each candidate applies exactly ONE rule. One candidate per
    applicable rule is picked first; if that yields fewer than three (few rules
    selected, or some rules have no applicable candidate) extra candidates are
    drawn from rules that have more, so the result has at least three whenever
    enough candidates exist (max 5).
  * mixed:  each candidate applies TWO different rules whose target slots do
    not overlap. One candidate per rule pair is picked first, then more
    combinations from the same pair(s) when the minimum is not reached.
"""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol

from ghtql.parser import GHTQLError, parse_string
from ghtql.unparser import unparse_entry
from hypolens.tools.validator import GHTQLValidator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── recommender registry ─────────────────────────────────────────────────────

class Recommender(Protocol):
    """A strategy that proposes follow-up hypotheses for a pattern entry."""

    name: str

    def recommend(self, entry: dict, options: dict[str, Any]) -> list[dict]: ...


_REGISTRY: dict[str, Recommender] = {}


def register_recommender(name: str, recommender: Recommender) -> None:
    _REGISTRY[name] = recommender


def get_recommender(name: str) -> Recommender | None:
    return _REGISTRY.get(name)


def available_recommenders() -> list[str]:
    return sorted(_REGISTRY)

# ── rule identifiers ─────────────────────────────────────────────────────────
ATTR_SUB  = "attribute_substitution"
VALUE_SUB = "attribute_value_substitution"
ADD       = "constraint_addition"
RELAX     = "constraint_relaxation"
TEST_MOD  = "test_target_modification"

RULE_ORDER = [ATTR_SUB, VALUE_SUB, ADD, RELAX, TEST_MOD]

RULE_TITLES = {
    ATTR_SUB:  "Attribute Substitution",
    VALUE_SUB: "Attribute Value Substitution",
    ADD:       "Constraint Addition",
    RELAX:     "Constraint Relaxation",
    TEST_MOD:  "Test Target Modification",
}

_NUMERIC_DTYPES = {"int", "float", "double", "long", "number"}
_CATEGORICAL_INT_ATTRS = {("movie", "genre")}
_SKIP_ATTRS = {"id", "title"}
_MIN_COVERAGE = 0.05
MIN_RECOMMENDATIONS = 3          # return at least this many when possible
_MAX_VALUE_SUBS_PER_CONSTRAINT = 4
_MAX_ADD_VALUES = 2
_MAX_TEST_ATTR_OPTIONS = 2


# ── schema / value metadata (cached — never mutated by callers) ──────────────

@lru_cache(maxsize=8)
def _load_schema_values(dataset: str) -> tuple[dict | None, dict]:
    schema_dir = _PROJECT_ROOT / "ghtql" / "schema"
    base = schema_dir / f"{dataset.lower()}"
    sp = base.with_suffix(".json")
    vp = base.with_name(base.name + "_values.json")
    if not sp.exists():
        return None, {}
    schema = json.loads(sp.read_text(encoding="utf-8"))
    values = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    return schema, values


def _norm_value(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _same(a, b) -> bool:
    a, b = _norm_value(a), _norm_value(b)
    return type(a) is type(b) and a == b


def _fmt_value(v) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, str) else str(v)


def _node_attrs(schema: dict, label: str) -> dict:
    return ((schema.get("node_types") or {}).get(label, {}) or {}).get("attributes") or {}


def _edge_attrs(schema: dict, edge_type: str | None) -> dict:
    if not edge_type:
        return {}
    return ((schema.get("edge_types") or {}).get(edge_type, {}) or {}).get("attributes") or {}


def _edge_type_for(schema: dict, label_a: str, label_b: str) -> str | None:
    for conn in schema.get("connectivity") or []:
        if {conn.get("src"), conn.get("dst")} == {label_a, label_b}:
            return conn.get("edge")
    for et in (schema.get("edge_types") or {}):
        if set(et.split("-")) == {label_a, label_b}:
            return et
    return None


def _is_numeric_dtype(dtype) -> bool:
    return str(dtype or "").lower() in _NUMERIC_DTYPES


def _numeric_node_attrs(schema: dict, label: str) -> list[str]:
    return [
        a for a, info in _node_attrs(schema, label).items()
        if _is_numeric_dtype((info or {}).get("dtype"))
        and (label, a) not in _CATEGORICAL_INT_ATTRS
    ]


def _numeric_edge_attrs(schema: dict, edge_type: str | None) -> list[str]:
    return [
        a for a, info in _edge_attrs(schema, edge_type).items()
        if _is_numeric_dtype((info or {}).get("dtype"))
    ]


def _pool(schema: dict, values: dict, label: str, attr: str,
          edge_type: str | None = None) -> list:
    """Frequency-ordered candidate values for an attribute (top-N, then samples)."""
    raw: list = []
    if edge_type is not None:
        src = ((values.get("edge_types") or {}).get(edge_type, {}) or {}).get(attr, [])
    else:
        src = ((values.get("node_types") or {}).get(label, {}) or {}).get(attr, [])
    for rec in src or []:
        raw.append(_norm_value(rec.get("value") if isinstance(rec, dict) else rec))
    if not raw:
        info = (_edge_attrs(schema, edge_type) if edge_type is not None
                else _node_attrs(schema, label)).get(attr, {}) or {}
        raw = [_norm_value(x) for x in (info.get("samples") or [])]
    out, seen = [], set()
    for v in raw:
        if v is None or v == "":
            continue
        if isinstance(v, float) and not float(v).is_integer():
            continue
        key = (type(v).__name__, v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# ── context ──────────────────────────────────────────────────────────────────

def _build_context(entry: dict, schema: dict, values: dict) -> dict:
    nodes = (entry.get("pattern") or {}).get("nodes") or []
    by_id = {n.get("id"): n for n in nodes}

    constraints = []  # (nid, label, attr, value)
    for n in nodes:
        for a, v in (n.get("attribute") or {}).items():
            constraints.append((n.get("id"), n.get("label"), a, _norm_value(v)))

    # Attributes that are themselves tested must not be turned into/removed as
    # constraints (that would change what the mean is taken over).
    protected: set[tuple] = set()
    tgt = entry.get("target") or {}
    if "node" in tgt:
        n = tgt.get("node") or {}
        protected.add((n.get("id"), n.get("attribute")))
    elif "node_diff" in tgt:
        nd = tgt.get("node_diff") or {}
        for nid in nd.get("nodes") or []:
            protected.add((nid, nd.get("attribute")))

    return {
        "entry": entry, "schema": schema, "values": values,
        "nodes": nodes, "by_id": by_id,
        "constraints": constraints, "protected": protected,
    }


# ── mutations ────────────────────────────────────────────────────────────────
# A mutation is {"rule", "desc", "keys", "ops"} where keys are the target slots
# it touches (for mixed-mode conflict detection) and ops are applied to a copy
# of the entry:  ("set", nid, attr, val) | ("del", nid, attr)
#                ("rhs", value) | ("target_attr", attr)

def _gen_attribute_substitution(ctx: dict) -> list[dict]:
    out = []
    schema, values = ctx["schema"], ctx["values"]
    by_id = ctx["by_id"]
    for nid, label, attr, val in ctx["constraints"]:
        if (nid, attr) in ctx["protected"]:
            continue
        existing = (by_id.get(nid) or {}).get("attribute") or {}
        for other, info in _node_attrs(schema, label).items():
            if other == attr or other in existing:
                continue
            if other in _SKIP_ATTRS or (label, other) in _CATEGORICAL_INT_ATTRS:
                continue
            if (nid, other) in ctx["protected"]:
                continue
            cov = (info or {}).get("coverage")
            if cov is not None and cov < _MIN_COVERAGE:
                continue
            if not any(_same(val, v) for v in _pool(schema, values, label, other)):
                continue
            out.append({
                "rule": ATTR_SUB,
                "desc": f"Replace {nid}.{attr} with {nid}.{other}.",
                "keys": {("node", nid, attr), ("node", nid, other)},
                "ops": [("del", nid, attr), ("set", nid, other, val)],
            })
    return out


def _gen_value_substitution(ctx: dict) -> list[dict]:
    out = []
    schema, values = ctx["schema"], ctx["values"]
    for nid, label, attr, old in ctx["constraints"]:
        if (nid, attr) in ctx["protected"]:
            continue
        pool = _pool(schema, values, label, attr)
        # All alternative values are allowed — including 1 -> 0 on boolean
        # flags ("movies that are NOT Sci-Fi"), which is a legitimate variant.
        others = [v for v in pool if not _same(v, old)][:_MAX_VALUE_SUBS_PER_CONSTRAINT]
        for new in others:
            out.append({
                "rule": VALUE_SUB,
                "desc": (f"Change the value of {nid}.{attr} from "
                         f"{_fmt_value(old)} to {_fmt_value(new)}."),
                "keys": {("node", nid, attr)},
                "ops": [("set", nid, attr, new)],
            })
    return out


def _attr_priority(label: str, attr: str, info: dict) -> tuple:
    """Categorical attributes make more meaningful filters than IDs/numbers."""
    dt = str((info or {}).get("dtype") or "").lower()
    pri = 0 if dt in ("bool", "str", "mixed") else 1
    cov = (info or {}).get("coverage")
    return (pri, -(cov if cov is not None else 0.0), attr)


def _gen_constraint_addition(ctx: dict) -> list[dict]:
    out = []
    schema, values = ctx["schema"], ctx["values"]
    for node in ctx["nodes"]:
        nid, label = node.get("id"), node.get("label")
        existing = node.get("attribute") or {}
        attrs = _node_attrs(schema, label)
        has_bool = any((attrs.get(a) or {}).get("dtype") == "bool" for a in existing)
        for attr, info in sorted(attrs.items(), key=lambda kv: _attr_priority(label, kv[0], kv[1])):
            if attr in existing or attr in _SKIP_ATTRS:
                continue
            if (label, attr) in _CATEGORICAL_INT_ATTRS:
                continue
            if (nid, attr) in ctx["protected"]:
                continue
            cov = (info or {}).get("coverage")
            if cov is not None and cov < _MIN_COVERAGE:
                continue
            dtype = (info or {}).get("dtype")
            if dtype == "bool" and has_bool:
                continue
            pool = _pool(schema, values, label, attr)
            if not pool:
                continue
            vals = [1] if dtype == "bool" and 1 in pool else pool[:_MAX_ADD_VALUES]
            for v in vals:
                out.append({
                    "rule": ADD,
                    "desc": f"Add the constraint {nid}.{attr} = {_fmt_value(v)}.",
                    "keys": {("node", nid, attr)},
                    "ops": [("set", nid, attr, v)],
                })
    return out


def _gen_constraint_relaxation(ctx: dict) -> list[dict]:
    """Relaxation = DROP an existing constraint (leaving a more general
    hypothesis).  This rule intentionally does NOT substitute values — value
    changes belong to attribute_value_substitution."""
    out = []
    for nid, _label, attr, old in ctx["constraints"]:
        if (nid, attr) in ctx["protected"]:
            continue
        out.append({
            "rule": RELAX,
            "desc": f"Remove the constraint {nid}.{attr} = {_fmt_value(old)}.",
            "keys": {("node", nid, attr)},
            "ops": [("del", nid, attr)],
        })
    return out


def _test_attr_options(ctx: dict) -> Iterable[tuple]:
    """Yield (old_attr, new_attr, old_element_str, new_element_str)."""
    entry, schema, by_id = ctx["entry"], ctx["schema"], ctx["by_id"]
    tgt = entry.get("target") or {}

    if "node" in tgt:
        nid = tgt["node"].get("id")
        cur = tgt["node"].get("attribute")
        label = (by_id.get(nid) or {}).get("label")
        constrained = set(((by_id.get(nid) or {}).get("attribute") or {}).keys())
        for a in _numeric_node_attrs(schema, label):
            if a == cur or a in constrained:
                continue
            yield cur, a, f"{nid}.{cur}", f"{nid}.{a}"

    elif "node_diff" in tgt:
        ids = tgt["node_diff"].get("nodes") or []
        cur = tgt["node_diff"].get("attribute")
        if len(ids) != 2:
            return
        labels = [(by_id.get(i) or {}).get("label") for i in ids]
        common = set(_numeric_node_attrs(schema, labels[0])) & set(_numeric_node_attrs(schema, labels[1]))
        constrained = set()
        for i in ids:
            constrained |= set(((by_id.get(i) or {}).get("attribute") or {}).keys())
        for a in (x for x in _numeric_node_attrs(schema, labels[0]) if x in common):
            if a == cur or a in constrained:
                continue
            yield cur, a, f"{ids[0]}.{cur}", f"{ids[0]}.{a}"

    elif "edge" in tgt:
        e = tgt["edge"]
        cur = e.get("attribute")
        et = _edge_type_for(
            schema,
            (by_id.get(e.get("from")) or {}).get("label"),
            (by_id.get(e.get("to")) or {}).get("label"),
        )
        el = f'{e.get("from")}->{e.get("to")}'
        for a in _numeric_edge_attrs(schema, et):
            if a == cur:
                continue
            yield cur, a, f"{el}.{cur}", f"{el}.{a}"

    elif "edge_diff" in tgt:
        ed = tgt["edge_diff"]
        cur = ed.get("attribute")
        edges = ed.get("edges") or []
        if not edges:
            return
        attr_sets = []
        for f, t in edges:
            et = _edge_type_for(
                schema,
                (by_id.get(f) or {}).get("label"),
                (by_id.get(t) or {}).get("label"),
            )
            attr_sets.append(set(_numeric_edge_attrs(schema, et)))
        common = set.intersection(*attr_sets) if attr_sets else set()
        f0, t0 = edges[0]
        el = f"{f0}->{t0}"
        ordered = [a for a in _numeric_edge_attrs(
            schema, _edge_type_for(schema,
                                   (by_id.get(f0) or {}).get("label"),
                                   (by_id.get(t0) or {}).get("label"))
        ) if a in common]
        for a in ordered:
            if a == cur:
                continue
            yield cur, a, f"{el}.{cur}", f"{el}.{a}"


def _gen_test_modification(ctx: dict) -> list[dict]:
    out = []
    entry = ctx["entry"]
    test = entry.get("test") or {}
    rhs = test.get("rhs")

    # (a) threshold change — only when rhs is a concrete number
    if isinstance(rhs, (int, float)) and not isinstance(rhs, bool):
        base = float(rhs)
        if base == 0:
            news = [0.5, -0.5]
        elif base.is_integer():
            step = max(1, round(abs(base) * 0.2))   # 25 -> 30 / 20
            news = [base + step, base - step]
        else:
            news = [base + 0.5, base - 0.5]
        for nv in news:
            num = int(nv) if float(nv).is_integer() else round(nv, 4)
            out.append({
                "rule": TEST_MOD,
                "desc": f"Change the test threshold from {_fmt_value(rhs)} to {_fmt_value(num)}.",
                "keys": {("test",)},
                "ops": [("rhs", num)],
            })

    # (b) replace the tested attribute with another numeric attribute
    count = 0
    for old_attr, new_attr, old_el, new_el in _test_attr_options(ctx):
        if count >= _MAX_TEST_ATTR_OPTIONS:
            break
        count += 1
        out.append({
            "rule": TEST_MOD,
            "desc": f"Replace the test attribute {old_el} with {new_el}.",
            "keys": {("test",)},
            "ops": [("target_attr", new_attr)],
        })
    return out


_GENERATORS = {
    ATTR_SUB:  _gen_attribute_substitution,
    VALUE_SUB: _gen_value_substitution,
    ADD:       _gen_constraint_addition,
    RELAX:     _gen_constraint_relaxation,
    TEST_MOD:  _gen_test_modification,
}


# ── apply + validate ─────────────────────────────────────────────────────────

def _apply_ops(entry: dict, ops: list[tuple]) -> None:
    nodes = {n.get("id"): n for n in (entry.get("pattern") or {}).get("nodes", [])}
    for op in ops:
        kind = op[0]
        if kind == "set":
            _, nid, attr, val = op
            nodes[nid].setdefault("attribute", {})[attr] = val
        elif kind == "del":
            _, nid, attr = op
            (nodes[nid].get("attribute") or {}).pop(attr, None)
        elif kind == "rhs":
            entry.setdefault("test", {})["rhs"] = op[1]
        elif kind == "target_attr":
            tgt = entry.get("target") or {}
            for key in ("node", "edge", "node_diff", "edge_diff"):
                if key in tgt:
                    tgt[key]["attribute"] = op[1]
                    break


def _finalize(entry: dict, mutations: list[dict],
              validator: GHTQLValidator, current_norm: str) -> dict | None:
    cand = copy.deepcopy(entry)
    ops: list[tuple] = []
    for m in mutations:
        ops.extend(m["ops"])
    try:
        _apply_ops(cand, ops)
    except (KeyError, TypeError):
        return None
    pattern_id = entry.get("id") or "current"
    try:
        ghtql = unparse_entry(pattern_id, cand)
    except Exception:  # noqa: BLE001
        return None
    norm = " ".join(ghtql.split())
    if not norm or norm == current_norm:
        return None
    if not validator.validate(ghtql).get("valid"):
        return None
    try:
        parsed = parse_string(ghtql)
    except GHTQLError:
        return None
    if not parsed or len(parsed) != 1:
        return None
    pentry = parsed[next(iter(parsed))]
    if pentry.get("type") not in ("node", "edge", "path"):
        return None
    if isinstance((pentry.get("test") or {}).get("rhs"), dict):
        return None
    return {
        # Opaque id only — the GHT-QL source is never sent to the browser.
        "id": hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16],
        "title": " + ".join(RULE_TITLES[m["rule"]] for m in mutations),
        "description": " ".join(m["desc"] for m in mutations),
        "rules": [m["rule"] for m in mutations],
        "parsed_pattern": pentry,
        "_norm": norm,
    }


# ── selection helpers ────────────────────────────────────────────────────────

def _collect(rec_iter, exclude: set[str], cap: int) -> dict:
    """
    Pull valid candidates from ``rec_iter`` and split them into not-yet-shown
    (``unseen``) and already-shown (``seen``) lists, each capped at ``cap``.
    Iteration stops early once both lists are full.
    """
    unseen, seen = [], []
    for rec in rec_iter:
        bucket = seen if rec["id"] in exclude else unseen
        if len(bucket) >= cap:
            continue
        bucket.append(rec)
        if len(unseen) >= cap and len(seen) >= cap:
            break
    return {"unseen": unseen, "seen": seen}


def _assemble(groups: list[dict], limit: int, fill_target: int) -> list[dict]:
    """
    Select candidates from several groups (rules or rule pairs):

      1. one candidate per group (unseen preferred, shown-as-fallback) — every
         group that has any candidate is represented before any group repeats;
      2. extra unseen candidates round-robin until the target;
      3. fall back to already-shown candidates if still below the target.

    The target is ``fill_target`` (the minimum) raised to one per group when
    there are more groups than that, so every selected rule/pair still shows
    a suggestion — including on Refresh, when all candidates are already seen.
    The RESULT is then grouped by rule (group order) so suggestions from the
    same rule sit together — not interleaved round-robin.  Never exceeds
    ``limit`` overall.
    """
    target = min(limit, max(fill_target, len(groups)))
    selected: list[tuple[int, dict]] = []   # (group index, candidate)
    used: set[str] = set()
    ptr = [{"u": 0, "s": 0} for _ in groups]

    def take(gi: int, seen: bool):
        lst = groups[gi]["seen" if seen else "unseen"]
        key = "s" if seen else "u"
        i = ptr[gi][key]
        while i < len(lst):
            rec = lst[i]
            i += 1
            if rec["id"] in used:
                continue
            ptr[gi][key] = i
            used.add(rec["id"])
            return rec
        ptr[gi][key] = i
        return None

    # 1) one per group — unseen first, but never starve a group on Refresh
    for gi in range(len(groups)):
        if len(selected) >= limit:
            break
        rec = take(gi, False) or take(gi, True)
        if rec:
            selected.append((gi, rec))

    # 2) extra unseen, round-robin, up to the target
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for gi in range(len(groups)):
            if len(selected) >= target:
                break
            rec = take(gi, False)
            if rec:
                selected.append((gi, rec))
                progressed = True

    # 3) already-shown fallback (Refresh has shown most of the pool)
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for gi in range(len(groups)):
            if len(selected) >= target:
                break
            rec = take(gi, True)
            if rec:
                selected.append((gi, rec))
                progressed = True

    # Group by rule/pair (stable: preserves discovery order inside a group).
    return [rec for _, rec in sorted(selected, key=lambda item: item[0])]


# ── recommender ──────────────────────────────────────────────────────────────

class RuleRecommender:
    name = "rules"

    def recommend(self, entry: dict, options: dict[str, Any] | None = None) -> list[dict]:
        options = options or {}
        limit = int(options.get("limit") or 5)
        mode = options.get("mode") or "single"
        rules_opt = options.get("rules")
        enabled = [r for r in (RULE_ORDER if rules_opt is None else rules_opt)
                   if r in RULE_ORDER]
        if not enabled:
            return []
        if mode == "mixed" and len(enabled) < 2:
            return []

        # exclude is a list of opaque recommendation ids (not GHT-QL strings)
        exclude = {x for x in (options.get("exclude") or []) if isinstance(x, str)}

        schema, values = _load_schema_values(entry.get("dataset") or "")
        if not schema or not (entry.get("pattern") or {}).get("nodes"):
            return []

        validator = GHTQLValidator()
        try:
            current_norm = " ".join(unparse_entry(entry.get("id") or "current", entry).split())
        except Exception:  # noqa: BLE001
            current_norm = ""

        ctx = _build_context(entry, schema, values)
        per_rule = {r: _GENERATORS[r](ctx) for r in enabled}

        if mode == "mixed":
            results = self._select_mixed(entry, per_rule, enabled, validator,
                                         current_norm, exclude, limit)
        else:
            results = self._select_single(entry, per_rule, enabled, validator,
                                          current_norm, exclude, limit)

        for r in results:
            r.pop("_norm", None)
        return results

    # -- single-rule: one candidate per rule first, topped up to three ───────
    @staticmethod
    def _select_single(entry, per_rule, enabled, validator,
                       current_norm, exclude, limit) -> list[dict]:
        def _rule_iter(rule):
            ids: set[str] = set()
            for mut in per_rule[rule]:
                rec = _finalize(entry, [mut], validator, current_norm)
                if not rec or rec["id"] in ids:
                    continue
                ids.add(rec["id"])
                yield rec

        groups = [_collect(_rule_iter(rule), exclude, limit) for rule in enabled]
        return _assemble(groups, limit, MIN_RECOMMENDATIONS)

    # -- mixed-rule: two non-overlapping rules per candidate, topped up to three
    @staticmethod
    def _select_mixed(entry, per_rule, enabled, validator,
                      current_norm, exclude, limit) -> list[dict]:
        n = len(enabled)
        # Order pairs by rule distance so the first few cover different rules
        # (AB, BC, CD, … before AC, BD, …), keeping the selection diverse.
        pairs = [(enabled[i], enabled[i + d])
                 for d in range(1, n) for i in range(n - d)][:limit]

        def _pair_iter(ra, rb):
            A, B = per_rule[ra], per_rule[rb]
            if not A or not B:
                return
            ids: set[str] = set()
            tried: set[tuple[int, int]] = set()
            state = {"attempts": 0}

            def attempt(i, j):
                if (i, j) in tried or state["attempts"] > 60:
                    return None
                tried.add((i, j))
                state["attempts"] += 1
                ma, mb = A[i], B[j]
                if ma["keys"] & mb["keys"]:
                    return None          # conflicting/overlapping edits
                rec = _finalize(entry, [ma, mb], validator, current_norm)
                if not rec or rec["id"] in ids:
                    return None
                ids.add(rec["id"])
                return rec

            # Diagonal pass first: A[0]+B[0], A[1]+B[1], … so successive
            # suggestions vary BOTH sides instead of reusing the first edit.
            for k in range(max(len(A), len(B))):
                rec = attempt(k % len(A), k % len(B))
                if rec:
                    yield rec
            # Then the remaining grid.
            for i in range(len(A)):
                for j in range(len(B)):
                    rec = attempt(i, j)
                    if rec:
                        yield rec
                    if state["attempts"] > 60:
                        return

        groups = [_collect(_pair_iter(ra, rb), exclude, limit) for ra, rb in pairs]
        return _assemble(groups, limit, MIN_RECOMMENDATIONS)


# Register the built-in strategy on import.
register_recommender("rules", RuleRecommender())

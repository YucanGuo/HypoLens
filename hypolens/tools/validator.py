"""
GHTQLValidator
==============
A higher-level validation tool that not only checks GHT-QL syntax but also
performs schema-level semantic checks:

  1. Syntax validation  — via the GHT-QL parser
  2. Label check        — every node label must exist in the dataset schema
  3. Attribute check    — every property key referenced in the pattern must be
                          a known attribute (or binary flag) for that node type
  4. TEST attribute     — the attribute referenced in TEST must be numeric
                          (dtype in {int, float}) for t-test validity

These checks provide precise error messages for patterns built in the Web
interface and for LLM-generated follow-up recommendations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ghtql.parser import parse_string, GHTQLError

SCHEMA_DIR = _PROJECT_ROOT / "ghtql" / "schema"

# Dataset name → schema file stem
_DATASET_MAP: dict[str, str] = {
    "Citation":  "citation",
    "MovieLens": "movielens",
    "Yelp":      "yelp",
}

_NUMERIC_DTYPES = {"int", "float"}


# ── schema cache ───────────────────────────────────────────────────────────────

_schema_cache: dict[str, dict] = {}

def _load_schema(dataset: str) -> dict | None:
    stem = _DATASET_MAP.get(dataset)
    if stem is None:
        return None
    if stem not in _schema_cache:
        path = SCHEMA_DIR / f"{stem}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            _schema_cache[stem] = json.load(f)
    return _schema_cache[stem]


def _all_attrs(node_info: dict) -> dict[str, str]:
    """
    Return {attr_name: dtype} for all known attributes + binary flags
    of a given node type info dict.
    """
    result: dict[str, str] = {}
    for attr, meta in node_info.get("attributes", {}).items():
        result[attr] = meta.get("dtype", "?")
    for flag in node_info.get("binary_flags", {}).keys():
        result[flag] = "bool"
    return result


# ── validator ──────────────────────────────────────────────────────────────────

class GHTQLValidator:
    """
    Validates GHT-QL source strings at both syntax and schema levels.

    Usage
    -----
    validator = GHTQLValidator()
    result = validator.validate(src)
    # result["valid"] is True/False
    # result["errors"] lists actionable error messages
    """

    def validate(self, src: str) -> dict[str, Any]:
        """
        Validate *src* and return a structured report.

        Returns
        -------
        {
          "valid":      bool,
          "errors":     list[str],   # non-empty when valid=False
          "warnings":   list[str],   # non-fatal issues
          "n_patterns": int,
        }
        """
        errors:   list[str] = []
        warnings: list[str] = []

        # ── step 1: syntax ─────────────────────────────────────────────────────
        try:
            patterns = parse_string(src)
        except GHTQLError as exc:
            return {
                "valid":      False,
                "errors":     [f"Syntax error: {exc}"],
                "warnings":   [],
                "n_patterns": 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "valid":      False,
                "errors":     [f"Unexpected parse error: {exc}"],
                "warnings":   [],
                "n_patterns": 0,
            }

        # ── step 2: schema-level checks ────────────────────────────────────────
        for pid, entry in patterns.items():
            dataset = entry.get("dataset", "")
            schema  = _load_schema(dataset)

            if schema is None:
                warnings.append(
                    f"[{pid}] Schema for dataset '{dataset}' not found; "
                    f"skipping semantic checks."
                )
                continue

            node_types_schema = schema.get("node_types", {})
            pat_nodes = entry["pattern"]["nodes"]
            pat_edges = entry["pattern"].get("edges", [])
            target    = entry["target"]
            test      = entry["test"]

            # Build a map: variable id → label
            var_to_label: dict[str, str] = {n["id"]: n["label"] for n in pat_nodes}

            for node in pat_nodes:
                label = node["label"]
                props = node.get("attribute", {})

                # ── label check ────────────────────────────────────────────────
                if label not in node_types_schema:
                    errors.append(
                        f"[{pid}] Unknown node label '{label}'. "
                        f"Available labels: {sorted(node_types_schema.keys())}"
                    )
                    continue   # skip attribute checks for unknown labels

                known_attrs = _all_attrs(node_types_schema[label])

                # ── attribute check ────────────────────────────────────────────
                for prop_key in props:
                    if prop_key not in known_attrs:
                        errors.append(
                            f"[{pid}] Node '{node['id']}:{label}' has unknown "
                            f"property '{prop_key}'. "
                            f"Known attributes: {sorted(known_attrs.keys())}"
                        )

            # ── edge existence check ───────────────────────────────────────────
            # The underlying graph is undirected (nx.Graph), so an edge in the
            # pattern is valid iff some schema connectivity entry connects the
            # two endpoint labels in *either* direction.
            conn_pairs: set[tuple[str, str]] = set()
            for c in schema.get("connectivity", []):
                s, d = c.get("src"), c.get("dst")
                if s and d:
                    conn_pairs.add((s, d))
                    conn_pairs.add((d, s))
            for e in pat_edges:
                frm, to = e.get("from"), e.get("to")
                if not frm or not to:
                    errors.append(f"[{pid}] Edge is missing 'from' or 'to'.")
                    continue
                lbl_a = var_to_label.get(frm)
                lbl_b = var_to_label.get(to)
                if not lbl_a or not lbl_b:
                    continue  # already reported via node label check
                if (lbl_a, lbl_b) not in conn_pairs:
                    errors.append(
                        f"[{pid}] No edge type connects {lbl_a} ↔ {lbl_b} "
                        f"in the schema (edge {frm}—{to})."
                    )

            # ── TEST attribute check ───────────────────────────────────────────
            # Map (label_a, label_b) → edge type name (either direction).
            edge_type_by_pair: dict[tuple[str, str], str] = {}
            edge_types_schema = schema.get("edge_types", {})
            for c in schema.get("connectivity", []):
                s, d, ename = c.get("src"), c.get("dst"), c.get("edge")
                if s and d and ename:
                    edge_type_by_pair[(s, d)] = ename
                    edge_type_by_pair[(d, s)] = ename

            for target_dict in _collect_targets(target, test["rhs"]):
                if "node" in target_dict:
                    ref_var  = target_dict["node"]["id"]
                    ref_attr = target_dict["node"]["attribute"]
                elif "node_diff" in target_dict:
                    ref_var  = target_dict["node_diff"]["nodes"][0]
                    ref_attr = target_dict["node_diff"]["attribute"]
                elif "edge_diff" in target_dict:
                    # validate the attribute on both referenced edges
                    for frm, to in target_dict["edge_diff"]["edges"]:
                        lbl_a = var_to_label.get(frm)
                        lbl_b = var_to_label.get(to)
                        ename = edge_type_by_pair.get((lbl_a, lbl_b)) \
                            if lbl_a and lbl_b else None
                        if not ename or ename not in edge_types_schema:
                            continue   # connectivity already reported above
                        ea = edge_types_schema[ename].get("attributes", {})
                        if target_dict["edge_diff"]["attribute"] not in ea:
                            errors.append(
                                f"[{pid}] TEST references unknown attribute "
                                f"'{target_dict['edge_diff']['attribute']}' on "
                                f"edge {frm}->{to} ({ename})."
                            )
                        elif ea[target_dict["edge_diff"]["attribute"]]\
                                .get("dtype") not in _NUMERIC_DTYPES:
                            warnings.append(
                                f"[{pid}] TEST attribute "
                                f"'{target_dict['edge_diff']['attribute']}' on "
                                f"edge {frm}->{to} ({ename}) is non-numeric; "
                                f"t-tests expect a numeric attribute."
                            )
                    continue
                else:
                    continue

                label = var_to_label.get(ref_var)
                if label and label in node_types_schema:
                    known_attrs = _all_attrs(node_types_schema[label])
                    if ref_attr not in known_attrs:
                        errors.append(
                            f"[{pid}] TEST references unknown attribute "
                            f"'{ref_attr}' on node '{ref_var}:{label}'."
                        )
                    else:
                        dtype = known_attrs[ref_attr]
                        if dtype not in _NUMERIC_DTYPES:
                            warnings.append(
                                f"[{pid}] TEST attribute '{ref_attr}' on "
                                f"'{ref_var}:{label}' has dtype='{dtype}'; "
                                f"t-tests expect a numeric attribute."
                            )

        valid = len(errors) == 0
        return {
            "valid":      valid,
            "errors":     errors,
            "warnings":   warnings,
            "n_patterns": len(patterns),
        }


# ── helper ─────────────────────────────────────────────────────────────────────

def _collect_targets(lhs_target: dict, rhs: Any) -> list[dict]:
    """Collect all target dicts that need attribute validation."""
    targets = [lhs_target]
    if isinstance(rhs, dict):   # two-sample: rhs is also a target dict
        targets.append(rhs)
    return targets

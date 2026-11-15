"""
GHT-QL Unparser  —  v1.0
Graph Hypothesis Testing Query Language

Converts a patterns dict (or JSON file) back into .ghtql source code.

CLI usage:
    python unparser.py patterns.json              # print result to stdout
    python unparser.py patterns.json -o out.ghtql # write result to .ghtql file

Python API:
    from ghtql.unparser import unparse, unparse_file
    src = unparse(patterns_dict)          # dict → str
    unparse_file("patterns.json", "out.ghtql")
"""

from __future__ import annotations

import json
import argparse
from typing import Any


# ---------------------------------------------------------------------------
# Core unparse logic
# ---------------------------------------------------------------------------

def _fmt_value(v: Any) -> str:
    """Format a property value as a GHT-QL literal."""
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)   # int / float → string as-is


def _fmt_prop_key(k: str) -> str:
    """Quote a property key when it is not a valid bare identifier."""
    # A bare identifier starts with a letter/underscore and contains only
    # letters, digits, underscores, or hyphens-between-alphanumerics. Anything
    # else (e.g. "Fast Food", "3D Printing", "A & B") must be quoted.
    import re
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*", k):
        return k
    return f'"{k}"'


def _fmt_props(attribute: dict) -> str:
    """Format a node attribute dict as {key:value, ...}. Returns '' for empty dicts."""
    if not attribute:
        return ""
    pairs = ", ".join(f"{_fmt_prop_key(k)}:{_fmt_value(v)}" for k, v in attribute.items())
    return f" {{{pairs}}}"


def _fmt_node(node: dict) -> str:
    """Format a node entry as (var:label {props})."""
    return f"({node['id']}:{node['label']}{_fmt_props(node['attribute'])})"


def _fmt_node_ref(var: str) -> str:
    """Format a back-reference to an already-declared node as (var)."""
    return f"({var})"


def _fmt_edge_decoration(edge: dict) -> str:
    """Format an edge decoration [var?:label? props?] if present; '' otherwise."""
    var   = edge.get("var")
    label = edge.get("label")
    attrs = edge.get("attribute", {})
    if not var and not label and not attrs:
        return ""
    inner = ""
    if var:
        inner = var
    if label:
        inner += f":{label}"
    if attrs:
        inner += _fmt_props(attrs)
    return f"-[{inner}]"


def _fmt_arrow(edge: dict) -> str:
    """Format the arrow string, including any edge decoration."""
    dec = _fmt_edge_decoration(edge)
    if dec:
        return f"{dec}->"
    return "->"


def _build_match_clauses(nodes: list[dict], edges: list[dict]) -> str:
    """
    Reconstruct the MATCH clause pattern string from a node list and edge list.

    Strategy:
    1. Build a directed adjacency list.
    2. Starting from the highest out-degree node, greedily extend chains to
       cover as many edges as possible in a single clause.
    3. Append any remaining uncovered edges as individual clauses.
    4. Each clause occupies one line; clauses are joined with ",\\n      ".
    """
    # id -> entry (used to emit full declarations on first occurrence)
    node_by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Mutable edge set — track which edges have been covered
    remaining: list[dict] = list(edges)

    # Adjacency list: from -> [to, ...]
    from collections import defaultdict
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["from"]].append(e["to"])

    clauses: list[str] = []
    declared: set[str] = set()   # nodes already declared in a previous clause

    def pop_edge(frm: str, to: str) -> bool:
        """Remove an edge from remaining; return True if found."""
        for i, e in enumerate(remaining):
            if e["from"] == frm and e["to"] == to:
                remaining.pop(i)
                return True
        return False

    def render_node(nid: str) -> str:
        """Emit a full declaration on first use; a bare reference thereafter."""
        if nid in declared:
            return _fmt_node_ref(nid)
        declared.add(nid)
        return _fmt_node(node_by_id[nid])

    def build_chain_from(start: str) -> str | None:
        """
        Greedily extend the longest chain starting from start along out-edges.
        Returns the clause string, or None if start has no remaining out-edges.
        """
        out_edges = [e for e in remaining if e["from"] == start]
        if not out_edges:
            return None
        e = out_edges[0]
        pop_edge(e["from"], e["to"])
        parts = [render_node(start), _fmt_arrow(e), render_node(e["to"])]
        cur = e["to"]
        # Keep extending
        while True:
            next_edges = [ex for ex in remaining if ex["from"] == cur]
            if not next_edges:
                break
            nxt = next_edges[0]
            pop_edge(nxt["from"], nxt["to"])
            parts += [_fmt_arrow(nxt), render_node(nxt["to"])]
            cur = nxt["to"]
        return "".join(parts)

    # Process nodes with out-edges first, ordered by descending out-degree
    from_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        from_counts[e["from"]] += 1
    start_order = sorted(from_counts.keys(), key=lambda x: -from_counts[x])

    for start in start_order:
        while True:
            chain = build_chain_from(start)
            if chain is None:
                break
            clauses.append(chain)

    # Handle any remaining uncovered edges (e.g. reverse edges not yet emitted)
    for e in list(remaining):
        pop_edge(e["from"], e["to"])
        clauses.append(render_node(e["from"]) + _fmt_arrow(e) + render_node(e["to"]))

    # Handle isolated nodes (no edges — rare)
    for n in nodes:
        if n["id"] not in declared:
            clauses.append(render_node(n["id"]))

    return ",\n      ".join(clauses)


def _fmt_agg_expr(target: dict, agg: str) -> str:
    """Format a single aggregation expression: mean(var.attr), mean(|...|), or mean(a->b.attr)."""
    if "node" in target:
        n = target["node"]
        return f"{agg}({n['id']}.{n['attribute']})"
    elif "node_diff" in target:
        d      = target["node_diff"]
        n1, n2 = d["nodes"]
        attr   = d["attribute"]
        return f"{agg}(|{n1}.{attr} - {n2}.{attr}|)"
    elif "edge" in target:
        e = target["edge"]
        return f"{agg}({e['from']}->{e['to']}.{e['attribute']})"
    elif "edge_diff" in target:
        d    = target["edge_diff"]
        (a1, b1), (a2, b2) = d["edges"]
        attr = d["attribute"]
        return f"{agg}({a1}->{b1}.{attr} - {a2}->{b2}.{attr})"
    else:
        raise ValueError(f"Unknown target format: {target}")


def _fmt_test(target: dict, test: dict) -> str:
    """
    Format the full TEST clause expression from the test dict.

    Handles three forms:
      - global mean  : mean(expr) = mu
      - scalar rhs   : mean(expr) OP <number>
      - aggregate rhs: mean(expr) OP mean(rhs_expr)
    """
    agg = test["agg"]
    op  = test["op"]
    rhs = test["rhs"]
    lhs = _fmt_agg_expr(target, agg)

    # Global population mean
    if rhs == "mu":
        return f"{lhs} {op} mu"

    # Scalar literal (int / float / string)
    if isinstance(rhs, (int, float)):
        return f"{lhs} {op} {rhs}"
    if isinstance(rhs, str):
        return f"{lhs} {op} \"{rhs}\""

    # Aggregate RHS (target dict)
    if isinstance(rhs, dict):
        rhs_expr = _fmt_agg_expr(rhs, agg)
        return f"{lhs} {op} {rhs_expr}"

    raise ValueError(f"Unrecognised rhs in test dict: {test}")


def unparse_entry(pid: str, entry: dict) -> str:
    """Convert a single pattern entry to a GHT-QL statement string (with comment line)."""
    nodes   = entry["pattern"]["nodes"]
    edges   = entry["pattern"]["edges"]
    dataset = entry["dataset"]
    target  = entry["target"]
    test    = entry["test"]

    match_body = _build_match_clauses(nodes, edges)
    test_expr  = _fmt_test(target, test)

    lines = [
        f"-- {pid}",
        f"MATCH {match_body}",
        f"ON {dataset}",
        f"TEST {test_expr}",
    ]
    return "\n".join(lines)


def unparse(patterns: dict[str, dict]) -> str:
    """
    Convert a complete patterns dict to a GHT-QL source string.
    Statements are separated by blank lines.
    """
    blocks = [unparse_entry(pid, entry) for pid, entry in patterns.items()]
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def unparse_file(input_json: str, output_ghtql: str | None = None) -> str:
    """
    Read a JSON file and convert it to GHT-QL source.
    If output_ghtql is provided, the result is also written to that file.
    Returns the generated source string.
    """
    with open(input_json, encoding="utf-8") as f:
        patterns = json.load(f)
    src = unparse(patterns)
    if output_ghtql:
        with open(output_ghtql, "w", encoding="utf-8") as f:
            f.write(src)
    return src


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="GHT-QL Unparser — JSON → .ghtql",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input",          help="input patterns.json file")
    ap.add_argument("-o", "--output", help="output .ghtql file (omit to print to stdout)")
    args = ap.parse_args()

    src = unparse_file(args.input, args.output)

    if args.output:
        print(f"✅ Unparse complete. Written to {args.output}")
    else:
        print(src)


if __name__ == "__main__":
    _cli()

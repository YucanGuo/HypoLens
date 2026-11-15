"""
schema_extractor.py
-------------------
Extract the schema of each heterogeneous graph stored as a NetworkX graph.pickle.

The schema includes:
  - Node types (node labels) and their attributes
      * attribute name
      * inferred dtype  (int | float | str | bool | mixed | null)
      * coverage        (fraction of nodes of that type that carry the attribute)
      * samples         (up to 5 distinct non-null values)
  - For nodes with large sets of binary (0/1) flag attributes (e.g. Yelp
    business categories, MovieLens genres), those flags are summarised as a
    separate "binary_flags" section listing all flag names and their coverage.
  - Edge types and their attributes (same structure as node attributes).
    Edges without an explicit "label" attribute are typed by the pair of
    node labels they connect: "<src_label>-<dst_label>".
  - Connectivity table: which (src_label, edge_type, dst_label) triples exist
    and how many edges of each triple there are.

CLI usage:
    python schema_extractor.py                         # all datasets, print to stdout
    python schema_extractor.py --dataset citation      # single dataset
    python schema_extractor.py -o schema.json          # write JSON output

Python API:
    from dataprep.schema_extractor import extract_schema
    schema = extract_schema("citation")   # returns a dict
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from ght_engine.paths import DATASET_DIR

DATASET_ROOT = DATASET_DIR
DATASETS = ("citation", "movielens", "yelp")

# Attributes that are always "structural" and should NOT be shown in the
# per-type attribute list (they are already captured at a higher level).
_SKIP_NODE_ATTRS = {"label"}
_SKIP_EDGE_ATTRS = {"label"}

# If a node type has more than this many binary (0/1 int) attributes whose
# names do NOT appear in a known "core" set, we treat the excess as a
# "binary_flags" category_set rather than listing them individually.
_BINARY_FLAG_THRESHOLD = 20


# ── dtype inference ────────────────────────────────────────────────────────────

def _infer_dtype(values: list) -> str:
    """Return a simple dtype string from a list of sample values.

    NOTE: bool must be checked with isinstance(v, bool) *before* int,
    because bool is a subclass of int in Python and type(True).__name__
    returns 'bool', but isinstance(True, int) is True.  Using type()
    alone would silently treat booleans as integers in mixed lists.
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "null"

    # Classify each value into a canonical type label, giving bool
    # priority over int (since bool is a subclass of int).
    def _label(v) -> str:
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        return "other"

    types = {_label(v) for v in non_null}

    if types == {"bool"}:
        return "bool"
    if types <= {"int", "bool"}:
        # Semantic inference: plain ints whose value domain is exactly
        # {0, 1} are treated as bool (e.g. genre flags in MovieLens).
        if set(non_null) <= {0, 1}:
            return "bool"
        return "int"
    if types <= {"float", "int", "bool"}:
        return "float"
    if types == {"str"}:
        return "str"
    return "mixed"


# ── attribute schema builder ───────────────────────────────────────────────────

def _build_attr_schema(values: list, total: int) -> dict:
    """
    Build the schema entry for one attribute given its value list.

    Returns {"dtype", "coverage", "samples"}.
    """
    present   = [v for v in values if v is not None]
    distinct: list = []
    seen: set = set()
    for v in present:
        key = str(v)
        if key not in seen:
            seen.add(key)
            distinct.append(v)
        if len(distinct) >= 5:
            break

    return {
        "dtype":    _infer_dtype(present),
        "coverage": round(len(present) / total, 4) if total else 0.0,
        "samples":  distinct,
    }


def _is_binary_flag(values: list) -> bool:
    """Return True if all non-null values are 0 or 1 (integer binary flag)."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    return all(isinstance(v, (int, bool)) and v in (0, 1) for v in non_null)


# ── edge type label ────────────────────────────────────────────────────────────

def _edge_type(G: Any, u: Any, v: Any, attrs: dict) -> str:
    """
    Return the edge type string.
    If the edge carries a 'label' attribute, use that.
    Otherwise derive it from the connected node labels.
    """
    if "label" in attrs:
        return attrs["label"]
    src = G.nodes[u].get("label", "?")
    dst = G.nodes[v].get("label", "?")
    return f"{src}-{dst}"


# ── main extraction ────────────────────────────────────────────────────────────

def extract_schema(dataset: str) -> dict:
    """
    Load <dataset>/graph.pickle and return its full schema as a dict.

    Schema structure:
    {
      "dataset":    str,
      "graph_type": "undirected" | "directed",
      "node_count": int,
      "edge_count": int,
      "node_types": {
          "<label>": {
              "count":        int,
              "attributes":   { "<attr>": {"dtype","coverage","samples"} },
              "binary_flags": { "<flag>": {"coverage"} }   # only if flags exist
          }
      },
      "edge_types": {
          "<label|src-dst>": {
              "count":      int,
              "attributes": { "<attr>": {"dtype","coverage","samples"} }
          }
      },
      "connectivity": [
          {"src": str, "edge": str, "dst": str, "count": int}
      ]
    }
    """
    pickle_path = DATASET_ROOT / dataset / "graph.pickle"
    with open(pickle_path, "rb") as f:
        G = pickle.load(f)

    # ── gather node attribute values per label ──────────────────────────────
    # node_vals[label][attr] = [v, ...]
    node_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    node_count: dict[str, int] = defaultdict(int)

    for _nid, attrs in G.nodes(data=True):
        label = attrs.get("label", "__unknown__")
        node_count[label] += 1
        for k, v in attrs.items():
            if k in _SKIP_NODE_ATTRS:
                continue
            node_vals[label][k].append(v)

    # ── gather edge attribute values per type ───────────────────────────────
    edge_vals: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    edge_count: dict[str, int] = defaultdict(int)
    connectivity_cnt: dict[tuple, int] = defaultdict(int)

    for u, v, attrs in G.edges(data=True):
        etype = _edge_type(G, u, v, attrs)
        edge_count[etype] += 1

        src_label = G.nodes[u].get("label", "?")
        dst_label = G.nodes[v].get("label", "?")
        connectivity_cnt[(src_label, etype, dst_label)] += 1

        for k, val in attrs.items():
            if k in _SKIP_EDGE_ATTRS:
                continue
            edge_vals[etype][k].append(val)

    # ── build node_types ────────────────────────────────────────────────────
    node_types: dict[str, Any] = {}
    for label in sorted(node_vals):
        cnt    = node_count[label]
        attrs  = node_vals[label]

        normal_attrs: dict[str, dict] = {}
        binary_flags: dict[str, dict] = {}

        for attr, values in sorted(attrs.items()):
            if not attr:  # skip empty-string attribute keys (malformed category splits)
                continue
            if _is_binary_flag(values):
                # Binary (0/1) integer attribute — candidate for flag folding
                binary_flags[attr] = {
                    "coverage": round(sum(1 for v in values if v == 1) / cnt, 4)
                }
            else:
                normal_attrs[attr] = _build_attr_schema(values, cnt)

        # If there are very few "binary_flags", promote them back to normal attrs
        if len(binary_flags) < _BINARY_FLAG_THRESHOLD:
            for attr, flag_info in binary_flags.items():
                # Reconstruct as a proper attr schema (values are all 0 or 1).
                # Use _infer_dtype on [0, 1] so that the semantic bool
                # inference added to _infer_dtype is honoured here too.
                normal_attrs[attr] = {
                    "dtype":    _infer_dtype([0, 1]),
                    "coverage": flag_info["coverage"],
                    "samples":  [0, 1],
                }
            binary_flags = {}

        entry: dict[str, Any] = {"count": cnt, "attributes": normal_attrs}
        if binary_flags:
            entry["binary_flags"] = binary_flags
        node_types[label] = entry

    # ── build edge_types ────────────────────────────────────────────────────
    # Use edge_count (not edge_vals) as the canonical set of edge types so that
    # edges with no attributes beyond "label" are still included in the output.
    edge_types: dict[str, Any] = {}
    for etype in sorted(edge_count):
        cnt   = edge_count[etype]
        attrs = {}
        for attr, values in sorted(edge_vals.get(etype, {}).items()):
            attrs[attr] = _build_attr_schema(values, cnt)
        edge_types[etype] = {"count": cnt, "attributes": attrs}

    # ── connectivity ────────────────────────────────────────────────────────
    connectivity = [
        {"src": src, "edge": edge, "dst": dst, "count": c}
        for (src, edge, dst), c in sorted(
            connectivity_cnt.items(), key=lambda x: -x[1]
        )
    ]

    return {
        "dataset":     dataset,
        "graph_type":  "directed" if G.is_directed() else "undirected",
        "node_count":  G.number_of_nodes(),
        "edge_count":  G.number_of_edges(),
        "node_types":  node_types,
        "edge_types":  edge_types,
        "connectivity": connectivity,
    }


# ── pretty-printer ─────────────────────────────────────────────────────────────

def _print_schema(schema: dict) -> None:
    sep = "=" * 68
    ds  = schema["dataset"]

    print(f"\n{sep}")
    print(f"  Dataset   : {ds}  ({schema['graph_type']})")
    print(f"  Nodes     : {schema['node_count']:,}")
    print(f"  Edges     : {schema['edge_count']:,}")
    print(sep)

    # ── node types ──────────────────────────────────────────────────────────
    print("\n── Node Types " + "─" * 54)
    for label, info in schema["node_types"].items():
        print(f"\n  [{label}]  count={info['count']:,}")

        for attr, meta in info["attributes"].items():
            samples_str = ", ".join(_fmt_sample(s) for s in meta["samples"])
            cov = f"{meta['coverage']*100:.1f}%"
            print(f"    {attr:<34}  {meta['dtype']:<7}  cov={cov:<7}  "
                  f"samples=[{samples_str}]")

        if "binary_flags" in info:
            flags = info["binary_flags"]
            # Show top-10 by coverage, then a count of the rest
            sorted_flags = sorted(flags.items(), key=lambda x: -x[1]["coverage"])
            top = sorted_flags[:10]
            rest = len(sorted_flags) - 10
            flag_strs = ", ".join(
                f"{name!r}({v['coverage']*100:.1f}%)" for name, v in top
            )
            suffix = f"  ... +{rest} more" if rest > 0 else ""
            print(f"    [binary_flags]  total={len(flags)}  "
                  f"top-10 by coverage: {flag_strs}{suffix}")

    # ── edge types ──────────────────────────────────────────────────────────
    print("\n── Edge Types " + "─" * 54)
    for etype, info in schema["edge_types"].items():
        print(f"\n  [{etype}]  count={info['count']:,}")
        for attr, meta in info["attributes"].items():
            samples_str = ", ".join(_fmt_sample(s) for s in meta["samples"])
            cov = f"{meta['coverage']*100:.1f}%"
            print(f"    {attr:<34}  {meta['dtype']:<7}  cov={cov:<7}  "
                  f"samples=[{samples_str}]")

    # ── connectivity ────────────────────────────────────────────────────────
    print("\n── Connectivity (src --[edge]--> dst) " + "─" * 31)
    for c in schema["connectivity"]:
        print(f"  ({c['src']}) --[{c['edge']}]--> ({c['dst']})   "
              f"count={c['count']:,}")


def _fmt_sample(v: Any) -> str:
    """Format a single sample value for display."""
    if isinstance(v, str):
        return repr(v)
    return str(v)


# Default output directory (relative to the repo root, next to ghtql/)
_SCHEMA_DIR = Path(__file__).parent.parent / "ghtql" / "schema"


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Extract schema from heterogeneous graph.pickle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "By default each dataset is written to ghtql/schema/<dataset>.json.\n"
            "Use -o to override with a single combined output file."
        ),
    )
    ap.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    ap.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Write all schemas into one combined JSON file at this path. "
            "If omitted, each dataset is written to ghtql/schema/<dataset>.json."
        ),
    )
    args = ap.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    # Extract
    schemas: dict[str, dict] = {}
    for ds in targets:
        print(f"Processing {ds} ...", flush=True)
        schemas[ds] = extract_schema(ds)

    # Write output
    if args.output:
        # Combined file at user-specified path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(schemas, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅  Schema written to {out_path}")
    else:
        # One file per dataset under ghtql/schema/
        _SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        for ds, schema in schemas.items():
            out_path = _SCHEMA_DIR / f"{ds}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(schema, f, ensure_ascii=False, indent=2, default=str)
            print(f"✅  {ds:12s} → {out_path}")


if __name__ == "__main__":
    _cli()

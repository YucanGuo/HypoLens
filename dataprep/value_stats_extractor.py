"""
value_stats_extractor.py
-------------------------
Extract, for every (node/edge type, attribute) pair in each heterogeneous
graph.pickle, the distribution of attribute values (value -> frequency),
and persist the top-N most frequent values to a standalone JSON file.

This is intentionally kept separate from `schema_extractor.py`:
  - schema_extractor.py describes structure (dtype/coverage/a few samples).
  - value_stats_extractor.py describes value popularity, which is only
    needed to power UI affordances (e.g. autocomplete candidates in the
    Visual Builder's attribute-value input), not for GHT-QL parsing/validation.

Output layout (one file per dataset), by default written next to the schema
files at ghtql/schema/<dataset>_values.json:

{
  "dataset": "movielens",
  "top_n": 10,
  "node_types": {
    "<label>": {
      "<attr>": [ {"value": <v>, "count": <int>}, ... up to top_n ... ],
      ...
    }
  },
  "edge_types": {
    "<etype>": {
      "<attr>": [ {"value": <v>, "count": <int>}, ... ],
      ...
    }
  }
}

CLI usage:
    python -m dataprep.value_stats_extractor                    # all datasets
    python -m dataprep.value_stats_extractor --dataset citation # single dataset
    python -m dataprep.value_stats_extractor --top-n 15         # keep more candidates

Python API:
    from dataprep.value_stats_extractor import extract_value_stats
    stats = extract_value_stats("citation", top_n=10)   # returns a dict
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ght_engine.paths import DATASET_DIR

DATASET_ROOT = DATASET_DIR
DATASETS = ("citation", "movielens", "yelp")

# Same structural attributes excluded by schema_extractor.py — they are not
# user-facing filterable attributes.
_SKIP_NODE_ATTRS = {"label"}
_SKIP_EDGE_ATTRS = {"label"}

DEFAULT_TOP_N = 10

# Default output directory (relative to the repo root, next to ghtql/schema/)
_OUTPUT_DIR = Path(__file__).parent.parent / "ghtql" / "schema"


def _edge_type(G: Any, u: Any, v: Any, attrs: dict) -> str:
    """Return the edge type string (same rule as schema_extractor.py)."""
    if "label" in attrs:
        return attrs["label"]
    src = G.nodes[u].get("label", "?")
    dst = G.nodes[v].get("label", "?")
    return f"{src}-{dst}"


def _top_values(counter: Counter, top_n: int) -> list[dict]:
    """Return the top-N (value, count) pairs, most frequent first.

    Ties are broken by insertion order via Counter.most_common, which is
    stable and deterministic enough for UI candidate lists.
    """
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(top_n)
    ]


def extract_value_stats(dataset: str, top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Load <dataset>/graph.pickle and return the top-N most frequent values
    for every node/edge attribute.
    """
    pickle_path = DATASET_ROOT / dataset / "graph.pickle"
    with open(pickle_path, "rb") as f:
        G = pickle.load(f)

    # node_counters[label][attr] = Counter({value: freq, ...})
    node_counters: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for _nid, attrs in G.nodes(data=True):
        label = attrs.get("label", "__unknown__")
        for k, v in attrs.items():
            if k in _SKIP_NODE_ATTRS or v is None:
                continue
            node_counters[label][k][v] += 1

    # edge_counters[etype][attr] = Counter({value: freq, ...})
    edge_counters: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for u, v, attrs in G.edges(data=True):
        etype = _edge_type(G, u, v, attrs)
        for k, val in attrs.items():
            if k in _SKIP_EDGE_ATTRS or val is None:
                continue
            edge_counters[etype][k][val] += 1

    node_types = {
        label: {
            attr: _top_values(counter, top_n)
            for attr, counter in sorted(attr_counters.items())
            if attr  # skip empty-string attribute keys
        }
        for label, attr_counters in sorted(node_counters.items())
    }
    edge_types = {
        etype: {
            attr: _top_values(counter, top_n)
            for attr, counter in sorted(attr_counters.items())
            if attr
        }
        for etype, attr_counters in sorted(edge_counters.items())
    }

    return {
        "dataset":    dataset,
        "top_n":      top_n,
        "node_types": node_types,
        "edge_types": edge_types,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract per-attribute value frequency (top-N) from heterogeneous "
            "graph.pickle files, for use as UI autocomplete candidates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "By default each dataset is written to ghtql/schema/<dataset>_values.json."
        ),
    )
    ap.add_argument(
        "--dataset", "-d",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="Which dataset to process (default: all)",
    )
    ap.add_argument(
        "--top-n", "-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"How many most-frequent values to keep per attribute (default: {DEFAULT_TOP_N})",
    )
    ap.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directory to write <dataset>_values.json files into (default: ghtql/schema/)",
    )
    args = ap.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    out_dir = Path(args.output_dir) if args.output_dir else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in targets:
        print(f"Processing {ds} ...", flush=True)
        stats = extract_value_stats(ds, top_n=args.top_n)
        out_path = out_dir / f"{ds}_values.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"✅  {ds:12s} → {out_path}")


if __name__ == "__main__":
    _cli()

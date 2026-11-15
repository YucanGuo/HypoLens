"""
GHTExecutor
===========
Bridges the Web interface to the Graph Hypothesis Testing (GHT) engine
contained in the ``ght_engine`` package.

Once the Visual Builder has produced a *parsed_pattern* — the
dict emitted by ``ghtql.parser.parse_string`` — this module:

  1. Converts the GHT-QL pattern (``dataset`` / ``pattern`` / ``target`` /
     ``test``) into the ``args.attribute`` structure understood by
     ``extraction.py`` (``getEdges`` / ``getNodes`` / ``getPaths``).
  2. Loads the real dataset graph (``utils.get_data``) — reusing the cached
     ``datasets/<name>/graph.pickle`` files.
  3. Computes the ground truth over the *entire* graph.
  4. Runs the ``Opt_PHASE`` sampler (``sampling.sample_graph``) for the
     requested sampling ratio(s) / repetitions, and aggregates the one-sample
     t-test results (mean, confidence interval, p-value, accuracy against the
     ground truth).

Only patterns whose topology is a linear chain (GHT-QL types ``node``,
``edge`` or ``path``) can be executed, because ``PHASE`` / ``Opt_PHASE`` walk
a single path of node/edge conditions.  Non-linear topologies (``star``,
``square``, ``tailed_square``, ``general``) are rejected with a clear error
telling the user which types are supported.

Public API
----------
    from ght_engine.executor import GHTExecutor
    executor = GHTExecutor()
    report = executor.run(parsed_pattern_entry, sampling_percent=[5])
"""

from __future__ import annotations

import io
import logging
import math
import time
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any, Callable

from .utils import get_data, HypothesisTesting, compute_accuracy
from .extraction import getEdges, getNodes, getPaths
from .sampling import sample_graph


# ── dataset name mapping ────────────────────────────────────────────────────────

_DATASET_TO_LOWER = {"Citation": "citation", "MovieLens": "movielens", "Yelp": "yelp"}

# GHT-QL pattern types that Opt_PHASE / extraction.py can actually execute.
_SUPPORTED_TYPES = {"node", "edge", "path"}

# extraction.py::getPaths() supports arbitrary-length path chains.
# The coverage statistics use r[0]/r[-1] (endpoints) and r[1:-1] (intermediate),
# so any chain length >= 2 works correctly.


class GHTExecutorError(Exception):
    """Raised when a parsed_pattern cannot be translated/executed."""


# ── args shim ────────────────────────────────────────────────────────────────────

class _Args:
    """Parameters shared by extraction, sampling, and hypothesis testing."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _make_silent_logger() -> logging.Logger:
    """A logger that swallows everything (we don't want to write log files for
    every ad-hoc web execution; progress is surfaced via progress_cb instead)."""
    logger = logging.getLogger(f"ght_executor.{id(object())}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    return logger


# ── pattern -> attribute translation ────────────────────────────────────────────

def _node_lookup(pattern: dict) -> dict[str, dict]:
    return {n["id"]: n for n in pattern["nodes"]}


def _linear_order(pattern: dict) -> list[str]:
    """
    Return the node-id chain implied by the pattern's edges, in traversal
    order, for 'path' / 'edge' / 'node' topologies (every node has undirected
    degree <= 2 and the edges form a single chain).
    """
    nodes = pattern["nodes"]
    edges = pattern["edges"]
    node_ids = [n["id"] for n in nodes]

    if len(node_ids) == 1:
        return node_ids

    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for e in edges:
        adjacency[e["from"]].append(e["to"])
        adjacency[e["to"]].append(e["from"])

    degrees = {nid: len(neigh) for nid, neigh in adjacency.items()}
    endpoints = [nid for nid, d in degrees.items() if d == 1]
    if len(endpoints) != 2 or any(d > 2 for d in degrees.values()):
        raise GHTExecutorError(
            "The pattern is not a simple linear chain (node -> node -> ...), "
            "so it cannot be translated into an Opt_PHASE path condition."
        )

    start = endpoints[0]
    order = [start]
    visited = {start}
    current = start
    while len(order) < len(node_ids):
        nxt = [n for n in adjacency[current] if n not in visited]
        if not nxt:
            raise GHTExecutorError("Failed to linearise the pattern into a single chain.")
        current = nxt[0]
        visited.add(current)
        order.append(current)
    return order


def _path_condition(pattern: dict) -> list[dict]:
    """Build the Opt_PHASE ``condition`` list: [{"type": label, "attribute": {...}}, ...]."""
    order  = _linear_order(pattern)
    lookup = _node_lookup(pattern)
    return [
        {"type": lookup[nid]["label"], "attribute": dict(lookup[nid].get("attribute", {}))}
        for nid in order
    ]


def build_attribute_dict(entry: dict, condition_name: str = "hypo") -> tuple[dict, int]:
    """
    Translate one parsed_pattern *entry* (as produced by
    ``ghtql.parser.parse_string``) into the ``args.attribute`` dict + ``args.hypo``
    code understood by ``extraction.py`` / ``Opt_PHASE`` / ``PHASE``.

    Returns
    -------
    (attribute_dict, hypo)
        attribute_dict : {condition_name: {...}} — extraction conditions.
        hypo           : 1 (edge) | 2 (node) | 3 (path)
    """
    ptype   = entry.get("type")
    pattern = entry["pattern"]
    target  = entry["target"]

    if ptype not in _SUPPORTED_TYPES:
        raise GHTExecutorError(
            f"Pattern type '{ptype}' is not executable by the GHT engine. "
            f"Only linear patterns are supported by Opt_PHASE/PHASE: "
            f"{sorted(_SUPPORTED_TYPES)} (star / square / tailed_square / general "
            f"require full subgraph matching, which the sampler does not implement)."
        )

    condition   = _path_condition(pattern)
    lookup      = _node_lookup(pattern)
    linear_order = _linear_order(pattern)

    # NOTE on supported (topology, target) combinations:
    #
    # The GHT-QL grammar produces four target shapes:
    #   {"node":      {"id": ..., "attribute": ...}}
    #   {"node_diff": {"nodes": [...], "attribute": ...}}
    #   {"edge":      {"from": ..., "to": ..., "attribute": ...}}
    #   {"edge_diff": {"edges": [[a,b],[c,d]], "attribute": ...}}
    #
    # Executable combinations:
    #   - path (N nodes, N≥3) + target.node      → getPaths node mode  (hypo=3)
    #   - path (N nodes, N≥3) + target.node_diff → getPaths node mode  (hypo=3)
    #   - path (N nodes, N≥3) + target.edge      → getPaths edge mode  (hypo=3)
    #   - path (N nodes, N≥3) + target.edge_diff → getPaths edge_diff mode (hypo=3)
    #   - edge (2 nodes) + target.node           → getEdges node mode  (hypo=1)
    #   - edge (2 nodes) + target.edge           → getEdges edge mode  (hypo=1)
    #   - node (1 node)  + target.node           → getNodes            (hypo=2)
    # node_diff on a 2-node edge is rejected (needs ≥3 positions).
    # Everything else is rejected explicitly so the user gets an actionable
    # message instead of a silent empty result.

    if "edge_diff" in target:
        ediff  = target["edge_diff"]
        edges  = ediff["edges"]
        attr   = ediff["attribute"]
        if len(edges) != 2:
            raise GHTExecutorError("edge_diff target must reference exactly two edges.")
        if ptype != "path":
            raise GHTExecutorError(
                f"edge_diff target is only supported on 'path' patterns "
                f"(at least 3 nodes), but this pattern is type '{ptype}'."
            )
        # Position index of each referenced edge along the linearised chain:
        # edge i connects linear_order[i] — linear_order[i+1]. Pattern edge
        # direction may differ from the chain direction, so match unordered.
        edge_positions: list[int] = []
        for frm, to in edges:
            for i in range(len(linear_order) - 1):
                if {linear_order[i], linear_order[i + 1]} == {frm, to}:
                    edge_positions.append(i)
                    break
            else:
                raise GHTExecutorError(
                    f"edge_diff references edge {frm!r}->{to!r} which is not part "
                    f"of the linearised pattern chain."
                )
        if edge_positions[0] == edge_positions[1]:
            raise GHTExecutorError(
                "edge_diff must reference two distinct pattern edges."
            )
        condition_dict: dict[str, Any] = {
            "path": condition,
            "edge_diff": {"indices": edge_positions, "attribute": attr},
            "mode": "edge_diff",
        }
        hypo = 3
    elif "node_diff" in target:
        diff    = target["node_diff"]
        var_ids = diff["nodes"]
        attr    = diff["attribute"]
        if len(var_ids) != 2:
            raise GHTExecutorError("node_diff target must reference exactly two nodes.")
        label_a = lookup[var_ids[0]]["label"]
        label_b = lookup[var_ids[1]]["label"]
        if label_a != label_b:
            raise GHTExecutorError(
                f"node_diff nodes must share the same label to be aggregated by the "
                f"GHT engine's path extractor (got '{label_a}' vs '{label_b}')."
            )
        if ptype != "path":
            raise GHTExecutorError(
                f"node_diff target is only supported on 'path' patterns, "
                f"but this pattern is type '{ptype}'."
            )
        # Position indices of the two diff nodes in the linearised chain.
        pos_a = linear_order.index(var_ids[0])
        pos_b = linear_order.index(var_ids[1])
        condition_dict: dict[str, Any] = {
            "path": condition,
            "node": {"indices": [pos_a, pos_b], "attribute": attr},
            "mode": "diff",
        }
        hypo = 3
    elif "node" in target:
        var_id = target["node"]["id"]
        attr   = target["node"]["attribute"]
        label  = lookup[var_id]["label"]

        if ptype == "path":
            pos = linear_order.index(var_id)
            condition_dict = {
                "path": condition,
                "node": {"index": pos, "attribute": attr},
            }
            hypo = 3
        elif ptype == "node":
            # Single-node pattern -> getNodes (hypo=2).
            #
            # getNodes sub-mode A expects condition_dict to have exactly 3 keys:
            #   - "dimension": {label: target_attr}  — what to extract
            #   - "path": [...]                       — for the Opt_PHASE sampler
            #   - <label>: {filter_attrs}             — node filter conditions
            # The len==3 guard in getNodes checks exactly this structure.
            #
            # Opt_PHASE with a 1-node path biases its random walk toward nodes
            # matching the single node condition, which is the right behaviour
            # for sampling a node-attribute distribution.
            node_filter = dict(lookup[var_id].get("attribute", {}))
            condition_dict: dict[str, Any] = {
                "dimension": {label: attr},
                "path": condition,
                label: node_filter,
            }
            hypo = 2
        elif ptype == "edge":
            # 2-node edge pattern + node-attribute target → getEdges node mode
            # (hypo=1). condition_dict["node"] = {"index": pos, "attribute": attr}
            # tells getEdges to extract the target endpoint's node attribute;
            # condition_dict["path"] drives the Opt_PHASE random walk.
            pos = linear_order.index(var_id)
            condition_dict: dict[str, Any] = {
                "node": {"index": pos, "attribute": attr},
                "path": condition,
            }
            for node_entry in pattern["nodes"]:
                if node_entry.get("attribute"):
                    condition_dict[node_entry["label"]] = dict(node_entry["attribute"])
            hypo = 1
        else:
            raise GHTExecutorError(
                f"Pattern type '{ptype}' with a node-attribute target is not "
                f"supported by the GHT engine."
            )
    elif "edge" in target:
        # Edge-attribute target: extract an attribute from the pattern's edge(s).
        #   - ptype == "edge" (2 nodes / 1 edge) → getEdges edge mode (hypo=1),
        #     which reads condition_dict["edge"] as the edge attribute name and
        #     filters endpoints via checkCondition.
        #   - ptype == "path" (N nodes, N≥3) → getPaths edge mode (hypo=3),
        #     which averages the edge attribute along the path's edges.
        e   = target["edge"]
        attr = e["attribute"]
        if ptype == "edge":
            condition_dict: dict[str, Any] = {"edge": attr, "path": condition}
            # Node attribute filters: checkCondition reads condition_dict[label]
            for node_entry in pattern["nodes"]:
                if node_entry.get("attribute"):
                    condition_dict[node_entry["label"]] = dict(node_entry["attribute"])
            hypo = 1
        elif ptype == "path":
            condition_dict = {"edge": attr, "path": condition}
            for node_entry in pattern["nodes"]:
                if node_entry.get("attribute"):
                    condition_dict[node_entry["label"]] = dict(node_entry["attribute"])
            hypo = 3
        else:
            raise GHTExecutorError(
                f"Edge-attribute targets are only supported on 'edge' (2-node) "
                f"and 'path' (N-node) patterns, not '{ptype}'."
            )
    else:
        raise GHTExecutorError(f"Unsupported TEST target shape: {target!r}")

    return {condition_name: condition_dict}, hypo


def _translate_test(entry: dict) -> tuple[str, float | None]:
    """Map the GHT-QL ``test`` clause to (comparison, scalar_c)."""
    test = entry["test"]
    op   = test["op"]
    rhs  = test["rhs"]

    op_map = {"=": "==", "==": "==", "!=": "!=", ">": ">", "<": "<", ">=": ">", "<=": "<"}
    if op not in op_map:
        raise GHTExecutorError(f"Unsupported comparison operator '{op}'.")
    comparison = op_map[op]

    if isinstance(rhs, dict):
        raise GHTExecutorError(
            "Two-sample TEST clauses (comparing two aggregate targets) are not yet "
            "supported by the GHT execution engine; only comparisons against 'mu' "
            "(the population mean) or a scalar constant are executable."
        )
    if rhs == "mu":
        if "edge_diff" in entry.get("target", {}) or "node_diff" in entry.get("target", {}):
            # A paired difference (two positional nodes or edges of the same
            # matched instance) has no meaningful population-mean reference;
            # the natural null is a scalar threshold (typically 0).
            raise GHTExecutorError(
                "node_diff / edge_diff TEST clauses must compare against a "
                "scalar threshold (typically 0), not 'mu'."
            )
        # Resolved later to the actual population-mean ground truth value.
        return comparison, None
    if isinstance(rhs, (int, float)):
        return comparison, float(rhs)
    raise GHTExecutorError(f"Unsupported TEST right-hand side: {rhs!r}")


# ── executor ─────────────────────────────────────────────────────────────────────

class GHTExecutor:
    """
    Runs the real GHT (Opt_PHASE) sampling + one-sample hypothesis-testing
    pipeline for a parsed GHT-QL pattern.

    A small in-process cache keeps loaded graphs around across calls (the
    Citation / Yelp graphs are large and slow to unpickle).
    """

    _graph_cache: dict[str, Any] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        entry: dict,
        sampling_percent: list[float] | None = None,
        num_samples: int = 10,
        seed: str = "2022",
        alpha: float = 0.05,
        sampling_method: str = "Opt_PHASE",
        mode: str = "sample",
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute Opt_PHASE sampling + hypothesis testing for one parsed_pattern
        entry (the value of a single key in the dict returned by
        ``ghtql.parser.parse_string``).

        Parameters
        ----------
        entry : dict
            One pattern entry: {"dataset", "type", "pattern", "target", "test", ...}
        sampling_percent : list[float]
            Sampling ratios expressed as % of total nodes, e.g. [5].
            Defaults to [5] (a single ratio).  Ignored when ``mode="full"``.
        num_samples : int
            Number of independent samples drawn (repeats), used for
            confidence-interval / p-value averaging.
        seed : str
            Base random seed.
        alpha : float
            Significance level (default 0.05); CI is computed at confidence
            level ``1 - alpha`` (i.e. 0.05 → 95% CI). Must match the p-value
            threshold hard-coded in utils.HypothesisTesting (0.05) to stay
            self-consistent.
        sampling_method : str
            Sampler to use — defaults to "Opt_PHASE"; "PHASE" also works.
        mode : str
            Precision mode — one of:

            * ``"sample"`` *(default)* — sampling approximation only.  Skips the
              full-graph pattern match and t-test; returns ``sampling_results``
              with CI / p-value, and ``ground_truth`` = null (except that μ is
              still computed when ``TEST ... = mu``).
            * ``"full"`` — full-graph exact result only.  Skips sampling;
              returns ``ground_truth`` (full-graph t-test verdict) and
              ``sampling_results`` = [].
            * ``"compare"`` — both; also reports ``accuracy`` (sampling verdict
              vs. full-graph verdict).
        progress_cb : callable(str) -> None, optional
            Called with human-readable progress messages as the pipeline runs.

        Returns
        -------
        A JSON-serialisable dict with ground truth, sampling results, and timing.
        """
        def report(msg: str) -> None:
            if progress_cb:
                progress_cb(msg)

        if mode not in ("sample", "full", "compare"):
            raise GHTExecutorError(
                f"Invalid mode '{mode}'; expected one of: sample, full, compare."
            )

        sampling_percent = sampling_percent or [5]

        dataset_pretty = entry.get("dataset")
        dataset_lower  = _DATASET_TO_LOWER.get(dataset_pretty)
        if dataset_lower is None:
            raise GHTExecutorError(f"Unsupported dataset '{dataset_pretty}'.")

        attribute, hypo = build_attribute_dict(entry)
        condition_name  = next(iter(attribute))
        comparison, scalar_c = _translate_test(entry)

        # NOTE on `comparison` direction:
        # In this codebase `comparison` describes the NULL hypothesis (H0),
        # not the alternative.  e.g. `comparison=">"` means "H0: mean > c".
        # utils.HypothesisTesting maps this to scipy's `alternative` as:
        #   ">" -> alternative="less"    (H1: mean < c,  i.e. H0: mean >= c)
        #   "<" -> alternative="greater" (H1: mean > c,  i.e. H0: mean <= c)
        # This is correct and intentional — do NOT flip it.  Verified via
        # direct experiment with utils.HypothesisTesting.

        args = _Args(
            dataset          = dataset_lower,
            seed             = seed,
            sampling_method  = sampling_method,
            sampling_percent = sampling_percent,
            num_samples      = num_samples,
            time_accuracy    = False,
            HTtype           = "one-sample",
            attribute        = attribute,
            agg              = "mean",
            hypo             = hypo,
            comparison       = comparison,
            c                = scalar_c if scalar_c is not None else 0.0,
            alpha            = alpha,
            logger           = _make_silent_logger(),
            H0               = entry.get("H0") or f"{condition_name} hypothesis test",
        )

        # ── 1. load graph (cached) ───────────────────────────────────────────
        report(f"Loading {dataset_pretty} graph…")
        t0 = time.time()
        graph = self._graph_cache.get(dataset_lower)
        if graph is None:
            with redirect_stdout(io.StringIO()):
                graph = get_data(args)
            self._graph_cache[dataset_lower] = graph
        load_time = round(time.time() - t0, 2)
        report(f"Graph loaded in {load_time}s "
               f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).")

        args.num_nodes = graph.number_of_nodes()
        args.num_edges = graph.number_of_edges()
        if hypo == 2:
            # Node extraction reads the dimension from the condition dict.
            args.dimension = attribute[condition_name]["dimension"]

        # ── 2. population mean (μ) — always computed for `TEST ... = mu` ─────
        # `mu` is the mean of the target attribute over ALL nodes/edges of the
        # relevant type in the full graph.  It is needed in every mode because
        # without it the `= mu` test has no defined reference value.
        if scalar_c is None:
            pop_mean, pop_count = self._population_mean(entry, graph, dataset_lower)
            args.c = pop_mean
            report(
                f"Population mean (μ) = {round(pop_mean, 4)} "
                f"over {pop_count} nodes/edges in the full graph."
            )

        # ── 3. ground truth (full-graph pattern match + t-test) ────────────
        # Skipped in "sample" mode: the full-graph match can be expensive on
        # large graphs, and the user has explicitly asked for a sampling
        # approximation only.  ground_truth stays None.
        ground_truth_value: float | None = None
        ground_truth_result: bool | None = None
        gt_n_matches: int = 0
        gt_time = 0.0
        gt_p_value: float | None = None

        if mode in ("full", "compare"):
            report("Computing ground truth over the full graph…")
            t0 = time.time()
            with redirect_stdout(io.StringIO()):
                gt_values = self._extract(args, graph, hypo)[condition_name]

            if len(gt_values) == 0:
                raise GHTExecutorError(
                    "No node/edge/path in the full graph satisfies this hypothesis pattern. "
                    "Try relaxing the attribute constraints."
                )
            ground_truth_value = round(sum(gt_values) / len(gt_values), 4)
            gt_n_matches = len(gt_values)
            gt_time = round(time.time() - t0, 2)

            # Set up capture buffers so HypothesisTesting stores the GT
            # p-value / CI on args (it only appends when these attrs exist).
            args.p_value = []
            args.CI = {"lower": [], "upper": []}
            with redirect_stdout(io.StringIO()):
                ground_truth_result = HypothesisTesting(args, gt_values, verbose=0)
            if args.p_value:
                gt_p_value = round(args.p_value[-1], 4)

            report(
                f"Ground truth computed in {gt_time}s: "
                f"subset mean={ground_truth_value} over {gt_n_matches} matches "
                f"(H0 {'accepted' if ground_truth_result else 'rejected'}, "
                f"p={gt_p_value})."
            )

        # ── 4. sampling + hypothesis test at each requested ratio ──────────
        # Skipped in "full" mode: the user wants the exact full-graph answer
        # only, so no sampling is performed.
        args.overall_time    = time.time()
        args.sampling_ratio  = [max(1, int(args.num_nodes * (p / 100))) for p in sampling_percent]
        args.coverage        = defaultdict(list)

        ratio_reports: list[dict[str, Any]] = []
        if mode in ("sample", "compare"):
            for pct, ratio in zip(sampling_percent, args.sampling_ratio):
                report(f"Sampling with {sampling_method} at {pct}% ({ratio} nodes)…")
                args.time_result  = defaultdict(list)
                args.valid_edges  = []
                args.variance     = []
                args.sample_means = []
                args.ratio        = ratio

                t0 = time.time()
                with redirect_stdout(io.StringIO()):
                    result_list, time_used_list = sample_graph(
                        args, graph, [], defaultdict(list), sampling_method
                    )
                elapsed = round(time.time() - t0, 2)

                sample_bools = [r[condition_name] for r in result_list if r.get(condition_name) is not None]

                # accuracy only makes sense when we have a ground-truth verdict
                # to compare against (i.e. "compare" mode).
                if ground_truth_result is not None:
                    accuracy = compute_accuracy(ground_truth_result, sample_bools)
                    accuracy_out: float | None = round(accuracy, 4)
                else:
                    accuracy_out = None

                # Build per-repeat detail rows.  All of the args.* lists and
                # sample_bools are aligned by repeat index and have length
                # num_samples (None / NaN entries are filtered in the averaging
                # but kept here so the table can show "—" for failed repeats).
                n_repeats = num_samples
                repeats: list[dict[str, Any]] = []
                for i in range(n_repeats):
                    sm_i  = args.sample_means[i] if i < len(args.sample_means) else None
                    vm_i  = args.valid_edges[i]  if i < len(args.valid_edges)  else None
                    pl_i  = args.CI["lower"][i]  if i < len(args.CI["lower"])  else None
                    pu_i  = args.CI["upper"][i]  if i < len(args.CI["upper"])  else None
                    pv_i  = args.p_value[i]      if i < len(args.p_value)      else None
                    t_i   = time_used_list["sampling"][i] if i < len(time_used_list["sampling"]) else None
                    vb_i  = sample_bools[i] if i < len(sample_bools) else None
                    repeats.append({
                        "index":          i + 1,
                        "valid_matches":  round(float(vm_i), 2) if vm_i is not None else None,
                        "sample_mean":    round(float(sm_i), 4) if sm_i is not None else None,
                        "ci_lower":       round(float(pl_i), 4) if pl_i is not None and not (isinstance(pl_i, float) and math.isnan(pl_i)) else None,
                        "ci_upper":       round(float(pu_i), 4) if pu_i is not None and not (isinstance(pu_i, float) and math.isnan(pu_i)) else None,
                        "p_value":        round(float(pv_i), 4) if pv_i is not None and not (isinstance(pv_i, float) and math.isnan(pv_i)) else None,
                        "accept_h0":      bool(vb_i) if vb_i is not None else None,
                        "sampling_time_seconds": round(float(t_i), 2) if t_i is not None else None,
                    })

                avg_valid = round(sum(args.valid_edges) / len(args.valid_edges), 2) if args.valid_edges else 0.0
                avg_sample_mean = round(sum(args.sample_means) / len(args.sample_means), 4) if args.sample_means else None
                ci_lower  = round(sum(args.CI["lower"]) / len(args.CI["lower"]), 4) if args.CI["lower"] else None
                ci_upper  = round(sum(args.CI["upper"]) / len(args.CI["upper"]), 4) if args.CI["upper"] else None
                p_value   = round(sum(args.p_value) / len(args.p_value), 4) if args.p_value else None
                avg_sampling_time = round(sum(time_used_list["sampling"]) / len(time_used_list["sampling"]), 2)

                ratio_reports.append({
                    "sampling_percent":    pct,
                    "sampling_size":       ratio,
                    "num_samples":         num_samples,
                    "avg_valid_matches":   avg_valid,
                    "avg_sample_mean":      avg_sample_mean,
                    "hypothesis_results":  [bool(r) if r is not None else None for r in sample_bools],
                    "accuracy":            accuracy_out,
                    "confidence_interval": {"lower": ci_lower, "upper": ci_upper},
                    "avg_p_value":         p_value,
                    "avg_sampling_time_seconds": avg_sampling_time,
                    "elapsed_seconds":     elapsed,
                    "repeats":             repeats,
                })
                acc_str = "N/A" if accuracy_out is None else accuracy_out
                report(
                    f"  -> {pct}% done in {elapsed}s: accuracy={acc_str}, "
                    f"CI=({ci_lower}, {ci_upper}), p-value={p_value}."
                )

        total_time = round(time.time() - args.overall_time + gt_time + load_time, 2)
        report(f"Finished. Total time {total_time}s.")

        # ── 5. overall verdict (the primary hypothesis-test conclusion) ──────
        # In "sample" / "compare" mode the primary verdict comes from the
        # (first) sampling ratio, decided by MAJORITY VOTE over the per-repeat
        # accept/reject outcomes (each repeat's outcome is True=accept when
        # its p-value >= 0.05).  This stays consistent with the `accuracy`
        # metric, which also compares each repeat's verdict to the full-graph
        # verdict.  In "full" mode the primary verdict is the full-graph
        # t-test result.
        sampling_verdict_accept: bool | None = None
        sampling_verdict_p: float | None = None
        sampling_vote: dict[str, int] | None = None
        if ratio_reports:
            first = ratio_reports[0]
            sampling_verdict_p = first["avg_p_value"]
            # Majority vote over the per-repeat hypothesis_results list.
            # None entries (p-value NaN) are ignored.
            outcomes = [r for r in (first.get("hypothesis_results") or []) if r is not None]
            if outcomes:
                accept_count = sum(1 for r in outcomes if r)
                reject_count = len(outcomes) - accept_count
                sampling_verdict_accept = accept_count > len(outcomes) / 2
                sampling_vote = {
                    "accept": accept_count,
                    "reject": reject_count,
                    "total":  len(outcomes),
                }

        if mode == "full":
            primary_accept = ground_truth_result
            primary_p = gt_p_value
            primary_source = "full-graph"
            primary_vote = None
        else:
            primary_accept = sampling_verdict_accept
            primary_p = sampling_verdict_p
            primary_source = "sampling"
            primary_vote = sampling_vote

        full_graph_ref = None
        if ground_truth_result is not None:
            full_graph_ref = {
                "accept_h0": ground_truth_result,
                "p_value":   gt_p_value,
            }

        # Human-readable description of what is being tested — set for the
        # paired edge-difference mode where "sample mean" alone is ambiguous.
        tgt = entry.get("target") or {}
        test_description = None
        if "edge_diff" in tgt:
            (f1, t1), (f2, t2) = tgt["edge_diff"]["edges"]
            _attr = tgt["edge_diff"]["attribute"]
            test_description = (
                f"paired comparison: the mean difference of {_attr} between "
                f"edge {f1}->{t1} and edge {f2}->{t2} on each matched path "
                f"(sample_mean is the mean of the per-path differences)"
            )

        return {
            "dataset":         dataset_pretty,
            "pattern_type":    entry.get("type"),
            "hypo":            hypo,
            "mode":            mode,
            "sampling_method": sampling_method,
            "H0":              entry.get("H0"),
            "attribute":       attribute,
            "test_description": test_description,
            "comparison":      comparison,
            "test_scalar":     args.c,
            "is_mu_test":      scalar_c is None,
            "verdict": {
                "accept_h0":  primary_accept,
                "p_value":    primary_p,
                "source":     primary_source,
                "vote":       primary_vote,
                "full_graph": full_graph_ref,
            },
            "ground_truth": {
                "value":               ground_truth_value,
                "n_matches":           gt_n_matches,
                "accept_h0":           ground_truth_result,
                "p_value":             gt_p_value,
                "compute_time_seconds": gt_time,
            } if ground_truth_result is not None else None,
            "graph_stats": {
                "num_nodes":         args.num_nodes,
                "num_edges":         args.num_edges,
                "load_time_seconds": load_time,
            },
            "sampling_results":   ratio_reports,
            "total_time_seconds": total_time,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    _popmean_cache: dict[tuple, tuple[float, int]] = {}

    @staticmethod
    def _population_mean(entry: dict, graph: Any, dataset: str) -> tuple[float, int]:
        """
        Compute the **population** mean of the target attribute across ALL
        nodes (or edges) of the relevant type in the full graph — NOT the
        pattern-matched subset.

        When ``TEST ... = mu`` is used, ``mu`` refers to this population mean.
        For example, in ``TEST mean(n1.age) > mu`` where ``n1:user``, ``mu`` is
        the average age over **all** user nodes, not just the users matching the
        pattern.

        The target type (node vs edge) and its label are read from the parsed
        ``entry["target"]`` and ``entry["pattern"]`` rather than from the
        engine-facing ``args.attribute`` dict, because the latter's key
        (``"node"`` vs ``"edge"``) names the *extraction mode* getEdges/getPaths
        should use, not the target shape — for an edge-topology pattern with a
        node-attribute target, build_attribute_dict stores it under ``"node"``
        to select getEdges node mode, which would mislead this function if it
        read ``args.attribute`` instead of ``entry["target"]``.

        Returns (mean, count).
        """
        target  = entry["target"]
        pattern = entry["pattern"]
        lookup  = _node_lookup(pattern)

        kind: str
        label: str | None
        attr: str

        if "node" in target:
            kind  = "node"
            var_id = target["node"]["id"]
            attr   = target["node"]["attribute"]
            label  = lookup[var_id]["label"]
        elif "node_diff" in target:
            kind  = "node"
            var_ids = target["node_diff"]["nodes"]
            attr    = target["node_diff"]["attribute"]
            label   = lookup[var_ids[0]]["label"]
        elif "edge" in target:
            kind  = "edge"
            attr   = target["edge"]["attribute"]
            label  = None  # edge attributes are not segmented by label
        else:
            raise GHTExecutorError(
                "Cannot determine the target attribute for 'mu'; the pattern "
                "must have a 'node', 'node_diff', or 'edge' target."
            )

        cache_key = (dataset, kind, label, attr)
        cached = GHTExecutor._popmean_cache.get(cache_key)
        if cached is not None:
            return cached

        values: list[float] = []
        if kind == "node":
            for n in graph.nodes():
                if graph.nodes[n].get("label") != label:
                    continue
                v = graph.nodes[n].get(attr)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv):
                    continue
                values.append(fv)
        else:  # edge
            for e in graph.edges():
                v = graph.edges[e].get(attr)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fv):
                    continue
                values.append(fv)

        if not values:
            raise GHTExecutorError(
                f"No {kind} with attribute '{attr}' found in the full graph; "
                f"cannot compute population mean (mu)."
            )

        result = (sum(values) / len(values), len(values))
        GHTExecutor._popmean_cache[cache_key] = result
        return result

    @staticmethod
    def _extract(args: _Args, graph: Any, hypo: int) -> dict[str, list]:
        if hypo == 1:
            return getEdges(args, graph)
        if hypo == 2:
            return getNodes(args, graph, dimension=args.dimension)
        if hypo == 3:
            args.total_valid = 0
            args.total_minus_reverse = 0
            return getPaths(args, graph)
        raise GHTExecutorError(f"Unsupported hypo code {hypo}.")

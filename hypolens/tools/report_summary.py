"""Format hypothesis-test reports as context for LLM tools."""

from __future__ import annotations


def build_report_summary(report: dict) -> str:
    """Build a compact, human-readable text snapshot of a GHT execution report.

    Shared by result interpretation and follow-up recommendation so both LLM
    prompts receive the same computed statistical evidence.
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
        # In "full" mode the primary verdict IS the full-graph verdict, so a
        # sampling-vs-full-graph comparison would be meaningless there.
        if verdict_str is not None and report.get("mode") != "full":
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

    # ── execution time (full-graph exact vs. sampling) ──────────────────────
    # Mirrors the "Execution Time" row in the UI so the summary can comment on
    # the efficiency trade-off between exact and sampled execution.
    gt_time    = gt.get("compute_time_seconds")
    samp_total = sr.get("elapsed_seconds")
    samp_runs  = sr.get("num_samples")
    samp_avg   = sr.get("avg_sampling_time_seconds")
    if gt_time is not None:
        lines.append(f"Full-graph exact time: {gt_time}s")
    if samp_total is not None and samp_runs:
        per_run = (round(float(samp_avg), 2) if samp_avg is not None
                   else round(float(samp_total) / float(samp_runs), 2))
        lines.append(
            f"Sampling time:     {per_run}s/run "
            f"({samp_total}s total, {samp_runs} runs)"
        )
        if gt_time and per_run > 0:
            lines.append(
                f"Sampling speedup:  {round(float(gt_time) / per_run, 1)}x "
                f"faster than full-graph exact"
            )
    return "\n".join(str(x) for x in lines)

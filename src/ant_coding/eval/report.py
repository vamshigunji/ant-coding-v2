"""
Report generation for experiment results.

Produces markdown reports, JSON export, and CSV export for single experiments
and multi-experiment comparisons with 4-tier metric tables, failure breakdowns,
statistical significance markers, and recommendations.

Reference: docs/success-metrics.md, Sprint-6-Epic-1.md
"""

import csv
import io
import json
from dataclasses import asdict
from typing import Dict, List, Optional

from ant_coding.eval.comparison import ComparisonResult
from ant_coding.eval.metrics import ExperimentMetrics


def generate_markdown(
    metrics: ExperimentMetrics,
    architecture: str = "",
    model: str = "",
    memory_mode: str = "",
    token_breakdown: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """
    Generate a markdown report for a single experiment.

    Args:
        metrics: ExperimentMetrics with all 4 tiers populated.
        architecture: Orchestration pattern name (e.g. "3-agent-sequential").
        model: Model identifier (e.g. "claude-sonnet-4-20250514").
        memory_mode: Memory mode (e.g. "shared", "isolated", "hybrid").
        token_breakdown: Optional per-agent token breakdown from EventLogger.

    Returns:
        Markdown-formatted report string.
    """
    lines = []

    # Header
    lines.append(f"# Experiment Report: {metrics.experiment_id}")
    lines.append("")

    # Metadata
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Experiment ID | {metrics.experiment_id} |")
    if architecture:
        lines.append(f"| Architecture | {architecture} |")
    if model:
        lines.append(f"| Model | {model} |")
    if memory_mode:
        lines.append(f"| Memory Mode | {memory_mode} |")
    lines.append(f"| Total Tasks | {metrics.total_tasks} |")
    lines.append(f"| Successful | {metrics.successful_tasks} |")
    lines.append(f"| Failed | {metrics.failed_tasks} |")
    lines.append(f"| Avg Duration | {metrics.avg_duration:.1f}s |")
    lines.append("")

    # 4-tier metrics table
    lines.append("## Metrics Summary")
    lines.append("")
    lines.append("| Tier | Metric | Value |")
    lines.append("|------|--------|-------|")

    # Tier 1
    lines.append(f"| 1 — Primary | Pass Rate | {metrics.pass_rate:.1%} |")
    lines.append(f"| 1 — Primary | Cost/Resolution | {_fmt_cost(metrics.cost_per_resolution)} |")

    # Tier 2
    lines.append(f"| 2 — Efficiency | Useful Token Ratio | {metrics.useful_token_ratio:.1%} |")
    lines.append(f"| 2 — Efficiency | Overhead Ratio | {metrics.overhead_ratio:.1f}x |")
    lines.append(f"| 2 — Efficiency | Tokens/Resolution | {_fmt_tokens(metrics.tokens_per_resolution)} |")
    lines.append(f"| 2 — Efficiency | Total Tokens | {metrics.total_tokens:,} |")
    lines.append(f"| 2 — Efficiency | Total Cost | ${metrics.total_cost:.2f} |")

    # Tier 3
    lines.append(f"| 3 — Quality | Patch Quality (1-5) | {metrics.avg_patch_quality:.1f} |")
    lines.append(f"| 3 — Quality | Patch Size Ratio | {metrics.avg_patch_size_ratio:.2f} |")

    # Tier 4
    lines.append(f"| 4 — Robustness | Variance (CV) | {metrics.resolution_variance_cv:.3f} |")
    lines.append(f"| 4 — Robustness | Error Recovery | {metrics.error_recovery_rate:.1%} |")
    lines.append("")

    # Per-agent token breakdown
    if token_breakdown:
        lines.append("## Per-Agent Token Breakdown")
        lines.append("")
        lines.append("| Agent | Input Tokens | Output Tokens | Total |")
        lines.append("|-------|-------------|---------------|-------|")
        for agent, tokens in sorted(token_breakdown.items()):
            input_t = tokens.get("input_tokens", 0)
            output_t = tokens.get("output_tokens", 0)
            total_t = input_t + output_t
            lines.append(f"| {agent} | {input_t:,} | {output_t:,} | {total_t:,} |")
        lines.append("")

    # Failure category breakdown
    if metrics.failure_categories and any(v > 0 for v in metrics.failure_categories.values()):
        lines.append("## Failure Category Breakdown")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, count in sorted(metrics.failure_categories.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"| {cat} | {count} |")
        lines.append("")

    return "\n".join(lines)


def generate_comparison_markdown(
    all_metrics: List[ExperimentMetrics],
    comparisons: Optional[List[ComparisonResult]] = None,
) -> str:
    """
    Generate a comparison markdown report for multiple experiments.

    Includes a side-by-side comparison table with statistical significance
    markers, breakeven analysis, and recommendation.

    Args:
        all_metrics: List of ExperimentMetrics (one per experiment).
        comparisons: Optional list of pairwise ComparisonResult objects.

    Returns:
        Markdown-formatted comparison report string.
    """
    if not all_metrics:
        return "# Comparison Report\n\nNo experiments to compare.\n"

    if len(all_metrics) == 1:
        return generate_markdown(all_metrics[0])

    lines = []
    lines.append("# Experiment Comparison Report")
    lines.append("")

    exp_ids = [m.experiment_id for m in all_metrics]
    lines.append(f"**Experiments:** {', '.join(exp_ids)}")
    lines.append("")

    # Build p-value lookup from comparisons
    p_values = _build_p_value_lookup(comparisons) if comparisons else {}

    # Side-by-side comparison table
    lines.append("## Side-by-Side Comparison")
    lines.append("")

    # Header row
    header = "| Metric |"
    separator = "|--------|"
    for m in all_metrics:
        header += f" {m.experiment_id} |"
        separator += "-------|"
    if comparisons:
        header += " Significance |"
        separator += "-------------|"
    lines.append(header)
    lines.append(separator)

    # Metric rows
    _add_comparison_row(lines, "Pass Rate (%)", all_metrics,
                        lambda m: f"{m.pass_rate:.1%}",
                        p_values.get("pass_rate"), comparisons)
    _add_comparison_row(lines, "Cost/Resolution ($)", all_metrics,
                        lambda m: _fmt_cost(m.cost_per_resolution),
                        None, comparisons)
    _add_comparison_row(lines, "Useful Token Ratio", all_metrics,
                        lambda m: f"{m.useful_token_ratio:.1%}",
                        None, comparisons)
    _add_comparison_row(lines, "Overhead Ratio", all_metrics,
                        lambda m: f"{m.overhead_ratio:.1f}x",
                        None, comparisons)
    _add_comparison_row(lines, "Tokens/Resolution", all_metrics,
                        lambda m: _fmt_tokens(m.tokens_per_resolution),
                        None, comparisons)
    _add_comparison_row(lines, "Total Cost ($)", all_metrics,
                        lambda m: f"${m.total_cost:.2f}",
                        p_values.get("total_cost"), comparisons)
    _add_comparison_row(lines, "Total Tokens", all_metrics,
                        lambda m: f"{m.total_tokens:,}",
                        p_values.get("total_tokens"), comparisons)
    _add_comparison_row(lines, "Patch Quality (1-5)", all_metrics,
                        lambda m: f"{m.avg_patch_quality:.1f}",
                        None, comparisons)
    _add_comparison_row(lines, "Patch Size Ratio", all_metrics,
                        lambda m: f"{m.avg_patch_size_ratio:.2f}",
                        None, comparisons)
    _add_comparison_row(lines, "Variance (CV)", all_metrics,
                        lambda m: f"{m.resolution_variance_cv:.3f}",
                        None, comparisons)
    _add_comparison_row(lines, "Error Recovery (%)", all_metrics,
                        lambda m: f"{m.error_recovery_rate:.1%}",
                        None, comparisons)
    lines.append("")

    # Significance legend
    if comparisons:
        lines.append("*\\* p < 0.05, \\*\\* p < 0.01*")
        lines.append("")

    # Breakeven analysis
    breakevens = [c.breakeven for c in (comparisons or []) if c.breakeven]
    if breakevens:
        lines.append("## Breakeven Analysis")
        lines.append("")
        for be in breakevens:
            lines.append(f"- {be['interpretation']}")
        lines.append("")

    # Pairwise recommendations
    if comparisons:
        lines.append("## Pairwise Comparisons")
        lines.append("")
        for comp in comparisons:
            lines.append(f"### {comp.experiment_a_id} vs {comp.experiment_b_id}")
            lines.append("")

            # Effect sizes
            if comp.effect_sizes:
                for metric, d in comp.effect_sizes.items():
                    from ant_coding.eval.comparison import interpret_effect_size
                    lines.append(f"- **{metric}**: {interpret_effect_size(d)}")
                lines.append("")

            # CIs
            if comp.confidence_intervals:
                lines.append("| Metric | 95% CI (A - B) |")
                lines.append("|--------|---------------|")
                for metric, ci in comp.confidence_intervals.items():
                    lines.append(
                        f"| {metric} | [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |"
                    )
                lines.append("")

    return "\n".join(lines)


# ── JSON Export ──


def generate_json(metrics: ExperimentMetrics) -> str:
    """
    Export ExperimentMetrics as a JSON string.

    All 4 tiers of metrics are included. The output round-trips
    back to ExperimentMetrics via metrics_from_json().

    Args:
        metrics: ExperimentMetrics with all fields populated.

    Returns:
        Pretty-printed JSON string.
    """
    data = asdict(metrics)
    # Handle infinity values (not valid JSON)
    for key, value in data.items():
        if isinstance(value, float) and value == float("inf"):
            data[key] = "Infinity"
    return json.dumps(data, indent=2, default=str)


def metrics_from_json(json_str: str) -> ExperimentMetrics:
    """
    Reconstruct ExperimentMetrics from a JSON string.

    Args:
        json_str: JSON string produced by generate_json().

    Returns:
        ExperimentMetrics instance.
    """
    data = json.loads(json_str)
    # Restore infinity values
    for key, value in data.items():
        if value == "Infinity":
            data[key] = float("inf")
    return ExperimentMetrics(**data)


# ── CSV Export ──


# Column order for CSV export
_CSV_COLUMNS = [
    "experiment_id",
    "total_tasks",
    "successful_tasks",
    "failed_tasks",
    "pass_rate",
    "total_tokens",
    "total_cost",
    "avg_duration",
    "cost_per_resolution",
    "useful_token_ratio",
    "overhead_ratio",
    "tokens_per_resolution",
    "avg_patch_quality",
    "avg_patch_size_ratio",
    "resolution_variance_cv",
    "error_recovery_rate",
]


def generate_csv(metrics_list: List[ExperimentMetrics]) -> str:
    """
    Export a list of ExperimentMetrics as a CSV string.

    One row per experiment, columns include all 11 success metrics.
    Importable into pandas without errors.

    Args:
        metrics_list: List of ExperimentMetrics to export.

    Returns:
        CSV-formatted string with header row.
    """
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
    writer.writeheader()

    for metrics in metrics_list:
        data = asdict(metrics)
        # Filter to only CSV columns and handle infinity
        row = {}
        for col in _CSV_COLUMNS:
            value = data.get(col, "")
            if isinstance(value, float) and value == float("inf"):
                value = "Infinity"
            row[col] = value
        writer.writerow(row)

    return output.getvalue()


# ── Helpers ──


def _fmt_cost(value: float) -> str:
    """Format cost value, handling infinity."""
    if value == float("inf"):
        return "inf"
    return f"${value:.2f}"


def _fmt_tokens(value: float) -> str:
    """Format token count, handling infinity."""
    if value == float("inf"):
        return "inf"
    return f"{value:,.0f}"


def _significance_marker(p_value: Optional[float]) -> str:
    """Return significance marker for p-value."""
    if p_value is None:
        return "—"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _build_p_value_lookup(
    comparisons: List[ComparisonResult],
) -> Dict[str, Optional[float]]:
    """Build a lookup of the first comparison's p-values by metric."""
    if not comparisons:
        return {}

    result = {}
    first = comparisons[0]
    for metric, test_result in first.statistical_tests.items():
        result[metric] = test_result.get("p_value")
    return result


def _add_comparison_row(
    lines: List[str],
    metric_name: str,
    all_metrics: List[ExperimentMetrics],
    formatter,
    p_value: Optional[float],
    comparisons: Optional[List[ComparisonResult]],
) -> None:
    """Add a single row to the comparison table."""
    row = f"| {metric_name} |"
    for m in all_metrics:
        row += f" {formatter(m)} |"
    if comparisons:
        row += f" {_significance_marker(p_value)} |"
    lines.append(row)

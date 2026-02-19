"""
Statistical comparison between experiments.

Provides paired statistical tests (McNemar's for pass rates, Wilcoxon signed-rank
for continuous metrics), bootstrap confidence intervals, effect size interpretation,
and breakeven analysis for cost-effectiveness evaluation.

Reference: docs/prd-plus.md Section 7, docs/success-metrics.md
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.tasks.types import TaskResult


@dataclass
class ComparisonResult:
    """Result of comparing two experiments."""

    experiment_a_id: str
    experiment_b_id: str

    # Paired test results: metric_name -> {statistic, p_value, significant}
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Breakeven analysis (only when baseline is configured)
    breakeven: Optional[Dict[str, Any]] = None

    # Effect sizes: metric_name -> effect_size
    effect_sizes: Dict[str, float] = field(default_factory=dict)

    # Bootstrap confidence intervals: metric_name -> {ci_lower, ci_upper, point_estimate}
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)


def breakeven_analysis(
    single_agent_metrics: ExperimentMetrics,
    multi_agent_cost_per_task: float,
) -> Dict[str, Any]:
    """
    Calculate at what resolution rate multi-agent breaks even with single-agent.

    breakeven_rate = multi_agent_cost_per_task / single_agent_cost_per_resolution

    Args:
        single_agent_metrics: Metrics from the single-agent baseline experiment.
        multi_agent_cost_per_task: Average cost per task for the multi-agent experiment.

    Returns:
        Dict with breakeven analysis results.
    """
    sa_cpr = single_agent_metrics.cost_per_resolution

    if sa_cpr <= 0 or sa_cpr == float("inf"):
        breakeven = 1.0
    else:
        breakeven = multi_agent_cost_per_task / sa_cpr

    capped = min(breakeven, 1.0)

    return {
        "single_agent_cost_per_resolution": sa_cpr,
        "multi_agent_cost_per_task": multi_agent_cost_per_task,
        "breakeven_resolution_rate": capped,
        "interpretation": (
            f"Multi-agent needs >{capped:.0%} resolution rate "
            f"to beat single-agent on cost-per-resolution"
        ),
    }


def mcnemar_test(
    results_a: List[TaskResult],
    results_b: List[TaskResult],
) -> Dict[str, Any]:
    """
    McNemar's test for paired binary outcomes (pass/fail).

    Compares whether the two experiments have significantly different pass rates
    on the same set of tasks.

    Args:
        results_a: Results from experiment A (one per task).
        results_b: Results from experiment B (one per task, same task order).

    Returns:
        Dict with statistic, p_value, and significant flag.
    """
    # Build lookup by task_id
    a_by_task = {r.task_id: r.success for r in results_a}
    b_by_task = {r.task_id: r.success for r in results_b}

    # Find shared tasks
    shared_tasks = set(a_by_task.keys()) & set(b_by_task.keys())

    if not shared_tasks:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n_tasks": 0}

    # McNemar contingency: count discordant pairs
    # b = A passes, B fails; c = A fails, B passes
    b_count = 0  # A pass, B fail
    c_count = 0  # A fail, B pass

    for task_id in shared_tasks:
        a_pass = a_by_task[task_id]
        b_pass = b_by_task[task_id]
        if a_pass and not b_pass:
            b_count += 1
        elif not a_pass and b_pass:
            c_count += 1

    # McNemar's chi-squared statistic (with continuity correction)
    if b_count + c_count == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_tasks": len(shared_tasks),
        }

    statistic = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)

    # Approximate p-value using chi-squared distribution with 1 df
    p_value = _chi2_survival(statistic, df=1)

    return {
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_tasks": len(shared_tasks),
        "discordant_a_wins": b_count,
        "discordant_b_wins": c_count,
    }


def wilcoxon_signed_rank(
    values_a: List[float],
    values_b: List[float],
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired continuous values.

    Non-parametric test for whether paired observations differ systematically.

    Args:
        values_a: Metric values from experiment A (one per task).
        values_b: Metric values from experiment B (same task order).

    Returns:
        Dict with statistic, p_value, and significant flag.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")

    n = len(values_a)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n": 0}

    # Compute differences and remove zeros
    diffs = []
    for a, b in zip(values_a, values_b):
        d = a - b
        if d != 0.0:
            diffs.append(d)

    nr = len(diffs)  # number of non-zero differences
    if nr == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n": n}

    # Rank absolute differences
    abs_diffs = [(abs(d), i) for i, d in enumerate(diffs)]
    abs_diffs.sort(key=lambda x: x[0])

    # Assign ranks (handling ties with average rank)
    ranks = [0.0] * nr
    i = 0
    while i < nr:
        j = i
        while j < nr and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[abs_diffs[k][1]] = avg_rank
        i = j

    # Sum of ranks for positive and negative differences
    w_plus = sum(ranks[i] for i in range(nr) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(nr) if diffs[i] < 0)

    # Test statistic is the smaller of W+ and W-
    w = min(w_plus, w_minus)

    # Normal approximation for p-value (valid for nr >= 10, approximate for smaller)
    mean_w = nr * (nr + 1) / 4.0
    std_w = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0)

    if std_w == 0:
        return {"statistic": round(w, 4), "p_value": 1.0, "significant": False, "n": n}

    z = (w - mean_w) / std_w
    # Two-tailed p-value
    p_value = 2.0 * _normal_survival(abs(z))

    return {
        "statistic": round(w, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n": n,
        "n_nonzero": nr,
        "w_plus": round(w_plus, 4),
        "w_minus": round(w_minus, 4),
    }


def compute_effect_size(
    values_a: List[float],
    values_b: List[float],
) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        values_a: Values from experiment A.
        values_b: Values from experiment B.

    Returns:
        Cohen's d. Positive means A > B.
    """
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return 0.0

    diffs = [a - b for a, b in zip(values_a, values_b)]
    n = len(diffs)
    mean_diff = sum(diffs) / n

    if n < 2:
        return 0.0

    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std_diff = math.sqrt(variance)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def compare_experiments(
    results_a: List[TaskResult],
    results_b: List[TaskResult],
    metrics_a: ExperimentMetrics,
    metrics_b: ExperimentMetrics,
    baseline_metrics: Optional[ExperimentMetrics] = None,
) -> ComparisonResult:
    """
    Run full statistical comparison between two experiments.

    Includes McNemar's test for pass rates, Wilcoxon for continuous metrics,
    effect sizes, and optional breakeven analysis.

    Args:
        results_a: TaskResults from experiment A.
        results_b: TaskResults from experiment B.
        metrics_a: Aggregated metrics for experiment A.
        metrics_b: Aggregated metrics for experiment B.
        baseline_metrics: Optional single-agent baseline for breakeven.

    Returns:
        ComparisonResult with all statistical tests and analyses.
    """
    comparison = ComparisonResult(
        experiment_a_id=metrics_a.experiment_id,
        experiment_b_id=metrics_b.experiment_id,
    )

    # McNemar's test for pass rate difference
    comparison.statistical_tests["pass_rate"] = mcnemar_test(results_a, results_b)

    # Pair up tasks for continuous metric comparison
    a_by_task = {r.task_id: r for r in results_a}
    b_by_task = {r.task_id: r for r in results_b}
    shared_tasks = sorted(set(a_by_task.keys()) & set(b_by_task.keys()))

    if shared_tasks:
        # Cost comparison
        costs_a = [a_by_task[t].total_cost for t in shared_tasks]
        costs_b = [b_by_task[t].total_cost for t in shared_tasks]
        comparison.statistical_tests["total_cost"] = wilcoxon_signed_rank(costs_a, costs_b)
        comparison.effect_sizes["total_cost"] = compute_effect_size(costs_a, costs_b)

        # Token comparison
        tokens_a = [float(a_by_task[t].total_tokens) for t in shared_tasks]
        tokens_b = [float(b_by_task[t].total_tokens) for t in shared_tasks]
        comparison.statistical_tests["total_tokens"] = wilcoxon_signed_rank(tokens_a, tokens_b)
        comparison.effect_sizes["total_tokens"] = compute_effect_size(tokens_a, tokens_b)

        # Duration comparison
        durations_a = [a_by_task[t].duration_seconds for t in shared_tasks]
        durations_b = [b_by_task[t].duration_seconds for t in shared_tasks]
        comparison.statistical_tests["duration"] = wilcoxon_signed_rank(durations_a, durations_b)
        comparison.effect_sizes["duration"] = compute_effect_size(durations_a, durations_b)

        # Bootstrap CIs for paired differences
        comparison.confidence_intervals["total_cost"] = bootstrap_paired_ci(
            costs_a, costs_b, seed=42
        )
        comparison.confidence_intervals["total_tokens"] = bootstrap_paired_ci(
            tokens_a, tokens_b, seed=42
        )
        comparison.confidence_intervals["duration"] = bootstrap_paired_ci(
            durations_a, durations_b, seed=42
        )

    # Breakeven analysis (if baseline provided)
    if baseline_metrics:
        multi_agent_cost_per_task = (
            metrics_b.total_cost / metrics_b.total_tasks
            if metrics_b.total_tasks > 0
            else 0.0
        )
        comparison.breakeven = breakeven_analysis(baseline_metrics, multi_agent_cost_per_task)

    return comparison


def generate_comparison_report(
    metrics_a: ExperimentMetrics,
    metrics_b: ExperimentMetrics,
    comparison: ComparisonResult,
) -> str:
    """
    Generate a markdown comparison report between two experiments.

    Includes a metric comparison table, statistical test results,
    effect size interpretations, and an overall recommendation.

    Args:
        metrics_a: Metrics from experiment A.
        metrics_b: Metrics from experiment B.
        comparison: ComparisonResult from compare_experiments().

    Returns:
        Markdown-formatted report string.
    """
    lines = []
    lines.append(f"# Experiment Comparison Report")
    lines.append(f"")
    lines.append(f"**Experiment A:** {metrics_a.experiment_id}")
    lines.append(f"**Experiment B:** {metrics_b.experiment_id}")
    lines.append(f"")

    # Metric comparison table
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Exp A | Exp B | p-value | Significant |")
    lines.append("|--------|-------|-------|---------|-------------|")

    # Tier 1
    p_pass = comparison.statistical_tests.get("pass_rate", {})
    lines.append(
        f"| Pass Rate (%) | {metrics_a.pass_rate:.1%} | {metrics_b.pass_rate:.1%} "
        f"| {p_pass.get('p_value', '—')} | {'Yes' if p_pass.get('significant') else 'No'} |"
    )
    lines.append(
        f"| Cost/Resolution ($) | {_fmt_cost(metrics_a.cost_per_resolution)} "
        f"| {_fmt_cost(metrics_b.cost_per_resolution)} | — | — |"
    )

    # Tier 2
    p_tokens = comparison.statistical_tests.get("total_tokens", {})
    lines.append(
        f"| Useful Token Ratio | {metrics_a.useful_token_ratio:.1%} "
        f"| {metrics_b.useful_token_ratio:.1%} | — | — |"
    )
    lines.append(
        f"| Overhead Ratio | {metrics_a.overhead_ratio:.1f}x "
        f"| {metrics_b.overhead_ratio:.1f}x | — | — |"
    )
    lines.append(
        f"| Tokens/Resolution | {_fmt_tokens(metrics_a.tokens_per_resolution)} "
        f"| {_fmt_tokens(metrics_b.tokens_per_resolution)} | — | — |"
    )

    p_cost = comparison.statistical_tests.get("total_cost", {})
    lines.append(
        f"| Total Cost ($) | {metrics_a.total_cost:.2f} | {metrics_b.total_cost:.2f} "
        f"| {p_cost.get('p_value', '—')} | {'Yes' if p_cost.get('significant') else 'No'} |"
    )
    lines.append(
        f"| Total Tokens | {metrics_a.total_tokens:,} | {metrics_b.total_tokens:,} "
        f"| {p_tokens.get('p_value', '—')} | {'Yes' if p_tokens.get('significant') else 'No'} |"
    )

    # Tier 3
    lines.append(
        f"| Patch Quality (1-5) | {metrics_a.avg_patch_quality:.1f} "
        f"| {metrics_b.avg_patch_quality:.1f} | — | — |"
    )
    lines.append(
        f"| Patch Size Ratio | {metrics_a.avg_patch_size_ratio:.2f} "
        f"| {metrics_b.avg_patch_size_ratio:.2f} | — | — |"
    )

    # Tier 4
    lines.append(
        f"| Variance (CV) | {metrics_a.resolution_variance_cv:.3f} "
        f"| {metrics_b.resolution_variance_cv:.3f} | — | — |"
    )
    lines.append(
        f"| Error Recovery (%) | {metrics_a.error_recovery_rate:.1%} "
        f"| {metrics_b.error_recovery_rate:.1%} | — | — |"
    )
    lines.append("")

    # Effect sizes
    if comparison.effect_sizes:
        lines.append("## Effect Sizes")
        lines.append("")
        for metric, d in comparison.effect_sizes.items():
            lines.append(f"- **{metric}**: {interpret_effect_size(d)}")
        lines.append("")

    # Confidence intervals
    if comparison.confidence_intervals:
        lines.append("## Bootstrap 95% Confidence Intervals (A - B)")
        lines.append("")
        lines.append("| Metric | Point Estimate | 95% CI |")
        lines.append("|--------|---------------|--------|")
        for metric, ci in comparison.confidence_intervals.items():
            lines.append(
                f"| {metric} | {ci['point_estimate']:.4f} "
                f"| [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |"
            )
        lines.append("")

    # Breakeven analysis
    if comparison.breakeven:
        be = comparison.breakeven
        lines.append("## Breakeven Analysis")
        lines.append("")
        lines.append(f"- Single-agent cost/resolution: {_fmt_cost(be['single_agent_cost_per_resolution'])}")
        lines.append(f"- Multi-agent cost/task: ${be['multi_agent_cost_per_task']:.2f}")
        lines.append(f"- Breakeven resolution rate: {be['breakeven_resolution_rate']:.0%}")
        lines.append(f"- {be['interpretation']}")
        lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    recommendation = _generate_recommendation(metrics_a, metrics_b, comparison)
    lines.append(recommendation)
    lines.append("")

    return "\n".join(lines)


def _generate_recommendation(
    metrics_a: ExperimentMetrics,
    metrics_b: ExperimentMetrics,
    comparison: ComparisonResult,
) -> str:
    """Generate a recommendation based on comparison results."""
    advantages_b = []
    advantages_a = []

    # Pass rate
    if metrics_b.pass_rate > metrics_a.pass_rate:
        diff = metrics_b.pass_rate - metrics_a.pass_rate
        sig = comparison.statistical_tests.get("pass_rate", {}).get("significant", False)
        advantages_b.append(
            f"higher pass rate (+{diff:.1%})"
            + (" (statistically significant)" if sig else " (not significant)")
        )
    elif metrics_a.pass_rate > metrics_b.pass_rate:
        diff = metrics_a.pass_rate - metrics_b.pass_rate
        advantages_a.append(f"higher pass rate (+{diff:.1%})")

    # Cost per resolution
    if (
        metrics_b.cost_per_resolution < metrics_a.cost_per_resolution
        and metrics_b.cost_per_resolution != float("inf")
    ):
        advantages_b.append("lower cost per resolution")
    elif (
        metrics_a.cost_per_resolution < metrics_b.cost_per_resolution
        and metrics_a.cost_per_resolution != float("inf")
    ):
        advantages_a.append("lower cost per resolution")

    # Variance
    if metrics_b.resolution_variance_cv < metrics_a.resolution_variance_cv:
        advantages_b.append("more consistent (lower variance)")
    elif metrics_a.resolution_variance_cv < metrics_b.resolution_variance_cv:
        advantages_a.append("more consistent (lower variance)")

    # Patch quality
    if metrics_b.avg_patch_quality > metrics_a.avg_patch_quality:
        advantages_b.append("higher patch quality")
    elif metrics_a.avg_patch_quality > metrics_b.avg_patch_quality:
        advantages_a.append("higher patch quality")

    if len(advantages_b) > len(advantages_a):
        winner = f"**Experiment B ({metrics_b.experiment_id})** is recommended"
        reasons = ", ".join(advantages_b)
        return f"{winner}: {reasons}."
    elif len(advantages_a) > len(advantages_b):
        winner = f"**Experiment A ({metrics_a.experiment_id})** is recommended"
        reasons = ", ".join(advantages_a)
        return f"{winner}: {reasons}."
    else:
        return (
            "**No clear winner.** Both experiments show comparable performance. "
            "Consider running more tasks or increasing runs per task for stronger signals."
        )


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


# ── Bootstrap Confidence Intervals ──


def bootstrap_ci(
    values: List[float],
    statistic: Callable[[List[float]], float] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Sample values.
        statistic: Function that computes the statistic from a sample.
            Defaults to mean.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Optional random seed for reproducibility.

    Returns:
        Dict with ci_lower, ci_upper, and point_estimate.
    """
    if statistic is None:
        statistic = lambda v: sum(v) / len(v) if v else 0.0

    if not values:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "point_estimate": 0.0}

    rng = random.Random(seed)
    n = len(values)
    point_estimate = statistic(values)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats.sort()
    alpha = 1.0 - confidence
    lower_idx = max(0, int(math.floor(alpha / 2.0 * n_bootstrap)) - 1)
    upper_idx = min(n_bootstrap - 1, int(math.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1)

    return {
        "ci_lower": round(bootstrap_stats[lower_idx], 6),
        "ci_upper": round(bootstrap_stats[upper_idx], 6),
        "point_estimate": round(point_estimate, 6),
    }


def bootstrap_paired_ci(
    values_a: List[float],
    values_b: List[float],
    statistic: Callable[[List[float]], float] = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute bootstrap CI for the difference in a statistic between paired samples.

    Args:
        values_a: Values from experiment A.
        values_b: Values from experiment B (same task order).
        statistic: Function to compute statistic. Defaults to mean.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Optional random seed.

    Returns:
        Dict with ci_lower, ci_upper for the difference (A - B),
        and point_estimate.
    """
    if len(values_a) != len(values_b):
        raise ValueError("values_a and values_b must have the same length")

    diffs = [a - b for a, b in zip(values_a, values_b)]
    return bootstrap_ci(diffs, statistic=statistic, n_bootstrap=n_bootstrap,
                        confidence=confidence, seed=seed)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size using standard thresholds.

    Args:
        d: Cohen's d value.

    Returns:
        Human-readable interpretation string.
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    if d > 0:
        direction = "A > B"
    elif d < 0:
        direction = "B > A"
    else:
        direction = "no difference"

    return f"{magnitude} ({direction}, d={d:.3f})"


# ── Internal math helpers (no scipy dependency) ──


def _normal_survival(z: float) -> float:
    """
    Approximate P(Z > z) for standard normal using Abramowitz & Stegun.

    Accurate to ~1e-5 for all z.
    """
    if z < 0:
        return 1.0 - _normal_survival(-z)

    # Abramowitz & Stegun approximation 26.2.17
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + p * z)
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    survival = phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    return max(0.0, min(1.0, survival))


def _chi2_survival(x: float, df: int = 1) -> float:
    """
    Approximate P(X > x) for chi-squared distribution.

    For df=1, chi2 is the square of a standard normal, so
    P(chi2 > x) = 2 * P(Z > sqrt(x)).
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return 2.0 * _normal_survival(math.sqrt(x))
    # For other df, use Wilson-Hilferty normal approximation
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return _normal_survival(z)

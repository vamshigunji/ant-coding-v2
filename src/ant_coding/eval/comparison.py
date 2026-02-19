"""
Statistical comparison between experiments.

Provides paired statistical tests (McNemar's for pass rates, Wilcoxon signed-rank
for continuous metrics) and breakeven analysis for cost-effectiveness evaluation.

Reference: docs/prd-plus.md Section 7, docs/success-metrics.md
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    # Breakeven analysis (if baseline provided)
    if baseline_metrics:
        multi_agent_cost_per_task = (
            metrics_b.total_cost / metrics_b.total_tasks
            if metrics_b.total_tasks > 0
            else 0.0
        )
        comparison.breakeven = breakeven_analysis(baseline_metrics, multi_agent_cost_per_task)

    return comparison


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

"""
Evaluation harness: computes all 4 tiers of PRD+ metrics from TaskResults.

Tier 1 — Primary: pass_rate, cost_per_resolution
Tier 2 — Efficiency: useful_token_ratio, overhead_ratio, tokens_per_resolution
Tier 3 — Quality: avg_patch_quality, avg_patch_size_ratio
Tier 4 — Robustness: resolution_variance_cv, error_recovery_rate, failure_categories

Also includes pass@k computation using the unbiased estimator.
"""

import math
from typing import Any, Dict, List, Optional

from ant_coding.eval.metrics import ExperimentMetrics, _default_failure_categories
from ant_coding.tasks.types import TaskResult


def _comb(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k). Returns 0 if k > n or k < 0."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k(results: List[TaskResult], k: int) -> float:
    """
    Compute the unbiased pass@k estimator averaged across tasks.

    For each task, given n total runs and c correct runs:
        pass@k = 1 - C(n-c, k) / C(n, k)

    This is the standard unbiased estimator from the Codex paper.

    Args:
        results: List of TaskResult objects (may include multiple runs per task).
        k: Number of attempts allowed.

    Returns:
        Average pass@k across all tasks. 0.0 if no results.
    """
    # Group by task_id
    task_runs: Dict[str, List[bool]] = {}
    for r in results:
        task_runs.setdefault(r.task_id, []).append(r.success)

    if not task_runs:
        return 0.0

    scores = []
    for task_id, successes in task_runs.items():
        n = len(successes)
        c = sum(1 for s in successes if s)

        if k > n:
            # If k > n, use min(k, n) — can't sample more than available
            effective_k = n
        else:
            effective_k = k

        denom = _comb(n, effective_k)
        if denom == 0:
            scores.append(0.0)
            continue

        numer = _comb(n - c, effective_k)
        scores.append(1.0 - numer / denom)

    return sum(scores) / len(scores)


def calculate_metrics(
    results: List[TaskResult],
    experiment_id: str,
    baseline_tokens: Optional[int] = None,
) -> ExperimentMetrics:
    """
    Compute all 4 tiers of PRD+ metrics from task results.

    Args:
        results: List of TaskResult objects from an experiment run.
        experiment_id: Unique identifier for this experiment.
        baseline_tokens: Total tokens from the baseline experiment
            (for overhead_ratio). None if no baseline configured.

    Returns:
        ExperimentMetrics with all 11 PRD+ metrics calculated.
    """
    total = len(results)
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    num_successful = len(successful)
    num_failed = len(failed)

    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.total_cost for r in results)
    avg_duration = (
        sum(r.duration_seconds for r in results) / total if total > 0 else 0.0
    )

    # ── Tier 1: Primary ──
    pass_rate = num_successful / total if total > 0 else 0.0
    cost_per_resolution = (
        total_cost / num_successful if num_successful > 0 else float("inf")
    )

    # ── Tier 2: Efficiency ──
    successful_tokens = sum(r.total_tokens for r in successful)
    useful_token_ratio = (
        successful_tokens / total_tokens if total_tokens > 0 else 0.0
    )

    overhead_ratio = 0.0
    if baseline_tokens is not None and baseline_tokens > 0:
        overhead_ratio = total_tokens / baseline_tokens

    tokens_per_resolution = (
        total_tokens / num_successful if num_successful > 0 else float("inf")
    )

    # ── Tier 3: Quality ──
    avg_patch_quality = _compute_avg_patch_quality(results)
    avg_patch_size_ratio = _compute_avg_patch_size_ratio(results)

    # ── Tier 4: Robustness ──
    resolution_variance_cv = _compute_resolution_variance_cv(results)
    error_recovery_rate = _compute_error_recovery_rate(results)
    failure_cats = _compute_failure_categories(results)

    return ExperimentMetrics(
        experiment_id=experiment_id,
        total_tasks=total,
        successful_tasks=num_successful,
        failed_tasks=num_failed,
        pass_rate=pass_rate,
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_duration=avg_duration,
        cost_per_resolution=cost_per_resolution,
        useful_token_ratio=useful_token_ratio,
        overhead_ratio=overhead_ratio,
        tokens_per_resolution=tokens_per_resolution,
        avg_patch_quality=avg_patch_quality,
        avg_patch_size_ratio=avg_patch_size_ratio,
        resolution_variance_cv=resolution_variance_cv,
        error_recovery_rate=error_recovery_rate,
        failure_categories=failure_cats,
    )


def _compute_avg_patch_quality(results: List[TaskResult]) -> float:
    """
    Average the 'overall' judge score across results that have judge_scores.

    Args:
        results: List of TaskResult objects.

    Returns:
        Average overall judge score, or 0.0 if no results have scores.
    """
    scores = []
    for r in results:
        if r.judge_scores and "overall" in r.judge_scores:
            scores.append(float(r.judge_scores["overall"]))
    return sum(scores) / len(scores) if scores else 0.0


def _compute_avg_patch_size_ratio(results: List[TaskResult]) -> float:
    """
    Average generated_patch_lines / gold_patch_lines for results with gold patches.

    Args:
        results: List of TaskResult objects.

    Returns:
        Average patch size ratio, or 0.0 if no results have gold patches.
    """
    ratios = []
    for r in results:
        if r.gold_patch_lines > 0:
            ratios.append(r.generated_patch_lines / r.gold_patch_lines)
    return sum(ratios) / len(ratios) if ratios else 0.0


def _compute_resolution_variance_cv(results: List[TaskResult]) -> float:
    """
    Compute the coefficient of variation of per-task pass rates.

    Groups results by task_id, computes per-task pass rate, then
    returns stdev/mean (CV) of those rates.

    Args:
        results: List of TaskResult objects (may include multiple runs per task).

    Returns:
        Coefficient of variation. 0.0 if insufficient data.
    """
    # Group by task_id
    task_runs: Dict[str, List[bool]] = {}
    for r in results:
        task_runs.setdefault(r.task_id, []).append(r.success)

    if not task_runs:
        return 0.0

    # Compute per-task pass rates
    rates = []
    for task_id, successes in task_runs.items():
        rate = sum(1 for s in successes if s) / len(successes)
        rates.append(rate)

    if len(rates) < 2:
        return 0.0

    mean = sum(rates) / len(rates)
    if mean == 0.0:
        return 0.0

    variance = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    stdev = math.sqrt(variance)
    return stdev / mean


def _compute_error_recovery_rate(results: List[TaskResult]) -> float:
    """
    Compute the error recovery rate from intermediate_test_results.

    Counts tasks that initially failed (first intermediate result is False)
    and then recovered (any subsequent result is True).

    Args:
        results: List of TaskResult objects.

    Returns:
        Fraction of initially-failed tasks that recovered. 0.0 if none failed initially.
    """
    initially_failed = 0
    recovered = 0

    for r in results:
        if not r.intermediate_test_results:
            continue
        if r.intermediate_test_results[0] is False:
            initially_failed += 1
            if any(r.intermediate_test_results[1:]):
                recovered += 1

    return recovered / initially_failed if initially_failed > 0 else 0.0


def _compute_failure_categories(results: List[TaskResult]) -> Dict[str, int]:
    """
    Aggregate failure_category counts from failed results.

    Args:
        results: List of TaskResult objects.

    Returns:
        Dict with counts per failure category.
    """
    cats = _default_failure_categories()
    for r in results:
        if not r.success and r.failure_category and r.failure_category in cats:
            cats[r.failure_category] += 1
    return cats

"""
Tests for statistical comparison: McNemar's, Wilcoxon, bootstrap CI,
effect size, breakeven analysis, and report generation.
"""


import pytest

from ant_coding.eval.comparison import (
    ComparisonResult,
    bootstrap_ci,
    bootstrap_paired_ci,
    breakeven_analysis,
    compare_experiments,
    compute_effect_size,
    generate_comparison_report,
    interpret_effect_size,
    mcnemar_test,
    wilcoxon_signed_rank,
)
from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.tasks.types import TaskResult


# ── Helpers ──


def _make_result(
    task_id: str = "t1",
    success: bool = True,
    total_tokens: int = 100,
    total_cost: float = 0.01,
    duration_seconds: float = 5.0,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        experiment_id="exp-1",
        success=success,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_seconds=duration_seconds,
        intermediate_test_results=[],
        agent_traces=[],
    )


def _make_metrics(
    experiment_id: str = "exp-1",
    pass_rate: float = 0.5,
    cost_per_resolution: float = 1.0,
    total_cost: float = 5.0,
    total_tokens: int = 1000,
    total_tasks: int = 10,
    successful_tasks: int = 5,
    failed_tasks: int = 5,
) -> ExperimentMetrics:
    return ExperimentMetrics(
        experiment_id=experiment_id,
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        pass_rate=pass_rate,
        total_tokens=total_tokens,
        total_cost=total_cost,
        cost_per_resolution=cost_per_resolution,
    )


# ── McNemar's Test ──


def test_mcnemar_identical_results():
    """No discordant pairs → p=1.0, not significant."""
    results_a = [_make_result(f"t{i}", success=True) for i in range(5)]
    results_b = [_make_result(f"t{i}", success=True) for i in range(5)]
    result = mcnemar_test(results_a, results_b)
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_mcnemar_all_discordant():
    """All tasks discordant in one direction."""
    results_a = [_make_result(f"t{i}", success=True) for i in range(10)]
    results_b = [_make_result(f"t{i}", success=False) for i in range(10)]
    result = mcnemar_test(results_a, results_b)
    assert result["p_value"] < 0.05
    assert result["significant"] is True
    assert result["discordant_a_wins"] == 10
    assert result["discordant_b_wins"] == 0


def test_mcnemar_no_shared_tasks():
    """No shared tasks → p=1.0."""
    results_a = [_make_result("t1", success=True)]
    results_b = [_make_result("t2", success=False)]
    result = mcnemar_test(results_a, results_b)
    assert result["p_value"] == 1.0
    assert result["n_tasks"] == 0


def test_mcnemar_symmetric_discordance():
    """Equal discordant pairs → not significant."""
    results_a = [
        _make_result("t1", success=True),
        _make_result("t2", success=False),
        _make_result("t3", success=True),
        _make_result("t4", success=False),
    ]
    results_b = [
        _make_result("t1", success=False),
        _make_result("t2", success=True),
        _make_result("t3", success=True),
        _make_result("t4", success=False),
    ]
    result = mcnemar_test(results_a, results_b)
    # b_count=1 (t1: A pass, B fail), c_count=1 (t2: A fail, B pass)
    assert result["significant"] is False


# ── Wilcoxon Signed-Rank ──


def test_wilcoxon_identical():
    """Identical values → p=1.0."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = wilcoxon_signed_rank(vals, vals)
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_wilcoxon_clear_difference():
    """Consistently larger values → significant."""
    vals_a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    vals_b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = wilcoxon_signed_rank(vals_a, vals_b)
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_wilcoxon_empty():
    """Empty values → p=1.0."""
    result = wilcoxon_signed_rank([], [])
    assert result["p_value"] == 1.0
    assert result["n"] == 0


def test_wilcoxon_mismatched_length():
    """Mismatched lengths → ValueError."""
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0], [1.0, 2.0])


# ── Effect Size ──


def test_effect_size_zero_difference():
    """Identical values → d=0."""
    d = compute_effect_size([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert d == 0.0


def test_effect_size_large():
    """Large consistent difference → large d."""
    d = compute_effect_size([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    assert abs(d) > 0.8  # large effect


def test_effect_size_empty():
    """Empty → 0.0."""
    assert compute_effect_size([], []) == 0.0


def test_interpret_effect_size_negligible():
    assert "negligible" in interpret_effect_size(0.1)


def test_interpret_effect_size_small():
    assert "small" in interpret_effect_size(0.3)


def test_interpret_effect_size_medium():
    assert "medium" in interpret_effect_size(0.6)


def test_interpret_effect_size_large():
    assert "large" in interpret_effect_size(1.0)


def test_interpret_effect_size_direction():
    assert "A > B" in interpret_effect_size(0.5)
    assert "B > A" in interpret_effect_size(-0.5)


# ── Breakeven Analysis ──


def test_breakeven_basic():
    """Standard breakeven calculation."""
    baseline = _make_metrics(cost_per_resolution=0.50)
    result = breakeven_analysis(baseline, multi_agent_cost_per_task=0.40)
    assert result["breakeven_resolution_rate"] == 0.40 / 0.50  # 0.8
    assert "80%" in result["interpretation"]


def test_breakeven_infinite_baseline():
    """Baseline with inf cost_per_resolution → rate=1.0."""
    baseline = _make_metrics(cost_per_resolution=float("inf"))
    result = breakeven_analysis(baseline, multi_agent_cost_per_task=0.40)
    assert result["breakeven_resolution_rate"] == 1.0


def test_breakeven_zero_baseline():
    """Baseline with 0 cost_per_resolution → rate=1.0."""
    baseline = _make_metrics(cost_per_resolution=0.0)
    result = breakeven_analysis(baseline, multi_agent_cost_per_task=0.40)
    assert result["breakeven_resolution_rate"] == 1.0


# ── Bootstrap CI ──


def test_bootstrap_ci_deterministic():
    """With seed, results are reproducible."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci1 = bootstrap_ci(vals, seed=42)
    ci2 = bootstrap_ci(vals, seed=42)
    assert ci1 == ci2


def test_bootstrap_ci_contains_mean():
    """CI should contain the sample mean."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ci(vals, seed=42)
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]


def test_bootstrap_ci_empty():
    """Empty input → all zeros."""
    ci = bootstrap_ci([])
    assert ci["point_estimate"] == 0.0


def test_bootstrap_paired_ci():
    """Paired CI for known difference."""
    vals_a = [10.0, 20.0, 30.0, 40.0, 50.0]
    vals_b = [5.0, 10.0, 15.0, 20.0, 25.0]
    ci = bootstrap_paired_ci(vals_a, vals_b, seed=42)
    # Diffs are [5, 10, 15, 20, 25], mean=15
    assert ci["point_estimate"] == 15.0
    assert ci["ci_lower"] > 0  # clearly positive difference


def test_bootstrap_paired_ci_mismatched():
    """Mismatched lengths → ValueError."""
    with pytest.raises(ValueError):
        bootstrap_paired_ci([1.0], [1.0, 2.0])


# ── compare_experiments ──


def test_compare_experiments_basic():
    """Full comparison returns structured result."""
    results_a = [
        _make_result("t1", success=True, total_cost=0.1),
        _make_result("t2", success=False, total_cost=0.2),
        _make_result("t3", success=True, total_cost=0.15),
    ]
    results_b = [
        _make_result("t1", success=True, total_cost=0.3),
        _make_result("t2", success=True, total_cost=0.4),
        _make_result("t3", success=False, total_cost=0.25),
    ]
    metrics_a = _make_metrics("exp-a")
    metrics_b = _make_metrics("exp-b")

    result = compare_experiments(results_a, results_b, metrics_a, metrics_b)
    assert result.experiment_a_id == "exp-a"
    assert result.experiment_b_id == "exp-b"
    assert "pass_rate" in result.statistical_tests
    assert "total_cost" in result.statistical_tests
    assert "total_tokens" in result.statistical_tests
    assert "total_cost" in result.effect_sizes
    assert "total_cost" in result.confidence_intervals


def test_compare_experiments_with_breakeven():
    """Breakeven analysis included when baseline provided."""
    results_a = [_make_result("t1")]
    results_b = [_make_result("t1")]
    metrics_a = _make_metrics("exp-a")
    metrics_b = _make_metrics("exp-b", total_cost=5.0, total_tasks=10)
    baseline = _make_metrics("baseline", cost_per_resolution=0.50)

    result = compare_experiments(results_a, results_b, metrics_a, metrics_b, baseline)
    assert result.breakeven is not None
    assert "breakeven_resolution_rate" in result.breakeven


# ── Report Generation ──


def test_report_contains_sections():
    """Report has all expected sections."""
    metrics_a = _make_metrics("single-agent", pass_rate=0.3, cost_per_resolution=0.50)
    metrics_b = _make_metrics("multi-agent", pass_rate=0.5, cost_per_resolution=0.73)

    comparison = ComparisonResult(
        experiment_a_id="single-agent",
        experiment_b_id="multi-agent",
        statistical_tests={
            "pass_rate": {"p_value": 0.03, "significant": True},
        },
        effect_sizes={"total_cost": 0.5},
        confidence_intervals={
            "total_cost": {"ci_lower": -0.1, "ci_upper": 0.3, "point_estimate": 0.1},
        },
    )

    report = generate_comparison_report(metrics_a, metrics_b, comparison)
    assert "# Experiment Comparison Report" in report
    assert "## Metric Summary" in report
    assert "## Effect Sizes" in report
    assert "## Bootstrap 95% Confidence Intervals" in report
    assert "## Recommendation" in report
    assert "single-agent" in report
    assert "multi-agent" in report


def test_report_with_breakeven():
    """Report includes breakeven section when present."""
    metrics_a = _make_metrics("exp-a")
    metrics_b = _make_metrics("exp-b")

    comparison = ComparisonResult(
        experiment_a_id="exp-a",
        experiment_b_id="exp-b",
        breakeven={
            "single_agent_cost_per_resolution": 0.50,
            "multi_agent_cost_per_task": 0.40,
            "breakeven_resolution_rate": 0.80,
            "interpretation": "Multi-agent needs >80% resolution rate",
        },
    )

    report = generate_comparison_report(metrics_a, metrics_b, comparison)
    assert "## Breakeven Analysis" in report
    assert "80%" in report


def test_report_recommendation_b_wins():
    """B has more advantages → B recommended."""
    metrics_a = _make_metrics("exp-a", pass_rate=0.3)
    metrics_a.avg_patch_quality = 3.0
    metrics_b = _make_metrics("exp-b", pass_rate=0.5)
    metrics_b.avg_patch_quality = 4.0

    comparison = ComparisonResult(
        experiment_a_id="exp-a",
        experiment_b_id="exp-b",
        statistical_tests={
            "pass_rate": {"p_value": 0.03, "significant": True},
        },
    )

    report = generate_comparison_report(metrics_a, metrics_b, comparison)
    assert "Experiment B" in report
    assert "recommended" in report.lower()

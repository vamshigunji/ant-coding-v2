"""
Tests for report generation: markdown, JSON, CSV.
"""

import csv
import io
import json

import pytest

from ant_coding.eval.comparison import ComparisonResult
from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.eval.report import (
    generate_comparison_markdown,
    generate_csv,
    generate_json,
    generate_markdown,
    metrics_from_json,
)


def _make_metrics(
    experiment_id: str = "exp-1",
    pass_rate: float = 0.6,
    total_tasks: int = 10,
    successful_tasks: int = 6,
    failed_tasks: int = 4,
    total_cost: float = 3.50,
    total_tokens: int = 50000,
    cost_per_resolution: float = 0.58,
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
        useful_token_ratio=0.7,
        overhead_ratio=1.5,
        tokens_per_resolution=8333.0,
        avg_patch_quality=3.8,
        avg_patch_size_ratio=1.1,
        resolution_variance_cv=0.2,
        error_recovery_rate=0.33,
        failure_categories={"planning": 2, "implementation": 1, "integration": 0,
                           "hallucination_cascade": 0, "timeout": 1, "tool_failure": 0},
    )


# ── Markdown ──


def test_markdown_single_has_tiers():
    """Single experiment markdown includes all 4 tiers."""
    m = _make_metrics()
    report = generate_markdown(m, architecture="sequential", model="claude", memory_mode="shared")
    assert "# Experiment Report" in report
    assert "1 — Primary" in report
    assert "2 — Efficiency" in report
    assert "3 — Quality" in report
    assert "4 — Robustness" in report
    assert "sequential" in report
    assert "claude" in report


def test_markdown_single_failure_categories():
    """Failure categories shown when non-zero."""
    m = _make_metrics()
    report = generate_markdown(m)
    assert "Failure Category Breakdown" in report
    assert "planning" in report
    assert "timeout" in report


def test_markdown_single_token_breakdown():
    """Per-agent token breakdown included when provided."""
    m = _make_metrics()
    breakdown = {"planner": {"input_tokens": 1000, "output_tokens": 500}}
    report = generate_markdown(m, token_breakdown=breakdown)
    assert "Per-Agent Token Breakdown" in report
    assert "planner" in report


def test_markdown_comparison_multiple():
    """Comparison markdown has side-by-side table."""
    m1 = _make_metrics("exp-a", pass_rate=0.3)
    m2 = _make_metrics("exp-b", pass_rate=0.5)
    report = generate_comparison_markdown([m1, m2])
    assert "# Experiment Comparison Report" in report
    assert "Side-by-Side Comparison" in report
    assert "exp-a" in report
    assert "exp-b" in report


def test_markdown_comparison_with_significance():
    """Comparison includes significance markers when comparisons provided."""
    m1 = _make_metrics("exp-a")
    m2 = _make_metrics("exp-b")
    comp = ComparisonResult(
        experiment_a_id="exp-a",
        experiment_b_id="exp-b",
        statistical_tests={"pass_rate": {"p_value": 0.03, "significant": True}},
    )
    report = generate_comparison_markdown([m1, m2], [comp])
    assert "Significance" in report
    assert "p < 0.05" in report


def test_markdown_comparison_single_fallback():
    """Single experiment comparison falls back to single report."""
    m = _make_metrics()
    report = generate_comparison_markdown([m])
    assert "# Experiment Report" in report


# ── JSON ──


def test_json_roundtrip():
    """JSON export round-trips back to ExperimentMetrics."""
    m = _make_metrics()
    j = generate_json(m)
    m2 = metrics_from_json(j)
    assert m2.experiment_id == m.experiment_id
    assert m2.pass_rate == m.pass_rate
    assert m2.cost_per_resolution == m.cost_per_resolution
    assert m2.useful_token_ratio == m.useful_token_ratio
    assert m2.avg_patch_quality == m.avg_patch_quality
    assert m2.failure_categories == m.failure_categories


def test_json_infinity_handling():
    """Infinity values round-trip correctly."""
    m = _make_metrics(cost_per_resolution=float("inf"))
    j = generate_json(m)
    assert '"Infinity"' in j
    m2 = metrics_from_json(j)
    assert m2.cost_per_resolution == float("inf")


def test_json_valid_format():
    """Output is valid JSON."""
    m = _make_metrics()
    j = generate_json(m)
    parsed = json.loads(j)
    assert parsed["experiment_id"] == "exp-1"
    assert parsed["pass_rate"] == 0.6


# ── CSV ──


def test_csv_header_and_rows():
    """CSV has header and one row per experiment."""
    m1 = _make_metrics("exp-a")
    m2 = _make_metrics("exp-b")
    csv_str = generate_csv([m1, m2])
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["experiment_id"] == "exp-a"
    assert rows[1]["experiment_id"] == "exp-b"


def test_csv_has_all_metrics():
    """CSV includes all 11 success metrics columns."""
    m = _make_metrics()
    csv_str = generate_csv([m])
    reader = csv.DictReader(io.StringIO(csv_str))
    fields = reader.fieldnames
    assert "pass_rate" in fields
    assert "cost_per_resolution" in fields
    assert "useful_token_ratio" in fields
    assert "avg_patch_quality" in fields
    assert "resolution_variance_cv" in fields
    assert "error_recovery_rate" in fields


def test_csv_infinity():
    """Infinity values appear as 'Infinity' string in CSV."""
    m = _make_metrics(cost_per_resolution=float("inf"))
    csv_str = generate_csv([m])
    assert "Infinity" in csv_str

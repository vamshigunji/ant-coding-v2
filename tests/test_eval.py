"""
Comprehensive tests for the evaluation layer: metrics, LLM judge, pass@k, failure classifier.
"""

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_coding.eval.failure_classifier import FailureClassifier
from ant_coding.eval.harness import calculate_metrics, pass_at_k
from ant_coding.eval.llm_judge import LLMJudge, _default_scores
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.tasks.types import TaskResult


# ── Helpers ──


def _make_result(
    task_id: str = "t1",
    success: bool = True,
    total_tokens: int = 100,
    total_cost: float = 0.01,
    duration_seconds: float = 5.0,
    failure_category: str = None,
    intermediate_test_results: list = None,
    judge_scores: dict = None,
    generated_patch_lines: int = 0,
    gold_patch_lines: int = 0,
    error: str = None,
    agent_traces: list = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        experiment_id="exp-1",
        success=success,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_seconds=duration_seconds,
        failure_category=failure_category,
        intermediate_test_results=intermediate_test_results or [],
        judge_scores=judge_scores,
        generated_patch_lines=generated_patch_lines,
        gold_patch_lines=gold_patch_lines,
        error=error,
        agent_traces=agent_traces or [],
    )


# ── Tier 1: Primary Metrics ──


def test_tier1_pass_rate():
    """pass_rate = successful / total."""
    results = [
        _make_result("t1", success=True),
        _make_result("t2", success=True),
        _make_result("t3", success=False),
        _make_result("t4", success=True),
        _make_result("t5", success=False),
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.pass_rate == 0.6
    assert metrics.successful_tasks == 3
    assert metrics.failed_tasks == 2


def test_tier1_cost_per_resolution():
    """cost_per_resolution = total_cost / successful_tasks."""
    results = [
        _make_result("t1", success=True, total_cost=1.0),
        _make_result("t2", success=True, total_cost=2.0),
        _make_result("t3", success=False, total_cost=0.5),
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.cost_per_resolution == 3.5 / 2  # 1.75


def test_tier1_cost_per_resolution_zero_pass():
    """cost_per_resolution = inf when 0 tasks pass."""
    results = [_make_result("t1", success=False)]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.cost_per_resolution == float("inf")


# ── Tier 2: Efficiency Metrics ──


def test_tier2_useful_token_ratio():
    """useful_token_ratio = tokens from successful / total tokens."""
    results = [
        _make_result("t1", success=True, total_tokens=400),
        _make_result("t2", success=False, total_tokens=200),
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert abs(metrics.useful_token_ratio - 400 / 600) < 0.001


def test_tier2_overhead_ratio_with_baseline():
    """overhead_ratio = total_tokens / baseline_tokens."""
    results = [_make_result("t1", total_tokens=600)]
    metrics = calculate_metrics(results, "exp-1", baseline_tokens=300)
    assert metrics.overhead_ratio == 2.0


def test_tier2_overhead_ratio_no_baseline():
    """overhead_ratio = 0.0 when no baseline."""
    results = [_make_result("t1", total_tokens=600)]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.overhead_ratio == 0.0


def test_tier2_tokens_per_resolution():
    """tokens_per_resolution = total_tokens / successful_tasks."""
    results = [
        _make_result("t1", success=True, total_tokens=300),
        _make_result("t2", success=True, total_tokens=200),
        _make_result("t3", success=False, total_tokens=100),
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.tokens_per_resolution == 600 / 2  # 300.0


# ── Tier 3: Quality Metrics ──


def test_tier3_avg_patch_quality():
    """avg_patch_quality = mean of judge_scores['overall']."""
    results = [
        _make_result("t1", judge_scores={"overall": 4.0}),
        _make_result("t2", judge_scores={"overall": 3.0}),
        _make_result("t3"),  # no judge scores
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.avg_patch_quality == 3.5


def test_tier3_avg_patch_quality_no_scores():
    """avg_patch_quality = 0.0 when no results have judge scores."""
    results = [_make_result("t1")]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.avg_patch_quality == 0.0


def test_tier3_avg_patch_size_ratio():
    """avg_patch_size_ratio = mean of generated/gold ratios."""
    results = [
        _make_result("t1", generated_patch_lines=10, gold_patch_lines=10),  # 1.0
        _make_result("t2", generated_patch_lines=15, gold_patch_lines=10),  # 1.5
        _make_result("t3", generated_patch_lines=8, gold_patch_lines=10),   # 0.8
    ]
    metrics = calculate_metrics(results, "exp-1")
    expected = (1.0 + 1.5 + 0.8) / 3
    assert abs(metrics.avg_patch_size_ratio - expected) < 0.001


def test_tier3_avg_patch_size_ratio_no_gold():
    """avg_patch_size_ratio = 0.0 when gold_patch_lines is 0."""
    results = [_make_result("t1", generated_patch_lines=10, gold_patch_lines=0)]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.avg_patch_size_ratio == 0.0


# ── Tier 4: Robustness Metrics ──


def test_tier4_error_recovery_rate():
    """error_recovery_rate = recovered / initially_failed."""
    results = [
        _make_result("t1", intermediate_test_results=[False, False, True]),  # recovered
        _make_result("t2", intermediate_test_results=[False, True]),          # recovered
        _make_result("t3", intermediate_test_results=[False, False, False]),  # not recovered
        _make_result("t4", intermediate_test_results=[True]),                 # never failed
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert abs(metrics.error_recovery_rate - 2 / 3) < 0.001


def test_tier4_failure_categories():
    """failure_categories counts by category."""
    results = [
        _make_result("t1", success=False, failure_category="planning"),
        _make_result("t2", success=False, failure_category="planning"),
        _make_result("t3", success=False, failure_category="timeout"),
        _make_result("t4", success=False, failure_category="hallucination_cascade"),
        _make_result("t5", success=True),  # not counted
    ]
    metrics = calculate_metrics(results, "exp-1")
    assert metrics.failure_categories["planning"] == 2
    assert metrics.failure_categories["timeout"] == 1
    assert metrics.failure_categories["hallucination_cascade"] == 1
    assert metrics.failure_categories["implementation"] == 0
    assert metrics.failure_categories["integration"] == 0
    assert metrics.failure_categories["tool_failure"] == 0


def test_tier4_resolution_variance_cv():
    """resolution_variance_cv = stdev/mean of per-task pass rates."""
    # 2 tasks, each run 2 times
    results = [
        _make_result("t1", success=True),
        _make_result("t1", success=False),  # t1: 50%
        _make_result("t2", success=True),
        _make_result("t2", success=True),   # t2: 100%
    ]
    metrics = calculate_metrics(results, "exp-1")
    # rates = [0.5, 1.0], mean=0.75, stdev=sqrt(0.125)≈0.354, cv≈0.471
    assert metrics.resolution_variance_cv > 0.0


# ── LLM Judge Tests ──


def test_judge_default_scores():
    """Default scores are all 1s with error note."""
    scores = _default_scores("test error")
    assert scores["correctness"] == 1
    assert scores["overall"] == 1.0
    assert "test error" in scores["reasoning"]


def test_judge_parse_valid_response():
    """Judge parses valid JSON response correctly."""
    judge = LLMJudge()
    response = json.dumps({
        "correctness": 4,
        "minimality": 3,
        "code_quality": 5,
        "completeness": 4,
        "reasoning": "Good fix",
    })
    result = judge._parse_response(response)
    assert result["correctness"] == 4
    assert result["minimality"] == 3
    assert result["code_quality"] == 5
    assert result["completeness"] == 4
    # overall = 4*0.4 + 3*0.2 + 5*0.2 + 4*0.2 = 1.6+0.6+1.0+0.8 = 4.0
    assert result["overall"] == 4.0
    assert result["reasoning"] == "Good fix"


def test_judge_parse_malformed_response():
    """Judge handles malformed JSON gracefully."""
    judge = LLMJudge()
    result = judge._parse_response("not json at all")
    assert result["correctness"] == 1
    assert result["overall"] == 1.0
    assert "Parse error" in result["reasoning"]


def test_judge_parse_with_code_fences():
    """Judge strips markdown code fences from response."""
    judge = LLMJudge()
    response = '```json\n{"correctness": 5, "minimality": 5, "code_quality": 5, "completeness": 5, "reasoning": "perfect"}\n```'
    result = judge._parse_response(response)
    assert result["correctness"] == 5
    assert result["overall"] == 5.0


def test_judge_clamps_scores():
    """Judge clamps out-of-range scores to 1-5."""
    judge = LLMJudge()
    response = json.dumps({
        "correctness": 10,
        "minimality": -1,
        "code_quality": 3,
        "completeness": 0,
        "reasoning": "test",
    })
    result = judge._parse_response(response)
    assert result["correctness"] == 5
    assert result["minimality"] == 1
    assert result["completeness"] == 1


# ── pass@k Tests ──


def test_pass_at_k_basic():
    """pass@1 for a task with 3/5 passing = 0.6."""
    results = [
        _make_result("t1", success=True),
        _make_result("t1", success=True),
        _make_result("t1", success=True),
        _make_result("t1", success=False),
        _make_result("t1", success=False),
    ]
    p1 = pass_at_k(results, k=1)
    assert abs(p1 - 0.6) < 0.001


def test_pass_at_k_all_pass():
    """pass@k = 1.0 when all runs pass."""
    results = [_make_result("t1", success=True) for _ in range(5)]
    assert pass_at_k(results, k=1) == 1.0
    assert pass_at_k(results, k=3) == 1.0


def test_pass_at_k_none_pass():
    """pass@k = 0.0 when no runs pass."""
    results = [_make_result("t1", success=False) for _ in range(5)]
    assert pass_at_k(results, k=1) == 0.0
    assert pass_at_k(results, k=3) == 0.0


def test_pass_at_k_increases_with_k():
    """pass@k increases as k increases (more attempts)."""
    results = [
        _make_result("t1", success=True),
        _make_result("t1", success=False),
        _make_result("t1", success=False),
        _make_result("t1", success=False),
        _make_result("t1", success=False),
    ]
    p1 = pass_at_k(results, k=1)
    p3 = pass_at_k(results, k=3)
    assert p3 > p1


def test_pass_at_k_empty_results():
    """pass@k returns 0.0 for empty results."""
    assert pass_at_k([], k=1) == 0.0


# ── Failure Classifier Tests ──


@pytest.mark.asyncio
async def test_classifier_timeout_shortcut():
    """Timeout error triggers deterministic shortcut."""
    classifier = FailureClassifier()
    result = _make_result("t1", success=False, error="Command timed out after 60s")
    category = await classifier.classify("fix bug", result)
    assert category == "timeout"


@pytest.mark.asyncio
async def test_classifier_tool_failure_shortcut():
    """Failed TOOL_CALL events trigger tool_failure shortcut."""
    classifier = FailureClassifier()
    result = _make_result("t1", success=False, error="execution failed")
    events = [
        Event(
            type=EventType.TOOL_CALL,
            task_id="t1",
            experiment_id="exp-1",
            payload={"success": False, "tool_name": "code_executor"},
        ),
    ]
    category = await classifier.classify("fix bug", result, events)
    assert category == "tool_failure"


@pytest.mark.asyncio
async def test_classifier_llm_fallback():
    """LLM classifier returns valid category from mocked response."""
    classifier = FailureClassifier()
    result = _make_result("t1", success=False, error="tests failed")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "category": "implementation",
        "reasoning": "Wrong logic in fix",
    })

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        category = await classifier.classify("fix bug", result)
        assert category == "implementation"


@pytest.mark.asyncio
async def test_classifier_malformed_output_fallback():
    """Malformed classifier output defaults to 'implementation'."""
    classifier = FailureClassifier()
    result = _make_result("t1", success=False, error="tests failed")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        category = await classifier.classify("fix bug", result)
        assert category == "implementation"

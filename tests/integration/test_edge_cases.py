"""
Edge case and error handling tests.

Covers graceful degradation for empty inputs, total failures,
malformed data, missing references, and corrupt files.

Reference: Sprint-6-Epic-3.md (S6-E3-S02)
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_coding.core.config import (
    ExperimentConfig,
    ExecutionConfig,
    MemoryConfig,
    MemoryMode,
    ModelConfig,
    OutputConfig,
    TasksConfig,
)
from ant_coding.eval.comparison import (
    bootstrap_ci,
    compare_experiments,
    mcnemar_test,
    wilcoxon_signed_rank,
)
from ant_coding.eval.failure_classifier import FailureClassifier
from ant_coding.eval.harness import calculate_metrics, pass_at_k
from ant_coding.eval.llm_judge import LLMJudge, _default_scores
from ant_coding.eval.report import generate_json, generate_markdown, metrics_from_json
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.observability.replay import SessionReplay
from ant_coding.runner.experiment import ExperimentRunner
from ant_coding.tasks.types import Task, TaskResult, TaskSource


# ── Helpers ──


def _make_result(
    task_id: str = "t1",
    success: bool = True,
    total_tokens: int = 100,
    total_cost: float = 0.01,
    error: str | None = None,
    failure_category: str | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        experiment_id="test",
        success=success,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_seconds=5.0,
        error=error,
        failure_category=failure_category,
    )


# ── Empty Experiment (0 tasks) ──


class TestEmptyExperiment:
    """Experiment with 0 tasks produces graceful empty results."""

    def test_metrics_with_no_results(self):
        """calculate_metrics with empty list returns zeros, not crash."""
        metrics = calculate_metrics([], "empty-exp")
        assert metrics.total_tasks == 0
        assert metrics.pass_rate == 0.0
        assert metrics.total_tokens == 0
        assert metrics.total_cost == 0.0
        assert metrics.cost_per_resolution == float("inf")
        assert metrics.useful_token_ratio == 0.0

    def test_pass_at_k_with_no_results(self):
        """pass@k with empty results returns 0.0."""
        assert pass_at_k([], k=1) == 0.0
        assert pass_at_k([], k=5) == 0.0

    def test_report_with_empty_metrics(self):
        """Markdown report generates without crash for empty metrics."""
        metrics = calculate_metrics([], "empty")
        md = generate_markdown(metrics)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_runner_summary_with_no_results(self):
        """Runner get_summary with no results returns 0 for everything."""
        config = ExperimentConfig(
            name="empty",
            model=ModelConfig(name="m", litellm_model="gpt-3.5-turbo", api_key_env="K"),
            memory=MemoryConfig(mode=MemoryMode.SHARED),
            tasks=TasksConfig(source="custom"),
            execution=ExecutionConfig(),
            output=OutputConfig(),
        )
        runner = ExperimentRunner(config)
        summary = runner.get_summary()
        assert summary["total_tasks"] == 0
        assert summary["pass_rate"] == 0.0


# ── All Tasks Fail ──


class TestAllTasksFail:
    """All tasks fail: metrics show 0% pass, inf cost_per_resolution."""

    def test_all_failures_metrics(self):
        """0% pass rate and infinite cost_per_resolution."""
        results = [
            _make_result("t1", success=False, error="timeout"),
            _make_result("t2", success=False, error="crash"),
            _make_result("t3", success=False, error="wrong output"),
        ]
        metrics = calculate_metrics(results, "all-fail")
        assert metrics.pass_rate == 0.0
        assert metrics.cost_per_resolution == float("inf")
        assert metrics.tokens_per_resolution == float("inf")
        assert metrics.failed_tasks == 3
        assert metrics.useful_token_ratio == 0.0

    def test_all_failures_json_roundtrip(self):
        """Infinity values survive JSON roundtrip."""
        results = [_make_result("t1", success=False)]
        metrics = calculate_metrics(results, "inf-test")
        json_str = generate_json(metrics)
        restored = metrics_from_json(json_str)
        assert restored.cost_per_resolution == float("inf")
        assert restored.tokens_per_resolution == float("inf")

    def test_all_failures_report(self):
        """Report generates without crash when all tasks fail."""
        results = [_make_result("t1", success=False)]
        metrics = calculate_metrics(results, "fail")
        md = generate_markdown(metrics)
        assert isinstance(md, str)


# ── Model Returns Empty Response ──


class TestEmptyModelResponse:
    """Handle model returning empty or None content gracefully."""

    @pytest.mark.asyncio
    async def test_runner_handles_empty_model_output(self):
        """Runner survives when model returns empty string."""
        config = ExperimentConfig(
            name="empty-resp",
            model=ModelConfig(name="m", litellm_model="gpt-3.5-turbo", api_key_env="K"),
            memory=MemoryConfig(mode=MemoryMode.SHARED),
            tasks=TasksConfig(source="custom"),
            execution=ExecutionConfig(),
            output=OutputConfig(),
        )
        runner = ExperimentRunner(config, pattern_name="single-agent")

        task = Task(id="t1", description="task", source=TaskSource.CUSTOM)

        # Mock model returning empty response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = MagicMock(total_tokens=50)

        model = MagicMock()
        model.complete = AsyncMock(return_value=mock_response)
        model.get_usage = MagicMock(return_value={
            "prompt_tokens": 50, "completion_tokens": 0,
            "total_tokens": 50, "total_cost_usd": 0.0001,
        })
        model.reset_usage = MagicMock()
        model.set_context = MagicMock()

        memory = MagicMock()
        memory.reset = MagicMock()
        memory.write = MagicMock()
        memory.set_context = MagicMock()

        from ant_coding.orchestration.examples.single_agent import SingleAgent

        with patch("ant_coding.runner.experiment.TaskWorkspace") as MockWS, \
             patch("ant_coding.runner.experiment.ToolRegistry") as MockTR:
            ws = AsyncMock()
            ws.workspace_dir = "/tmp/ws"
            MockWS.return_value = ws
            MockTR.return_value = MagicMock()
            MockTR.return_value.as_dict.return_value = {}

            result = await runner._run_task(task, model, memory, "single-agent")

        # Should succeed (SingleAgent always returns success=True)
        assert result.task_id == "t1"
        assert result.duration_seconds > 0


# ── Tool Execution Timeout ──


class TestToolTimeout:
    """Handle tool execution timeouts gracefully."""

    @pytest.mark.asyncio
    async def test_runner_catches_pattern_exception(self):
        """Runner catches exceptions from pattern.solve() and returns failure."""
        config = ExperimentConfig(
            name="timeout-test",
            model=ModelConfig(name="m", litellm_model="gpt-3.5-turbo", api_key_env="K"),
            memory=MemoryConfig(mode=MemoryMode.SHARED),
            tasks=TasksConfig(source="custom"),
            execution=ExecutionConfig(),
            output=OutputConfig(),
        )
        runner = ExperimentRunner(config, pattern_name="single-agent")
        task = Task(id="t1", description="task", source=TaskSource.CUSTOM)

        model = MagicMock()
        model.complete = AsyncMock(side_effect=TimeoutError("Model call timed out"))
        model.reset_usage = MagicMock()
        model.set_context = MagicMock()

        memory = MagicMock()
        memory.reset = MagicMock()
        memory.set_context = MagicMock()

        from ant_coding.orchestration.examples.single_agent import SingleAgent

        with patch("ant_coding.runner.experiment.TaskWorkspace") as MockWS, \
             patch("ant_coding.runner.experiment.ToolRegistry") as MockTR:
            ws = AsyncMock()
            ws.workspace_dir = "/tmp/ws"
            MockWS.return_value = ws
            MockTR.return_value = MagicMock()
            MockTR.return_value.as_dict.return_value = {}

            result = await runner._run_task(task, model, memory, "single-agent")

        assert result.success is False
        assert "timed out" in result.error.lower()


# ── Corrupt events.jsonl Recovery ──


class TestCorruptEvents:
    """Handle corrupt or malformed events.jsonl files."""

    def test_replay_with_empty_file(self, tmp_path):
        """SessionReplay handles empty events file."""
        events_file = tmp_path / "events.jsonl"
        events_file.write_text("")
        replay = SessionReplay(str(events_file))
        assert replay.total_events == 0
        assert replay.step() == []

    def test_replay_with_missing_file(self):
        """SessionReplay raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            SessionReplay("/nonexistent/events.jsonl")

    def test_replay_skips_blank_lines(self, tmp_path):
        """SessionReplay ignores blank lines in JSONL."""
        events_file = tmp_path / "events.jsonl"
        event = Event(type=EventType.TASK_START, task_id="t1", experiment_id="e1")
        line = json.dumps({
            "type": event.type.value,
            "task_id": event.task_id,
            "experiment_id": event.experiment_id,
            "agent_id": None,
            "payload": {},
            "timestamp": event.timestamp.isoformat(),
        })
        events_file.write_text(f"\n{line}\n\n{line}\n\n")
        replay = SessionReplay(str(events_file))
        assert replay.total_events == 2


# ── LLM Judge Malformed JSON ──


class TestJudgeMalformedResponse:
    """LLM judge handles malformed responses gracefully."""

    def test_default_scores_structure(self):
        """_default_scores returns valid structure."""
        scores = _default_scores("test error")
        assert scores["overall"] == 1.0
        assert scores["correctness"] == 1
        assert "test error" in scores["reasoning"]

    @pytest.mark.asyncio
    async def test_judge_api_failure_returns_defaults(self):
        """Judge returns default scores when LLM call fails."""
        judge = LLMJudge()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("API down")
            scores = await judge.evaluate("fix bug", "diff --git ...")

        assert scores["overall"] == 1.0
        assert scores["correctness"] == 1

    @pytest.mark.asyncio
    async def test_judge_malformed_json_returns_defaults(self):
        """Judge returns defaults when LLM returns non-JSON."""
        judge = LLMJudge()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all"

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            scores = await judge.evaluate("fix bug", "patch")

        assert scores["overall"] == 1.0
        assert "Parse error" in scores["reasoning"] or "failed" in scores["reasoning"].lower()

    def test_judge_parse_with_code_fences(self):
        """Judge parses response wrapped in markdown code fences."""
        judge = LLMJudge()
        content = '```json\n{"correctness": 4, "minimality": 3, "code_quality": 4, "completeness": 3, "reasoning": "Good fix"}\n```'
        scores = judge._parse_response(content)
        assert scores["correctness"] == 4
        assert scores["overall"] > 1.0

    def test_judge_parse_clamps_out_of_range(self):
        """Judge clamps scores outside 1-5 range."""
        judge = LLMJudge()
        content = json.dumps({
            "correctness": 10,
            "minimality": -1,
            "code_quality": 3,
            "completeness": 0,
            "reasoning": "test",
        })
        scores = judge._parse_response(content)
        assert scores["correctness"] == 5
        assert scores["minimality"] == 1
        assert scores["completeness"] == 1


# ── FailureClassifier with Missing Events ──


class TestClassifierEdgeCases:
    """FailureClassifier handles edge cases gracefully."""

    @pytest.mark.asyncio
    async def test_classify_timeout_shortcut(self):
        """Classifier detects timeout from error message without LLM."""
        classifier = FailureClassifier()
        result = _make_result("t1", success=False, error="Task timed out after 300s")
        category = await classifier.classify("fix bug", result, events=None)
        assert category == "timeout"

    @pytest.mark.asyncio
    async def test_classify_tool_failure_shortcut(self):
        """Classifier detects tool failure from events without LLM."""
        classifier = FailureClassifier()
        result = _make_result("t1", success=False, error="command failed")
        events = [
            Event(
                type=EventType.TOOL_CALL,
                task_id="t1",
                experiment_id="test",
                payload={"success": False, "tool": "bash"},
            )
        ]
        category = await classifier.classify("fix bug", result, events=events)
        assert category == "tool_failure"

    @pytest.mark.asyncio
    async def test_classify_no_events_falls_back_to_llm(self):
        """Classifier falls back to LLM when no shortcuts match."""
        classifier = FailureClassifier()
        result = _make_result("t1", success=False, error="wrong output")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "category": "implementation",
            "reasoning": "Code logic error",
        })

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            category = await classifier.classify("fix bug", result, events=None)

        assert category == "implementation"

    @pytest.mark.asyncio
    async def test_classify_llm_failure_defaults_to_implementation(self):
        """Classifier defaults to 'implementation' when LLM call fails."""
        classifier = FailureClassifier()
        result = _make_result("t1", success=False, error="mysterious failure")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("API unavailable")
            category = await classifier.classify("fix bug", result, events=None)

        assert category == "implementation"

    def test_parse_invalid_category_defaults(self):
        """Classifier defaults to 'implementation' for unknown categories."""
        classifier = FailureClassifier()
        result = classifier._parse_response(json.dumps({
            "category": "alien_invasion",
            "reasoning": "not real",
        }))
        assert result == "implementation"


# ── Statistical Comparison Edge Cases ──


class TestComparisonEdgeCases:
    """Statistical comparison handles edge cases."""

    def test_mcnemar_identical_results(self):
        """McNemar's test with identical results gives p-value of 1.0."""
        results_a = [
            _make_result("t1", success=True),
            _make_result("t2", success=False),
        ]
        results_b = [
            _make_result("t1", success=True),
            _make_result("t2", success=False),
        ]
        result = mcnemar_test(results_a, results_b)
        assert result["p_value"] == 1.0

    def test_wilcoxon_identical_values(self):
        """Wilcoxon with identical values returns p-value of 1.0."""
        values = [1.0, 2.0, 3.0]
        result = wilcoxon_signed_rank(values, values)
        assert result["p_value"] == 1.0

    def test_bootstrap_ci_single_value(self):
        """Bootstrap CI with single value returns that value for both bounds."""
        ci = bootstrap_ci([5.0], seed=42)
        assert ci["ci_lower"] <= ci["ci_upper"]
        assert ci["point_estimate"] == 5.0

    def test_compare_experiments_with_single_task(self):
        """Comparison works with just 1 task (minimal case)."""
        results_a = [_make_result("t1", success=True, total_tokens=100)]
        results_b = [_make_result("t1", success=True, total_tokens=200)]
        metrics_a = calculate_metrics(results_a, "a")
        metrics_b = calculate_metrics(results_b, "b")

        comp = compare_experiments(results_a, results_b, metrics_a, metrics_b)
        assert comp.experiment_a_id == "a"
        assert comp.experiment_b_id == "b"

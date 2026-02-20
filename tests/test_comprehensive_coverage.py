"""
Comprehensive coverage tests for ant-coding.

Targets the specific untested code paths identified during review:
- CodeExecutor: unsupported language, custom cwd, subprocess exceptions
- GitOperations: checkout, unstaged diff
- ToolRegistry: log_tool_call with/without logger
- CodebaseSearch: hidden files, invalid regex fallback, subdirectory grep
- A2AServer: submit_task success, register_all, pattern exception
- LLMJudge: full evaluate flow, missing dimensions, non-integer scores
- FailureClassifier: _format_events, _format_memory_summary, _parse_response edge cases
- Harness: pass@k edge cases, calculate_metrics with zero tokens
- EventLogger: combined filters, output_path when memory-only, clear + re-add
- ExperimentRunner: from_config_file, model/memory string config errors
- CLI: verbose logging, exit codes
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_coding.eval.failure_classifier import FailureClassifier
from ant_coding.eval.harness import (
    _compute_avg_patch_quality,
    _compute_avg_patch_size_ratio,
    _compute_error_recovery_rate,
    _compute_failure_categories,
    _compute_resolution_variance_cv,
    calculate_metrics,
    pass_at_k,
)
from ant_coding.eval.llm_judge import LLMJudge, _default_scores
from ant_coding.observability.event_logger import Event, EventLogger, EventType
from ant_coding.tasks.types import Task, TaskDifficulty, TaskResult, TaskSource


# =============================================================================
# CodeExecutor Tests
# =============================================================================


class TestCodeExecutorUnsupportedLanguage:
    """Test CodeExecutor with non-Python languages."""

    @pytest.mark.asyncio
    async def test_execute_unsupported_language_bash(self):
        from ant_coding.tools.code_executor import CodeExecutor

        executor = CodeExecutor()
        result = await executor.execute("echo hello", language="bash")
        assert result["success"] is False
        assert result["exit_code"] == -1
        assert "Unsupported language" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_unsupported_language_javascript(self):
        from ant_coding.tools.code_executor import CodeExecutor

        executor = CodeExecutor()
        result = await executor.execute("console.log(1)", language="javascript")
        assert result["success"] is False
        assert "javascript" in result["stderr"]

    @pytest.mark.asyncio
    async def test_execute_language_case_insensitive(self):
        from ant_coding.tools.code_executor import CodeExecutor

        executor = CodeExecutor()
        result = await executor.execute("print(1)", language="Python")
        # Should succeed since "Python".lower() == "python"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_command_with_custom_cwd(self):
        from ant_coding.tools.code_executor import CodeExecutor

        executor = CodeExecutor()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await executor.run_command("pwd", cwd=tmpdir)
            assert result["success"] is True
            assert Path(tmpdir).name in result["stdout"]

    @pytest.mark.asyncio
    async def test_run_command_subprocess_exception(self):
        from ant_coding.tools.code_executor import CodeExecutor

        executor = CodeExecutor()
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("spawn failed")):
            result = await executor.run_command("echo test")
            assert result["success"] is False
            assert "spawn failed" in result["stderr"]
            assert result["exit_code"] == -1


# =============================================================================
# GitOperations Tests
# =============================================================================


class TestGitOperationsExtended:
    """Test GitOperations methods not covered by existing tests."""

    def test_checkout_branch(self, tmp_path):
        from ant_coding.tools.git_ops import GitOperations

        git = GitOperations(tmp_path)
        # Create a file and commit on main
        (tmp_path / "file.txt").write_text("hello")
        git.add(".")
        git.commit("initial commit")

        # Create and switch to a new branch
        git.create_branch("feature")
        (tmp_path / "feature.txt").write_text("feature work")
        git.add(".")
        git.commit("feature commit")

        # Checkout back to master
        git.checkout("master")
        assert not (tmp_path / "feature.txt").exists()

        # Checkout feature again
        git.checkout("feature")
        assert (tmp_path / "feature.txt").exists()

    def test_diff_unstaged(self, tmp_path):
        from ant_coding.tools.git_ops import GitOperations

        git = GitOperations(tmp_path)
        (tmp_path / "file.txt").write_text("original")
        git.add(".")
        git.commit("initial")

        # Modify file without staging
        (tmp_path / "file.txt").write_text("modified")

        unstaged_diff = git.get_diff(staged=False)
        assert "modified" in unstaged_diff

        staged_diff = git.get_diff(staged=True)
        assert staged_diff == ""  # Nothing staged

    def test_init_on_non_repo_directory(self, tmp_path):
        from ant_coding.tools.git_ops import GitOperations

        # Should auto-init a repo
        git = GitOperations(tmp_path)
        assert git.repo is not None
        assert (tmp_path / ".git").exists()

    def test_get_status_empty_repo(self, tmp_path):
        from ant_coding.tools.git_ops import GitOperations

        git = GitOperations(tmp_path)
        status = git.get_status()
        assert status == []


# =============================================================================
# ToolRegistry Tests
# =============================================================================


class TestToolRegistryLogToolCall:
    """Test ToolRegistry.log_tool_call method."""

    def test_log_tool_call_with_logger(self, tmp_path):
        logger = EventLogger(experiment_id="test-exp")
        registry = ToolRegistry(
            tmp_path,
            event_logger=logger,
            experiment_id="test-exp",
            task_id="task-1",
        )

        registry.log_tool_call(
            tool_name="file_ops",
            method="edit_file",
            args_summary="file.py:old->new",
            success=True,
            duration_ms=42.567,
            agent_id="coder",
        )

        events = logger.get_events(event_type=EventType.TOOL_CALL)
        assert len(events) == 1
        e = events[0]
        assert e.payload["tool_name"] == "file_ops"
        assert e.payload["method"] == "edit_file"
        assert e.payload["success"] is True
        assert e.payload["duration_ms"] == 42.6  # Rounded to 1 decimal
        assert e.agent_id == "coder"
        assert e.task_id == "task-1"

    def test_log_tool_call_without_logger(self, tmp_path):
        registry = ToolRegistry(tmp_path, event_logger=None)

        # Should return early without error
        registry.log_tool_call(
            tool_name="git_ops",
            method="commit",
            args_summary="test",
            success=True,
            duration_ms=10.0,
        )

    def test_log_tool_call_no_agent_id(self, tmp_path):
        logger = EventLogger(experiment_id="test-exp")
        registry = ToolRegistry(tmp_path, event_logger=logger, experiment_id="e", task_id="t")

        registry.log_tool_call(
            tool_name="search",
            method="grep",
            args_summary="pattern",
            success=False,
            duration_ms=5.0,
        )

        events = logger.get_events()
        assert len(events) == 1
        assert events[0].agent_id is None


# Need to import ToolRegistry after reading tools
from ant_coding.tools.registry import ToolRegistry  # noqa: E402


# =============================================================================
# CodebaseSearch Tests
# =============================================================================


class TestCodebaseSearchExtended:
    """Test CodebaseSearch edge cases."""

    def test_skips_hidden_directories(self, tmp_path):
        from ant_coding.tools.search import CodebaseSearch

        # Create normal and hidden files
        (tmp_path / "visible.py").write_text("hello = 1")
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "secret.py").write_text("hello = 2")

        search = CodebaseSearch(tmp_path)
        results = search.grep("hello")
        files = [r["file"] for r in results]
        assert "visible.py" in files
        assert ".hidden/secret.py" not in files

    def test_grep_invalid_regex_fallback(self, tmp_path):
        from ant_coding.tools.search import CodebaseSearch

        (tmp_path / "test.py").write_text("data = [1, 2, 3]")
        search = CodebaseSearch(tmp_path)

        # Invalid regex (unclosed bracket) should fall back to literal match
        results = search.grep("[1, 2", file_pattern="*.py")
        assert len(results) == 1
        assert "[1, 2" in results[0]["line_content"]

    def test_grep_with_subdirectory(self, tmp_path):
        from ant_coding.tools.search import CodebaseSearch

        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "app.py").write_text("target_value = 42")
        (tmp_path / "root.py").write_text("target_value = 99")

        search = CodebaseSearch(tmp_path)
        results = search.grep("target_value", path="src")
        assert len(results) == 1
        assert "42" in results[0]["line_content"]

    def test_grep_nonexistent_subdirectory(self, tmp_path):
        from ant_coding.tools.search import CodebaseSearch

        search = CodebaseSearch(tmp_path)
        results = search.grep("anything", path="does_not_exist")
        assert results == []

    def test_find_definition_skips_unsupported_file_types(self, tmp_path):
        from ant_coding.tools.search import CodebaseSearch

        (tmp_path / "code.go").write_text("func MyFunc() {}")
        (tmp_path / "code.py").write_text("def MyFunc(): pass")

        search = CodebaseSearch(tmp_path)
        results = search.find_definition("MyFunc")
        # Should only find Python definition, not Go
        files = [r["file"] for r in results]
        assert "code.py" in files
        assert "code.go" not in files


# =============================================================================
# A2AServer Tests
# =============================================================================


class TestA2AServerExtended:
    """Test A2AServer submit_task success and register_all."""

    @pytest.mark.asyncio
    async def test_submit_task_success(self):
        from ant_coding.orchestration.examples.single_agent import SingleAgent  # noqa: F401
        from ant_coding.protocols.a2a_server import A2AServer

        server = A2AServer()

        # Create a mock pattern and register it
        mock_pattern = MagicMock()
        mock_pattern.name.return_value = "test-pattern"
        mock_pattern.description.return_value = "A test pattern"
        mock_pattern.get_agent_definitions.return_value = []

        server.register_pattern(mock_pattern)

        # Mock OrchestrationRegistry.get to return a pattern whose solve returns a TaskResult
        mock_result = TaskResult(
            task_id="task-1",
            experiment_id="exp-1",
            success=True,
            total_tokens=1000,
            total_cost=0.01,
            duration_seconds=5.0,
        )

        mock_solve_pattern = AsyncMock()
        mock_solve_pattern.solve.return_value = mock_result

        with patch(
            "ant_coding.protocols.a2a_server.OrchestrationRegistry.get",
            return_value=mock_solve_pattern,
        ):
            result = await server.submit_task(
                "test-pattern",
                {"task_description": "fix the bug"},
            )

        assert result["success"] is True
        assert result["task_id"] == "task-1"
        assert result["total_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_submit_task_pattern_exception(self):
        from ant_coding.protocols.a2a_server import A2AServer

        server = A2AServer()

        mock_pattern = MagicMock()
        mock_pattern.name.return_value = "fail-pattern"
        mock_pattern.description.return_value = "Fails"
        mock_pattern.get_agent_definitions.return_value = []
        server.register_pattern(mock_pattern)

        mock_solve_pattern = AsyncMock()
        mock_solve_pattern.solve.side_effect = RuntimeError("Pattern exploded")

        with patch(
            "ant_coding.protocols.a2a_server.OrchestrationRegistry.get",
            return_value=mock_solve_pattern,
        ):
            result = await server.submit_task("fail-pattern", {"task_description": "x"})

        assert result["success"] is False
        assert "Pattern exploded" in result["error"]

    def test_register_all(self):
        from ant_coding.orchestration.examples.single_agent import SingleAgent  # noqa: F401
        from ant_coding.protocols.a2a_server import A2AServer

        server = A2AServer()
        cards = server.register_all()

        # At least single_agent should be registered
        assert len(cards) >= 1
        names = [c.name for c in cards]
        assert "single-agent" in names


# =============================================================================
# LLMJudge Tests
# =============================================================================


class TestLLMJudgeExtended:
    """Test LLMJudge evaluate flow and parsing edge cases."""

    @pytest.mark.asyncio
    async def test_evaluate_success_with_mock(self):
        judge = LLMJudge()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "correctness": 4,
            "minimality": 5,
            "code_quality": 3,
            "completeness": 4,
            "reasoning": "Good fix",
        })

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch.dict(os.environ, {"JUDGE_API_KEY": "fake-key"}):
                result = await judge.evaluate("fix bug", "--- a/f.py\n+++ b/f.py")

        assert result["correctness"] == 4
        assert result["minimality"] == 5
        assert "overall" in result
        assert result["overall"] > 0

    @pytest.mark.asyncio
    async def test_evaluate_acompletion_exception(self):
        judge = LLMJudge()

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = await judge.evaluate("fix bug", "patch content")

        # Should return default scores
        assert result["correctness"] == 1
        assert "API down" in result["reasoning"]

    def test_parse_missing_dimensions(self):
        judge = LLMJudge()
        result = judge._parse_response(json.dumps({
            "correctness": 4,
            # minimality, code_quality, completeness are missing
        }))

        assert result["correctness"] == 4
        assert result["minimality"] == 1  # default
        assert result["code_quality"] == 1  # default
        assert result["completeness"] == 1  # default
        assert "overall" in result

    def test_parse_non_integer_scores(self):
        judge = LLMJudge()
        result = judge._parse_response(json.dumps({
            "correctness": 3.7,
            "minimality": 4.2,
            "code_quality": "not a number",
            "completeness": 5,
            "reasoning": "test",
        }))

        assert result["correctness"] == 3  # int(3.7) clamped
        assert result["minimality"] == 4
        assert result["code_quality"] == 1  # "not a number" fails isinstance check
        assert result["completeness"] == 5

    def test_parse_out_of_range_clamped(self):
        judge = LLMJudge()
        result = judge._parse_response(json.dumps({
            "correctness": 0,
            "minimality": 10,
            "code_quality": -5,
            "completeness": 100,
            "reasoning": "extreme",
        }))

        # All should be clamped to [1, 5]
        assert result["correctness"] == 1
        assert result["minimality"] == 5
        assert result["code_quality"] == 1
        assert result["completeness"] == 5

    def test_parse_missing_reasoning(self):
        judge = LLMJudge()
        result = judge._parse_response(json.dumps({
            "correctness": 3,
            "minimality": 3,
            "code_quality": 3,
            "completeness": 3,
        }))

        assert result["reasoning"] == "No reasoning provided"

    def test_default_scores_with_note(self):
        scores = _default_scores("test error")
        assert scores["correctness"] == 1
        assert scores["overall"] == 1.0
        assert "test error" in scores["reasoning"]

    def test_default_scores_without_note(self):
        scores = _default_scores()
        assert "No evaluation performed" in scores["reasoning"]


# =============================================================================
# FailureClassifier Tests
# =============================================================================


class TestFailureClassifierExtended:
    """Test FailureClassifier formatting and parsing edge cases."""

    def test_format_events_with_events(self):
        classifier = FailureClassifier()
        events = [
            Event(
                type=EventType.LLM_CALL,
                task_id="t1",
                experiment_id="e1",
                agent_id="coder",
                payload={"model": "gpt-4"},
            ),
            Event(
                type=EventType.TOOL_CALL,
                task_id="t1",
                experiment_id="e1",
                agent_id="coder",
                payload={"tool": "file_ops"},
            ),
        ]

        result = classifier._format_events(events, "t1")
        assert "coder" in result
        assert "llm_call" in result
        assert "tool_call" in result

    def test_format_events_no_events(self):
        classifier = FailureClassifier()
        result = classifier._format_events(None, "t1")
        assert "no events" in result.lower()

    def test_format_events_no_matching_task(self):
        classifier = FailureClassifier()
        events = [
            Event(type=EventType.LLM_CALL, task_id="other-task", experiment_id="e1"),
        ]
        result = classifier._format_events(events, "t1")
        assert "no events for this task" in result.lower()

    def test_format_memory_summary_with_reads(self):
        classifier = FailureClassifier()
        events = [
            Event(
                type=EventType.MEMORY_READ,
                task_id="t1",
                experiment_id="e1",
                payload={"agent": "planner", "key": "context", "found": True},
            ),
            Event(
                type=EventType.MEMORY_READ,
                task_id="t1",
                experiment_id="e1",
                payload={"agent": "coder", "key": "plan", "found": False},
            ),
        ]

        result = classifier._format_memory_summary(events, "t1")
        assert "planner" in result
        assert "found" in result
        assert "NOT FOUND" in result

    def test_format_memory_summary_no_events(self):
        classifier = FailureClassifier()
        result = classifier._format_memory_summary(None, "t1")
        assert "no events" in result.lower()

    def test_format_memory_summary_no_reads(self):
        classifier = FailureClassifier()
        events = [
            Event(type=EventType.LLM_CALL, task_id="t1", experiment_id="e1"),
        ]
        result = classifier._format_memory_summary(events, "t1")
        assert "no memory reads" in result.lower()

    def test_parse_response_valid_category(self):
        classifier = FailureClassifier()
        result = classifier._parse_response(json.dumps({
            "category": "hallucination_cascade",
            "reasoning": "Agent used non-existent API",
        }))
        assert result == "hallucination_cascade"

    def test_parse_response_invalid_category_defaults(self):
        classifier = FailureClassifier()
        result = classifier._parse_response(json.dumps({
            "category": "unknown_thing",
            "reasoning": "something",
        }))
        assert result == "implementation"

    def test_parse_response_with_code_fences(self):
        classifier = FailureClassifier()
        content = '```json\n{"category": "planning", "reasoning": "bad plan"}\n```'
        result = classifier._parse_response(content)
        assert result == "planning"

    def test_parse_response_malformed_json(self):
        classifier = FailureClassifier()
        result = classifier._parse_response("not json at all")
        assert result == "implementation"

    def test_check_shortcuts_timeout_variations(self):
        classifier = FailureClassifier()

        for error_msg in ["Command timed out", "TIMEOUT reached", "Timed Out after 30s"]:
            result = TaskResult(
                task_id="t1", experiment_id="e1", success=False, error=error_msg
            )
            assert classifier._check_shortcuts(result) == "timeout"

    def test_check_shortcuts_no_match(self):
        classifier = FailureClassifier()
        result = TaskResult(
            task_id="t1", experiment_id="e1", success=False, error="syntax error"
        )
        assert classifier._check_shortcuts(result) is None


# =============================================================================
# Evaluation Harness Tests
# =============================================================================


class TestHarnessEdgeCases:
    """Test harness edge cases in metrics computation."""

    def test_pass_at_k_zero_k(self):
        results = [TaskResult(task_id="t1", experiment_id="e", success=True)]
        score = pass_at_k(results, k=0)
        # k=0 means 0 attempts, so pass@0 = 0
        assert score == 0.0

    def test_pass_at_k_k_much_larger_than_n(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=True),
            TaskResult(task_id="t1", experiment_id="e", success=False),
        ]
        score = pass_at_k(results, k=100)
        # k >> n should clamp to n, and with 1/2 success the score should be 1.0
        assert score == 1.0

    def test_calculate_metrics_zero_tokens(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=True, total_tokens=0),
            TaskResult(task_id="t2", experiment_id="e", success=False, total_tokens=0),
        ]
        metrics = calculate_metrics(results, "exp-zero")
        assert metrics.total_tokens == 0
        assert metrics.useful_token_ratio == 0.0
        assert metrics.pass_rate == 0.5

    def test_calculate_metrics_empty_results(self):
        metrics = calculate_metrics([], "exp-empty")
        assert metrics.total_tasks == 0
        assert metrics.pass_rate == 0.0
        assert metrics.cost_per_resolution == float("inf")

    def test_avg_patch_quality_non_numeric_overall_raises(self):
        """Non-numeric judge_scores['overall'] raises ValueError.

        This documents current behavior — _compute_avg_patch_quality does not
        guard against non-numeric overall values. Callers should ensure
        judge_scores are numeric before passing to calculate_metrics.
        """
        results = [
            TaskResult(
                task_id="t1", experiment_id="e", success=True,
                judge_scores={"overall": "not_a_number"},
            ),
        ]
        with pytest.raises(ValueError):
            _compute_avg_patch_quality(results)

    def test_avg_patch_size_ratio_zero_generated(self):
        results = [
            TaskResult(
                task_id="t1", experiment_id="e", success=True,
                generated_patch_lines=0, gold_patch_lines=10,
            ),
        ]
        ratio = _compute_avg_patch_size_ratio(results)
        assert ratio == 0.0  # 0/10

    def test_resolution_variance_cv_single_task(self):
        results = [TaskResult(task_id="t1", experiment_id="e", success=True)]
        cv = _compute_resolution_variance_cv(results)
        assert cv == 0.0  # Need at least 2 tasks

    def test_resolution_variance_cv_all_same_rate(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=True),
            TaskResult(task_id="t2", experiment_id="e", success=True),
            TaskResult(task_id="t3", experiment_id="e", success=True),
        ]
        cv = _compute_resolution_variance_cv(results)
        assert cv == 0.0  # All rates = 1.0, zero variance

    def test_error_recovery_rate_no_intermediate_results(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=False),
        ]
        rate = _compute_error_recovery_rate(results)
        assert rate == 0.0

    def test_error_recovery_rate_all_initially_pass(self):
        results = [
            TaskResult(
                task_id="t1", experiment_id="e", success=True,
                intermediate_test_results=[True, True],
            ),
        ]
        rate = _compute_error_recovery_rate(results)
        assert rate == 0.0  # No initially-failed tasks

    def test_failure_categories_unknown_category_ignored(self):
        results = [
            TaskResult(
                task_id="t1", experiment_id="e", success=False,
                failure_category="made_up_category",
            ),
        ]
        cats = _compute_failure_categories(results)
        # Unknown category should not appear; counts should all be 0
        assert all(v == 0 for v in cats.values())

    def test_overhead_ratio_with_baseline(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=True, total_tokens=2000),
        ]
        metrics = calculate_metrics(results, "exp", baseline_tokens=1000)
        assert metrics.overhead_ratio == 2.0

    def test_overhead_ratio_without_baseline(self):
        results = [
            TaskResult(task_id="t1", experiment_id="e", success=True, total_tokens=2000),
        ]
        metrics = calculate_metrics(results, "exp", baseline_tokens=None)
        assert metrics.overhead_ratio == 0.0


# =============================================================================
# EventLogger Tests
# =============================================================================


class TestEventLoggerExtended:
    """Test EventLogger edge cases."""

    def test_combined_filters(self):
        logger = EventLogger(experiment_id="test")

        logger.log(Event(type=EventType.LLM_CALL, task_id="t1", experiment_id="e", agent_id="a1"))
        logger.log(Event(type=EventType.LLM_CALL, task_id="t2", experiment_id="e", agent_id="a1"))
        logger.log(Event(type=EventType.TOOL_CALL, task_id="t1", experiment_id="e", agent_id="a1"))
        logger.log(Event(type=EventType.LLM_CALL, task_id="t1", experiment_id="e", agent_id="a2"))

        # Filter by all three
        results = logger.get_events(agent_name="a1", event_type=EventType.LLM_CALL, task_id="t1")
        assert len(results) == 1

    def test_output_path_memory_only(self):
        logger = EventLogger(experiment_id="test")
        assert logger.output_path is None

    def test_output_path_with_dir(self, tmp_path):
        logger = EventLogger(experiment_id="test", output_dir=str(tmp_path))
        assert logger.output_path is not None
        assert "events.jsonl" in str(logger.output_path)

    def test_clear_and_readd(self):
        logger = EventLogger(experiment_id="test")
        logger.log(Event(type=EventType.LLM_CALL, task_id="t1", experiment_id="e"))
        assert logger.event_count == 1

        logger.clear()
        assert logger.event_count == 0

        logger.log(Event(type=EventType.TOOL_CALL, task_id="t2", experiment_id="e"))
        assert logger.event_count == 1
        assert logger.get_events()[0].type == EventType.TOOL_CALL

    def test_token_breakdown_missing_payload_keys(self):
        logger = EventLogger(experiment_id="test")
        logger.log(Event(
            type=EventType.LLM_CALL,
            task_id="t1",
            experiment_id="e",
            agent_id="agent",
            payload={},  # No token fields
        ))

        breakdown = logger.get_token_breakdown()
        assert breakdown["agent"]["prompt"] == 0
        assert breakdown["agent"]["completion"] == 0
        assert breakdown["agent"]["total"] == 0


# =============================================================================
# ExperimentRunner Tests
# =============================================================================


class TestExperimentRunnerExtended:
    """Test ExperimentRunner error paths."""

    def test_init_model_string_config_raises(self):
        from ant_coding.core.config import ExperimentConfig

        config = MagicMock(spec=ExperimentConfig)
        config.name = "test"
        config.model = "configs/models/claude.yaml"  # Unresolved string
        config.output = MagicMock()
        config.output.dir = "/tmp/test-output"

        from ant_coding.runner.experiment import ExperimentRunner

        runner = ExperimentRunner(config)
        with pytest.raises(ValueError, match="Model config must be resolved"):
            runner._init_model()

    def test_init_memory_string_config_raises(self):
        from ant_coding.core.config import ExperimentConfig

        config = MagicMock(spec=ExperimentConfig)
        config.name = "test"
        config.memory = "configs/memory/shared.yaml"  # Unresolved string
        config.output = MagicMock()
        config.output.dir = "/tmp/test-output"

        from ant_coding.runner.experiment import ExperimentRunner

        runner = ExperimentRunner(config)
        with pytest.raises(ValueError, match="Memory config must be resolved"):
            runner._init_memory()


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLIExtended:
    """Test CLI argument parsing edge cases."""

    def test_parse_args_verbose_flag(self):
        from ant_coding.cli.run import parse_args

        args = parse_args(["config.yaml", "--verbose"])
        assert args.verbose is True
        assert args.config == "config.yaml"

    def test_parse_args_short_verbose(self):
        from ant_coding.cli.run import parse_args

        args = parse_args(["config.yaml", "-v"])
        assert args.verbose is True

    def test_parse_args_all_options(self):
        from ant_coding.cli.run import parse_args

        args = parse_args([
            "config.yaml",
            "--pattern", "single_agent",
            "--output", "/tmp/results",
            "-v",
        ])
        assert args.config == "config.yaml"
        assert args.pattern == "single_agent"
        assert args.output == "/tmp/results"
        assert args.verbose is True

    def test_parse_args_missing_config_raises(self):
        from ant_coding.cli.run import parse_args

        with pytest.raises(SystemExit):
            parse_args([])


# =============================================================================
# Cross-module Integration Tests
# =============================================================================


class TestCrossModuleIntegration:
    """Integration tests spanning multiple modules."""

    def test_metrics_to_report_roundtrip(self):
        """Metrics computed from results can be serialized and deserialized."""
        from ant_coding.eval.report import generate_json, metrics_from_json

        results = [
            TaskResult(
                task_id="t1", experiment_id="e", success=True,
                total_tokens=500, total_cost=0.01, duration_seconds=10.0,
                judge_scores={"overall": 4.0, "correctness": 4, "minimality": 5,
                              "code_quality": 4, "completeness": 3},
                generated_patch_lines=10, gold_patch_lines=8,
            ),
            TaskResult(
                task_id="t2", experiment_id="e", success=False,
                total_tokens=300, total_cost=0.005, duration_seconds=5.0,
                failure_category="implementation",
            ),
        ]

        metrics = calculate_metrics(results, "roundtrip-test")
        json_str = generate_json(metrics)
        restored = metrics_from_json(json_str)

        assert restored.experiment_id == "roundtrip-test"
        assert restored.pass_rate == metrics.pass_rate
        assert restored.total_tokens == metrics.total_tokens
        assert restored.avg_patch_quality == metrics.avg_patch_quality

    def test_event_logger_feeds_into_failure_classifier(self):
        """Events from EventLogger can be used by FailureClassifier."""
        logger = EventLogger(experiment_id="test")

        # Simulate a tool failure
        logger.log(Event(
            type=EventType.TOOL_CALL,
            task_id="t1",
            experiment_id="test",
            agent_id="coder",
            payload={"tool": "code_executor", "success": False},
        ))

        result = TaskResult(task_id="t1", experiment_id="test", success=False, error="test failed")
        classifier = FailureClassifier()

        category = classifier._check_shortcuts(result, logger.get_events())
        assert category == "tool_failure"

    @pytest.mark.asyncio
    async def test_runner_handles_pattern_exception(self):
        """Runner produces failed TaskResult when pattern.solve() raises."""
        from ant_coding.core.config import ExperimentConfig
        from ant_coding.runner.experiment import ExperimentRunner

        config = MagicMock(spec=ExperimentConfig)
        config.name = "exception-test"
        config.output = MagicMock()
        config.output.dir = tempfile.mkdtemp()

        runner = ExperimentRunner(config)

        task = Task(
            id="fail-task",
            description="This will fail",
            source=TaskSource.CUSTOM,
        )

        model = MagicMock()
        memory = MagicMock()

        # Mock workspace and pattern
        with patch("ant_coding.runner.experiment.TaskWorkspace") as MockWS, \
             patch("ant_coding.runner.experiment.ToolRegistry"), \
             patch("ant_coding.runner.experiment.OrchestrationRegistry") as MockReg:

            MockWS.return_value.setup = AsyncMock()
            MockWS.return_value.teardown = AsyncMock()
            MockWS.return_value.workspace_dir = Path(config.output.dir)

            mock_pattern = AsyncMock()
            mock_pattern.solve.side_effect = RuntimeError("boom")
            MockReg.get.return_value = mock_pattern

            result = await runner._run_task(task, model, memory, "single_agent")

        assert result.success is False
        assert "boom" in result.error
        assert result.task_id == "fail-task"

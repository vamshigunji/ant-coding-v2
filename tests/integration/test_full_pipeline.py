"""
Full pipeline integration test: config → tasks → tools → orchestration → memory → eval → report.

Exercises every layer end-to-end with mocked LLM calls.
Validates PRD+ 4-tier metrics, comparison, and report generation.

Reference: Sprint-6-Epic-3.md (S6-E3-S01)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_coding.core.config import (
    EvalConfig,
    ExperimentConfig,
    ExecutionConfig,
    MemoryConfig,
    MemoryMode,
    ModelConfig,
    OutputConfig,
    TasksConfig,
)
from ant_coding.eval.comparison import compare_experiments, generate_comparison_report
from ant_coding.eval.harness import calculate_metrics, pass_at_k
from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.eval.report import (
    generate_csv,
    generate_json,
    generate_markdown,
    generate_comparison_markdown,
    metrics_from_json,
)
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.runner.experiment import ExperimentRunner
from ant_coding.runner.output import ResultWriter
from ant_coding.tasks.types import Task, TaskResult, TaskSource


# ── Fixtures ──


def _make_model_config(name: str = "test-model") -> ModelConfig:
    return ModelConfig(
        name=name,
        litellm_model="gpt-3.5-turbo",
        api_key_env="TEST_API_KEY",
    )


def _make_config(
    name: str,
    memory_mode: MemoryMode = MemoryMode.SHARED,
    output_dir: str = "results",
    baseline_id: str | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model=_make_model_config(),
        memory=MemoryConfig(mode=memory_mode),
        tasks=TasksConfig(source="custom", subset="tasks/custom/example-task.yaml"),
        execution=ExecutionConfig(max_workers=1, timeout_seconds=60),
        eval=EvalConfig(),
        output=OutputConfig(dir=output_dir),
        baseline_experiment_id=baseline_id,
    )


def _make_task(task_id: str, description: str = "Fix the bug") -> Task:
    return Task(id=task_id, description=description, source=TaskSource.CUSTOM)


def _make_mock_response(content: str = "def fix(): return True"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 100
    response.usage.total_tokens = 150
    return response


def _make_result(
    task_id: str,
    experiment_id: str,
    success: bool = True,
    total_tokens: int = 150,
    total_cost: float = 0.01,
    duration: float = 5.0,
    failure_category: str | None = None,
    intermediate_results: list | None = None,
    judge_scores: dict | None = None,
    gen_patch: int = 10,
    gold_patch: int = 10,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        experiment_id=experiment_id,
        success=success,
        total_tokens=total_tokens,
        total_cost=total_cost,
        duration_seconds=duration,
        failure_category=failure_category,
        intermediate_test_results=intermediate_results or [],
        judge_scores=judge_scores,
        generated_patch_lines=gen_patch,
        gold_patch_lines=gold_patch,
    )


# ── Pipeline Integration: Metrics Calculation ──


class TestMetricsPipeline:
    """Test the full metrics calculation pipeline with realistic data."""

    def _build_results(self, exp_id: str, pass_count: int, fail_count: int):
        results = []
        for i in range(pass_count):
            results.append(
                _make_result(
                    task_id=f"t{i}",
                    experiment_id=exp_id,
                    success=True,
                    total_tokens=150,
                    total_cost=0.01,
                    judge_scores={"overall": 4.0, "correctness": 5, "style": 3},
                    gen_patch=12,
                    gold_patch=10,
                    intermediate_results=[False, True],
                )
            )
        for i in range(fail_count):
            results.append(
                _make_result(
                    task_id=f"f{i}",
                    experiment_id=exp_id,
                    success=False,
                    total_tokens=200,
                    total_cost=0.015,
                    failure_category="implementation",
                    intermediate_results=[False, False],
                    gen_patch=5,
                    gold_patch=10,
                )
            )
        return results

    def test_four_tier_metrics_all_populated(self):
        """All 4 tiers of PRD+ metrics are computed from results."""
        results = self._build_results("exp-a", pass_count=3, fail_count=2)
        metrics = calculate_metrics(results, "exp-a")

        # Tier 1
        assert metrics.total_tasks == 5
        assert metrics.successful_tasks == 3
        assert metrics.failed_tasks == 2
        assert metrics.pass_rate == pytest.approx(0.6)
        assert metrics.cost_per_resolution > 0
        assert metrics.cost_per_resolution < float("inf")

        # Tier 2
        assert metrics.total_tokens == 3 * 150 + 2 * 200
        assert metrics.total_cost == pytest.approx(3 * 0.01 + 2 * 0.015)
        assert metrics.useful_token_ratio > 0
        assert metrics.tokens_per_resolution > 0

        # Tier 3
        assert metrics.avg_patch_quality == pytest.approx(4.0)
        assert metrics.avg_patch_size_ratio > 0

        # Tier 4
        assert metrics.error_recovery_rate == pytest.approx(0.6)  # 3 of 5 recovered from initial False
        assert metrics.failure_categories["implementation"] == 2

    def test_pass_at_k(self):
        """pass@k computation works across tasks."""
        results = self._build_results("exp-a", pass_count=4, fail_count=1)
        p1 = pass_at_k(results, k=1)
        assert 0 < p1 <= 1.0

    def test_metrics_json_roundtrip(self):
        """ExperimentMetrics survives JSON serialization and deserialization."""
        results = self._build_results("exp-rt", pass_count=2, fail_count=1)
        metrics = calculate_metrics(results, "exp-rt")

        json_str = generate_json(metrics)
        restored = metrics_from_json(json_str)

        assert restored.experiment_id == "exp-rt"
        assert restored.total_tasks == 3
        assert restored.pass_rate == pytest.approx(metrics.pass_rate)
        assert restored.cost_per_resolution == pytest.approx(metrics.cost_per_resolution)
        assert restored.failure_categories == metrics.failure_categories


# ── Pipeline Integration: Two-Experiment Comparison ──


class TestComparisonPipeline:
    """Test full comparison flow between two experiments."""

    def _two_experiments(self):
        results_a = [
            _make_result("t1", "single-agent", success=True, total_tokens=100, total_cost=0.01),
            _make_result("t2", "single-agent", success=False, total_tokens=200, total_cost=0.02),
            _make_result("t3", "single-agent", success=True, total_tokens=150, total_cost=0.015),
            _make_result("t4", "single-agent", success=False, total_tokens=180, total_cost=0.018),
        ]
        results_b = [
            _make_result("t1", "multi-agent", success=True, total_tokens=300, total_cost=0.03),
            _make_result("t2", "multi-agent", success=True, total_tokens=400, total_cost=0.04),
            _make_result("t3", "multi-agent", success=True, total_tokens=350, total_cost=0.035),
            _make_result("t4", "multi-agent", success=False, total_tokens=250, total_cost=0.025),
        ]
        metrics_a = calculate_metrics(results_a, "single-agent")
        metrics_b = calculate_metrics(results_b, "multi-agent")
        return results_a, results_b, metrics_a, metrics_b

    def test_compare_experiments_produces_all_fields(self):
        """compare_experiments fills statistical_tests, effect_sizes, confidence_intervals."""
        results_a, results_b, metrics_a, metrics_b = self._two_experiments()
        comp = compare_experiments(results_a, results_b, metrics_a, metrics_b)

        assert comp.experiment_a_id == "single-agent"
        assert comp.experiment_b_id == "multi-agent"
        assert "pass_rate" in comp.statistical_tests
        assert len(comp.effect_sizes) > 0
        assert len(comp.confidence_intervals) > 0

    def test_comparison_report_is_valid_markdown(self):
        """Comparison report generates valid markdown with metric table."""
        results_a, results_b, metrics_a, metrics_b = self._two_experiments()
        comp = compare_experiments(results_a, results_b, metrics_a, metrics_b)
        report = generate_comparison_report(metrics_a, metrics_b, comp)

        assert "single-agent" in report
        assert "multi-agent" in report
        assert "pass_rate" in report.lower() or "Pass Rate" in report

    def test_comparison_markdown_multi_experiment(self):
        """generate_comparison_markdown handles list of metrics."""
        _, _, metrics_a, metrics_b = self._two_experiments()
        md = generate_comparison_markdown([metrics_a, metrics_b])

        assert "single-agent" in md
        assert "multi-agent" in md

    def test_csv_export_multiple_experiments(self):
        """CSV export works for multiple experiments."""
        _, _, metrics_a, metrics_b = self._two_experiments()
        csv_str = generate_csv([metrics_a, metrics_b])

        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "single-agent" in lines[1]
        assert "multi-agent" in lines[2]


# ── Pipeline Integration: Report Generation ──


class TestReportPipeline:
    """Test report generation from metrics."""

    def test_markdown_report_has_all_tiers(self):
        """Single-experiment markdown report includes all 4 tiers."""
        results = [
            _make_result("t1", "exp", success=True, judge_scores={"overall": 4.5}),
            _make_result("t2", "exp", success=False, failure_category="timeout"),
        ]
        metrics = calculate_metrics(results, "exp")
        md = generate_markdown(
            metrics,
            architecture="single-agent",
            model="gpt-3.5-turbo",
            memory_mode="shared",
        )

        assert "Tier 1" in md or "Primary" in md or "pass_rate" in md.lower()
        assert "single-agent" in md
        assert "gpt-3.5-turbo" in md


# ── Pipeline Integration: ResultWriter Output Structure ──


class TestResultOutputPipeline:
    """Test the full output pipeline from results to files."""

    def test_save_all_creates_complete_directory(self, tmp_path):
        """ResultWriter.save_all creates all expected files."""
        writer = ResultWriter(str(tmp_path), "integration-exp")

        results = [
            TaskResult(task_id="t1", experiment_id="integration-exp", success=True, total_tokens=100),
            TaskResult(task_id="t2", experiment_id="integration-exp", success=False, error="timeout"),
        ]
        events = [
            Event(type=EventType.TASK_START, task_id="t1", experiment_id="integration-exp"),
            Event(type=EventType.TASK_END, task_id="t1", experiment_id="integration-exp"),
        ]

        output = writer.save_all(
            config={"name": "integration-exp", "model": "gpt-3.5-turbo"},
            results=results,
            summary={"pass_rate": 0.5, "total_tasks": 2},
            events=events,
        )

        assert (output / "config.json").exists()
        assert (output / "results.json").exists()
        assert (output / "summary.json").exists()
        assert (output / "events.jsonl").exists()

        # Verify content integrity
        results_data = json.loads((output / "results.json").read_text())
        assert len(results_data) == 2
        assert results_data[0]["task_id"] == "t1"

        events_lines = (output / "events.jsonl").read_text().strip().split("\n")
        assert len(events_lines) == 2

    def test_metrics_saved_and_restored(self, tmp_path):
        """Metrics can be saved as JSON and fully restored."""
        results = [
            _make_result("t1", "exp", success=True),
            _make_result("t2", "exp", success=False, failure_category="planning"),
        ]
        metrics = calculate_metrics(results, "exp")
        json_str = generate_json(metrics)

        # Save to file
        json_path = tmp_path / "metrics.json"
        json_path.write_text(json_str)

        # Restore
        restored = metrics_from_json(json_path.read_text())
        assert restored.experiment_id == "exp"
        assert restored.total_tasks == 2
        assert restored.failure_categories["planning"] == 1


# ── Pipeline Integration: Runner + Eval End-to-End ──


class TestRunnerEvalPipeline:
    """Test running experiments through the runner then evaluating."""

    @pytest.mark.asyncio
    async def test_runner_task_produces_evaluable_results(self):
        """Runner._run_task produces TaskResults that can be evaluated."""
        config = _make_config("pipeline-test")
        runner = ExperimentRunner(config, pattern_name="single-agent")

        task = _make_task("pipeline-t1", "Add error handling")
        mock_response = _make_mock_response("def handle_error(): pass")

        # Mock model
        model = MagicMock()
        model.complete = AsyncMock(return_value=mock_response)
        model.get_usage = MagicMock(return_value={
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
            "total_cost_usd": 0.001,
        })
        model.reset_usage = MagicMock()
        model.set_context = MagicMock()

        # Mock memory
        memory = MagicMock()
        memory.reset = MagicMock()
        memory.write = MagicMock()
        memory.set_context = MagicMock()

        # Import to register pattern
        from ant_coding.orchestration.examples.single_agent import SingleAgent

        # Mock workspace setup/teardown
        with patch("ant_coding.runner.experiment.TaskWorkspace") as MockWorkspace, \
             patch("ant_coding.runner.experiment.ToolRegistry") as MockToolReg:
            workspace_instance = AsyncMock()
            workspace_instance.workspace_dir = "/tmp/test-ws"
            MockWorkspace.return_value = workspace_instance
            MockToolReg.return_value = MagicMock()
            MockToolReg.return_value.as_dict.return_value = {}

            result = await runner._run_task(task, model, memory, "single-agent")

        # Verify result is evaluable
        assert result.task_id == "pipeline-t1"
        assert result.success is True
        assert result.total_tokens == 150
        assert result.duration_seconds > 0

        # Now evaluate
        metrics = calculate_metrics([result], "pipeline-test")
        assert metrics.total_tasks == 1
        assert metrics.pass_rate == 1.0
        assert metrics.total_tokens == 150

        # Generate report
        md = generate_markdown(metrics, architecture="single-agent")
        assert "pipeline-test" in md or "1" in md

    @pytest.mark.asyncio
    async def test_two_experiment_full_pipeline(self):
        """Two experiments: run → eval → compare → report (full pipeline)."""
        # Simulate results from two experiments
        results_a = [
            _make_result("t1", "exp-baseline", True, 100, 0.01, judge_scores={"overall": 3.5}),
            _make_result("t2", "exp-baseline", True, 120, 0.012, judge_scores={"overall": 4.0}),
            _make_result("t3", "exp-baseline", False, 200, 0.02, failure_category="planning"),
        ]
        results_b = [
            _make_result("t1", "exp-variant", True, 300, 0.03, judge_scores={"overall": 4.5}),
            _make_result("t2", "exp-variant", True, 280, 0.028, judge_scores={"overall": 4.0}),
            _make_result("t3", "exp-variant", True, 350, 0.035, judge_scores={"overall": 3.5}),
        ]

        # Step 1: Calculate metrics
        metrics_a = calculate_metrics(results_a, "exp-baseline")
        metrics_b = calculate_metrics(results_b, "exp-variant")

        assert metrics_a.pass_rate == pytest.approx(2 / 3)
        assert metrics_b.pass_rate == pytest.approx(1.0)
        assert metrics_b.total_cost > metrics_a.total_cost

        # Step 2: Compare
        comp = compare_experiments(results_a, results_b, metrics_a, metrics_b)
        assert comp.experiment_a_id == "exp-baseline"
        assert comp.experiment_b_id == "exp-variant"

        # Step 3: Generate comparison report
        report = generate_comparison_report(metrics_a, metrics_b, comp)
        assert isinstance(report, str)
        assert len(report) > 100

        # Step 4: Generate comparison markdown table
        md = generate_comparison_markdown([metrics_a, metrics_b], [comp])
        assert "exp-baseline" in md
        assert "exp-variant" in md

        # Step 5: CSV export
        csv = generate_csv([metrics_a, metrics_b])
        lines = csv.strip().split("\n")
        assert len(lines) == 3

        # Step 6: JSON roundtrip
        for m in [metrics_a, metrics_b]:
            restored = metrics_from_json(generate_json(m))
            assert restored.experiment_id == m.experiment_id
            assert restored.pass_rate == pytest.approx(m.pass_rate)

    @pytest.mark.asyncio
    async def test_full_pipeline_with_result_output(self, tmp_path):
        """Full pipeline: results → metrics → report → save to disk."""
        results = [
            _make_result("t1", "full-pipe", True, 100, 0.01, judge_scores={"overall": 4.0}),
            _make_result("t2", "full-pipe", False, 200, 0.02, failure_category="timeout"),
            _make_result("t3", "full-pipe", True, 150, 0.015, judge_scores={"overall": 3.5}),
        ]

        # Calculate metrics
        metrics = calculate_metrics(results, "full-pipe")
        assert metrics.total_tasks == 3
        assert metrics.successful_tasks == 2

        # Generate all reports
        md_report = generate_markdown(metrics, architecture="single-agent", model="gpt-3.5-turbo")
        json_report = generate_json(metrics)
        csv_report = generate_csv([metrics])

        # Save everything
        writer = ResultWriter(str(tmp_path), "full-pipe")
        events = [
            Event(type=EventType.TASK_START, task_id="t1", experiment_id="full-pipe"),
            Event(type=EventType.TASK_END, task_id="t1", experiment_id="full-pipe"),
        ]
        output = writer.save_all(
            config={"name": "full-pipe"},
            results=results,
            summary={"pass_rate": metrics.pass_rate, "total_tasks": 3},
            events=events,
        )

        # Save reports alongside
        (output / "report.md").write_text(md_report)
        (output / "metrics.json").write_text(json_report)
        (output / "metrics.csv").write_text(csv_report)

        # Verify all files exist and are non-empty
        for fname in ["config.json", "results.json", "summary.json", "events.jsonl",
                       "report.md", "metrics.json", "metrics.csv"]:
            fpath = output / fname
            assert fpath.exists(), f"Missing: {fname}"
            assert fpath.stat().st_size > 0, f"Empty: {fname}"

        # Verify metrics can be restored from disk
        restored = metrics_from_json((output / "metrics.json").read_text())
        assert restored.pass_rate == pytest.approx(metrics.pass_rate)


# ── Pipeline Integration: Events Flow Through Layers ──


class TestEventsPipeline:
    """Test that events flow through the runner correctly."""

    @pytest.mark.asyncio
    async def test_events_logged_during_task_run(self):
        """Runner logs TASK_START and TASK_END events."""
        config = _make_config("events-test")
        runner = ExperimentRunner(config, pattern_name="single-agent")

        task = _make_task("evt-t1")
        mock_response = _make_mock_response()

        model = MagicMock()
        model.complete = AsyncMock(return_value=mock_response)
        model.get_usage = MagicMock(return_value={
            "prompt_tokens": 50, "completion_tokens": 100,
            "total_tokens": 150, "total_cost_usd": 0.001,
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

            await runner._run_task(task, model, memory, "single-agent")

        assert len(runner.events) >= 2
        types = [e.type for e in runner.events]
        assert EventType.TASK_START in types
        assert EventType.TASK_END in types

        # Verify event has correct task_id
        start_event = next(e for e in runner.events if e.type == EventType.TASK_START)
        assert start_event.task_id == "evt-t1"

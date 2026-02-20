"""
End-to-end integration tests for ExperimentRunner and output.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ant_coding.cli.run import parse_args
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
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.runner.experiment import ExperimentRunner
from ant_coding.runner.output import ResultWriter
from ant_coding.tasks.types import Task, TaskResult, TaskSource


# ── Helpers ──


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        litellm_model="gpt-3.5-turbo",
        api_key_env="TEST_API_KEY",
    )


def _make_experiment_config(output_dir: str = "results") -> ExperimentConfig:
    return ExperimentConfig(
        name="test-experiment",
        model=_make_model_config(),
        memory=MemoryConfig(mode=MemoryMode.SHARED),
        tasks=TasksConfig(source="custom", subset="tasks/custom/example-task.yaml"),
        execution=ExecutionConfig(max_workers=1, timeout_seconds=60),
        eval=EvalConfig(),
        output=OutputConfig(dir=output_dir),
    )


def _make_task(task_id: str = "test-1") -> Task:
    return Task(id=task_id, description="Fix the bug", source=TaskSource.CUSTOM)


def _make_mock_response(content: str = "solution"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 100
    response.usage.total_tokens = 150
    return response


# ── ResultWriter Tests ──


def test_result_writer_creates_directory(tmp_path):
    writer = ResultWriter(str(tmp_path), "my-exp")
    assert writer.output_path.exists()
    assert "my-exp" in str(writer.output_path)


def test_result_writer_save_config(tmp_path):
    writer = ResultWriter(str(tmp_path), "exp")
    path = writer.save_config({"name": "test", "model": "gpt-4"})
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "test"


def test_result_writer_save_results(tmp_path):
    writer = ResultWriter(str(tmp_path), "exp")
    results = [
        TaskResult(task_id="t1", experiment_id="exp", success=True, total_tokens=100),
        TaskResult(task_id="t2", experiment_id="exp", success=False, error="timeout"),
    ]
    path = writer.save_results(results)
    data = json.loads(path.read_text())
    assert len(data) == 2
    assert data[0]["task_id"] == "t1"
    assert data[1]["success"] is False


def test_result_writer_save_events(tmp_path):
    writer = ResultWriter(str(tmp_path), "exp")
    events = [
        Event(type=EventType.TASK_START, task_id="t1", experiment_id="exp"),
        Event(type=EventType.TASK_END, task_id="t1", experiment_id="exp"),
    ]
    path = writer.save_events(events)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "task_start"


def test_result_writer_save_all(tmp_path):
    writer = ResultWriter(str(tmp_path), "full-exp")
    output_path = writer.save_all(
        config={"name": "full"},
        results=[TaskResult(task_id="t1", experiment_id="full", success=True)],
        summary={"pass_rate": 1.0},
        events=[Event(type=EventType.TASK_START, task_id="t1", experiment_id="full")],
    )
    assert (output_path / "config.json").exists()
    assert (output_path / "results.json").exists()
    assert (output_path / "summary.json").exists()
    assert (output_path / "events.jsonl").exists()


# ── ExperimentRunner Tests ──


def test_runner_init():
    config = _make_experiment_config()
    runner = ExperimentRunner(config)
    assert runner.config.name == "test-experiment"
    assert runner.results == []
    assert runner.events == []


def test_runner_get_summary_empty():
    config = _make_experiment_config()
    runner = ExperimentRunner(config)
    summary = runner.get_summary()
    assert summary["total_tasks"] == 0
    assert summary["pass_rate"] == 0.0


def test_runner_get_summary_with_results():
    config = _make_experiment_config()
    runner = ExperimentRunner(config)
    runner.results = [
        TaskResult(task_id="t1", experiment_id="exp", success=True, total_tokens=100, total_cost=0.01, duration_seconds=5.0),
        TaskResult(task_id="t2", experiment_id="exp", success=False, total_tokens=200, total_cost=0.02, duration_seconds=10.0),
    ]
    summary = runner.get_summary()
    assert summary["total_tasks"] == 2
    assert summary["successful_tasks"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["total_tokens"] == 300
    assert summary["avg_duration_seconds"] == 7.5


@pytest.mark.asyncio
async def test_runner_run_task_success():
    """Test running a single task with mocked model and pattern."""
    config = _make_experiment_config()
    runner = ExperimentRunner(config, pattern_name="single-agent")

    task = _make_task()
    mock_response = _make_mock_response("def fix(): return True")

    # Mock model
    from ant_coding.models.provider import ModelProvider
    model = MagicMock(spec=ModelProvider)
    model.complete = AsyncMock(return_value=mock_response)
    model.get_usage = MagicMock(return_value={
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150,
        "total_cost_usd": 0.001,
    })
    model.reset_usage = MagicMock()

    # Mock memory
    from ant_coding.memory.manager import MemoryManager
    memory = MagicMock(spec=MemoryManager)
    memory.reset = MagicMock()
    memory.write = MagicMock()

    # Need to import SingleAgent to register it
    from ant_coding.orchestration.examples.single_agent import SingleAgent  # noqa: F401

    result = await runner._run_task(task, model, memory, "single-agent")

    assert result.task_id == "test-1"
    assert result.duration_seconds > 0
    # Check events were logged
    assert any(e.type == EventType.TASK_START for e in runner.events)
    assert any(e.type == EventType.TASK_END for e in runner.events)


# ── CLI Tests ──


def test_cli_parse_args():
    args = parse_args(["configs/experiments/test.yaml"])
    assert args.config == "configs/experiments/test.yaml"
    assert args.pattern is None
    assert args.output is None
    assert args.verbose is False


def test_cli_parse_args_with_options():
    args = parse_args([
        "config.yaml",
        "--pattern", "minimal-sequential",
        "--output", "/tmp/out",
        "-v",
    ])
    assert args.config == "config.yaml"
    assert args.pattern == "minimal-sequential"
    assert args.output == "/tmp/out"
    assert args.verbose is True

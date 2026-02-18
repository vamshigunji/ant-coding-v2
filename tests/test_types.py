import pytest
from ant_coding.tasks.types import Task, TaskResult, TaskSource, TaskDifficulty
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.eval.metrics import ExperimentMetrics

def test_task_creation():
    task = Task(id="test-1", description="Fix bug", source=TaskSource.CUSTOM)
    assert task.id == "test-1"
    assert task.difficulty == TaskDifficulty.MEDIUM
    assert task.max_tokens_budget == 100_000
    assert task.timeout_seconds == 600
    assert task.files_context == []

def test_task_result_creation():
    result = TaskResult(task_id="test-1", experiment_id="exp-1", success=True)
    assert result.total_tokens == 0
    assert result.agent_traces == []

def test_event_creation():
    event = Event(type=EventType.LLM_CALL, task_id="test-1", experiment_id="exp-1")
    assert event.type == EventType.LLM_CALL
    assert event.payload == {}
    
    # Check some enum values
    assert EventType.AGENT_START == "agent_start"
    assert EventType.TASK_END == "task_end"

def test_experiment_metrics_creation():
    metrics = ExperimentMetrics(experiment_id="exp-1", pass_rate=0.8)
    assert metrics.experiment_id == "exp-1"
    assert metrics.pass_rate == 0.8
    assert metrics.total_tasks == 0
    assert metrics.total_cost == 0.0

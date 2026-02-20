from ant_coding.tasks.types import Task, TaskResult, TaskSource, TaskDifficulty, VALID_FAILURE_CATEGORIES
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


# ── PRD+ TaskResult Tests ──


def test_task_result_prd_plus_defaults():
    """New PRD+ fields have backward-compatible defaults."""
    result = TaskResult(task_id="t1", experiment_id="e1", success=True)
    assert result.intermediate_test_results == []
    assert result.failure_category is None
    assert result.generated_patch_lines == 0
    assert result.gold_patch_lines == 0
    assert result.judge_scores is None


def test_task_result_intermediate_test_results():
    """intermediate_test_results tracks error recovery across attempts."""
    result = TaskResult(
        task_id="t1",
        experiment_id="e1",
        success=True,
        intermediate_test_results=[False, False, True],
    )
    assert len(result.intermediate_test_results) == 3
    assert result.intermediate_test_results[-1] is True
    # Recovery happened: failed twice then succeeded
    assert result.intermediate_test_results.count(False) == 2


def test_task_result_judge_scores():
    """judge_scores stores LLM-as-Judge evaluation dimensions."""
    scores = {
        "correctness": 4,
        "minimality": 3,
        "code_quality": 4,
        "completeness": 3,
        "overall": 3.5,
        "reasoning": "Good implementation with minor style issues.",
    }
    result = TaskResult(
        task_id="t1",
        experiment_id="e1",
        success=True,
        judge_scores=scores,
    )
    assert result.judge_scores["overall"] == 3.5
    assert result.judge_scores["correctness"] == 4
    assert isinstance(result.judge_scores["reasoning"], str)


def test_task_result_failure_category_valid():
    """failure_category accepts valid category strings."""
    for category in VALID_FAILURE_CATEGORIES:
        result = TaskResult(
            task_id="t1",
            experiment_id="e1",
            success=False,
            failure_category=category,
        )
        assert result.failure_category == category
        assert result.failure_category in VALID_FAILURE_CATEGORIES


# ── PRD+ ExperimentMetrics Tests ──


def test_experiment_metrics_prd_plus_defaults():
    """New PRD+ tier fields have zero defaults."""
    metrics = ExperimentMetrics(experiment_id="e1")
    # Tier 1
    assert metrics.cost_per_resolution == 0.0
    # Tier 2
    assert metrics.useful_token_ratio == 0.0
    assert metrics.overhead_ratio == 0.0
    assert metrics.tokens_per_resolution == 0.0
    # Tier 3
    assert metrics.avg_patch_quality == 0.0
    assert metrics.avg_patch_size_ratio == 0.0
    # Tier 4
    assert metrics.resolution_variance_cv == 0.0
    assert metrics.error_recovery_rate == 0.0


def test_experiment_metrics_failure_categories_dict():
    """failure_categories dict has all 6 categories defaulting to 0."""
    metrics = ExperimentMetrics(experiment_id="e1")
    assert len(metrics.failure_categories) == 6
    assert metrics.failure_categories["hallucination_cascade"] == 0
    assert metrics.failure_categories["planning"] == 0
    assert metrics.failure_categories["timeout"] == 0
    assert all(v == 0 for v in metrics.failure_categories.values())


def test_experiment_metrics_failure_categories_independent():
    """Each ExperimentMetrics instance gets its own failure_categories dict."""
    m1 = ExperimentMetrics(experiment_id="e1")
    m2 = ExperimentMetrics(experiment_id="e2")
    m1.failure_categories["planning"] = 5
    assert m2.failure_categories["planning"] == 0

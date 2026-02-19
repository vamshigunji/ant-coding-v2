"""
Comprehensive tests for the orchestration layer.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ant_coding.core.config import MemoryConfig, MemoryMode, ModelConfig
from ant_coding.memory.manager import MemoryManager
from ant_coding.models.provider import ModelProvider
from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import (
    DuplicatePatternError,
    OrchestrationRegistry,
    PatternNotFoundError,
)
from ant_coding.tasks.types import Task, TaskSource


# ── Helpers ──


def _make_task(task_id: str = "test-task-1") -> Task:
    return Task(
        id=task_id,
        description="Fix the off-by-one error in utils.py",
        source=TaskSource.CUSTOM,
    )


def _make_mock_model(responses: list[str] | None = None) -> ModelProvider:
    """Create a mock ModelProvider that returns predetermined responses."""
    if responses is None:
        responses = ["Mock LLM response"]

    model = MagicMock(spec=ModelProvider)

    call_count = 0
    prompt_tokens = 0
    completion_tokens = 0

    async def mock_complete(messages, **kwargs):
        nonlocal call_count, prompt_tokens, completion_tokens
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        prompt_tokens += 50
        completion_tokens += 100

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = responses[idx]
        return response

    model.complete = AsyncMock(side_effect=mock_complete)
    model.get_usage = lambda: {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "total_cost_usd": (prompt_tokens + completion_tokens) * 0.00001,
    }
    model.reset_usage = MagicMock()
    return model


def _make_memory(mode: MemoryMode = MemoryMode.SHARED) -> MemoryManager:
    return MemoryManager(MemoryConfig(mode=mode))


# ── ABC Tests ──


def test_abc_enforcement():
    """Cannot instantiate OrchestrationPattern without implementing abstract methods."""

    class IncompletePattern(OrchestrationPattern):
        pass

    with pytest.raises(TypeError):
        IncompletePattern()


def test_abc_concrete_subclass():
    """Concrete subclass with all methods implemented can be instantiated."""

    class ConcretePattern(OrchestrationPattern):
        def name(self):
            return "concrete"

        def description(self):
            return "A concrete pattern"

        async def solve(self, task, model, memory, tools, workspace_dir):
            pass

    p = ConcretePattern()
    assert p.name() == "concrete"
    assert p.description() == "A concrete pattern"
    assert p.get_agent_definitions() == []


# ── Registry Tests ──


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear and restore registry between tests."""
    # Save current state
    saved = dict(OrchestrationRegistry._registry)
    OrchestrationRegistry.clear()
    yield
    # Restore
    OrchestrationRegistry._registry = saved


def _register_dummy(name: str):
    """Helper to register a dummy pattern with the given name."""

    class DummyPattern(OrchestrationPattern):
        def __init__(self):
            self._name = name

        def name(self):
            return self._name

        def description(self):
            return f"Dummy {self._name}"

        async def solve(self, task, model, memory, tools, workspace_dir):
            pass

    OrchestrationRegistry._registry[name] = DummyPattern
    return DummyPattern


def test_registry_register_and_list():
    _register_dummy("test-a")
    _register_dummy("test-b")
    available = OrchestrationRegistry.list_available()
    assert "test-a" in available
    assert "test-b" in available


def test_registry_get():
    _register_dummy("get-test")
    pattern = OrchestrationRegistry.get("get-test")
    assert pattern.name() == "get-test"


def test_registry_get_not_found():
    with pytest.raises(PatternNotFoundError):
        OrchestrationRegistry.get("nonexistent")


def test_registry_duplicate_rejection():
    _register_dummy("dup-test")
    with pytest.raises(DuplicatePatternError):

        @OrchestrationRegistry.register
        class DupPattern(OrchestrationPattern):
            def name(self):
                return "dup-test"

            def description(self):
                return "duplicate"

            async def solve(self, task, model, memory, tools, workspace_dir):
                pass


# ── SingleAgent Tests ──


@pytest.mark.asyncio
async def test_single_agent_solve():
    from ant_coding.orchestration.examples.single_agent import SingleAgent

    pattern = SingleAgent()
    assert pattern.name() == "single-agent"
    assert len(pattern.get_agent_definitions()) == 1
    assert pattern.get_agent_definitions()[0]["name"] == "SoloAgent"

    task = _make_task()
    model = _make_mock_model(["def fix(): return x + 1"])
    memory = _make_memory()

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.task_id == "test-task-1"
    assert result.success is True
    assert result.total_tokens > 0
    model.complete.assert_called_once()


# ── MinimalSequential Tests ──


@pytest.mark.asyncio
async def test_sequential_solve_shared_memory():
    from ant_coding.orchestration.examples.sequential import MinimalSequential

    pattern = MinimalSequential()
    assert pattern.name() == "minimal-sequential"

    task = _make_task()
    model = _make_mock_model(["Step 1: Read file\nStep 2: Fix bug", "def fix(): pass"])
    memory = _make_memory(MemoryMode.SHARED)

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.task_id == "test-task-1"
    assert result.total_tokens > 0
    assert len(result.agent_traces) == 2
    assert model.complete.call_count == 2

    # In shared mode, coder can read the plan
    plan = memory.read("coder", "implementation_plan")
    assert plan is not None
    assert "Step 1" in plan


@pytest.mark.asyncio
async def test_sequential_solve_isolated_memory():
    from ant_coding.orchestration.examples.sequential import MinimalSequential

    pattern = MinimalSequential()
    task = _make_task()
    model = _make_mock_model(["Plan: fix the bug", "Code: done"])
    memory = _make_memory(MemoryMode.ISOLATED)

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.success is True
    # In isolated mode, coder CANNOT read planner's data
    plan = memory.read("coder", "implementation_plan")
    assert plan is None


# ── MinimalParallel Tests ──


@pytest.mark.asyncio
async def test_parallel_concurrent_execution():
    from ant_coding.orchestration.examples.parallel import MinimalParallel

    pattern = MinimalParallel()
    assert pattern.name() == "minimal-parallel"
    assert len(pattern.get_agent_definitions()) >= 2

    task = _make_task()
    model = _make_mock_model(["Analysis output", "Implementation output", "Merged result"])
    memory = _make_memory()

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.success is True
    assert result.total_tokens > 0
    # 3 calls: analyzer, implementer (concurrent), then merger
    assert model.complete.call_count == 3
    assert len(result.agent_traces) == 3


# ── MinimalLoop Tests ──


@pytest.mark.asyncio
async def test_loop_iteration_success_on_third():
    from ant_coding.orchestration.examples.loop import MinimalLoop

    pattern = MinimalLoop(max_iterations=5)
    assert pattern.name() == "minimal-loop"

    task = _make_task()
    model = _make_mock_model(["attempt 1", "attempt 2", "attempt 3"])
    memory = _make_memory()

    call_count = 0

    async def mock_run_tests(tools, workspace_dir):
        nonlocal call_count
        call_count += 1
        return call_count >= 3  # Pass on 3rd attempt

    pattern._run_tests = mock_run_tests

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.success is True
    assert result.intermediate_test_results == [False, False, True]
    assert model.complete.call_count == 3


@pytest.mark.asyncio
async def test_loop_max_iterations_reached():
    from ant_coding.orchestration.examples.loop import MinimalLoop

    pattern = MinimalLoop(max_iterations=5)
    task = _make_task()
    model = _make_mock_model(["fail"] * 5)
    memory = _make_memory()

    # Tests always fail
    pattern._run_tests = AsyncMock(return_value=False)

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.success is False
    assert len(result.intermediate_test_results) == 5
    assert all(r is False for r in result.intermediate_test_results)
    assert model.complete.call_count == 5


@pytest.mark.asyncio
async def test_loop_intermediate_results_tracking():
    from ant_coding.orchestration.examples.loop import MinimalLoop

    pattern = MinimalLoop(max_iterations=4)
    task = _make_task()
    model = _make_mock_model(["a", "b", "c", "d"])
    memory = _make_memory()

    results_sequence = [False, True]  # Pass on 2nd
    call_idx = 0

    async def mock_tests(tools, workspace_dir):
        nonlocal call_idx
        r = results_sequence[min(call_idx, len(results_sequence) - 1)]
        call_idx += 1
        return r

    pattern._run_tests = mock_tests

    result = await pattern.solve(task, model, memory, {}, "/tmp/ws")

    assert result.intermediate_test_results == [False, True]
    assert result.success is True


# ── Pattern Name Uniqueness ──


def test_all_pattern_names_unique():
    """All reference patterns have distinct names."""
    from ant_coding.orchestration.examples.loop import MinimalLoop
    from ant_coding.orchestration.examples.parallel import MinimalParallel
    from ant_coding.orchestration.examples.sequential import MinimalSequential
    from ant_coding.orchestration.examples.single_agent import SingleAgent

    names = [
        SingleAgent().name(),
        MinimalSequential().name(),
        MinimalParallel().name(),
        MinimalLoop().name(),
    ]
    assert len(names) == len(set(names))

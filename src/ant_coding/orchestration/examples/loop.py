"""
MinimalLoop: iterative refinement pattern.

A single agent iteratively solves the task, running tests after each attempt.
Continues until tests pass or max_iterations is reached. Tracks
intermediate_test_results for error_recovery_rate calculation.
"""

import logging
from typing import Dict, Any, List

from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

SOLVER_SYSTEM_PROMPT = (
    "You are an iterative coding agent. Given a task and possibly feedback "
    "from previous attempts, write or improve the code to solve the task. "
    "Use the available tools to write files and fix issues."
)

DEFAULT_MAX_ITERATIONS = 5


@OrchestrationRegistry.register
class MinimalLoop(OrchestrationPattern):
    """Iterative refinement: solve, test, fix until tests pass or max iterations."""

    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        """
        Initialize MinimalLoop.

        Args:
            max_iterations: Maximum number of solve-test-fix cycles.
        """
        self.max_iterations = max_iterations

    def name(self) -> str:
        return "minimal-loop"

    def description(self) -> str:
        return (
            "Iterative refinement pattern. Agent solves, tests run, agent fixes "
            f"based on feedback. Up to {self.max_iterations} iterations."
        )

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        return [
            {"name": "Solver", "role": "Iterative task solver with test feedback"},
        ]

    async def solve(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        tools: Dict[str, Any],
        workspace_dir: str,
    ) -> TaskResult:
        """
        Iteratively solve the task until tests pass.

        Args:
            task: The task to solve.
            model: LLM provider.
            memory: Memory manager.
            tools: Tool instances.
            workspace_dir: Workspace path.

        Returns:
            TaskResult with intermediate_test_results tracking all iterations.
        """
        agent_id = "solver"
        traces: List[Dict[str, Any]] = []
        intermediate_test_results: List[bool] = []
        feedback = ""

        for iteration in range(self.max_iterations):
            # Build prompt with task and any previous feedback
            user_content = f"Task: {task.description}"
            if feedback:
                user_content += f"\n\nPrevious attempt feedback:\n{feedback}"

            messages = [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            response = await model.complete(messages)
            output = response.choices[0].message.content
            traces.append({
                "agent": "Solver",
                "action": "solve",
                "iteration": iteration + 1,
                "output": output,
            })

            # Write current attempt to memory
            memory.write(agent_id, f"attempt_{iteration + 1}", output)

            # Run tests (simulated via code_executor if available)
            test_passed = await self._run_tests(tools, workspace_dir)
            intermediate_test_results.append(test_passed)

            if test_passed:
                logger.info(f"Tests passed on iteration {iteration + 1}")
                break

            # Build feedback for next iteration
            feedback = f"Iteration {iteration + 1} failed. Please review and fix the issues."

        usage = model.get_usage()
        success = any(intermediate_test_results)

        return TaskResult(
            task_id=task.id,
            experiment_id=self.name(),
            success=success,
            total_tokens=usage["total_tokens"],
            total_cost=usage["total_cost_usd"],
            agent_traces=traces,
            intermediate_test_results=intermediate_test_results,
        )

    async def _run_tests(
        self, tools: Dict[str, Any], workspace_dir: str
    ) -> bool:
        """
        Run tests in the workspace.

        Uses CodeExecutor if available, otherwise returns False
        (tests not yet implemented).

        Args:
            tools: Tool instances dict.
            workspace_dir: Workspace path.

        Returns:
            True if tests pass, False otherwise.
        """
        code_executor = tools.get("code_executor")
        if code_executor is None:
            return False

        try:
            result = await code_executor.run_command(
                "python -m pytest --tb=short -q", cwd=workspace_dir
            )
            return result.get("success", False)
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            return False

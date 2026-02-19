"""
SingleAgent: baseline control group for multi-agent experiments.

Uses a single LLM call to solve the task end-to-end. Provides the
performance floor and the denominator for overhead_ratio calculation.
"""

import logging
from typing import Dict, Any, List

from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

SOLO_AGENT_SYSTEM_PROMPT = (
    "You are a software engineering agent. Given a task description, "
    "analyze the problem, plan your approach, and implement the solution. "
    "Use the available tools to write files, run commands, and verify your work."
)


@OrchestrationRegistry.register
class SingleAgent(OrchestrationPattern):
    """Single-agent baseline — one LLM call, no inter-agent coordination."""

    def name(self) -> str:
        return "single-agent"

    def description(self) -> str:
        return (
            "Single-agent baseline. One agent solves the entire task end-to-end. "
            "Serves as the control group for overhead_ratio calculation."
        )

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        return [{"name": "SoloAgent", "role": "End-to-end task solver"}]

    async def solve(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        tools: Dict[str, Any],
        workspace_dir: str,
    ) -> TaskResult:
        """
        Solve the task with a single model call.

        Args:
            task: The task to solve.
            model: LLM provider for completions.
            memory: Memory manager (writes for contract consistency).
            tools: Dict of tool instances.
            workspace_dir: Path to workspace.

        Returns:
            TaskResult with outcome and usage stats.
        """
        agent_id = "solo_agent"
        traces: List[Dict[str, Any]] = []

        # Single model call with full task context
        messages = [
            {"role": "system", "content": SOLO_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task.description}"},
        ]

        response = await model.complete(messages)
        output = response.choices[0].message.content
        traces.append({"agent": "SoloAgent", "action": "solve", "output": output})

        # Write to memory for framework contract consistency
        memory.write(agent_id, "solution", output)

        # Gather usage
        usage = model.get_usage()

        return TaskResult(
            task_id=task.id,
            experiment_id=self.name(),
            success=True,
            total_tokens=usage["total_tokens"],
            total_cost=usage["total_cost_usd"],
            agent_traces=traces,
        )

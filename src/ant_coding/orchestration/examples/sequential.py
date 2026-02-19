"""
MinimalSequential: 2-agent sequential pipeline (Planner -> Coder).

Demonstrates the full orchestration plugin contract with shared/isolated
memory as the research variable. The Planner creates an implementation plan,
the Coder reads it and generates code.
"""

import logging
from typing import Dict, Any, List

from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = (
    "You are a planning agent. Given a task description, create a step-by-step "
    "implementation plan. Be specific about which files to create or modify "
    "and what each change should accomplish. Output only the plan."
)

CODER_SYSTEM_PROMPT = (
    "You are a coding agent. You will be given a task and possibly an "
    "implementation plan. Write the code to solve the task. "
    "Use the available tools to write files and run tests."
)


@OrchestrationRegistry.register
class MinimalSequential(OrchestrationPattern):
    """Two-agent sequential pipeline: Planner creates a plan, Coder implements it."""

    def name(self) -> str:
        return "minimal-sequential"

    def description(self) -> str:
        return (
            "2-agent sequential pipeline. Planner writes an implementation plan "
            "to memory, Coder reads it and generates code. Memory mode determines "
            "whether the plan is visible to the Coder (shared vs isolated)."
        )

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        return [
            {"name": "Planner", "role": "Creates implementation plan from task description"},
            {"name": "Coder", "role": "Implements code based on plan and task"},
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
        Execute the Planner -> Coder pipeline.

        Args:
            task: The task to solve.
            model: LLM provider for completions.
            memory: Memory manager (shared/isolated/hybrid).
            tools: Dict of tool instances.
            workspace_dir: Path to workspace.

        Returns:
            TaskResult with outcome and usage stats.
        """
        traces: List[Dict[str, Any]] = []
        planner_id = "planner"
        coder_id = "coder"

        # Phase 1: Planner creates implementation plan
        planner_messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task.description}"},
        ]

        planner_response = await model.complete(planner_messages)
        plan = planner_response.choices[0].message.content
        traces.append({"agent": "Planner", "action": "plan", "output": plan})

        # Write plan to memory
        memory.write(planner_id, "implementation_plan", plan)

        # Phase 2: Coder reads plan and implements
        retrieved_plan = memory.read(coder_id, "implementation_plan")

        coder_context = f"Task: {task.description}"
        if retrieved_plan:
            coder_context += f"\n\nImplementation Plan:\n{retrieved_plan}"

        coder_messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": coder_context},
        ]

        coder_response = await model.complete(coder_messages)
        code_output = coder_response.choices[0].message.content
        traces.append({"agent": "Coder", "action": "implement", "output": code_output})

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

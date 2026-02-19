"""
MinimalParallel: fan-out pattern with concurrent agents.

Multiple agents work on the task simultaneously (via asyncio.gather),
then results are merged. Demonstrates parallel orchestration.
"""

import asyncio
import logging
from typing import Dict, Any, List

from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

ANALYZER_PROMPT = (
    "You are a code analysis agent. Analyze the task and identify "
    "the key files, functions, and patterns involved. "
    "Output your analysis concisely."
)

IMPLEMENTER_PROMPT = (
    "You are an implementation agent. Given a task description, "
    "write the code to solve it. Focus on correctness and clarity."
)

MERGER_PROMPT = (
    "You are a merge agent. Given analysis and implementation from "
    "two parallel agents, synthesize the best solution. "
    "Combine insights from the analysis with the implementation."
)


@OrchestrationRegistry.register
class MinimalParallel(OrchestrationPattern):
    """Fan-out pattern: Analyzer and Implementer run concurrently, Merger combines."""

    def name(self) -> str:
        return "minimal-parallel"

    def description(self) -> str:
        return (
            "Parallel fan-out pattern. An Analyzer and Implementer work "
            "concurrently on the task, then a Merger combines their outputs."
        )

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        return [
            {"name": "Analyzer", "role": "Analyzes task and identifies key patterns"},
            {"name": "Implementer", "role": "Writes initial implementation"},
            {"name": "Merger", "role": "Combines analysis and implementation"},
        ]

    async def _run_agent(
        self,
        agent_id: str,
        system_prompt: str,
        user_content: str,
        model: ModelProvider,
        memory: MemoryManager,
        memory_key: str,
    ) -> str:
        """Run a single agent and write output to memory."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = await model.complete(messages)
        output = response.choices[0].message.content
        memory.write(agent_id, memory_key, output)
        return output

    async def solve(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        tools: Dict[str, Any],
        workspace_dir: str,
    ) -> TaskResult:
        """
        Execute parallel fan-out then merge.

        Args:
            task: The task to solve.
            model: LLM provider.
            memory: Memory manager.
            tools: Tool instances.
            workspace_dir: Workspace path.

        Returns:
            TaskResult with merged outcome.
        """
        traces: List[Dict[str, Any]] = []

        # Phase 1: Fan-out — run Analyzer and Implementer concurrently
        analysis_task = self._run_agent(
            "analyzer", ANALYZER_PROMPT,
            f"Task: {task.description}", model, memory, "analysis",
        )
        impl_task = self._run_agent(
            "implementer", IMPLEMENTER_PROMPT,
            f"Task: {task.description}", model, memory, "implementation",
        )

        analysis, implementation = await asyncio.gather(analysis_task, impl_task)
        traces.append({"agent": "Analyzer", "action": "analyze", "output": analysis})
        traces.append({"agent": "Implementer", "action": "implement", "output": implementation})

        # Phase 2: Merge
        merger_context = (
            f"Task: {task.description}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Implementation:\n{implementation}"
        )
        merge_messages = [
            {"role": "system", "content": MERGER_PROMPT},
            {"role": "user", "content": merger_context},
        ]
        merge_response = await model.complete(merge_messages)
        merged = merge_response.choices[0].message.content
        traces.append({"agent": "Merger", "action": "merge", "output": merged})
        memory.write("merger", "final_solution", merged)

        usage = model.get_usage()

        return TaskResult(
            task_id=task.id,
            experiment_id=self.name(),
            success=True,
            total_tokens=usage["total_tokens"],
            total_cost=usage["total_cost_usd"],
            agent_traces=traces,
        )

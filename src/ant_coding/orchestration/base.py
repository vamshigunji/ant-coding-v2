"""
Abstract base class for orchestration patterns.

All orchestration patterns (single-agent, sequential, parallel, loop, etc.)
must subclass OrchestrationPattern and implement the solve() method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager


class OrchestrationPattern(ABC):
    """
    Abstract base class that defines the plugin contract for orchestration patterns.

    Subclasses must implement:
        - name() -> str
        - description() -> str
        - solve() -> TaskResult
    """

    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this orchestration pattern."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the pattern."""
        ...

    @abstractmethod
    async def solve(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        tools: Dict[str, Any],
        workspace_dir: str,
    ) -> TaskResult:
        """
        Execute the orchestration pattern to solve a task.

        Args:
            task: The task to solve.
            model: The LLM provider for completions.
            memory: The memory manager for inter-agent state.
            tools: Dict of tool instances (from ToolRegistry.as_dict()).
            workspace_dir: Path to the task workspace directory.

        Returns:
            TaskResult with outcome, token usage, and traces.
        """
        ...

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        """
        Return the list of agents used by this pattern.

        Each agent is a dict with at least 'name' and 'role' keys.
        Override in subclasses to declare agents.

        Returns:
            List of agent definition dicts. Empty list by default.
        """
        return []

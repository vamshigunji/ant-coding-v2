"""
Tool registry that wires all tool instances together for a workspace.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from ant_coding.tools.code_executor import CodeExecutor
from ant_coding.tools.file_ops import FileOperations
from ant_coding.tools.git_ops import GitOperations
from ant_coding.tools.search import CodebaseSearch

if TYPE_CHECKING:
    from ant_coding.observability.event_logger import EventLogger


class ToolRegistry:
    """
    Central registry providing access to all tool instances scoped to a workspace.

    Args:
        workspace_dir: The workspace directory that file, git, and search operations are scoped to.
        code_timeout: Timeout in seconds for code execution.
        event_logger: Optional EventLogger for TOOL_CALL event logging.
        experiment_id: Experiment ID for event context.
        task_id: Task ID for event context.
    """

    def __init__(
        self,
        workspace_dir: Union[str, Path],
        code_timeout: int = 30,
        event_logger: Optional["EventLogger"] = None,
        experiment_id: str = "",
        task_id: str = "",
    ):
        """
        Initialize all tools scoped to the given workspace directory.

        Args:
            workspace_dir: Root directory for workspace-scoped operations.
            code_timeout: Timeout for code execution in seconds.
            event_logger: Optional EventLogger for TOOL_CALL events.
            experiment_id: Experiment ID for event context.
            task_id: Task ID for event context.
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.code_executor = CodeExecutor(timeout=code_timeout)
        self.file_ops = FileOperations(workspace_dir)
        self.git_ops = GitOperations(workspace_dir)
        self.search = CodebaseSearch(workspace_dir)
        self._event_logger = event_logger
        self._experiment_id = experiment_id
        self._task_id = task_id

    def log_tool_call(
        self,
        tool_name: str,
        method: str,
        args_summary: str,
        success: bool,
        duration_ms: float,
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Log a TOOL_CALL event.

        Args:
            tool_name: Name of the tool (e.g., "file_ops").
            method: Method called (e.g., "edit_file").
            args_summary: Brief summary of the arguments.
            success: Whether the call succeeded.
            duration_ms: Call duration in milliseconds.
            agent_id: Optional agent that invoked the tool.
        """
        if self._event_logger is None:
            return

        from ant_coding.observability.event_logger import Event, EventType

        self._event_logger.log(Event(
            type=EventType.TOOL_CALL,
            task_id=self._task_id,
            experiment_id=self._experiment_id,
            agent_id=agent_id,
            payload={
                "tool_name": tool_name,
                "method": method,
                "args_summary": args_summary,
                "success": success,
                "duration_ms": round(duration_ms, 1),
            },
        ))

    def as_dict(self) -> Dict[str, Any]:
        """
        Return all tool instances as a dictionary.

        Returns:
            Dict mapping tool names to their instances.
        """
        return {
            "code_executor": self.code_executor,
            "file_ops": self.file_ops,
            "git_ops": self.git_ops,
            "search": self.search,
        }

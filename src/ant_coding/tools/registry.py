"""
Tool registry that wires all tool instances together for a workspace.
"""

from pathlib import Path
from typing import Dict, Any, Union

from ant_coding.tools.code_executor import CodeExecutor
from ant_coding.tools.file_ops import FileOperations
from ant_coding.tools.git_ops import GitOperations
from ant_coding.tools.search import CodebaseSearch


class ToolRegistry:
    """
    Central registry providing access to all tool instances scoped to a workspace.

    Args:
        workspace_dir: The workspace directory that file, git, and search operations are scoped to.
        code_timeout: Timeout in seconds for code execution.
    """

    def __init__(
        self,
        workspace_dir: Union[str, Path],
        code_timeout: int = 30,
    ):
        """
        Initialize all tools scoped to the given workspace directory.

        Args:
            workspace_dir: Root directory for workspace-scoped operations.
            code_timeout: Timeout for code execution in seconds.
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.code_executor = CodeExecutor(timeout=code_timeout)
        self.file_ops = FileOperations(workspace_dir)
        self.git_ops = GitOperations(workspace_dir)
        self.search = CodebaseSearch(workspace_dir)

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

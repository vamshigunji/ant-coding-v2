"""
MCP (Model Context Protocol) server for exposing framework tools.

Wraps CodeExecutor, FileOperations, GitOperations, and CodebaseSearch
as MCP-compliant tool endpoints.

Reference: docs/prd.md Section 8, Sprint-6-Epic-2.md (S6-E2-S01)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ant_coding.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Tool definitions with name, description, and input schema
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "code_execute",
        "description": "Execute code in a sandboxed environment with timeout.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {
                    "type": "string",
                    "description": "Programming language (python, bash)",
                    "default": "python",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "file_read",
        "description": "Read the contents of a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": "Write content to a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_list",
        "description": "List files in a directory within the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "git_diff",
        "description": "Show the git diff of current changes.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "git_apply_patch",
        "description": "Apply a unified diff patch to the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff patch content"},
            },
            "required": ["patch"],
        },
    },
    {
        "name": "search_code",
        "description": "Search for a pattern in the codebase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex)"},
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py')",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
]


class MCPToolServer:
    """
    MCP-compliant tool server wrapping framework tools.

    Provides tool registration, listing, and invocation with
    structured error handling.
    """

    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the MCP server with a ToolRegistry.

        Args:
            tool_registry: The tool registry providing tool instances.
        """
        self._registry = tool_registry
        self._handlers: Dict[str, Callable] = {
            "code_execute": self._handle_code_execute,
            "file_read": self._handle_file_read,
            "file_write": self._handle_file_write,
            "file_list": self._handle_file_list,
            "git_diff": self._handle_git_diff,
            "git_apply_patch": self._handle_git_apply_patch,
            "search_code": self._handle_search_code,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return MCP tool definitions for all registered tools.

        Returns:
            List of tool definition dicts with name, description, input_schema.
        """
        return TOOL_DEFINITIONS

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke an MCP tool by name with arguments.

        Args:
            name: Tool name (must match a registered tool).
            arguments: Tool input arguments matching the input_schema.

        Returns:
            Dict with "content" (result) or "error" (error message).
        """
        handler = self._handlers.get(name)
        if not handler:
            return _error_response(f"Unknown tool: {name}")

        try:
            result = handler(arguments)
            return {"content": result, "is_error": False}
        except Exception as e:
            logger.warning(f"MCP tool '{name}' failed: {e}")
            return _error_response(str(e))

    # ── Tool handlers ──

    def _handle_code_execute(self, args: Dict[str, Any]) -> str:
        """Execute code via CodeExecutor."""
        code = args["code"]
        language = args.get("language", "python")
        result = self._registry.code_executor.execute(code, language=language)
        return result.output if hasattr(result, "output") else str(result)

    def _handle_file_read(self, args: Dict[str, Any]) -> str:
        """Read file via FileOperations."""
        return self._registry.file_ops.read(args["path"])

    def _handle_file_write(self, args: Dict[str, Any]) -> str:
        """Write file via FileOperations."""
        self._registry.file_ops.write(args["path"], args["content"])
        return f"Written to {args['path']}"

    def _handle_file_list(self, args: Dict[str, Any]) -> List[str]:
        """List files via FileOperations."""
        path = args.get("path", ".")
        return self._registry.file_ops.list_files(path)

    def _handle_git_diff(self, args: Dict[str, Any]) -> str:
        """Get git diff via GitOperations."""
        return self._registry.git_ops.diff()

    def _handle_git_apply_patch(self, args: Dict[str, Any]) -> str:
        """Apply patch via GitOperations."""
        self._registry.git_ops.apply_patch(args["patch"])
        return "Patch applied successfully"

    def _handle_search_code(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search code via CodebaseSearch."""
        pattern = args["pattern"]
        file_pattern = args.get("file_pattern", "*")
        return self._registry.search.search(pattern, file_pattern=file_pattern)


def _error_response(message: str) -> Dict[str, Any]:
    """Create an MCP error response."""
    return {"content": message, "is_error": True}

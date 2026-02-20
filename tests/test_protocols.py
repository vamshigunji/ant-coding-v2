"""
Tests for MCP and A2A protocol layers.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ant_coding.protocols.mcp_server import MCPToolServer, TOOL_DEFINITIONS
from ant_coding.protocols.a2a_server import A2AServer, AgentCard


# ── MCP Tests ──


def _mock_tool_registry():
    """Create a mock ToolRegistry for MCP testing."""
    registry = MagicMock()
    registry.code_executor.execute.return_value = MagicMock(output="Hello, World!")
    registry.file_ops.read.return_value = "file content"
    registry.file_ops.write.return_value = None
    registry.file_ops.list_files.return_value = ["a.py", "b.py"]
    registry.git_ops.diff.return_value = "diff --git ..."
    registry.git_ops.apply_patch.return_value = None
    registry.search.search.return_value = [{"file": "a.py", "line": 1, "match": "hello"}]
    return registry


def test_mcp_list_tools():
    """list_tools returns all tool definitions."""
    server = MCPToolServer(_mock_tool_registry())
    tools = server.list_tools()
    assert len(tools) == 7
    names = {t["name"] for t in tools}
    assert "code_execute" in names
    assert "file_read" in names
    assert "search_code" in names


def test_mcp_tool_has_schema():
    """Each tool has name, description, and input_schema."""
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_mcp_call_code_execute():
    """call_tool('code_execute') delegates to CodeExecutor."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("code_execute", {"code": "print('hi')"})
    assert result["is_error"] is False
    assert "Hello" in result["content"]


def test_mcp_call_file_read():
    """call_tool('file_read') delegates to FileOperations."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("file_read", {"path": "test.py"})
    assert result["is_error"] is False
    assert result["content"] == "file content"


def test_mcp_call_file_write():
    """call_tool('file_write') delegates to FileOperations."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("file_write", {"path": "test.py", "content": "new"})
    assert result["is_error"] is False


def test_mcp_call_file_list():
    """call_tool('file_list') returns file list."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("file_list", {"path": "."})
    assert result["is_error"] is False
    assert "a.py" in result["content"]


def test_mcp_call_git_diff():
    """call_tool('git_diff') returns diff output."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("git_diff", {})
    assert result["is_error"] is False
    assert "diff" in result["content"]


def test_mcp_call_search_code():
    """call_tool('search_code') returns search results."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("search_code", {"pattern": "hello"})
    assert result["is_error"] is False


def test_mcp_call_unknown_tool():
    """Unknown tool returns error response."""
    server = MCPToolServer(_mock_tool_registry())
    result = server.call_tool("nonexistent", {})
    assert result["is_error"] is True
    assert "Unknown tool" in result["content"]


def test_mcp_call_tool_exception():
    """Tool exception returns error response, not crash."""
    registry = _mock_tool_registry()
    registry.code_executor.execute.side_effect = RuntimeError("sandbox failed")
    server = MCPToolServer(registry)
    result = server.call_tool("code_execute", {"code": "boom"})
    assert result["is_error"] is True
    assert "sandbox failed" in result["content"]


# ── A2A Tests ──


def test_agent_card_to_dict():
    """AgentCard serializes to dict."""
    card = AgentCard(
        name="test-agent",
        description="A test agent",
        capabilities=["coding"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    d = card.to_dict()
    assert d["name"] == "test-agent"
    assert d["capabilities"] == ["coding"]


def test_a2a_register_pattern():
    """register_pattern creates an AgentCard."""
    server = A2AServer()
    pattern = MagicMock()
    pattern.name.return_value = "test-pattern"
    pattern.description.return_value = "A test pattern"
    pattern.get_agent_definitions.return_value = [
        {"name": "Agent1", "role": "coder"},
        {"name": "Agent2", "role": "reviewer"},
    ]

    card = server.register_pattern(pattern)
    assert card.name == "test-pattern"
    assert card.metadata["agent_count"] == 2
    assert "Agent1" in card.metadata["agent_names"]


def test_a2a_discover():
    """discover() returns all registered Agent Cards."""
    server = A2AServer()
    pattern1 = MagicMock()
    pattern1.name.return_value = "pattern-a"
    pattern1.description.return_value = "A"
    pattern1.get_agent_definitions.return_value = []

    pattern2 = MagicMock()
    pattern2.name.return_value = "pattern-b"
    pattern2.description.return_value = "B"
    pattern2.get_agent_definitions.return_value = []

    server.register_pattern(pattern1)
    server.register_pattern(pattern2)

    cards = server.discover()
    assert len(cards) == 2
    names = {c["name"] for c in cards}
    assert "pattern-a" in names
    assert "pattern-b" in names


def test_a2a_get_agent():
    """get_agent returns card for known agent, None for unknown."""
    server = A2AServer()
    pattern = MagicMock()
    pattern.name.return_value = "my-agent"
    pattern.description.return_value = "desc"
    pattern.get_agent_definitions.return_value = []

    server.register_pattern(pattern)
    assert server.get_agent("my-agent") is not None
    assert server.get_agent("nonexistent") is None


@pytest.mark.asyncio
async def test_a2a_submit_task_unknown_agent():
    """submit_task with unknown agent returns error."""
    server = A2AServer()
    result = await server.submit_task("nonexistent", {"task_description": "fix bug"})
    assert result["success"] is False
    assert "not found" in result["error"]

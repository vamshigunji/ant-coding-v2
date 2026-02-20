# Branch Summary: feature/S6-E2-S01

## Story
S6-E2-S01: MCP Tool Wrapping

## What Changed
- Created `src/ant_coding/protocols/__init__.py`
- Created `src/ant_coding/protocols/mcp_server.py` with:
  - `TOOL_DEFINITIONS`: 7 MCP tool schemas (code_execute, file_read, file_write, file_list, git_diff, git_apply_patch, search_code)
  - `MCPToolServer`: MCP-compliant server wrapping ToolRegistry
  - `list_tools()`: Returns all tool definitions with input schemas
  - `call_tool()`: Invokes tools by name with structured error handling
  - Handler methods for each tool delegating to ToolRegistry instances

## Key Decisions
- Tool definitions follow MCP input_schema format (JSON Schema)
- Error handling returns `{content, is_error}` instead of raising (MCP convention)
- No MCP SDK dependency — tool definitions and invocation are framework-native, can be wrapped by any MCP SDK

## Files Touched
- `src/ant_coding/protocols/__init__.py` (new)
- `src/ant_coding/protocols/mcp_server.py` (new)
- `.agent/sprint.yml` (S6-E2-S01 done, S6-E2 in-progress)

## How to Verify
```bash
pytest tests/ -v  # full suite: 224 passed, 1 skipped
```

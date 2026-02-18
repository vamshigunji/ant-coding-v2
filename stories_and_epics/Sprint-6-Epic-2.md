# Sprint 6 — Epic 2: Protocol Layer (MCP + A2A)

**Epic ID:** S6-E2  
**Sprint:** 6  
**Priority:** P2 — Polish  
**Goal:** Wrap tools as MCP servers and register agents as A2A endpoints. This enables cross-framework experiments in the future.

**Dependencies:** S3-E2 (tools), S4-E1 (orchestration)

---

## Story S6-E2-S01: MCP Tool Wrapping

**Branch:** `feature/S6-E2-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given the ToolRegistry with CodeExecutor, FileOps, GitOps, Search
When I call registry.as_mcp_server()
Then it returns an MCP server instance that exposes all tools
And each tool method is callable via the MCP protocol

Given an MCP client connects to the server
When it calls list_tools()
Then it returns tool descriptors for: execute_code, read_file, write_file, edit_file, get_diff, grep, etc.

Given an MCP client
When it calls execute_code(code="print('hello')", language="python")
Then it returns the execution result through MCP protocol
```

**Files to Create:**
- `src/ant_coding/tools/mcp_server.py`

---

## Story S6-E2-S02: A2A Agent Registration

**Branch:** `feature/S6-E2-S02`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given an OrchestrationPattern registered in the framework
When I call register_as_a2a(pattern)
Then it creates an A2A Agent Card with:
  - name: pattern.name()
  - description: pattern.description()
  - capabilities: list of supported task types

Given an A2A-registered agent
When a remote A2A client sends a task request
Then the agent receives it, runs solve(), and returns results via A2A protocol

Given the A2A server is running
When I call discover_agents()
Then it returns Agent Cards for all registered patterns
```

**Files to Create:**
- `src/ant_coding/protocols/a2a_server.py`
- `src/ant_coding/protocols/__init__.py`

---

## Story S6-E2-S03: Protocol Tests

**Branch:** `feature/S6-E2-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
When I run `pytest tests/test_protocols.py -v`
Then all tests pass covering:
  - MCP server tool listing
  - MCP tool invocation (mocked)
  - A2A agent card generation
  - A2A task dispatch (mocked)
```

**Files to Create:**
- `tests/test_protocols.py`

---

## Epic Completion Checklist

- [ ] MCP server exposes all tools via standard protocol
- [ ] A2A Agent Cards generated from orchestration patterns
- [ ] `pytest tests/test_protocols.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

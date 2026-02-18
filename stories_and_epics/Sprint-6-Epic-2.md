# Sprint 6 — Epic 2: Protocol Layer (MCP + A2A)

**Epic ID:** S6-E2  
**Sprint:** 6  
**Priority:** P3 — Extension  
**Goal:** Wrap framework tools as MCP endpoints and register agents via A2A protocol. After this epic, ant-coding tools can be accessed by external clients (Claude Desktop, etc.) and agents can be discovered by external orchestrators.

**Dependencies:** S3-E2 (tools), S4-E1 (orchestration patterns)  
**Reference:** `docs/prd.md` Section 8 (Protocol Layer)

---

## Story S6-E2-S01: MCP Tool Wrapping

**Branch:** `feature/S6-E2-S01`  
**Points:** 5

**Description:**  
Expose CodeExecutor, FileOperations, GitOperations, and CodebaseSearch as MCP-compliant tool endpoints using the MCP SDK.

**Acceptance Criteria:**

```gherkin
Given the ant-coding framework tools
When I start the MCP server
Then each tool is registered with name, description, and input schema

Given an MCP client (e.g., Claude Desktop)
When it calls the "code_execute" tool via MCP protocol
Then it executes code in the sandbox and returns structured output

Given an MCP tool call that fails
When the error propagates
Then it returns a proper MCP error response (not a crash)

Given all MCP-wrapped tools
When I call list_tools() via MCP
Then it returns tool definitions matching the ToolRegistry entries
```

**Files to Create:**
- `src/ant_coding/protocols/mcp_server.py`

---

## Story S6-E2-S02: A2A Agent Registration

**Branch:** `feature/S6-E2-S02`  
**Points:** 5

**Description:**  
Register orchestration patterns as discoverable A2A agents with Agent Cards.

**Acceptance Criteria:**

```gherkin
Given a registered OrchestrationPattern (e.g., "minimal-sequential")
When I register it as an A2A agent
Then it generates an Agent Card with: name, description, capabilities, input/output schema

Given an A2A discovery request
When a client queries available agents
Then it returns Agent Cards for all registered patterns including "single-agent" (PRD+)

Given an A2A task submission
When a client sends a Task with task description
Then the framework routes it to the specified agent and returns the result
```

**Files to Create:**
- `src/ant_coding/protocols/a2a_server.py`

---

## Story S6-E2-S03: Protocol Tests

**Branch:** `feature/S6-E2-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
When I run `pytest tests/test_protocols.py -v`
Then all tests pass covering:
  - MCP tool registration and schema generation
  - MCP tool invocation with mock tools
  - MCP error handling
  - A2A agent card generation
  - A2A agent discovery
  - A2A task routing
```

**Files to Create:**
- `tests/test_protocols.py`

---

## Epic Completion Checklist

- [ ] MCP server exposes all framework tools
- [ ] A2A server exposes all orchestration patterns (including SingleAgent)
- [ ] Protocol error handling is robust
- [ ] `pytest tests/test_protocols.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
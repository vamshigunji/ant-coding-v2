# Sprint 5 — Epic 1: Event Logger & Observability

**Epic ID:** S5-E1  
**Sprint:** 5  
**Priority:** P1 — Core  
**Goal:** Build the immutable event logging system that records every LLM call, tool call, and memory access. After this epic, every experiment produces a complete JSONL event trace.

**Dependencies:** S4-E2 (runner — events are logged during execution)

---

## Story S5-E1-S01: EventLogger with JSONL Output

**Branch:** `feature/S5-E1-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given an EventLogger initialized with experiment_id="exp-001"
When I call log(Event(type=EventType.LLM_CALL, agent_name="planner", ...))
Then the event is appended to results/exp-001/events.jsonl
And each line in the JSONL file is a valid JSON object

Given an EventLogger with 10 logged events
When I call get_events()
Then it returns all 10 events in chronological order

Given an EventLogger with events from "planner" and "coder" agents
When I call get_events(agent_name="planner")
Then it returns only events from the planner agent

Given an EventLogger with multiple LLM_CALL events containing token data
When I call get_token_breakdown()
Then it returns a dict like:
  {"planner": {"prompt": N, "completion": N, "total": N}, "coder": {...}}
And the totals are correct sums of individual call tokens
```

**Files to Create:**
- `src/ant_coding/observability/event_logger.py` (full implementation, replacing skeleton)

---

## Story S5-E1-S02: Wire EventLogger into Runner and Layers

**Branch:** `feature/S5-E1-S02`  
**Points:** 5

**Description:**  
Integrate the EventLogger into ModelProvider, MemoryManager, ToolRegistry, and ExperimentRunner so events are automatically logged during execution.

**Acceptance Criteria:**

```gherkin
Given an ExperimentRunner with EventLogger enabled
When a ModelProvider.complete() call is made
Then an LLM_CALL event is logged with: model, tokens, cost, duration_ms

Given an ExperimentRunner with EventLogger enabled
When MemoryManager.write() is called
Then a MEMORY_WRITE event is logged with: agent, key, resolved_key, value_size

Given an ExperimentRunner with EventLogger enabled
When MemoryManager.read() is called
Then a MEMORY_READ event is logged with: agent, key, resolved_key, found

Given an ExperimentRunner with EventLogger enabled
When a tool is invoked (e.g., file_ops.edit_file)
Then a TOOL_CALL event is logged with: tool_name, method, args_summary, success

Given an ExperimentRunner that completes a full task
Then events.jsonl contains TASK_START, AGENT_START, LLM_CALL(s), TOOL_CALL(s),
  MEMORY_READ/WRITE(s), AGENT_END, TASK_END in chronological order
```

**Files to Modify:**
- `src/ant_coding/models/provider.py` (add event logging)
- `src/ant_coding/memory/manager.py` (add event logging)
- `src/ant_coding/tools/code_executor.py` (add event logging)
- `src/ant_coding/tools/file_ops.py` (add event logging)
- `src/ant_coding/core/runner.py` (pass logger to all layers)

---

## Story S5-E1-S03: Latency Tracking

**Branch:** `feature/S5-E1-S03`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given an LLM_CALL event
When I inspect its data
Then it includes "duration_ms" with the actual API call latency

Given a TOOL_CALL event
When I inspect its data
Then it includes "duration_ms" with the tool execution time

Given an EventLogger with completed task events
When I calculate total wall time from TASK_START to TASK_END
Then it matches the TaskResult.duration_seconds (within 100ms tolerance)
```

**Files to Modify/Create:**
- `src/ant_coding/observability/latency.py`

---

## Story S5-E1-S04: Observability Tests

**Branch:** `feature/S5-E1-S04`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_observability.py -v`
Then all tests pass with at least 10 test cases covering:
  - JSONL writing and reading
  - Event filtering by agent, type, task
  - Token breakdown calculation
  - Latency tracking
  - Agent timeline ordering
  - JSONL file format validation
```

**Files to Create:**
- `tests/test_observability.py`

---

## Epic Completion Checklist

- [ ] EventLogger writes append-only JSONL with all event types
- [ ] All layers automatically log events during execution
- [ ] Token breakdown per agent is accurate
- [ ] Latency tracked on LLM and tool calls
- [ ] `pytest tests/test_observability.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
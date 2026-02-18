# Sprint 2 — Epic 2: Memory Layer

**Epic ID:** S2-E2  
**Sprint:** 2  
**Priority:** P0 — Foundation (Core Research Variable)  
**Goal:** Build the MemoryManager that enforces shared/isolated/hybrid access patterns via key-prefix routing. This is the dependent variable in the research — it must be bulletproof.

**Dependencies:** S1-E1 (config system, type definitions)

---

## Story S2-E2-S01: MemoryManager Base with Mode Enum

**Branch:** `feature/S2-E2-S01`  
**Points:** 3

**Description:**  
Create the MemoryManager class with MemoryMode enum and the key resolution logic that routes reads/writes to the correct state prefix.

**Acceptance Criteria:**

```gherkin
Given MemoryMode enum is imported
Then it has exactly 3 values: SHARED, ISOLATED, HYBRID

Given a MemoryManager with mode=SHARED
When I call _resolve_key("planner", "plan")
Then it returns "app:plan"

Given a MemoryManager with mode=ISOLATED
When I call _resolve_key("planner", "plan")
Then it returns "temp:planner:plan"

Given a MemoryManager with mode=HYBRID and shared_keys=["plan"]
When I call _resolve_key("planner", "plan")
Then it returns "app:plan"

Given a MemoryManager with mode=HYBRID and shared_keys=["plan"]
When I call _resolve_key("planner", "scratch")
Then it returns "temp:planner:scratch"
```

**Files to Create:**
- `src/ant_coding/memory/manager.py`

---

## Story S2-E2-S02: Shared Memory Implementation

**Branch:** `feature/S2-E2-S02`  
**Points:** 3

**Description:**  
Implement write/read for shared mode. All agents see the same namespace.

**Acceptance Criteria:**

```gherkin
Given a MemoryManager with mode=SHARED
When Agent "planner" writes key "plan" with value "step 1, step 2"
And Agent "coder" reads key "plan"
Then the read returns "step 1, step 2"

Given a MemoryManager with mode=SHARED
When Agent "planner" writes key "plan" with value "v1"
And Agent "reviewer" writes key "plan" with value "v2" (overwrite)
And Agent "coder" reads key "plan"
Then the read returns "v2" (last-write-wins)

Given a MemoryManager with mode=SHARED
When no agent has written key "missing"
And Agent "coder" reads key "missing"
Then the read returns None

Given a MemoryManager with mode=SHARED
When Agent "planner" writes two keys: "plan" and "context"
And I call list_keys("coder")
Then it returns ["plan", "context"] (both visible)
```

**Files to Modify:**
- `src/ant_coding/memory/manager.py`

---

## Story S2-E2-S03: Isolated Memory Implementation

**Branch:** `feature/S2-E2-S03`  
**Points:** 3

**Description:**  
Implement write/read for isolated mode. Each agent can only see its own writes.

**Acceptance Criteria:**

```gherkin
Given a MemoryManager with mode=ISOLATED
When Agent "planner" writes key "plan" with value "step 1, step 2"
And Agent "coder" reads key "plan"
Then the read returns None (coder cannot see planner's data)

Given a MemoryManager with mode=ISOLATED
When Agent "planner" writes key "plan" with value "planner's plan"
And Agent "coder" writes key "plan" with value "coder's plan"
And Agent "planner" reads key "plan"
Then the read returns "planner's plan" (not coder's)

Given a MemoryManager with mode=ISOLATED
When Agent "planner" writes keys "plan" and "context"
And Agent "coder" writes keys "plan" and "draft"
And I call list_keys("planner")
Then it returns ["plan", "context"] (only planner's keys)
And list_keys("coder") returns ["plan", "draft"] (only coder's keys)
```

**Files to Modify:**
- `src/ant_coding/memory/manager.py`

---

## Story S2-E2-S04: Hybrid Memory Implementation

**Branch:** `feature/S2-E2-S04`  
**Points:** 3

**Description:**  
Implement write/read for hybrid mode. Shared_keys are globally visible; everything else is private.

**Acceptance Criteria:**

```gherkin
Given a MemoryManager with mode=HYBRID and shared_keys=["plan", "test_results"]
When Agent "planner" writes key "plan" with value "shared plan"
And Agent "coder" reads key "plan"
Then the read returns "shared plan" (plan is a shared key)

Given a MemoryManager with mode=HYBRID and shared_keys=["plan"]
When Agent "planner" writes key "scratch_notes" with value "private thinking"
And Agent "coder" reads key "scratch_notes"
Then the read returns None (scratch_notes is not in shared_keys)

Given a MemoryManager with mode=HYBRID and shared_keys=["plan"]
When Agent "planner" writes "plan" (shared) and "scratch" (private)
And I call list_keys("coder")
Then it returns ["plan"] (shared key visible, private key not visible)
```

**Files to Modify:**
- `src/ant_coding/memory/manager.py`

---

## Story S2-E2-S05: Memory Access Logging

**Branch:** `feature/S2-E2-S05`  
**Points:** 2

**Description:**  
Every read and write must be recorded in an access log for post-hoc analysis of information flow patterns.

**Acceptance Criteria:**

```gherkin
Given a MemoryManager (any mode)
When Agent "planner" writes key "plan" and Agent "coder" reads key "plan"
Then get_access_log() returns a list of 2 entries
And entry[0] has: action="write", agent="planner", key="plan", resolved_key=<prefixed>
And entry[1] has: action="read", agent="coder", key="plan", found=True/False

Given a MemoryManager
When Agent "coder" reads a key that doesn't exist
Then the access log entry has found=False

Given a MemoryManager with several operations
When I call get_state_snapshot()
Then it returns the full current state dict (all resolved keys and values)

Given a MemoryManager with operations
When I call reset()
Then get_access_log() returns []
And get_state_snapshot() returns {}
```

**Files to Modify:**
- `src/ant_coding/memory/manager.py`

---

## Story S2-E2-S06: Memory Config Loading

**Branch:** `feature/S2-E2-S06`  
**Points:** 2

**Description:**  
Wire the MemoryManager to the config system so it's instantiated from YAML.

**Acceptance Criteria:**

```gherkin
Given configs/memory/shared.yaml with mode: "shared"
When MemoryManager is created from this config
Then manager.mode == MemoryMode.SHARED

Given configs/memory/hybrid.yaml with mode: "hybrid" and shared_keys: ["plan", "tests"]
When MemoryManager is created from this config
Then manager.shared_keys == ["plan", "tests"]

Given a config with mode: "unknown"
When MemoryManager creation is attempted
Then it raises ConfigError
```

**Files to Modify:**
- `src/ant_coding/memory/manager.py` (add factory method)

---

## Story S2-E2-S07: Memory Layer Unit Tests

**Branch:** `feature/S2-E2-S07`  
**Points:** 3

**Description:**  
Comprehensive test suite for all memory modes, access logging, and edge cases.

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_memory.py -v`
Then all tests pass

Given the test file
When I count test functions
Then there are at least 15 test cases covering:
  - Shared: cross-agent visibility
  - Shared: last-write-wins
  - Shared: read nonexistent key
  - Isolated: cross-agent invisibility
  - Isolated: same key different agents
  - Isolated: list_keys scoping
  - Hybrid: shared key visibility
  - Hybrid: private key invisibility
  - Hybrid: list_keys mixed
  - Access log: write recorded
  - Access log: read recorded (found + not-found)
  - State snapshot
  - Reset clears everything
  - Config loading (all 3 modes)
  - Invalid mode rejection
```

**Files to Create:**
- `tests/test_memory.py`

---

## Epic Completion Checklist

- [ ] All 3 memory modes work correctly (shared, isolated, hybrid)
- [ ] Access logging captures every read/write with resolved keys
- [ ] Config-based initialization works for all modes
- [ ] `pytest tests/test_memory.py` passes with 15+ test cases
- [ ] Cross-agent visibility/invisibility is iron-clad
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

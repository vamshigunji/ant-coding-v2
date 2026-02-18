# Sprint 4 — Epic 1: Orchestration Plugin Interface

**Epic ID:** S4-E1  
**Sprint:** 4  
**Priority:** P1 — Core  
**Goal:** Build the abstract base class, registry, and reference implementations for orchestration patterns. After this epic, new architectures can be created by subclassing `OrchestrationPattern` and registering with a decorator.

**Dependencies:** S2-E1 (models), S2-E2 (memory), S3-E2 (tools)

---

## Story S4-E1-S01: OrchestrationPattern Abstract Base Class

**Branch:** `feature/S4-E1-S01`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given I subclass OrchestrationPattern without implementing solve()
When I try to instantiate the subclass
Then it raises TypeError (abstract method not implemented)

Given I subclass OrchestrationPattern and implement name(), description(), and solve()
When I instantiate the subclass
Then it succeeds and name() and description() return strings

Given the OrchestrationPattern base class
When I inspect its solve() signature
Then it accepts: task (Task), model (ModelProvider), memory (MemoryManager), tools (dict), workspace_dir (str)
And it returns TaskResult
And it is an async method

Given a concrete OrchestrationPattern
When I call get_agent_definitions() without overriding
Then it returns an empty list (default behavior)
```

**Files to Create:**
- `src/ant_coding/orchestration/base.py`

---

## Story S4-E1-S02: OrchestrationRegistry with Decorator

**Branch:** `feature/S4-E1-S02`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given a class decorated with @OrchestrationRegistry.register
When I call OrchestrationRegistry.list_available()
Then it includes the registered pattern's name

Given a registered pattern "minimal-sequential"
When I call OrchestrationRegistry.get("minimal-sequential")
Then it returns an instance of that pattern class

Given no pattern registered with name "nonexistent"
When I call OrchestrationRegistry.get("nonexistent")
Then it raises PatternNotFoundError

Given two patterns registered with the same name
When the second registration happens
Then it raises DuplicatePatternError
```

**Files to Create:**
- `src/ant_coding/orchestration/registry.py`

---

## Story S4-E1-S03: MinimalSequential Reference Implementation

**Branch:** `feature/S4-E1-S03`  
**Points:** 5

**Description:**  
Build a working 2-agent sequential pipeline (Planner → Coder) that demonstrates the full plugin contract. This is the baseline architecture all experiments compare against.

**Acceptance Criteria:**

```gherkin
Given the MinimalSequential pattern
When I call pattern.name()
Then it returns "minimal-sequential"

Given MinimalSequential with a mock model that returns predetermined responses
When I call await pattern.solve(task, model, memory, tools, workspace_dir)
Then it returns a TaskResult with:
  - task_id matching the input task
  - experiment_id == "minimal-sequential"
  - total_tokens > 0
  - prompt_tokens > 0

Given MinimalSequential with SHARED memory
When solve() executes
Then the Planner agent writes "implementation_plan" to memory
And the Coder agent reads "implementation_plan" from memory
And the read returns the plan (shared mode)

Given MinimalSequential with ISOLATED memory
When solve() executes
Then the Planner agent writes "implementation_plan" to memory
And the Coder agent reads "implementation_plan" from memory
And the read returns None (isolated mode — demonstrating the difference)

Given MinimalSequential with a model and working tools
When solve() completes
Then model.get_usage()["total_tokens"] > 0 (at least 2 LLM calls were made)
```

**Files to Create:**
- `src/ant_coding/orchestration/examples/sequential.py`

---

## Story S4-E1-S04: MinimalParallel and MinimalLoop Reference Implementations

**Branch:** `feature/S4-E1-S04`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given the MinimalParallel pattern
When I call pattern.name()
Then it returns "minimal-parallel"
And get_agent_definitions() returns at least 2 agents
And the agents run concurrently (asyncio.gather)

Given the MinimalLoop pattern
When I call pattern.name()
Then it returns "minimal-loop"
And get_agent_definitions() returns agents in a loop configuration
And solve() iterates until a quality threshold or max iterations

Given both patterns registered
When I call OrchestrationRegistry.list_available()
Then it includes "minimal-parallel" and "minimal-loop"
```

**Files to Create:**
- `src/ant_coding/orchestration/examples/parallel.py`
- `src/ant_coding/orchestration/examples/loop.py`

---

## Story S4-E1-S05: Orchestration Tests

**Branch:** `feature/S4-E1-S05`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_orchestration.py -v`
Then all tests pass

Given the test file
Then there are at least 10 test cases covering:
  - ABC enforcement (missing methods raises TypeError)
  - Registry: register, get, list, duplicate rejection
  - MinimalSequential: full solve with mocked model
  - MinimalSequential: shared vs isolated memory behavior difference
  - MinimalParallel: concurrent execution
  - MinimalLoop: iteration and termination
```

**Files to Create:**
- `tests/test_orchestration.py`

---

## Epic Completion Checklist

- [ ] OrchestrationPattern ABC enforces solve() implementation
- [ ] OrchestrationRegistry supports decorator registration
- [ ] 3 reference implementations work end-to-end with mocked models
- [ ] Shared vs isolated memory produces different behavior in sequential pattern
- [ ] `pytest tests/test_orchestration.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

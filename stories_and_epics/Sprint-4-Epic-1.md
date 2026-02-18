# Sprint 4 — Epic 1: Orchestration Plugin Interface

**Epic ID:** S4-E1  
**Sprint:** 4  
**Priority:** P1 — Core  
**Goal:** Build the abstract base class, registry, and reference implementations for orchestration patterns. Includes the SingleAgent baseline (PRD+) as the control group for all multi-agent experiments. After this epic, new architectures can be created by subclassing `OrchestrationPattern` and registering with a decorator.

**Dependencies:** S2-E1 (models), S2-E2 (memory), S3-E2 (tools)  
**Reference:** `docs/prd.md` Section 7, `docs/prd-plus.md` Section 5

---

## Story S4-E1-S01: OrchestrationPattern Abstract Base Class

**Branch:** `feature/S4-E1-S01`  
**Points:** 3

**Description:**  
Create the ABC that all orchestration patterns must implement. This is the plugin contract — Vamshi implements subclasses, the framework calls `solve()`.

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

**Description:**  
Build a registry that allows patterns to be registered via decorator and retrieved by name.

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
Build a working 2-agent sequential pipeline (Planner → Coder) that demonstrates the full plugin contract. This is the baseline multi-agent architecture all experiments compare against.

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

Given MinimalSequential with SHARED memory
When solve() executes
Then the Planner agent writes "implementation_plan" to memory
And the Coder agent reads "implementation_plan" from memory
And the read returns the plan (shared mode)

Given MinimalSequential with ISOLATED memory
When solve() executes
Then the Planner agent writes "implementation_plan" to memory
And the Coder agent reads "implementation_plan" from memory
And the read returns None (isolated mode — demonstrating the research variable)

Given MinimalSequential with a model and working tools
When solve() completes
Then model.get_usage()["total_tokens"] > 0 (at least 2 LLM calls were made)
```

**Files to Create:**
- `src/ant_coding/orchestration/examples/sequential.py`

---

## Story S4-E1-S04: SingleAgent Baseline (PRD+)

**Branch:** `feature/S4-E1-S04`  
**Points:** 5

**Description:**  
Build the SingleAgent pattern — the control group for all multi-agent experiments. This establishes the performance floor and provides the denominator for `overhead_ratio` calculation. Every multi-agent experiment should reference a SingleAgent experiment via `baseline_experiment_id`.

**Reference:** `docs/prd-plus.md` Section 5, `docs/success-metrics.md` Tier 2 (overhead_ratio)

**Acceptance Criteria:**

```gherkin
Given the SingleAgent pattern
When I call pattern.name()
Then it returns "single-agent"

Given SingleAgent is registered
When I call OrchestrationRegistry.list_available()
Then it includes "single-agent"

Given SingleAgent with a mock model
When I call await pattern.solve(task, model, memory, tools, workspace_dir)
Then it returns a TaskResult with:
  - task_id matching the input task
  - total_tokens > 0 (exactly 1 LLM call in the simplest case)

Given SingleAgent
When I inspect get_agent_definitions()
Then it returns [{"name": "SoloAgent", "role": "End-to-end task solver"}]

Given SingleAgent
When solve() executes
Then it makes a SINGLE model.complete() call with task + tools
And it executes any tool calls returned by the model
And it runs tests via workspace
And it returns the result

Given SingleAgent with any memory mode
When solve() executes
Then memory writes happen for framework contract consistency
But no inter-agent reads occur (there is only one agent)

Given SingleAgent results and MinimalSequential results on the same tasks
When overhead_ratio is calculated
Then overhead_ratio = multi_agent_tokens / single_agent_tokens
And this ratio is meaningful (typically 1.5x-3x)
```

**Files to Create:**
- `src/ant_coding/orchestration/examples/single_agent.py`

---

## Story S4-E1-S05: MinimalParallel and MinimalLoop References

**Branch:** `feature/S4-E1-S05`  
**Points:** 5

**Description:**  
Build parallel (fan-out) and loop (iterative refinement) reference patterns.

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
And solve() iterates until tests pass or max iterations reached

Given MinimalLoop with intermediate_test_results tracking
When solve() runs 3 iterations: [fail, fail, pass]
Then result.intermediate_test_results == [False, False, True]
And result.success == True

Given MinimalLoop with max_iterations=5 and all iterations fail
When solve() completes
Then result.intermediate_test_results has 5 entries, all False
And result.success == False

Given all patterns registered
When I call OrchestrationRegistry.list_available()
Then it includes "single-agent", "minimal-sequential", "minimal-parallel", "minimal-loop"
```

**Files to Create:**
- `src/ant_coding/orchestration/examples/parallel.py`
- `src/ant_coding/orchestration/examples/loop.py`

---

## Story S4-E1-S06: Orchestration Tests

**Branch:** `feature/S4-E1-S06`  
**Points:** 3

**Description:**  
Comprehensive tests for all orchestration components.

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_orchestration.py -v`
Then all tests pass

Given the test file
Then there are at least 12 test cases covering:
  - ABC enforcement (missing methods raises TypeError)
  - Registry: register, get, list, duplicate rejection
  - SingleAgent: full solve with mocked model
  - MinimalSequential: full solve with mocked model
  - MinimalSequential: shared vs isolated memory behavior difference
  - MinimalParallel: concurrent execution
  - MinimalLoop: iteration and termination
  - MinimalLoop: intermediate_test_results tracking
  - MinimalLoop: max iterations cap
  - Pattern name uniqueness
```

**Files to Create:**
- `tests/test_orchestration.py`

---

## Epic Completion Checklist

- [ ] OrchestrationPattern ABC enforces solve() implementation
- [ ] OrchestrationRegistry supports decorator registration
- [ ] SingleAgent baseline works end-to-end with mocked models (PRD+)
- [ ] 3 multi-agent reference implementations work end-to-end
- [ ] MinimalLoop tracks intermediate_test_results (PRD+)
- [ ] Shared vs isolated memory produces different behavior in sequential pattern
- [ ] `pytest tests/test_orchestration.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
# Sprint 6 — Epic 3: Comprehensive Tests & Documentation

**Epic ID:** S6-E3  
**Sprint:** 6  
**Priority:** P2 — Polish  
**Goal:** Full integration tests, edge case coverage, and complete documentation. After this epic, a new developer (or agent) can understand, run, and extend ant-coding from the README alone.

**Dependencies:** All previous epics

---

## Story S6-E3-S01: Full Pipeline Integration Test

**Branch:** `feature/S6-E3-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given a complete experiment config with "minimal-sequential" + "shared" memory + mocked model
When the integration test runs the full pipeline:
  config → tasks → model → orchestration → memory → tools → eval → report
Then it completes without errors
And results directory contains: config.yaml, metrics.json, events.jsonl, task_results/, patches/, memory_logs/, report.md
And metrics.json has valid aggregated metrics
And events.jsonl has chronologically ordered events
And memory_logs show access patterns

Given the same config but with "isolated" memory
When the integration test runs
Then it also completes without errors
And memory_logs show different access patterns (reads returning None)
And this demonstrates the framework correctly isolates the memory variable

Given both integration test results
When I run compare_results.py on them
Then it produces a valid comparison report
```

**Files to Create:**
- `tests/test_integration.py`

---

## Story S6-E3-S02: Edge Case and Error Handling Tests

**Branch:** `feature/S6-E3-S02`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_edge_cases.py -v`
Then all tests pass covering:
  - Empty task list (0 tasks)
  - Task with 0 timeout (immediate timeout)
  - Model that always returns errors
  - Memory with 1000+ keys (performance)
  - Extremely large patches (>1MB)
  - Special characters in task descriptions
  - Concurrent workspace setup (no collisions)
  - Config with all optional fields omitted
  - Runner interrupted mid-task (cleanup happens)
```

**Files to Create:**
- `tests/test_edge_cases.py`

---

## Story S6-E3-S03: README and Developer Documentation

**Branch:** `feature/S6-E3-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the README.md
When a new developer reads it
Then it contains:
  - Project overview (what ant-coding is, research question)
  - Quick start (3 commands to run first experiment)
  - Architecture diagram (link to docs/architecture/)
  - How to create a new orchestration pattern (5-step guide)
  - How to add a new model (1 YAML file)
  - How to compare experiments (CLI command)
  - Configuration reference (link to example YAMLs)
  - Project structure explanation

Given the quick start section
When a developer follows the steps
Then they can run a baseline experiment within 5 minutes

Given docs/CONTRIBUTING.md
When a developer reads it
Then they understand: branching strategy, commit convention, test requirements, PR process
```

**Files to Create/Update:**
- `README.md` (comprehensive)
- `docs/CONTRIBUTING.md`

---

## Story S6-E3-S04: Linting and Code Quality

**Branch:** `feature/S6-E3-S04`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given the full codebase
When I run `ruff check src/`
Then there are 0 linting errors

Given the full codebase
When I run `ruff format --check src/`
Then all files are properly formatted

Given pyproject.toml
When I inspect [tool.ruff]
Then it has sensible defaults (line-length=100, target-version="py311")

Given the full test suite
When I run `pytest tests/ -v --tb=short`
Then all tests pass with 0 failures
And total test count is at least 80
```

**Files to Modify:**
- `pyproject.toml` (add ruff config)

---

## Epic Completion Checklist

- [ ] Full pipeline integration test passes (shared + isolated)
- [ ] Edge case tests cover failure modes
- [ ] README enables 5-minute quickstart
- [ ] CONTRIBUTING.md documents workflow
- [ ] `ruff check src/` — 0 errors
- [ ] `pytest tests/ -v` — 80+ tests, 0 failures
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

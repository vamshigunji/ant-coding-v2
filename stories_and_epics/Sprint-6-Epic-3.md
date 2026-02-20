# Sprint 6 — Epic 3: Comprehensive Tests & Documentation

**Epic ID:** S6-E3  
**Sprint:** 6  
**Priority:** P2 — Polish  
**Goal:** Full pipeline integration tests, edge case coverage, developer documentation, and code quality. After this epic, the framework is production-ready for experimentation.

**Dependencies:** All prior epics  
**Reference:** `docs/spec/prd.md`, `docs/spec/prd-plus.md`, `docs/guides/experimentation-playbook.md`

---

## Story S6-E3-S01: Full Pipeline Integration Test

**Branch:** `feature/S6-E3-S01`  
**Points:** 5

**Description:**  
End-to-end test that runs a complete experiment with mocked LLM, collects all 4 tiers of metrics, generates a comparison report, and validates the entire pipeline.

**Acceptance Criteria:**

```gherkin
Given a complete experiment config using "single-agent" pattern
And a complete experiment config using "minimal-sequential" with shared memory
And mocked model responses for both
When I run both experiments through ExperimentRunner
Then both produce valid results directories with all PRD+ fields

Given the two experiment results
When I run statistical comparison
Then it produces a comparison table with all 11 metrics
And breakeven analysis is included (since baseline is configured)
And the markdown report is valid

Given the pipeline test
When I run it end-to-end
Then it completes within 60 seconds (no real API calls)
And every layer was exercised: config → tasks → tools → orchestration → memory → eval → report
```

**Files to Create:**
- `tests/integration/test_full_pipeline.py`

---

## Story S6-E3-S02: Edge Case and Error Handling Tests

**Branch:** `feature/S6-E3-S02`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/integration/test_edge_cases.py -v`
Then all tests pass covering:
  - Experiment with 0 tasks (graceful empty result)
  - All tasks fail (metrics show 0% pass, inf cost_per_resolution)
  - Model returns empty response
  - Tool execution timeout
  - Invalid experiment config YAML
  - Missing baseline_experiment_id reference
  - Memory mode mismatch detection
  - Corrupt events.jsonl recovery
  - LLM judge returning malformed JSON
  - FailureClassifier with missing event logs
```

**Files to Create:**
- `tests/integration/test_edge_cases.py`

---

## Story S6-E3-S03: README and Developer Documentation

**Branch:** `feature/S6-E3-S03`  
**Points:** 3

**Description:**  
Update README with quickstart, architecture overview, and experimentation guide. Create developer docs for extending the framework.

**Acceptance Criteria:**

```gherkin
Given the README.md
When a new developer reads it
Then they can:
  - Understand the project purpose (multi-agent benchmarking)
  - Install dependencies and run smoke test
  - Run their first experiment with `python scripts/run_experiment.py`
  - Understand the 4-tier metrics framework
  - Know where to find detailed docs

Given the developer documentation
Then it includes:
  - How to create a new OrchestrationPattern (subclass + register)
  - How to add new tools to the ToolRegistry
  - How to configure experiment YAML with baseline_experiment_id (PRD+)
  - How to use the experiment registry and follow the playbook
  - Architecture diagram showing layer interactions
```

**Files to Create/Modify:**
- `README.md` (update)
- `docs/guides/developer-guide.md` (new)

---

## Story S6-E3-S04: Linting and Code Quality

**Branch:** `feature/S6-E3-S04`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given the codebase
When I run `ruff check src/ tests/`
Then 0 linting errors

Given the codebase
When I run `mypy src/ --ignore-missing-imports`
Then 0 type errors (or documented exceptions)

Given all test files
When I run `pytest tests/ -v --tb=short`
Then all tests pass with 0 failures
And test coverage > 80% on src/ant_coding/
```

---

## Epic Completion Checklist

- [ ] Full pipeline integration test exercises every layer
- [ ] Edge case tests cover failure modes gracefully
- [ ] README enables new developer onboarding
- [ ] Developer guide covers extension patterns
- [ ] Linting and type checking pass
- [ ] Test coverage > 80%
- [ ] sprint.yml updated with final status
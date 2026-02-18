# Sprint 4 — Epic 2: Experiment Runner

**Epic ID:** S4-E2  
**Sprint:** 4  
**Priority:** P1 — Core  
**Goal:** Wire all layers together into the ExperimentRunner — the main entry point that takes a YAML config and produces structured results. After this epic, `python scripts/run_experiment.py config.yaml` runs end-to-end.

**Dependencies:** S4-E1 (orchestration), S3-E1 (tasks), S3-E2 (tools), S2-E1 (models), S2-E2 (memory)

---

## Story S4-E2-S01: ExperimentRunner Core

**Branch:** `feature/S4-E2-S01`  
**Points:** 8

**Description:**  
The central engine that loads config, initializes all layers, runs tasks through the orchestration pattern, and collects results.

**Acceptance Criteria:**

```gherkin
Given a valid experiment config YAML
When I create ExperimentRunner(config_path) 
Then it loads and validates the config without errors
And it initializes: ModelProvider, MemoryManager, TaskLoader

Given an ExperimentRunner
When I call await runner.run()
Then it iterates over all tasks
And for each task it calls: workspace.setup(), pattern.solve(), workspace.run_tests(), workspace.teardown()
And it returns ExperimentMetrics with aggregated results

Given an experiment with runs_per_task=3 and 2 tasks
When runner.run() completes
Then exactly 6 task runs are executed (2 tasks × 3 runs each)
And results directory contains 6 task_result JSON files

Given a task that exceeds timeout_per_task_seconds
When the task is running
Then it is terminated and recorded as errored (not failed)
And the runner continues to the next task

Given a task where the orchestration pattern raises an exception
When the runner catches the error
Then it records the task as errored with the error message
And continues to the next task (no crash)
```

**Files to Create:**
- `src/ant_coding/core/runner.py`

---

## Story S4-E2-S02: Result Output Structure

**Branch:** `feature/S4-E2-S02`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given an experiment with id="test-exp-001"
When the runner completes
Then the following directory structure exists:
  results/test-exp-001/
  ├── config.yaml        (copy of input config)
  ├── metrics.json       (ExperimentMetrics as JSON)
  ├── events.jsonl       (empty for now — populated in Sprint 5)
  ├── task_results/
  │   └── {task_id}-run-{n}.json
  ├── patches/
  │   └── {task_id}-run-{n}.patch
  └── memory_logs/
      └── {task_id}-access-log.json

Given a TaskResult with success=True and patch="diff content"
When it is saved to task_results/
Then the JSON file contains all TaskResult fields
And the patch file contains the raw diff

Given a MemoryManager access log for a task
When the task completes
Then memory_logs/{task_id}-access-log.json contains the full access log
```

**Files to Create:**
- `src/ant_coding/core/output.py`

---

## Story S4-E2-S03: CLI Entry Point

**Branch:** `feature/S4-E2-S03`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given a valid experiment config YAML
When I run `python scripts/run_experiment.py configs/experiments/baseline-sequential.yaml`
Then it prints progress to console (task N/M, current status)
And it creates the results directory with all outputs
And exits with code 0

Given an invalid config path
When I run `python scripts/run_experiment.py nonexistent.yaml`
Then it prints an error message and exits with code 1

Given multiple config paths
When I run `python scripts/run_experiment.py config1.yaml config2.yaml`
Then it runs both experiments sequentially
```

**Files to Create:**
- `scripts/run_experiment.py`

---

## Story S4-E2-S04: End-to-End Integration Test

**Branch:** `feature/S4-E2-S04`  
**Points:** 5

**Description:**  
A full integration test with mocked model responses that validates the entire pipeline from config → execution → results.

**Acceptance Criteria:**

```gherkin
Given a test experiment config using "minimal-sequential" pattern
And a mocked ModelProvider that returns predetermined responses
And a custom task YAML with 1 simple task
When I run the ExperimentRunner end-to-end
Then it completes without errors
And results/test-experiment/metrics.json exists
And metrics.json shows num_tasks=1 and total_tokens > 0
And task_results/ contains exactly 1 JSON file (runs_per_task=1)
And memory_logs/ contains the access log

Given the integration test
When I run `pytest tests/test_runner.py -v`
Then it passes within 30 seconds (no real API calls)
```

**Files to Create:**
- `tests/test_runner.py`
- `tests/fixtures/mock_experiment.yaml`
- `tests/fixtures/mock_tasks.yaml`

---

## Epic Completion Checklist

- [ ] ExperimentRunner wires all layers and runs tasks end-to-end
- [ ] Results directory has correct structure with config copy, metrics, task results, patches, memory logs
- [ ] CLI entry point works with progress output
- [ ] Integration test passes with mocked models
- [ ] Error handling: timeout tasks and exceptions don't crash the runner
- [ ] `pytest tests/test_runner.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

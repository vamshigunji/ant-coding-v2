# Sprint 6 — Epic 1: Reports, Replay & Experiment Registry

**Epic ID:** S6-E1  
**Sprint:** 6  
**Priority:** P2 — Polish  
**Goal:** Build the report generator, session replay, cross-experiment comparison CLI, and experiment registry. After this epic, `scripts/compare_results.py` produces a publication-ready markdown report and the experiment registry tracks lineage.

**Dependencies:** S5-E2 (eval harness), S5-E3 (statistical), S5-E1 (event logger)  
**Reference:** `docs/experimentation-playbook.md`, `docs/success-metrics.md`

---

## Story S6-E1-S01: Markdown Report Generator

**Branch:** `feature/S6-E1-S01`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given ExperimentMetrics for a single experiment (all 4 tiers populated)
When I call generate_markdown(metrics)
Then it returns a markdown string with:
  - Experiment ID, architecture, model, memory mode
  - 4-tier summary table (all 11 metrics from success-metrics.md)
  - Per-agent token breakdown
  - Failure category breakdown table

Given ExperimentMetrics for 3 experiments
When I call generate_comparison_markdown(all_metrics, all_comparisons)
Then it includes:
  - Side-by-side comparison table (matching format from success-metrics.md)
  - Statistical significance markers (* for p<0.05, ** for p<0.01)
  - Breakeven analysis section (when baseline exists)
  - Recommendation summary
```

**Files to Modify:**
- `src/ant_coding/eval/report.py` (extend with full markdown generation)

---

## Story S6-E1-S02: JSON and CSV Export

**Branch:** `feature/S6-E1-S02`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given ExperimentMetrics (with all PRD+ fields)
When I call generate_json(metrics)
Then it returns valid JSON that includes all 4 tiers of metrics
And it round-trips back to ExperimentMetrics

Given a list of ExperimentMetrics
When I call generate_csv(metrics_list)
Then it returns a CSV with one row per experiment
And columns include all 11 success metrics
And it's importable into pandas without errors
```

**Files to Modify:**
- `src/ant_coding/eval/report.py`

---

## Story S6-E1-S03: Session Replay

**Branch:** `feature/S6-E1-S03`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given an events.jsonl file from a completed experiment
When I create SessionReplay(events_path)
Then it loads all events successfully

Given a SessionReplay with 50 events
When I call step(5)
Then it returns the next 5 events in order

Given a SessionReplay
When I call state_at(event_index=25)
Then it reconstructs the memory state by replaying MEMORY_WRITE events up to index 25
And returns a dict of all state keys and values at that point

Given a SessionReplay
When I call token_curve()
Then it returns a list of (event_index, cumulative_tokens) tuples
```

**Files to Create:**
- `src/ant_coding/observability/replay.py`

---

## Story S6-E1-S04: Cross-Experiment Comparison CLI

**Branch:** `feature/S6-E1-S04`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given two experiment result directories
When I run `python scripts/compare_results.py results/exp-a results/exp-b`
Then it prints a 4-tier comparison table to console
And saves comparison_report.md to the current directory

Given three or more experiment directories
When I run the comparison
Then it produces pairwise comparisons for all pairs

Given one experiment directory
When I run the comparison
Then it prints the single experiment's metrics (all 4 tiers)
```

**Files to Create:**
- `scripts/compare_results.py`

---

## Story S6-E1-S05: Experiment Registry Setup (PRD+)

**Branch:** `feature/S6-E1-S05`  
**Points:** 3

**Description:**  
Implement the experiment registry system from `docs/experimentation-playbook.md`. This enables experiment lineage tracking: each experiment records its parent, the single variable changed, a hypothesis, and post-run insights.

**Reference:** `docs/experimentation-playbook.md` (Experiment Journal section)

**Acceptance Criteria:**

```gherkin
Given the experiments/registry.yml file
When I call registry.add_experiment(config) before running
Then a new entry is added with:
  id, date, parent, variable_changed, hypothesis, config path, status="planned"
And the outcome fields are null

Given an experiment that has completed
When I call registry.update_outcome(experiment_id, metrics)
Then the outcome section is populated with all 4 tiers of metrics
And status changes to "complete"

Given the registry
When I call registry.get_lineage(experiment_id)
Then it returns the chain of parent experiments back to the root baseline

Given the registry
When I call registry.validate()
Then it checks that every experiment (except baselines) has exactly one variable_changed
And it warns about experiments with status="planned" older than 7 days

Given the ExperimentRunner
When an experiment starts
Then it automatically updates registry status to "running"
When an experiment completes
Then it automatically updates registry with outcome metrics

Given the experiment naming convention
When a new experiment is created with parent="baseline-seq-shared-claude"
And variable_changed="memory: shared → isolated"
Then suggested id = "baseline-seq-shared-claude--memory-isolated"
```

**Files to Create:**
- `src/ant_coding/core/experiment_registry.py`
- Initialize `experiments/registry.yml` with the template from experimentation-playbook.md

---

## Story S6-E1-S06: Polish Tests

**Branch:** `feature/S6-E1-S06`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
When I run `pytest tests/test_reports.py tests/test_replay.py tests/test_registry.py -v`
Then all tests pass covering:
  - Markdown generation with 4-tier metrics
  - JSON round-trip with all PRD+ fields
  - CSV format validation
  - Session replay stepping and state reconstruction
  - Experiment registry: add, update, lineage, validate
  - Compare CLI with mock results
```

**Files to Create:**
- `tests/test_reports.py`
- `tests/test_replay.py`
- `tests/test_registry.py`

---

## Epic Completion Checklist

- [ ] Markdown reports with 4-tier comparison tables and significance markers
- [ ] JSON/CSV export with all PRD+ metrics
- [ ] Session replay reconstructs state at any point
- [ ] Experiment registry tracks lineage and enforces single-variable changes
- [ ] `scripts/compare_results.py` works for 1-N experiments
- [ ] All tests pass
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
# Sprint 6 — Epic 1: Reports, Replay & Cross-Experiment Comparison

**Epic ID:** S6-E1  
**Sprint:** 6  
**Priority:** P2 — Polish  
**Goal:** Build the report generator, session replay, and cross-experiment comparison CLI. After this epic, `scripts/compare_results.py` produces a publication-ready markdown report.

**Dependencies:** S5-E2 (eval harness), S5-E3 (statistical), S5-E1 (event logger)

---

## Story S6-E1-S01: Markdown Report Generator

**Branch:** `feature/S6-E1-S01`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given ExperimentMetrics for a single experiment
When I call generate_markdown(metrics)
Then it returns a markdown string with:
  - Experiment ID, architecture, model, memory mode
  - Summary table (pass rate, tokens, cost, time)
  - Per-agent token breakdown

Given ExperimentMetrics for 3 experiments
When I call generate_comparison_markdown(all_metrics, all_comparisons)
Then it includes:
  - Side-by-side comparison table
  - Statistical significance markers (* for p<0.05, ** for p<0.01)
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
Given ExperimentMetrics
When I call generate_json(metrics)
Then it returns valid JSON that round-trips back to ExperimentMetrics

Given a list of ExperimentMetrics
When I call generate_csv(metrics_list)
Then it returns a CSV string with one row per experiment
And columns match ExperimentMetrics fields
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
And subsequent step() calls continue from position 5

Given a SessionReplay
When I call state_at(event_index=25)
Then it reconstructs the memory state by replaying all MEMORY_WRITE events up to index 25
And returns a dict of all state keys and values at that point

Given a SessionReplay
When I call token_curve()
Then it returns a list of (event_index, cumulative_tokens) tuples
And cumulative_tokens is monotonically non-decreasing
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
Then it prints a comparison table to console
And saves comparison_report.md to the current directory

Given three or more experiment directories
When I run `python scripts/compare_results.py results/exp-a results/exp-b results/exp-c`
Then it produces pairwise comparisons for all pairs (A vs B, A vs C, B vs C)
And the report includes all comparisons

Given one experiment directory
When I run `python scripts/compare_results.py results/exp-a`
Then it prints the single experiment's metrics (no comparison)
```

**Files to Create:**
- `scripts/compare_results.py`

---

## Story S6-E1-S05: Polish Tests

**Branch:** `feature/S6-E1-S05`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
When I run `pytest tests/test_reports.py tests/test_replay.py -v`
Then all tests pass covering:
  - Markdown generation with single and multiple experiments
  - JSON round-trip
  - CSV format validation
  - Session replay stepping and state reconstruction
  - Token curve monotonicity
  - Compare CLI with mock results
```

**Files to Create:**
- `tests/test_reports.py`
- `tests/test_replay.py`

---

## Epic Completion Checklist

- [ ] Markdown reports with comparison tables and significance markers
- [ ] JSON/CSV export for programmatic analysis
- [ ] Session replay reconstructs state at any point
- [ ] `scripts/compare_results.py` works for 1-N experiments
- [ ] All tests pass
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

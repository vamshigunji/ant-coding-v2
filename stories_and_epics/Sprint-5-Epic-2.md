# Sprint 5 — Epic 2: Evaluation Harness

**Epic ID:** S5-E2  
**Sprint:** 5  
**Priority:** P1 — Core  
**Goal:** Build the evaluation system that computes metrics, runs LLM-as-Judge scoring, and calculates pass@k. This is where raw results become publishable evidence.

**Dependencies:** S5-E1 (event logger for token data), S4-E2 (runner for TaskResults)

---

## Story S5-E2-S01: Metrics Calculation

**Branch:** `feature/S5-E2-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given 10 TaskResults: 7 success, 2 failed, 1 errored
When I call calculate_metrics(task_results, experiment_config)
Then metrics.pass_rate == 0.7 (7/10)
And metrics.tasks_passed == 7
And metrics.tasks_failed == 2
And metrics.tasks_errored == 1

Given TaskResults with token counts [1000, 2000, 3000, 4000, 5000]
When metrics are calculated
Then metrics.avg_tokens_per_task == 3000
And metrics.median_tokens_per_task == 3000
And metrics.total_tokens == 15000

Given TaskResults with prompt_tokens and completion_tokens
When metrics are calculated
Then metrics.prompt_completion_ratio == avg_prompt / avg_completion

Given TaskResults with wall_time_seconds
When metrics are calculated
Then metrics.avg_wall_time_seconds and median are correct

Given TaskResults with cost_usd values
When metrics are calculated
Then metrics.total_cost_usd == sum of all costs
And metrics.avg_cost_per_task == total / num_tasks
```

**Files to Create:**
- `src/ant_coding/eval/harness.py`
- Update `src/ant_coding/eval/metrics.py` (add calculation logic to existing dataclass file)

---

## Story S5-E2-S02: LLM-as-Judge Scoring

**Branch:** `feature/S5-E2-S02`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given a Task and a generated patch
When I call await judge.evaluate(task, patch, test_output)
Then it returns a dict with keys: correctness, code_quality, completeness, efficiency (each 1-5)
And it includes "overall" as a weighted average
And it includes "reasoning" as a string explanation

Given the LLM judge is configured with judge_model="gemini/gemini-2.5-flash"
When evaluate() is called
Then it uses a DIFFERENT model than the one being tested
And the judge prompt asks for structured JSON output

Given the judge model returns malformed output
When evaluate() parses the response
Then it handles errors gracefully (returns default scores with error note)

Given the judge prompt
When I inspect it
Then it includes: the task description, the generated patch, test results
And it asks for scoring on the 4 dimensions with 1-5 scale and reasoning
```

**Files to Create:**
- `src/ant_coding/eval/llm_judge.py`

---

## Story S5-E2-S03: pass@k Computation

**Branch:** `feature/S5-E2-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given 10 tasks each run 5 times (50 total results)
When I compute pass_at_k(results, k=1)
Then it calculates the unbiased estimator: 1 - C(n-c, k) / C(n, k) per task
And returns the average across tasks

Given a task with 5 runs where 3 pass
When I compute pass_at_k for this task at k=1
Then pass@1 ≈ 0.6 (3/5)

Given a task with 5 runs where 3 pass
When I compute pass_at_k for this task at k=3
Then pass@3 > pass@1 (more attempts increases probability)

Given a task with 5 runs where 0 pass
When I compute pass_at_k at k=1, k=3, k=5
Then all return 0.0

Given a task with 5 runs where 5 pass
When I compute pass_at_k at k=1, k=3, k=5
Then all return 1.0
```

**Files to Create:**
- Update `src/ant_coding/eval/harness.py` (add pass@k method)

---

## Story S5-E2-S04: Evaluation Tests

**Branch:** `feature/S5-E2-S04`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_eval.py -v`
Then all tests pass with at least 12 test cases covering:
  - Metrics: pass rate, tokens, cost, time calculations
  - Metrics: edge cases (0 tasks, all pass, all fail)
  - LLM Judge: successful evaluation (mocked)
  - LLM Judge: malformed response handling
  - pass@k: mathematical correctness for known inputs
  - pass@k: boundary cases (all pass, none pass)
```

**Files to Create:**
- `tests/test_eval.py`

---

## Epic Completion Checklist

- [ ] Metrics calculated correctly for pass rate, tokens, cost, time
- [ ] LLM-as-Judge produces structured scores on 4 dimensions
- [ ] pass@k uses unbiased estimator formula
- [ ] `pytest tests/test_eval.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

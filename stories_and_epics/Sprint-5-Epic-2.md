# Sprint 5 — Epic 2: Evaluation Harness

**Epic ID:** S5-E2  
**Sprint:** 5  
**Priority:** P1 — Core  
**Goal:** Build the evaluation system that computes all 4 tiers of metrics (PRD+), runs the updated LLM-as-Judge scoring, calculates pass@k, and classifies failures. This is where raw results become publishable evidence.

**Dependencies:** S5-E1 (event logger for token data), S4-E2 (runner for TaskResults), S3-E3 (PRD+ type fields)  
**Reference:** `docs/spec/prd-plus.md` Sections 1, 3, 4, 6; `docs/spec/success-metrics.md`

---

## Story S5-E2-S01: 4-Tier Metrics Calculation

**Branch:** `feature/S5-E2-S01`  
**Points:** 8

**Description:**  
Implement the full `calculate_metrics()` function that computes all 11 success metrics across 4 tiers. This is the core evaluation logic specified in PRD+ Section 6.

**Acceptance Criteria:**

```gherkin
# ── Tier 1: Primary ──

Given 10 TaskResults: 7 success, 2 failed, 1 errored
When I call calculate_metrics(results, config)
Then metrics.pass_rate == 0.7 (7/10)
And metrics.successful_tasks == 7
And metrics.failed_tasks == 3

Given TaskResults with total_cost summing to $3.50 and 7 tasks passed
When metrics are calculated
Then metrics.cost_per_resolution == $0.50 (3.50 / 7)

Given 0 tasks passed
When metrics are calculated
Then metrics.cost_per_resolution == float('inf')

# ── Tier 2: Efficiency ──

Given successful results using 40K tokens and total across all runs = 60K tokens
When metrics are calculated
Then metrics.useful_token_ratio == 0.667 (40K / 60K)

Given a config with baseline_experiment_id and baseline_metrics.total_tokens = 30K
And current experiment total_tokens = 60K
When metrics are calculated
Then metrics.overhead_ratio == 2.0 (60K / 30K)

Given no baseline_experiment_id configured
When metrics are calculated
Then metrics.overhead_ratio == 0.0 (no baseline)

Given total_tokens = 75K and 5 tasks passed
When metrics are calculated
Then metrics.tokens_per_resolution == 15000 (75K / 5)

# ── Tier 3: Quality ──

Given TaskResults with judge_scores averaging overall=3.8
When metrics are calculated
Then metrics.avg_patch_quality == 3.8

Given TaskResults where generated_patch_lines / gold_patch_lines ratios are [1.0, 1.5, 0.8]
When metrics are calculated
Then metrics.avg_patch_size_ratio == 1.1 (mean of ratios)

Given TaskResults with gold_patch_lines == 0 (custom tasks, no gold standard)
When metrics are calculated
Then metrics.avg_patch_size_ratio == 0.0

# ── Tier 4: Robustness ──

Given 5 tasks each run 3 times with pass rates [0.33, 0.67, 1.0, 0.0, 0.67]
When metrics are calculated
Then metrics.resolution_variance_cv is computed as stdev/mean of those rates

Given TaskResults where 3 initially failed (first intermediate=False) and 2 of those recovered
When metrics are calculated
Then metrics.error_recovery_rate == 0.667 (2/3)

Given TaskResults with failure_categories: 2 planning, 1 timeout, 1 hallucination_cascade
When metrics are calculated
Then metrics.failure_categories == {"planning": 2, "implementation": 0, "integration": 0,
  "hallucination_cascade": 1, "timeout": 1, "tool_failure": 0}
```

**Files to Create/Modify:**
- `src/ant_coding/eval/harness.py`
- Update `src/ant_coding/eval/metrics.py` (add calculation logic)

---

## Story S5-E2-S02: LLM-as-Judge Scoring (PRD+ Dimensions)

**Branch:** `feature/S5-E2-S02`  
**Points:** 5

**Description:**  
Build the LLM-as-Judge that scores patches on the PRD+ dimensions: correctness, minimality, code_quality, completeness (replacing the original PRD's "efficiency" with "minimality").

**Reference:** `docs/spec/prd-plus.md` Section 3

**Acceptance Criteria:**

```gherkin
Given a Task and a generated patch
When I call await judge.evaluate(task, patch, test_output)
Then it returns a dict with keys: correctness, minimality, code_quality, completeness (each 1-5)
And it includes "overall" as a weighted average
And it includes "reasoning" as a string explanation

Given the judge dimensions
Then the scoring rubric is:
  | Dimension       | 5 (best)        | 3 (mid)           | 1 (worst)          |
  | correctness     | Root cause fixed | Symptom patched    | Wrong fix          |
  | minimality      | Minimal diff     | Some extra changes | Rewrote everything |
  | code_quality    | Production-grade | Acceptable         | Hacky              |
  | completeness    | Comprehensive    | Main case only     | Incomplete         |

Given the LLM judge is configured with judge_model="gemini/gemini-2.5-flash"
When evaluate() is called
Then it uses a DIFFERENT model than the one being tested
And the judge prompt asks for structured JSON output

Given the judge model returns malformed output
When evaluate() parses the response
Then it handles errors gracefully (returns default scores with error note)

Given a successful judge evaluation result
When it is stored on TaskResult.judge_scores
Then it matches the schema: {"correctness": int, "minimality": int, "code_quality": int,
  "completeness": int, "overall": float, "reasoning": str}
```

**Files to Create:**
- `src/ant_coding/eval/llm_judge.py`

---

## Story S5-E2-S03: pass@k Computation

**Branch:** `feature/S5-E2-S03`  
**Points:** 3

**Description:**  
Implement the unbiased pass@k estimator for measuring first-attempt and multi-attempt success rates.

**Acceptance Criteria:**

```gherkin
Given 10 tasks each run 5 times (50 total results)
When I compute pass_at_k(results, k=1)
Then it uses the unbiased estimator: 1 - C(n-c, k) / C(n, k) per task
And returns the average across tasks

Given a task with 5 runs where 3 pass
When I compute pass_at_k for k=1
Then pass@1 ≈ 0.6

Given a task with 5 runs where 3 pass
When I compute pass_at_k for k=3
Then pass@3 > pass@1

Given a task with 5 runs where 0 pass
Then pass@k returns 0.0 for all k

Given a task with 5 runs where 5 pass
Then pass@k returns 1.0 for all k
```

**Files to Modify:**
- `src/ant_coding/eval/harness.py` (add pass@k method)

---

## Story S5-E2-S04: Failure Classification (PRD+)

**Branch:** `feature/S5-E2-S04`  
**Points:** 5

**Description:**  
Build the FailureClassifier that categorizes WHY failed tasks failed. Uses a cheap/fast LLM to classify each failure into one of 6 categories. This enables the failure_categories breakdown in ExperimentMetrics.

**Reference:** `docs/spec/prd-plus.md` Section 4

**Acceptance Criteria:**

```gherkin
Given a failed TaskResult with patch, test output, and event log
When I call await classifier.classify(task, result, events)
Then it returns one of: "planning", "implementation", "integration",
  "hallucination_cascade", "timeout", "tool_failure"

Given a task that timed out (result.error contains "timeout")
When classifier runs
Then it returns "timeout" without making an LLM call (deterministic shortcut)

Given a task where tool execution failed
When classifier runs
Then it returns "tool_failure" (deterministic shortcut based on event log)

Given the classifier LLM prompt
When I inspect it
Then it includes:
  - Task description (what should have been fixed)
  - Generated patch (or "no patch generated")
  - Test output (failure evidence)
  - Last 20 events from event log (agent decision trail)
  - Memory access summary (reads that returned None — information gaps)

Given the classifier model returns malformed output
When classify() handles the error
Then it defaults to "implementation" and logs a warning

Given a FailureClassifier
When I inspect the default model
Then it uses "gemini/gemini-2.5-flash" (cheap/fast for scale)

Given results from classification
When TaskResult.failure_category is set
Then it feeds into ExperimentMetrics.failure_categories aggregation
```

**Files to Create:**
- `src/ant_coding/eval/failure_classifier.py`

---

## Story S5-E2-S05: Evaluation Tests

**Branch:** `feature/S5-E2-S05`  
**Points:** 3

**Description:**  
Comprehensive test coverage for all evaluation components.

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_eval.py -v`
Then all tests pass with at least 15 test cases covering:

  Metrics (4-tier):
  - Tier 1: pass rate, cost_per_resolution (normal + zero-pass edge case)
  - Tier 2: useful_token_ratio, overhead_ratio (with + without baseline), tokens_per_resolution
  - Tier 3: avg_patch_quality, avg_patch_size_ratio (with + without gold patches)
  - Tier 4: resolution_variance_cv, error_recovery_rate, failure_categories

  LLM Judge:
  - Successful evaluation with all 4 dimensions (mocked)
  - Malformed response handling
  - Judge scores stored on TaskResult correctly

  pass@k:
  - Mathematical correctness for known inputs
  - Boundary cases (all pass, none pass)

  Failure Classifier:
  - Timeout shortcut classification
  - Tool failure shortcut classification
  - LLM-based classification (mocked)
  - Malformed classifier output fallback
```

**Files to Create:**
- `tests/test_eval.py`

---

## Epic Completion Checklist

- [ ] All 4 tiers of metrics calculated correctly (11 metrics total)
- [ ] LLM-as-Judge uses PRD+ dimensions: correctness, minimality, code_quality, completeness
- [ ] pass@k uses unbiased estimator formula
- [ ] FailureClassifier categorizes failures into 6 categories
- [ ] Baseline overhead_ratio works when baseline_experiment_id is configured
- [ ] Edge cases handled: zero tasks passed, no gold patches, no baseline
- [ ] `pytest tests/test_eval.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
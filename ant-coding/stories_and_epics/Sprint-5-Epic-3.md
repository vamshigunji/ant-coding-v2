# Sprint 5 — Epic 3: Statistical Comparison

**Epic ID:** S5-E3  
**Sprint:** 5  
**Priority:** P1 — Core  
**Goal:** Build the statistical testing framework that compares experiment results with mathematical rigor. After this epic, "Architecture A is better than Architecture B" is backed by p-values and confidence intervals.

**Dependencies:** S5-E2 (evaluation harness for metrics)

---

## Story S5-E3-S01: Paired Statistical Tests

**Branch:** `feature/S5-E3-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given two experiments A and B that ran the same 20 tasks
When I call compare(metrics_a, metrics_b, results_a, results_b)
Then it returns a dict with pass_rate comparison including p_value (paired t-test)
And token_efficiency comparison including p_value (Mann-Whitney U)
And each comparison has: a_value, b_value, p_value, significant (p < 0.05)

Given experiment A with pass_rate=0.8 and B with pass_rate=0.3 on 50 tasks
When I compare them
Then pass_rate.significant == True (large difference, enough samples)

Given experiment A and B with pass_rates 0.50 and 0.51 on 5 tasks
When I compare them
Then pass_rate.significant == False (too small difference, too few samples)

Given two experiments with identical results
When I compare them
Then all p_values > 0.05 and significant == False for all metrics
```

**Files to Create:**
- `src/ant_coding/eval/statistical.py`

---

## Story S5-E3-S02: Bootstrap Confidence Intervals and Effect Size

**Branch:** `feature/S5-E3-S02`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given experiment results for pass@k metrics
When I compute bootstrap confidence intervals with n_bootstraps=10000
Then each metric has ci_95 as [lower, upper] bounds
And the true mean falls within ci_95 approximately 95% of the time

Given two experiments with different pass rates
When I compute Cohen's d effect size
Then it returns a float where:
  |d| < 0.2 is "negligible"
  0.2 ≤ |d| < 0.5 is "small"
  0.5 ≤ |d| < 0.8 is "medium"
  |d| ≥ 0.8 is "large"
And the interpretation string is included in the result

Given the comparison result
When I inspect the output
Then it includes effect_size, effect_interpretation, and ci_95 for each metric
```

**Files to Modify:**
- `src/ant_coding/eval/statistical.py`

---

## Story S5-E3-S03: Comparison Report with Recommendation

**Branch:** `feature/S5-E3-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given a statistical comparison result
When I access result["recommendation"]
Then it contains a plain-English summary like:
  "Architecture A significantly outperforms B on pass rate (p=0.003, d=1.2 large)
   while using 23% fewer tokens. Recommend Architecture A."

Given two experiments with no significant differences
When I access result["recommendation"]  
Then it says something like:
  "No statistically significant differences found between A and B.
   Larger sample sizes may be needed."

Given the comparison
When I call generate_comparison_table(results)
Then it returns a markdown table with columns: Metric, A, B, Δ, p-value, Significant, Effect Size
```

**Files to Create:**
- `src/ant_coding/eval/report.py`

---

## Story S5-E3-S04: Statistical Tests Validation

**Branch:** `feature/S5-E3-S04`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_statistical.py -v`
Then all tests pass with at least 10 test cases covering:
  - Paired t-test with known distributions
  - Mann-Whitney U with known rankings
  - Bootstrap CI contains true mean
  - Cohen's d calculation and interpretation
  - Comparison with identical results (no significance)
  - Comparison with clearly different results (significant)
  - Recommendation generation
  - Markdown table format
```

**Files to Create:**
- `tests/test_statistical.py`

---

## Epic Completion Checklist

- [ ] Paired t-test and Mann-Whitney U produce correct p-values
- [ ] Bootstrap CI with 10,000 samples
- [ ] Cohen's d with interpretation strings
- [ ] Plain-English recommendation generated
- [ ] Markdown comparison table
- [ ] `pytest tests/test_statistical.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

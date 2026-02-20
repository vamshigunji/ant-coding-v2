# Sprint 3 — Epic 3: PRD+ Type & Config Updates

**Epic ID:** S3-E3  
**Sprint:** 3  
**Priority:** P0 — Foundation (retroactive)  
**Goal:** Extend the already-built type definitions and config models with fields required by PRD+ (`docs/spec/prd-plus.md`) and the success metrics framework (`docs/spec/success-metrics.md`). These changes are prerequisites for the evaluation harness in Sprint 5.

**Dependencies:** S1-E1 (config, types — already done), S2-E2 (memory — already done)  
**Reference:** `docs/spec/prd-plus.md` Sections 1, 2; `docs/spec/success-metrics.md`

---

## Story S3-E3-S01: Extend TaskResult with PRD+ Fields

**Branch:** `feature/S3-E3-S01`  
**Points:** 3

**Description:**  
Add fields to the existing `TaskResult` dataclass that enable error recovery tracking, failure classification, patch size comparison, and per-task LLM judge scores.

**Acceptance Criteria:**

```gherkin
Given the existing TaskResult in src/ant_coding/tasks/types.py
When I add the PRD+ fields
Then TaskResult has these new fields with correct defaults:
  | Field                       | Type                 | Default                |
  | intermediate_test_results   | list[bool]           | []                     |
  | failure_category            | Optional[str]        | None                   |
  | generated_patch_lines       | int                  | 0                      |
  | gold_patch_lines            | int                  | 0                      |
  | judge_scores                | Optional[dict]       | None                   |

Given a TaskResult with intermediate_test_results=[False, False, True]
When I inspect the field
Then it indicates: failed twice, succeeded on 3rd attempt (used for error_recovery_rate)

Given a TaskResult with failure_category="hallucination_cascade"
When I inspect the field
Then it is one of the 6 valid categories:
  "planning", "implementation", "integration", "hallucination_cascade", "timeout", "tool_failure"

Given a TaskResult with judge_scores={"correctness": 4, "minimality": 3, "code_quality": 4, "completeness": 3, "overall": 3.5, "reasoning": "..."}
When I access result.judge_scores["overall"]
Then it returns 3.5

Given existing tests that create TaskResult objects
When I run pytest tests/test_types.py
Then all existing tests still pass (backward compatible defaults)
```

**Files to Modify:**
- `src/ant_coding/tasks/types.py`
- `tests/test_types.py` (add tests for new fields)

---

## Story S3-E3-S02: Extend ExperimentMetrics with 4-Tier Fields

**Branch:** `feature/S3-E3-S02`  
**Points:** 3

**Description:**  
Add the 11 success metrics fields from `docs/spec/success-metrics.md` to the `ExperimentMetrics` dataclass. These are storage fields only — calculation logic is Sprint 5.

**Acceptance Criteria:**

```gherkin
Given the existing ExperimentMetrics in src/ant_coding/eval/metrics.py
When I add the PRD+ fields
Then ExperimentMetrics has all of these new fields:

  Tier 1 — Primary:
  | cost_per_resolution         | float  | 0.0   |

  Tier 2 — Efficiency:
  | useful_token_ratio          | float  | 0.0   |
  | overhead_ratio              | float  | 0.0   |
  | tokens_per_resolution       | float  | 0.0   |

  Tier 3 — Quality:
  | avg_patch_quality           | float  | 0.0   |
  | avg_patch_size_ratio        | float  | 0.0   |

  Tier 4 — Robustness:
  | resolution_variance_cv      | float  | 0.0   |
  | error_recovery_rate         | float  | 0.0   |
  | failure_categories          | dict   | {"planning": 0, "implementation": 0, "integration": 0, "hallucination_cascade": 0, "timeout": 0, "tool_failure": 0} |

Given existing code that creates ExperimentMetrics
When I run pytest tests/test_types.py
Then all existing tests still pass (backward compatible defaults)

Given a new ExperimentMetrics instance
When I access metrics.failure_categories["hallucination_cascade"]
Then it returns 0 (default)
```

**Files to Modify:**
- `src/ant_coding/eval/metrics.py`
- `tests/test_types.py` (add tests for new fields)

---

## Story S3-E3-S03: Add baseline_experiment_id to ExperimentConfig

**Branch:** `feature/S3-E3-S03`  
**Points:** 2

**Description:**  
Add the `baseline_experiment_id` optional field to ExperimentConfig. This enables overhead_ratio calculation by referencing a single-agent experiment's results.

**Acceptance Criteria:**

```gherkin
Given the existing ExperimentConfig in src/ant_coding/core/config.py
When I add baseline_experiment_id: Optional[str] = None
Then loading an experiment YAML without baseline_experiment_id succeeds (defaults to None)

Given an experiment YAML with:
  """
  name: "test"
  model: "claude-sonnet"
  memory: "shared"
  tasks:
    source: "custom"
  baseline_experiment_id: "single-agent-claude"
  """
When I call load_experiment_config(path)
Then config.baseline_experiment_id == "single-agent-claude"

Given an experiment YAML without baseline_experiment_id
When I call load_experiment_config(path)
Then config.baseline_experiment_id is None

Given existing experiment config YAML files (e.g., baseline-sequential.yaml)
When I run pytest tests/test_config.py
Then all existing tests still pass
```

**Files to Modify:**
- `src/ant_coding/core/config.py`
- `tests/test_config.py` (add test for new field)

---

## Story S3-E3-S04: PRD+ Type & Config Tests

**Branch:** `feature/S3-E3-S04`  
**Points:** 2

**Description:**  
Comprehensive test coverage for all PRD+ type and config additions. Ensure backward compatibility with all existing tests.

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_types.py tests/test_config.py -v`
Then all tests pass

Given the test files
When I count NEW test functions added for PRD+ fields
Then there are at least 8 new tests covering:
  - TaskResult new field defaults
  - TaskResult with intermediate_test_results populated
  - TaskResult with judge_scores populated
  - TaskResult failure_category validation
  - ExperimentMetrics new field defaults
  - ExperimentMetrics failure_categories dict
  - ExperimentConfig baseline_experiment_id present
  - ExperimentConfig baseline_experiment_id absent (None)

Given all existing tests across the codebase
When I run `pytest tests/ -v`
Then zero regressions (all pre-existing tests still pass)
```

**Files to Modify:**
- `tests/test_types.py`
- `tests/test_config.py`

---

## Epic Completion Checklist

- [ ] TaskResult has all 5 new PRD+ fields with backward-compatible defaults
- [ ] ExperimentMetrics has all 9 new PRD+ fields including failure_categories dict
- [ ] ExperimentConfig has baseline_experiment_id optional field
- [ ] All existing tests still pass (zero regressions)
- [ ] New tests cover all added fields
- [ ] `pytest tests/test_types.py tests/test_config.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated
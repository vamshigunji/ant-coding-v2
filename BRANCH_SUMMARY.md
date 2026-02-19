# Branch Summary: feature/S5-E2-S01

## Story
S5-E2-S01: 4-Tier Metrics Calculation (PRD+)

## What Changed
- Added `src/ant_coding/eval/harness.py` with `calculate_metrics()` function
- Implements all 11 PRD+ metrics across 4 tiers:
  - Tier 1 (Primary): pass_rate, cost_per_resolution (inf when 0 pass)
  - Tier 2 (Efficiency): useful_token_ratio, overhead_ratio (with baseline), tokens_per_resolution
  - Tier 3 (Quality): avg_patch_quality (from judge_scores), avg_patch_size_ratio
  - Tier 4 (Robustness): resolution_variance_cv (sample stdev/mean), error_recovery_rate, failure_categories

## Key Decisions
- cost_per_resolution returns float('inf') when 0 tasks pass (matches acceptance criteria)
- tokens_per_resolution also returns float('inf') when 0 pass
- resolution_variance_cv uses sample standard deviation (n-1 denominator), returns 0.0 if <2 tasks
- overhead_ratio = 0.0 when no baseline provided
- avg_patch_size_ratio = 0.0 when no gold patches exist
- error_recovery_rate looks at intermediate_test_results: counts initial failures that later recovered

## Files Touched
- `src/ant_coding/eval/harness.py` (new)
- `.agent/sprint.yml` (S5-E2-S01 done, S5-E2 in-progress)

## How to Verify
```bash
pytest tests/ -v  # full suite: 131 passed, 1 skipped
```

## Notes for Reviewer
- Tests for calculate_metrics will be in S5-E2-S05 (Evaluation Tests).

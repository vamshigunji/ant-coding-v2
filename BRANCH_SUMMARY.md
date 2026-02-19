# Branch Summary: feature/S5-E2-S05

## Story
S5-E2-S05: Evaluation Tests

## What Changed
- Added `tests/test_eval.py` with 28 comprehensive tests covering:
  - Tier 1 (3 tests): pass_rate, cost_per_resolution, zero-pass edge case
  - Tier 2 (4 tests): useful_token_ratio, overhead_ratio (with/without baseline), tokens_per_resolution
  - Tier 3 (4 tests): avg_patch_quality (with/without scores), avg_patch_size_ratio (with/without gold)
  - Tier 4 (3 tests): error_recovery_rate, failure_categories, resolution_variance_cv
  - LLM Judge (5 tests): default scores, valid parse, malformed parse, code fences, score clamping
  - pass@k (5 tests): basic, all pass, none pass, monotonic increase, empty results
  - Failure Classifier (4 tests): timeout shortcut, tool_failure shortcut, LLM fallback, malformed output
- Marked S5-E2 epic as review

## Key Decisions
- Mock `litellm.acompletion` (not module-level) since FailureClassifier uses lazy import
- Tests cover all edge cases: zero division, empty inputs, clamping, malformed JSON

## Files Touched
- `tests/test_eval.py` (new)
- `.agent/sprint.yml` (S5-E2-S05 done, S5-E2 review)

## How to Verify
```bash
pytest tests/ -v  # full suite: 159 passed, 1 skipped
```

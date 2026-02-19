# Branch Summary: feature/S5-E2-S03

## Story
S5-E2-S03: pass@k Computation

## What Changed
- Added `pass_at_k()` function to `src/ant_coding/eval/harness.py`
- Implements unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k) per task
- Groups results by task_id, averages pass@k across all tasks

## Key Decisions
- Uses math.comb for binomial coefficient (Python 3.8+)
- When k > n, uses effective_k = n (can't sample more than available)
- Returns 0.0 for empty results

## Files Touched
- `src/ant_coding/eval/harness.py` (modified)
- `.agent/sprint.yml` (S5-E2-S03 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 131 passed, 1 skipped
```

## Notes for Reviewer
- Tests for pass@k will be in S5-E2-S05 (Evaluation Tests).

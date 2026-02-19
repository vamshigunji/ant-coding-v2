# Branch Summary: feature/S5-E3-S02

## Story
S5-E3-S02: Bootstrap CI and Effect Size

## What Changed
- Added to `src/ant_coding/eval/comparison.py`:
  - `bootstrap_ci()`: Bootstrap confidence intervals for any statistic (default: mean)
  - `bootstrap_paired_ci()`: Bootstrap CI for paired difference between experiments
  - `interpret_effect_size()`: Human-readable Cohen's d interpretation (negligible/small/medium/large)
  - `ComparisonResult.confidence_intervals` field for storing CIs
  - `compare_experiments()` now computes bootstrap CIs for cost, tokens, and duration

## Key Decisions
- Uses Python's built-in `random` module with optional seed for reproducibility
- Default 1000 bootstrap resamples (fast, adequate for experiment comparison)
- Percentile method for CI bounds (simple, robust)
- Cohen's d thresholds: <0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large

## Files Touched
- `src/ant_coding/eval/comparison.py` (modified)
- `.agent/sprint.yml` (S5-E3-S02 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 159 passed, 1 skipped
```

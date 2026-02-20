# Branch Summary: feature/S6-E1-S04

## Story
S6-E1-S04: Cross-Experiment Comparison CLI

## What Changed
- Created `scripts/compare_results.py`:
  - Single experiment: prints all 4 tiers of metrics
  - Two+ experiments: pairwise comparison with statistical tests
  - Loads metrics from `{result_dir}/metrics.json`
  - Saves `comparison_report.md` to current directory
  - Supports N experiments (produces pairwise comparisons for all pairs)

## Key Decisions
- Uses `metrics_from_json()` for loading (JSON round-trip from S6-E1-S02)
- Per-task results not available from metrics.json alone, so statistical tests
  may have limited data (McNemar's needs per-task results)
- Report saved alongside console output for easy sharing

## Files Touched
- `scripts/compare_results.py` (new)
- `.agent/sprint.yml` (S6-E1-S04 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 188 passed, 1 skipped
```

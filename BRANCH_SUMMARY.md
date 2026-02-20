# Branch Summary: feature/S6-E1-S01

## Story
S6-E1-S01: Markdown Report Generator

## What Changed
- Created `src/ant_coding/eval/report.py` with:
  - `generate_markdown()`: Single experiment report with config table, 4-tier metrics summary, per-agent token breakdown, failure category breakdown
  - `generate_comparison_markdown()`: Multi-experiment comparison with side-by-side table, significance markers (* p<0.05, ** p<0.01), breakeven analysis, pairwise effect sizes and CIs
  - Helper functions for formatting costs/tokens/significance

## Key Decisions
- Significance markers: * for p<0.05, ** for p<0.01, ns for not significant
- Single experiment falls back to `generate_markdown()` when only 1 experiment provided
- Failure categories only shown if any count > 0
- Token breakdown optional (requires EventLogger data)

## Files Touched
- `src/ant_coding/eval/report.py` (new)
- `.agent/sprint.yml` (S6-E1-S01 done, S6-E1 in-progress)

## How to Verify
```bash
pytest tests/ -v  # full suite: 188 passed, 1 skipped
```

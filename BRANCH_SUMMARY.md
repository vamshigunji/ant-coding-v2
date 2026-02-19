# Branch Summary: feature/S5-E1-S03

## Story
S5-E1-S03: Latency Tracking

## What Changed
- Added `src/ant_coding/observability/latency.py` with latency utilities:
  - `get_task_wall_time()`: calculates wall time from TASK_START to TASK_END events
  - `get_llm_latencies()`: extracts duration_ms from LLM_CALL events
  - `get_tool_latencies()`: extracts duration_ms from TOOL_CALL events
  - `get_latency_summary()`: aggregate stats (total, avg, count for LLM and tool calls)
  - `measure_duration_ms()`: context manager for timing code blocks

## Key Decisions
- duration_ms is already recorded in LLM_CALL events (S5-E1-S02), this story adds extraction/analysis utilities
- Wall time computed from event timestamps rather than separate timer to ensure consistency

## Files Touched
- `src/ant_coding/observability/latency.py` (new)
- `.agent/sprint.yml` (S5-E1-S03 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 112 passed, 1 skipped
```

## Notes for Reviewer
- Tests for latency utilities will be part of S5-E1-S04 (Observability Tests).

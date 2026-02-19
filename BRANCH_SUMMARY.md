# Branch Summary: feature/S5-E1-S01

## Story
S5-E1-S01: EventLogger with JSONL Output

## What Changed
- Replaced the skeleton `event_logger.py` with a full `EventLogger` class
- Added `log()` for append-only JSONL writing + in-memory storage
- Added `get_events()` with filtering by agent_name, event_type, task_id
- Added `get_token_breakdown()` for per-agent prompt/completion/total token stats
- Added `clear()`, `event_count`, and `output_path` property

## Key Decisions
- EventLogger supports both file-backed (JSONL) and memory-only modes (output_dir=None)
- Events are written immediately on `log()` — no buffering, crash-safe
- Token breakdown only considers LLM_CALL events, uses payload keys `prompt_tokens`, `completion_tokens`, `total_tokens`

## Files Touched
- `src/ant_coding/observability/event_logger.py` (rewritten)
- `.agent/sprint.yml` (S5-E1-S01 in-progress → done, sprint 5 started)

## How to Verify
```bash
pytest tests/ -v  # full suite: 112 passed, 1 skipped
```

## Notes for Reviewer
- Sprint 5 has begun. This is the first story in E1 (Observability).
- The existing Event and EventType dataclasses are preserved with same interface.
- Tests for EventLogger will be added in S5-E1-S04 (Observability Tests).

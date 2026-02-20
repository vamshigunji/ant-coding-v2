# Branch Summary: feature/S6-E1-S03

## Story
S6-E1-S03: Session Replay

## What Changed
- Created `src/ant_coding/observability/replay.py` with `SessionReplay` class:
  - Loads events from JSONL file, deserializes back to Event objects
  - `step(count)`: Returns next N events, advances cursor
  - `state_at(event_index)`: Reconstructs memory state by replaying MEMORY_WRITE events
  - `token_curve()`: Returns cumulative token curve as (event_index, tokens) tuples
  - `get_events()`: Filtered event retrieval by type and task_id
  - `reset()`: Resets cursor to beginning

## Key Decisions
- Events deserialized back to full Event objects (not raw dicts) for type safety
- State reconstruction replays all MEMORY_WRITE events up to the index
- Token curve only includes LLM_CALL events (where tokens are consumed)

## Files Touched
- `src/ant_coding/observability/replay.py` (new)
- `.agent/sprint.yml` (S6-E1-S03 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 188 passed, 1 skipped
```

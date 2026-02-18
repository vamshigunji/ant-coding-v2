# Branch Summary: feature/S1-E1-S05

## Story
S1-E1-S05: Core Type Definitions

## What Changed
- Defined `Task`, `TaskResult`, `TaskSource`, and `TaskDifficulty` in `src/ant_coding/tasks/types.py`.
- Defined `Event` and `EventType` in `src/ant_coding/observability/event_logger.py`.
- Defined `ExperimentMetrics` in `src/ant_coding/eval/metrics.py`.
- Added unit tests for these types in `tests/test_types.py`.

## Key Decisions
- Used Python `dataclasses` for data objects to keep them lightweight and focused on state.
- Used `Enum` for fixed sets of values (like `EventType`, `TaskSource`) to ensure type safety.
- Provided sensible defaults for all numeric and collection fields in dataclasses.

## Files Touched
- `src/ant_coding/tasks/types.py`
- `src/ant_coding/observability/event_logger.py`
- `src/ant_coding/eval/metrics.py`
- `tests/test_types.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_types.py -v
```

## Notes for Reviewer
- All types are currently importable and behave as expected according to the PRD requirements.

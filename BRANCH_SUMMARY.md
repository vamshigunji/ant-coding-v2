# Branch Summary: feature/S3-E1-S04

## Story
S3-E1-S04: TaskLoader Unified Interface

## What Changed
- Integrated `load_swebench` into the `TaskLoader` class in `src/ant_coding/tasks/loader.py`.
- Updated `load_from_config` to dispatch to either `load_custom` or `load_swebench` based on the `TasksConfig.source` field.
- Mapped `TasksConfig.subset` to the appropriate parameter for each source (file path for `custom`, dataset split for `swe-bench`).
- Added unit tests for the unified loader interface in `tests/test_tasks.py`.

## Key Decisions
- Chose to use `subset` as the path for custom tasks to avoid adding new fields to the Pydantic config model prematurely.
- Ensured that `load_from_config` correctly passes through the `limit` parameter to the SWE-bench adapter.

## Files Touched
- `src/ant_coding/tasks/loader.py`
- `tests/test_tasks.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_tasks.py -v
```

## Notes for Reviewer
- The unified interface makes it easy to switch task sources via experiment configuration.

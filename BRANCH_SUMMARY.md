# Branch Summary: feature/S3-E1-S01

## Story
S3-E1-S01: Custom YAML Task Loader

## What Changed
- Implemented `TaskLoader` class in `src/ant_coding/tasks/loader.py`.
- Added `load_custom` method to load and validate tasks from YAML files.
- Added support for mapping string difficulty values to `TaskDifficulty` enum.
- Captured extra fields in YAML (like `test_command`) into the `Task.metadata` dictionary.
- Created an example task file in `tasks/custom/example-task.yaml`.
- Added unit tests for custom task loading in `tests/test_tasks.py`.

## Key Decisions
- Placed any non-standard task fields into the `metadata` dictionary to keep the `Task` dataclass clean while remaining flexible.
- Implemented a unified `load_from_config` entry point that will eventually route between custom and SWE-bench loaders.

## Files Touched
- `src/ant_coding/tasks/loader.py`
- `tasks/custom/example-task.yaml`
- `tests/test_tasks.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_tasks.py -v
```

## Notes for Reviewer
- The `load_from_config` method currently uses the `subset` field of `TasksConfig` as the file path when the source is `custom`.

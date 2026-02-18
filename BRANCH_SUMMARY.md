# Branch Summary: feature/S3-E1-S03

## Story
S3-E1-S03: SWE-bench Adapter

## What Changed
- Implemented `load_swebench` in `src/ant_coding/tasks/swebench.py`.
- Added support for loading tasks from Hugging Face Datasets (principally `SWE-bench_Lite` and `SWE-bench_Verified`).
- Mapped SWE-bench instance fields to the unified `Task` dataclass.
- Implemented graceful handling of missing dependencies (`datasets` and `swebench`).
- Added unit tests with comprehensive mocking of the `datasets` library.

## Key Decisions
- Defaulted all SWE-bench tasks to `TaskDifficulty.HARD` and a higher token budget (200k) given their typical complexity.
- Used `sys.modules` patching in tests to simulate the presence/absence of the `datasets` library without requiring it in the base environment.

## Files Touched
- `src/ant_coding/tasks/swebench.py`
- `tests/test_tasks.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_tasks.py -v
```

## Notes for Reviewer
- The adapter supports `split` and `subset` parameters to control which part of SWE-bench is loaded.
- Real usage requires `pip install datasets`.

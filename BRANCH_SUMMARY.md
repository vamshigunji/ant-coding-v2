# Branch Summary: feature/S1-E1-S03

## Story
S1-E1-S03: Environment Configuration

## What Changed
- Created `.env.example` with placeholders for API keys.
- Created `.gitignore` to prevent committing sensitive files and build artifacts.
- Implemented `get_env` utility in `src/ant_coding/core/config.py` for safe environment variable access.
- Added unit tests for `get_env` in `tests/test_env.py`.

## Key Decisions
- Implemented a custom `ConfigError` for better error reporting when required keys are missing.
- Used `python-dotenv` to support loading from `.env` files locally.

## Files Touched
- `.env.example`
- `.gitignore`
- `src/ant_coding/core/config.py`
- `tests/test_env.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_env.py -v
```

## Notes for Reviewer
- The `get_env` function is tested for success, failure (raises `ConfigError`), and default value support.

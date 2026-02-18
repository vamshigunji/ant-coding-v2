# Branch Summary: feature/S2-E1-S02

## Story
S2-E1-S02: Token Budget Enforcement

## What Changed
- Added `TokenBudgetExceeded` custom exception in `src/ant_coding/models/provider.py`.
- Updated `ModelProvider` to accept an optional `token_budget`.
- Implemented pre-call and post-call token budget checks.
- Ensured that `TokenBudgetExceeded` exceptions do not trigger retries.
- Added unit tests for budget enforcement in `tests/test_models.py`.

## Key Decisions
- Placed the budget check both before the call (to avoid unnecessary API calls) and after the usage update (to catch budget violations from the latest call).
- Ensured `TokenBudgetExceeded` is a subclass of `ModelError` for consistent exception hierarchy.

## Files Touched
- `src/ant_coding/models/provider.py`
- `tests/test_models.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_models.py -v
```

## Notes for Reviewer
- The `TokenBudgetExceeded` exception provides detailed information about current usage, the budget limit, and the tokens used in the last call.

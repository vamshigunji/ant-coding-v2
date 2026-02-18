# Branch Summary: feature/S2-E1-S01

## Story
S2-E1-S01: ModelConfig and ModelProvider Core

## What Changed
- Implemented `ModelProvider` class in `src/ant_coding/models/provider.py`.
- Integrated `litellm.acompletion` for async LLM calls.
- Added automatic retry logic with exponential backoff for transient errors.
- Implemented token and cost tracking using LiteLLM's `usage` and `completion_cost`.
- Added usage reset and retrieval methods.
- Created `tests/test_models.py` with initial unit tests.

## Key Decisions
- Chose `litellm.acompletion` to support async orchestration.
- Implemented exponential backoff for retries to handle rate limits gracefully.
- Used LiteLLM's utility for cost calculation to ensure accuracy across different providers.

## Files Touched
- `src/ant_coding/models/provider.py`
- `tests/test_models.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_models.py -v
```

## Notes for Reviewer
- LiteLLM warnings regarding `asyncio.iscoroutinefunction` are internal to the library and can be ignored for now.
- Retries are currently set to 3 attempts with a base delay of 1s.

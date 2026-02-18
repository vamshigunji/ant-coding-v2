# Branch Summary: feature/S2-E1-S03

## Story
S2-E1-S03: ModelRegistry with YAML Loading

## What Changed
- Implemented `ModelRegistry` class in `src/ant_coding/models/registry.py`.
- Added `load_from_yaml` method to bulk load configurations from a directory.
- Added `get` method to instantiate a fresh `ModelProvider` from a registered config.
- Added `list_available` method to see registered model names.
- Added `ModelNotFoundError` exception.
- Added unit tests for registry functionality in `tests/test_models.py`.

## Key Decisions
- Registry returns a *new* instance of `ModelProvider` on every `get()` call to ensure token counters and usage tracking are fresh and independent per usage context.
- Gracefully skips malformed YAML files during bulk load to prevent a single bad file from breaking the registry initialization.

## Files Touched
- `src/ant_coding/models/registry.py`
- `tests/test_models.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_models.py -v
```

## Notes for Reviewer
- The registry expects YAML files to be in the format validated by `ModelConfig`.

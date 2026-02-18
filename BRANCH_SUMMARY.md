# Branch Summary: feature/S1-E1-S04

## Story
S1-E1-S04: YAML Config Loader with Pydantic Validation

## What Changed
- Implemented Pydantic models for `ModelConfig`, `MemoryConfig`, `TasksConfig`, `ExecutionConfig`, `EvalConfig`, `OutputConfig`, and `ExperimentConfig`.
- Developed `load_model_config`, `load_memory_config`, and `load_experiment_config` functions in `src/ant_coding/core/config.py`.
- Added support for resolving nested configs in `load_experiment_config` from string identifiers (e.g., resolving `model: "claude-sonnet"` to `configs/models/claude-sonnet.yaml`).
- Created default configuration files in `configs/models/` and `configs/memory/`.
- Created a baseline experiment config in `configs/experiments/baseline-sequential.yaml`.
- Added unit tests for the configuration system in `tests/test_config.py`.

## Key Decisions
- Used `Union[str, ModelConfig]` in `ExperimentConfig` to allow both inline config and file-based references.
- Centralized YAML loading and error handling in a private helper `_load_yaml`.
- Followed the PRD Section 12.2 schema for all config models.

## Files Touched
- `src/ant_coding/core/config.py`
- `configs/models/claude-sonnet.yaml`
- `configs/models/gpt-4o.yaml`
- `configs/models/gemini-flash.yaml`
- `configs/memory/shared.yaml`
- `configs/memory/isolated.yaml`
- `configs/memory/hybrid.yaml`
- `configs/experiments/baseline-sequential.yaml`
- `tests/test_config.py`

## How to Verify
```bash
source .venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/test_config.py -v
```

## Notes for Reviewer
- The config system correctly validates mandatory fields and enum values (like `MemoryMode`).
- Resolving nested configs assumes they live in `configs/models/` and `configs/memory/`.

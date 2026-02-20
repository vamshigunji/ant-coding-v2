# Branch Summary: feature/S6-E1-S05

## Story
S6-E1-S05: Experiment Registry Setup (PRD+)

## What Changed
- Created `src/ant_coding/core/experiment_registry.py` with `ExperimentRegistry`:
  - `add_experiment()`: Register planned experiments with parent, variable_changed, hypothesis
  - `update_status()`: Transition between planned/running/complete
  - `update_outcome()`: Populate all 4 tiers of metrics from ExperimentMetrics
  - `get_lineage()`: Trace parent chain back to root baseline
  - `validate()`: Check for missing variable_changed and stale planned experiments
  - `suggest_id()`: Auto-generate ID from parent + variable_changed slug
  - `list_experiments()`, `get_experiment()`: Lookup helpers
- Created `experiments/registry.yml` with empty template and documentation

## Key Decisions
- YAML-based registry (human-readable, version-controllable)
- Naming convention: {parent}--{variable-slug} for automatic lineage in names
- Infinity values stored as None in YAML (not representable)
- Validation warns (not errors) for stale planned experiments

## Files Touched
- `src/ant_coding/core/experiment_registry.py` (new)
- `experiments/registry.yml` (new)
- `.agent/sprint.yml` (S6-E1-S05 done)

## How to Verify
```bash
pytest tests/ -v  # full suite: 188 passed, 1 skipped
```

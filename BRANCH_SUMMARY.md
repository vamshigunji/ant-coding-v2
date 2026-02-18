# Branch Summary: feature/S3-E2-S05

## Story
S3-E2-S05: ToolRegistry

## What Changed
- Added `ToolRegistry` class that wires all tools (CodeExecutor, FileOperations, GitOperations, CodebaseSearch) together
- `as_dict()` method returns all tool instances keyed by name
- Added 2 test cases to `tests/test_tools.py`

## Key Decisions
- ToolRegistry accepts optional `code_timeout` parameter for CodeExecutor configuration
- All workspace-scoped tools share the same `workspace_dir`

## Files Touched
- `src/ant_coding/tools/registry.py` (new)
- `tests/test_tools.py` (modified — added registry tests)
- `.agent/sprint.yml` (status update)

## How to Verify
```bash
pytest tests/test_tools.py -v -k "registry"
```

## Notes for Reviewer
- Simple wiring class. Could be extended with tool configuration from experiment config in future sprints.

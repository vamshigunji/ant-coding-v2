# Branch Summary: feature/S5-E1-S02

## Story
S5-E1-S02: Wire EventLogger into Runner and Layers

## What Changed
- ModelProvider: logs LLM_CALL events with model, tokens, cost, duration_ms after each complete() call
- MemoryManager: logs MEMORY_WRITE and MEMORY_READ events with agent, key, resolved_key, value_size/found
- ToolRegistry: adds log_tool_call() method for TOOL_CALL events with tool_name, method, args_summary, success, duration_ms
- ExperimentRunner: creates EventLogger instance, passes it to model/memory/tools, sets task context per-task

## Key Decisions
- Used optional injection (event_logger=None) to keep all layers backward-compatible
- Added set_context() methods to ModelProvider and MemoryManager for per-task context switching
- ToolRegistry exposes log_tool_call() for orchestration patterns to call; does not auto-wrap tool methods
- Used TYPE_CHECKING import guards to avoid circular imports between layers and event_logger

## Files Touched
- `src/ant_coding/models/provider.py` (modified)
- `src/ant_coding/memory/manager.py` (modified)
- `src/ant_coding/tools/registry.py` (modified)
- `src/ant_coding/runner/experiment.py` (modified)

## How to Verify
```bash
pytest tests/ -v  # full suite: 112 passed, 1 skipped
```

## Notes for Reviewer
- All layer changes are backward-compatible — event_logger defaults to None.
- Event logging is synchronous (append to list + file write) which is appropriate for benchmarking.

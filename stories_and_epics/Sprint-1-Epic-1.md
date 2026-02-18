# Sprint 1 — Epic 1: Project Scaffolding & Configuration

**Epic ID:** S1-E1  
**Sprint:** 1  
**Priority:** P0 — Foundation  
**Goal:** Create the project skeleton, dependency management, config system, and core type definitions. After this epic, `from ant_coding.core.config import load_config` works and all YAML schemas validate.

**Dependencies:** None (first epic)

---

## Story S1-E1-S01: Initialize Python Project

**Branch:** `feature/S1-E1-S01`  
**Points:** 2

**Description:**  
Create the `ant-coding` Python project with `pyproject.toml`, virtual environment support, and all dependencies declared.

**Acceptance Criteria:**

```gherkin
Given the ant-coding repository is cloned
When I run `pip install -e ".[dev]"`
Then all dependencies install without errors
And `python -c "import ant_coding"` succeeds

Given pyproject.toml exists
When I inspect the [project] section
Then the project name is "ant-coding"
And requires-python is ">=3.11"
And dependencies include: pyyaml, pydantic, litellm, google-adk, gitpython, scipy, numpy, rich, aiofiles, python-dotenv

Given pyproject.toml exists  
When I inspect [project.optional-dependencies]
Then "dev" includes: pytest, pytest-asyncio, ruff
And "swebench" includes: swe-bench, datasets
```

**Files to Create:**
- `pyproject.toml`
- `src/ant_coding/__init__.py` (with `__version__ = "0.1.0"`)
- `README.md` (minimal — project name + one-line description)

---

## Story S1-E1-S02: Create Directory Structure

**Branch:** `feature/S1-E1-S02`  
**Points:** 1

**Description:**  
Create the full directory tree with `__init__.py` files for all packages. Every module referenced in the PRD must have a placeholder.

**Acceptance Criteria:**

```gherkin
Given the project directory exists
When I list the directory tree
Then the following directories exist with __init__.py files:
  | Directory                            |
  | src/ant_coding/core/                 |
  | src/ant_coding/models/               |
  | src/ant_coding/orchestration/        |
  | src/ant_coding/orchestration/examples/ |
  | src/ant_coding/memory/               |
  | src/ant_coding/tools/                |
  | src/ant_coding/tasks/                |
  | src/ant_coding/eval/                 |
  | src/ant_coding/observability/        |
And the following non-Python directories exist:
  | Directory         |
  | configs/models/   |
  | configs/memory/   |
  | configs/experiments/ |
  | configs/tasks/    |
  | tasks/custom/     |
  | results/          |
  | tests/            |
  | scripts/          |

Given I run `find src/ant_coding -name "__init__.py" | wc -l`
Then the count is at least 10
```

**Files to Create:**
- All `__init__.py` files (can be empty or have minimal docstrings)
- `results/.gitkeep`
- `tasks/custom/.gitkeep`

---

## Story S1-E1-S03: Environment Configuration

**Branch:** `feature/S1-E1-S03`  
**Points:** 1

**Description:**  
Create `.env.example` template and environment loading utility. API keys must never be committed.

**Acceptance Criteria:**

```gherkin
Given .env.example exists
When I inspect its contents
Then it contains placeholders for:
  | Variable          | Example Value       |
  | ANTHROPIC_API_KEY | sk-ant-xxx          |
  | OPENAI_API_KEY    | sk-xxx              |
  | GOOGLE_API_KEY    | AIzaxxx             |

Given .gitignore exists
When I inspect its contents
Then it includes: .env, __pycache__, *.pyc, .venv/, results/, *.egg-info

Given a .env file with ANTHROPIC_API_KEY=test-key
When I call `from ant_coding.core.config import get_env`
And I call `get_env("ANTHROPIC_API_KEY")`
Then it returns "test-key"

Given no .env file exists
When I call `get_env("MISSING_KEY")`
Then it raises a ConfigError with message containing "MISSING_KEY"
```

**Files to Create:**
- `.env.example`
- `.gitignore`
- `src/ant_coding/core/config.py` (env loading portion)

---

## Story S1-E1-S04: YAML Config Loader with Pydantic Validation

**Branch:** `feature/S1-E1-S04`  
**Points:** 5

**Description:**  
Build the configuration system that loads YAML files and validates them with Pydantic models. This is the backbone — every other layer reads config through this.

**Acceptance Criteria:**

```gherkin
Given a valid model config YAML file:
  """
  name: "claude-sonnet"
  litellm_model: "anthropic/claude-sonnet-4-5-20250929"
  api_key_env: "ANTHROPIC_API_KEY"
  max_tokens: 8192
  temperature: 0.0
  """
When I call `load_model_config("configs/models/claude-sonnet.yaml")`
Then it returns a ModelConfig with name="claude-sonnet" and max_tokens=8192

Given a model config YAML missing required field "litellm_model"
When I call `load_model_config(path)`
Then it raises a ValidationError mentioning "litellm_model"

Given a valid memory config YAML:
  """
  mode: "hybrid"
  shared_keys: ["implementation_plan", "test_results"]
  """
When I call `load_memory_config(path)`
Then it returns a MemoryConfig with mode=MemoryMode.HYBRID and 2 shared_keys

Given a memory config with mode="invalid_mode"
When I call `load_memory_config(path)`
Then it raises a ValidationError

Given a valid experiment config YAML (full schema from PRD Section 12.2)
When I call `load_experiment_config(path)`
Then it returns an ExperimentConfig with all nested configs resolved
And model config is loaded from the referenced model YAML
And memory config is loaded from the referenced memory YAML

Given an experiment config referencing a nonexistent model "gpt-5"
When I call `load_experiment_config(path)`
Then it raises ConfigError with message containing "gpt-5"
```

**Pydantic Models to Define:**

```python
class ModelConfig(BaseModel): ...
class MemoryConfig(BaseModel): ...
class TasksConfig(BaseModel): ...
class ExecutionConfig(BaseModel): ...
class EvalConfig(BaseModel): ...
class OutputConfig(BaseModel): ...
class ExperimentConfig(BaseModel): ...  # Top-level, contains all above
```

**Files to Create:**
- `src/ant_coding/core/config.py` (complete config system)
- `configs/models/claude-sonnet.yaml`
- `configs/models/gpt-4o.yaml`
- `configs/models/gemini-flash.yaml`
- `configs/memory/shared.yaml`
- `configs/memory/isolated.yaml`
- `configs/memory/hybrid.yaml`
- `configs/experiments/baseline-sequential.yaml`

---

## Story S1-E1-S05: Core Type Definitions

**Branch:** `feature/S1-E1-S05`  
**Points:** 3

**Description:**  
Define all shared data types used across layers: Task, TaskResult, Event, EventType, ExperimentMetrics. These are dataclasses, not Pydantic models (config uses Pydantic; data objects use dataclasses).

**Acceptance Criteria:**

```gherkin
Given I import Task from ant_coding.tasks.types
When I create a Task with id="test-1" and description="Fix bug" and source=TaskSource.CUSTOM
Then task.id == "test-1"
And task.difficulty == TaskDifficulty.MEDIUM (default)
And task.max_tokens_budget == 100_000 (default)
And task.timeout_seconds == 600 (default)
And task.files_context == [] (default)

Given I import TaskResult from ant_coding.tasks.types
When I create a TaskResult with task_id="test-1" and experiment_id="exp-1" and success=True
Then result.total_tokens == 0 (default)
And result.agent_traces == [] (default)

Given I import Event and EventType from ant_coding.observability.event_logger
When I create an Event with type=EventType.LLM_CALL
Then event.type == EventType.LLM_CALL
And EventType has values: AGENT_START, AGENT_END, LLM_CALL, TOOL_CALL, MEMORY_READ, MEMORY_WRITE, ERROR, TASK_START, TASK_END

Given I import ExperimentMetrics from ant_coding.eval.metrics
When I create metrics with experiment_id="exp-1" and pass_rate=0.8
Then all fields have sensible defaults (0 for counts, 0.0 for floats)
```

**Files to Create:**
- `src/ant_coding/core/types.py` (if shared types needed)
- `src/ant_coding/tasks/types.py` (Task, TaskResult, TaskSource, TaskDifficulty)
- `src/ant_coding/eval/metrics.py` (ExperimentMetrics — dataclass only, no logic yet)
- `src/ant_coding/observability/event_logger.py` (Event, EventType — dataclass only, no logic yet)

---

## Story S1-E1-S06: Setup Script and Smoke Test

**Branch:** `feature/S1-E1-S06`  
**Points:** 2

**Description:**  
Create the one-command setup script and a smoke test that validates the entire foundation works.

**Acceptance Criteria:**

```gherkin
Given a fresh clone of the repository
When I run `bash scripts/setup.sh`
Then a virtual environment is created at .venv/
And all dependencies are installed
And it prints "Setup complete"

Given the project is installed
When I run `python -c "from ant_coding.core.config import load_experiment_config; print('OK')"`
Then it prints "OK"

Given the project is installed
When I run `python -m pytest tests/test_config.py -v`
Then all config validation tests pass

Given the project is installed
When I run `python -m pytest tests/test_types.py -v`
Then all type definition tests pass
```

**Files to Create:**
- `scripts/setup.sh`
- `tests/test_config.py`
- `tests/test_types.py`

---

## Epic Completion Checklist

Before marking this epic as `review`, the dev agent must verify:

- [ ] `pip install -e ".[dev]"` succeeds
- [ ] All `__init__.py` files present
- [ ] Config loader handles valid + invalid YAML correctly
- [ ] All type dataclasses importable
- [ ] `pytest tests/test_config.py tests/test_types.py` passes
- [ ] No linting errors (`ruff check src/`)
- [ ] All 6 stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated with all branch links

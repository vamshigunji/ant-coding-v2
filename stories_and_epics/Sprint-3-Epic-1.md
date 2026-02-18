# Sprint 3 — Epic 1: Task Management

**Epic ID:** S3-E1  
**Sprint:** 3  
**Priority:** P0 — Foundation  
**Goal:** Build the task loading, workspace management, and SWE-bench integration. After this epic, tasks can be loaded from YAML and SWE-bench, and each task gets an isolated workspace with git capabilities.

**Dependencies:** S1-E1 (config, types)

---

## Story S3-E1-S01: Custom YAML Task Loader

**Branch:** `feature/S3-E1-S01`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given a custom task YAML file:
  """
  tasks:
    - id: "fix-auth-bug"
      description: "Fix the authentication bypass in login.py"
      files_context: ["src/auth/login.py"]
      test_command: "pytest tests/test_auth.py -v"
      difficulty: "medium"
      timeout_seconds: 300
  """
When I call loader.load_custom("tasks/custom/example.yaml")
Then it returns a list with 1 Task object
And task.id == "fix-auth-bug"
And task.source == TaskSource.CUSTOM
And task.difficulty == TaskDifficulty.MEDIUM

Given a YAML with 5 tasks
When I call loader.load_custom(path)
Then it returns a list of 5 Task objects with unique ids

Given a YAML with a task missing required "id" field
When I call loader.load_custom(path)
Then it raises TaskLoadError with message about missing id
```

**Files to Create:**
- `src/ant_coding/tasks/loader.py`
- `tasks/custom/example-task.yaml`

---

## Story S3-E1-S02: TaskWorkspace Setup and Teardown

**Branch:** `feature/S3-E1-S02`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given a Task with id="test-task"
When I call await workspace.setup()
Then a unique directory is created at /tmp/ant-coding/test-task_{hash}/
And the directory exists and is writable

Given a TaskWorkspace that has been set up
When I create a file in the workspace and call await workspace.get_patch()
Then it returns a non-empty git diff string

Given a TaskWorkspace with modified files
When I call await workspace.run_tests() with test_command="python -m pytest tests/ -v"
Then it returns (bool_passed, str_output) tuple

Given a TaskWorkspace that has been used
When I call await workspace.teardown()
Then the workspace directory is deleted

Given a Task with repo_url and base_commit (SWE-bench style)
When I call await workspace.setup()
Then the repo is cloned and checked out to base_commit
```

**Files to Create:**
- `src/ant_coding/tasks/workspace.py`

---

## Story S3-E1-S03: SWE-bench Adapter

**Branch:** `feature/S3-E1-S03`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given the swe-bench package is installed
When I call loader.load_swebench(split="lite", limit=5)
Then it returns exactly 5 Task objects
And each task has: repo_url, base_commit, description, test_command
And each task.source == TaskSource.SWEBENCH

Given split="lite"
When I inspect the loaded tasks
Then task.description contains the GitHub issue text
And task.test_command contains the test patch command

Given the swe-bench package is NOT installed
When I call loader.load_swebench()
Then it raises ImportError with message "pip install ant-coding[swebench]"
```

**Files to Create:**
- `src/ant_coding/tasks/swebench.py`

---

## Story S3-E1-S04: TaskLoader Unified Interface

**Branch:** `feature/S3-E1-S04`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given a tasks config with source="custom" and custom_path="tasks/custom/example-task.yaml"
When I call loader.load_from_config(config)
Then it delegates to load_custom() and returns Task objects

Given a tasks config with source="swebench" and split="lite" and limit=10
When I call loader.load_from_config(config)
Then it delegates to load_swebench() and returns Task objects

Given a tasks config with source="invalid"
When I call loader.load_from_config(config)
Then it raises TaskLoadError
```

**Files to Modify:**
- `src/ant_coding/tasks/loader.py`

---

## Story S3-E1-S05: Task Management Tests

**Branch:** `feature/S3-E1-S05`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite
When I run `pytest tests/test_tasks.py -v`
Then all tests pass

Given the test file
When I count test functions
Then there are at least 10 test cases covering:
  - Custom YAML loading (valid, invalid, missing fields)
  - Workspace setup/teardown
  - Workspace get_patch (with actual file changes)
  - Workspace run_tests (mock subprocess)
  - SWE-bench loading (mocked dataset)
  - Unified loader dispatch
```

**Files to Create:**
- `tests/test_tasks.py`

---

## Epic Completion Checklist

- [ ] Custom YAML tasks load and validate
- [ ] TaskWorkspace creates isolated directories with git init
- [ ] SWE-bench adapter loads from dataset (with graceful import error)
- [ ] Unified TaskLoader dispatches based on config
- [ ] `pytest tests/test_tasks.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

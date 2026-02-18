# Sprint 3 — Epic 2: Tool Layer

**Epic ID:** S3-E2  
**Sprint:** 3  
**Priority:** P1 — Core  
**Goal:** Build all tools (code execution, file ops, git ops, search) that agents use to interact with code. Tools are scoped to the task workspace for security.

**Dependencies:** S3-E1 (TaskWorkspace for filesystem scoping)

---

## Story S3-E2-S01: CodeExecutor with Sandboxing

**Branch:** `feature/S3-E2-S01`  
**Points:** 5

**Acceptance Criteria:**

```gherkin
Given a CodeExecutor
When I call execute(code='print("hello")', language="python")
Then it returns {"success": True, "stdout": "hello\n", "stderr": "", "exit_code": 0}

Given a CodeExecutor with timeout=2
When I call execute(code="import time; time.sleep(10)")
Then it returns {"success": False, "exit_code": -1} within ~3 seconds
And stderr contains timeout-related message

Given a CodeExecutor
When I call run_command("echo 'test'", cwd=workspace_dir)
Then it returns {"success": True, "stdout": "test\n", "exit_code": 0}

Given a CodeExecutor
When I call run_command("exit 1")
Then it returns {"success": False, "exit_code": 1}

Given a CodeExecutor
When I call execute(code="import os; os.system('rm -rf /')")
Then the execution is contained to the sandbox (no system damage)
```

**Files to Create:**
- `src/ant_coding/tools/code_executor.py`

---

## Story S3-E2-S02: FileOperations with Workspace Scoping

**Branch:** `feature/S3-E2-S02`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given FileOperations initialized with workspace_dir="/tmp/test-ws"
When I call write_file("src/main.py", "print('hello')")
Then /tmp/test-ws/src/main.py exists with that content

Given a file exists at src/main.py in the workspace
When I call read_file("src/main.py")
Then it returns the file contents as a string

Given a file with content "old_function()" exists
When I call edit_file("src/main.py", "old_function()", "new_function()")
Then the file now contains "new_function()"
And edit_file returns True

Given a file where "target_string" appears 0 times
When I call edit_file(path, "target_string", "replacement")
Then it returns False (no replacement made)

Given FileOperations with workspace_dir="/tmp/test-ws"
When I call read_file("../../etc/passwd")
Then it raises SecurityError (path traversal blocked)

Given a workspace with files: src/a.py, src/b.py, tests/test_a.py
When I call list_files("**/*.py")
Then it returns ["src/a.py", "src/b.py", "tests/test_a.py"]

Given a workspace with files containing "TODO" in some lines
When I call search_files("TODO")
Then it returns a list of dicts with file, line_number, line_content
```

**Files to Create:**
- `src/ant_coding/tools/file_ops.py`

---

## Story S3-E2-S03: GitOperations

**Branch:** `feature/S3-E2-S03`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given a git-initialized workspace with modified files
When I call get_diff()
Then it returns a non-empty unified diff string

Given a git-initialized workspace
When I call get_status()
Then it returns a list of {"file": str, "status": str} dicts

Given a workspace with staged changes
When I call commit("feat: initial implementation")
Then it returns a commit hash (40-char hex string)

Given a workspace
When I call create_branch("feature/test")
Then the current branch is now "feature/test"
```

**Files to Create:**
- `src/ant_coding/tools/git_ops.py`

---

## Story S3-E2-S04: CodebaseSearch

**Branch:** `feature/S3-E2-S04`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given a workspace with Python files
When I call grep("def calculate", path="src/")
Then it returns matches with file, line_number, line_content

Given a workspace with a class "UserManager" defined in src/models.py
When I call find_definition("UserManager")
Then the results include {"file": "src/models.py", "line_number": N, ...}

Given a workspace where "UserManager" is imported in 3 files
When I call find_references("UserManager")
Then it returns at least 3 results
```

**Files to Create:**
- `src/ant_coding/tools/search.py`

---

## Story S3-E2-S05: ToolRegistry

**Branch:** `feature/S3-E2-S05`  
**Points:** 2

**Acceptance Criteria:**

```gherkin
Given a workspace_dir
When I create ToolRegistry(workspace_dir)
Then registry.code_executor is a CodeExecutor instance
And registry.file_ops is a FileOperations instance scoped to workspace_dir
And registry.git_ops is a GitOperations instance scoped to workspace_dir
And registry.search is a CodebaseSearch instance

Given a ToolRegistry
When I call as_dict()
Then it returns a dict with keys: "code_executor", "file_ops", "git_ops", "search"
And each value is the corresponding tool instance
```

**Files to Create:**
- `src/ant_coding/tools/registry.py`

---

## Story S3-E2-S06: Tool Layer Tests

**Branch:** `feature/S3-E2-S06`  
**Points:** 3

**Acceptance Criteria:**

```gherkin
Given the test suite using temp directories
When I run `pytest tests/test_tools.py -v`
Then all tests pass

Given the test file
When I count test functions
Then there are at least 15 test cases covering:
  - CodeExecutor: success, failure, timeout
  - FileOps: read, write, edit, list, search, path traversal block
  - GitOps: diff, status, commit, branch
  - Search: grep, find_definition
  - ToolRegistry: initialization, as_dict
```

**Files to Create:**
- `tests/test_tools.py`

---

## Epic Completion Checklist

- [ ] CodeExecutor runs Python with timeout and sandboxing
- [ ] FileOperations scoped to workspace, blocks path traversal
- [ ] GitOperations produces diffs and commits
- [ ] CodebaseSearch finds definitions and references
- [ ] ToolRegistry wires everything together
- [ ] `pytest tests/test_tools.py` passes
- [ ] No linting errors
- [ ] All stories have BRANCH_SUMMARY.md
- [ ] sprint.yml updated

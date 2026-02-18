import pytest
import asyncio
from ant_coding.tools.code_executor import CodeExecutor
from ant_coding.tools.file_ops import FileOperations, SecurityError
from ant_coding.tools.git_ops import GitOperations
from ant_coding.tools.search import CodebaseSearch

@pytest.fixture
def temp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws

@pytest.fixture
def git_workspace(temp_workspace):
    from git import Repo
    Repo.init(temp_workspace)
    return temp_workspace

def test_git_ops_status_and_commit(git_workspace):
    git = GitOperations(git_workspace)
    
    # Create and add a file
    test_file = git_workspace / "test.txt"
    test_file.write_text("initial")
    git.add("test.txt")
    
    status = git.get_status()
    assert any(s["file"] == "test.txt" and s["status"] == "staged" for s in status)
    
    commit_hash = git.commit("initial commit")
    assert len(commit_hash) == 40
    
    # Modify and check diff
    test_file.write_text("modified")
    diff = git.get_diff(staged=False)
    assert "modified" in diff

def test_git_ops_branch(git_workspace):
    git = GitOperations(git_workspace)
    # Need at least one commit before branching
    test_file = git_workspace / "init.txt"
    test_file.write_text("init")
    git.add(".")
    git.commit("init")
    
    git.create_branch("feature/test")
    assert git.repo.active_branch.name == "feature/test"

def test_file_ops_write_read(temp_workspace):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws

def test_file_ops_write_read(temp_workspace):
    ops = FileOperations(temp_workspace)
    ops.write_file("src/main.py", "print('hello')")
    
    assert (temp_workspace / "src/main.py").exists()
    assert ops.read_file("src/main.py") == "print('hello')"

def test_file_ops_edit(temp_workspace):
    ops = FileOperations(temp_workspace)
    ops.write_file("app.py", "def old(): pass")
    
    success = ops.edit_file("app.py", "old()", "new()")
    assert success is True
    assert "def new()" in ops.read_file("app.py")
    
    success = ops.edit_file("app.py", "nonexistent", "new")
    assert success is False

def test_file_ops_security(temp_workspace):
    ops = FileOperations(temp_workspace)
    with pytest.raises(SecurityError):
        ops.read_file("../../../etc/passwd")

def test_file_ops_list_and_search(temp_workspace):
    ops = FileOperations(temp_workspace)
    ops.write_file("src/a.py", "TODO: fix this")
    ops.write_file("src/b.py", "no tasks here")
    ops.write_file("tests/test.py", "TODO: add tests")
    
    files = ops.list_files("**/*.py")
    assert "src/a.py" in files
    assert "tests/test.py" in files
    
    results = ops.search_files("TODO")
    assert len(results) == 2
    assert results[0]["file"] == "src/a.py"
    assert results[1]["file"] == "tests/test.py"

@pytest.mark.asyncio
async def test_execute_python_success():
    executor = CodeExecutor()
    result = await executor.execute("print('hello world')")
    
    assert result["success"] is True
    assert result["stdout"].strip() == "hello world"
    assert result["exit_code"] == 0

@pytest.mark.asyncio
async def test_execute_python_error():
    executor = CodeExecutor()
    result = await executor.execute("raise ValueError('oops')")
    
    assert result["success"] is False
    assert "ValueError: oops" in result["stderr"]
    assert result["exit_code"] != 0

@pytest.mark.asyncio
async def test_execute_timeout():
    executor = CodeExecutor(timeout=1)
    result = await executor.execute("import time; time.sleep(5)")
    
    assert result["success"] is False
    assert "timed out" in result["stderr"]
    assert result["exit_code"] == -1

@pytest.mark.asyncio
async def test_run_command_success():
    executor = CodeExecutor()
    result = await executor.run_command("echo 'it works'")
    
    assert result["success"] is True
    assert result["stdout"].strip() == "it works"

@pytest.mark.asyncio
async def test_run_command_failure():
    executor = CodeExecutor()
    result = await executor.run_command("ls nonexistent_file_xyz")

    assert result["success"] is False
    assert result["exit_code"] != 0


# ── CodebaseSearch Tests ──

@pytest.fixture
def search_workspace(tmp_path):
    """Create a workspace with Python files for search testing."""
    ws = tmp_path / "search_ws"
    ws.mkdir()

    # src/models.py — has a class definition
    src = ws / "src"
    src.mkdir()
    (src / "models.py").write_text(
        "class UserManager:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "    def get_user(self, user_id):\n"
        "        return None\n"
        "\n"
        "def calculate(x, y):\n"
        "    return x + y\n"
    )

    # src/views.py — imports and uses UserManager
    (src / "views.py").write_text(
        "from models import UserManager\n"
        "\n"
        "manager = UserManager()\n"
        "result = manager.get_user(1)\n"
    )

    # src/utils.py — also references UserManager
    (src / "utils.py").write_text(
        "from models import UserManager\n"
        "\n"
        "def create_manager() -> UserManager:\n"
        "    return UserManager()\n"
    )

    # tests/test_models.py — imports UserManager for testing
    tests = ws / "tests"
    tests.mkdir()
    (tests / "test_models.py").write_text(
        "from src.models import UserManager\n"
        "\n"
        "def test_user_manager():\n"
        "    mgr = UserManager()\n"
        "    assert mgr is not None\n"
    )

    return ws


def test_search_grep_basic(search_workspace):
    search = CodebaseSearch(search_workspace)
    results = search.grep("def calculate", path="src/")

    assert len(results) >= 1
    assert any(
        r["file"] == "src/models.py" and "def calculate" in r["line_content"]
        for r in results
    )


def test_search_grep_regex(search_workspace):
    search = CodebaseSearch(search_workspace)
    results = search.grep(r"def \w+\(self", path="src/")

    # Should match __init__ and get_user
    assert len(results) >= 2


def test_search_find_definition(search_workspace):
    search = CodebaseSearch(search_workspace)
    results = search.find_definition("UserManager")

    assert len(results) >= 1
    assert any(
        r["file"] == "src/models.py" and "class UserManager" in r["line_content"]
        for r in results
    )


def test_search_find_references(search_workspace):
    search = CodebaseSearch(search_workspace)
    results = search.find_references("UserManager")

    # UserManager is imported/used in views.py, utils.py, and tests/test_models.py
    assert len(results) >= 3

    referenced_files = {r["file"] for r in results}
    assert "src/views.py" in referenced_files
    assert "src/utils.py" in referenced_files


def test_search_grep_no_matches(search_workspace):
    search = CodebaseSearch(search_workspace)
    results = search.grep("nonexistent_pattern_xyz")

    assert results == []


def test_search_skips_binary_files(tmp_path):
    ws = tmp_path / "bin_ws"
    ws.mkdir()
    (ws / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (ws / "code.py").write_text("def hello(): pass\n")

    search = CodebaseSearch(ws)
    results = search.grep("hello")

    assert len(results) == 1
    assert results[0]["file"] == "code.py"

import pytest
import asyncio
from ant_coding.tools.code_executor import CodeExecutor
from ant_coding.tools.file_ops import FileOperations, SecurityError

@pytest.fixture
def temp_workspace(tmp_path):
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

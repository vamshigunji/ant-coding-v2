import pytest
import asyncio
from ant_coding.tools.code_executor import CodeExecutor

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

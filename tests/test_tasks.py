import pytest
from pathlib import Path
from ant_coding.tasks.loader import TaskLoader, TaskLoadError
from ant_coding.tasks.types import TaskSource, TaskDifficulty, Task
from ant_coding.tasks.workspace import TaskWorkspace

@pytest.mark.asyncio
async def test_workspace_setup_teardown():
    task = Task(id="test-task", description="test", source=TaskSource.CUSTOM)
    workspace = TaskWorkspace(task)
    
    await workspace.setup()
    assert workspace.workspace_dir.exists()
    assert (workspace.workspace_dir / ".git").exists()
    
    workspace_path = workspace.workspace_dir
    await workspace.teardown()
    assert not workspace_path.exists()

@pytest.mark.asyncio
async def test_workspace_get_patch():
    task = Task(id="patch-test", description="test", source=TaskSource.CUSTOM)
    workspace = TaskWorkspace(task)
    await workspace.setup()
    
    # Create a file
    test_file = workspace.workspace_dir / "hello.txt"
    test_file.write_text("hello world")
    
    patch_str = await workspace.get_patch()
    assert "hello.txt" in patch_str
    assert "hello world" in patch_str
    
    await workspace.teardown()

@pytest.mark.asyncio
async def test_workspace_run_command():
    task = Task(id="cmd-test", description="test", source=TaskSource.CUSTOM)
    workspace = TaskWorkspace(task)
    await workspace.setup()
    
    success, output = await workspace.run_command("echo 'it works'")
    assert success is True
    assert "it works" in output
    
    success, output = await workspace.run_command("exit 1")
    assert success is False
    
    await workspace.teardown()

def test_load_custom_success():
    loader = TaskLoader()
    tasks = loader.load_custom("tasks/custom/example-task.yaml")
    
    assert len(tasks) == 2
    assert tasks[0].id == "fix-auth-bug"
    assert tasks[0].source == TaskSource.CUSTOM
    assert tasks[0].difficulty == TaskDifficulty.MEDIUM
    assert tasks[0].metadata["test_command"] == "pytest tests/test_auth.py -v"
    
    assert tasks[1].id == "optimize-query"
    assert tasks[1].difficulty == TaskDifficulty.HARD

def test_load_custom_missing_id():
    path = Path("tasks/custom/invalid.yaml")
    path.write_text("tasks: [{description: 'no id'}]")
    
    loader = TaskLoader()
    with pytest.raises(TaskLoadError) as excinfo:
        loader.load_custom(path)
    assert "missing required 'id'" in str(excinfo.value)
    path.unlink()

def test_load_custom_not_found():
    loader = TaskLoader()
    with pytest.raises(TaskLoadError) as excinfo:
        loader.load_custom("nonexistent.yaml")
    assert "not found" in str(excinfo.value)

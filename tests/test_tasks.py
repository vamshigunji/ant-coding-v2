import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from ant_coding.tasks.loader import TaskLoader, TaskLoadError
from ant_coding.tasks.types import TaskSource, TaskDifficulty, Task
from ant_coding.tasks.workspace import TaskWorkspace
from ant_coding.tasks.swebench import load_swebench

from ant_coding.core.config import TasksConfig

def test_loader_from_config_custom():
    config = TasksConfig(source="custom", subset="tasks/custom/example-task.yaml")
    loader = TaskLoader()
    tasks = loader.load_from_config(config)
    assert len(tasks) == 2
    assert tasks[0].source == TaskSource.CUSTOM

def test_loader_from_config_swebench():
    config = TasksConfig(source="swe-bench", subset="lite", limit=5)
    loader = TaskLoader()
    
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [{"instance_id": f"id-{i}", "problem_statement": "p", "repo": "r", "base_commit": "c", "version": "v", "test_patch": "t"} for i in range(5)]
    
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        tasks = loader.load_from_config(config)
    
    assert len(tasks) == 5
    assert tasks[0].source == TaskSource.SWE_BENCH

def test_loader_from_config_invalid():
    config = TasksConfig(source="invalid")
    loader = TaskLoader()
    with pytest.raises(TaskLoadError):
        loader.load_from_config(config)

@pytest.mark.asyncio
async def test_load_swebench_import_error(monkeypatch):
    """Verify ImportError if datasets is missing."""
    import sys
    # Simulate missing 'datasets'
    with patch.dict(sys.modules, {'datasets': None}):
        with pytest.raises(ImportError) as excinfo:
            load_swebench()
        assert "pip install datasets" in str(excinfo.value)

def test_load_swebench_mocked():
    """Verify SWE-bench adapter mapping with mocked dataset."""
    mock_item = {
        "instance_id": "test-id",
        "problem_statement": "Fix this",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "version": "1.0",
        "test_patch": "diff --git ..."
    }
    
    # Create a mock for the datasets module
    mock_datasets = MagicMock()
    mock_datasets.load_dataset.return_value = [mock_item]
    
    # Patch sys.modules to include our mock datasets
    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        tasks = load_swebench(limit=1)
    
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "test-id"
    assert task.source == TaskSource.SWE_BENCH
    assert task.metadata["base_commit"] == "abc123"
    assert task.metadata["repo_url"] == "https://github.com/owner/repo"

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

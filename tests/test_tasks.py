import pytest
from pathlib import Path
from ant_coding.tasks.loader import TaskLoader, TaskLoadError
from ant_coding.tasks.types import TaskSource, TaskDifficulty

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

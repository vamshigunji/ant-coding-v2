"""
Utilities for loading tasks from various sources.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from ant_coding.tasks.types import Task, TaskSource, TaskDifficulty
from ant_coding.core.config import TasksConfig

class TaskLoadError(Exception):
    """Exception raised when task loading fails."""
    pass

class TaskLoader:
    """
    Loader for tasks from YAML files and external datasets.
    """
    
    def load_custom(self, path: Union[str, Path]) -> List[Task]:
        """
        Load custom tasks from a YAML file.
        
        Args:
            path: Path to the YAML file.
            
        Returns:
            A list of Task objects.
            
        Raises:
            TaskLoadError: If loading or validation fails.
        """
        path = Path(path)
        if not path.exists():
            raise TaskLoadError(f"Task file not found: {path}")
            
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise TaskLoadError(f"Failed to parse YAML file {path}: {e}")
            
        if not isinstance(data, dict) or "tasks" not in data:
            raise TaskLoadError(f"Invalid task file format: {path}. Expected 'tasks' key.")
            
        tasks = []
        for task_data in data["tasks"]:
            if "id" not in task_data:
                raise TaskLoadError(f"Task missing required 'id' field in {path}")
            if "description" not in task_data:
                raise TaskLoadError(f"Task {task_data.get('id')} missing 'description' field")
                
            # Map string difficulty to enum
            diff_str = task_data.get("difficulty", "medium").upper()
            try:
                difficulty = TaskDifficulty[diff_str]
            except KeyError:
                difficulty = TaskDifficulty.MEDIUM
                
            task = Task(
                id=task_data["id"],
                description=task_data["description"],
                source=TaskSource.CUSTOM,
                difficulty=difficulty,
                max_tokens_budget=task_data.get("max_tokens_budget", 100_000),
                timeout_seconds=task_data.get("timeout_seconds", 600),
                files_context=task_data.get("files_context", []),
                metadata=task_data.get("metadata", {})
            )
            
            # Store any extra fields in metadata
            extra_keys = set(task_data.keys()) - {
                "id", "description", "difficulty", "max_tokens_budget", 
                "timeout_seconds", "files_context", "metadata"
            }
            for k in extra_keys:
                task.metadata[k] = task_data[k]
                
            tasks.append(task)
            
        return tasks

    def load_from_config(self, config: TasksConfig) -> List[Task]:
        """
        Unified interface to load tasks based on configuration.
        
        Args:
            config: TasksConfig object.
            
        Returns:
            A list of Task objects.
        """
        if config.source == "custom":
            # Assuming custom_path is in metadata or we need to add it to TasksConfig
            # For now, we'll check if custom_path is in metadata or use a default
            custom_path = config.subset or "tasks/custom/example-task.yaml"
            return self.load_custom(custom_path)
            
        if config.source == "swe-bench":
             # Placeholder for story S3-E1-S03
             raise NotImplementedError("SWE-bench loader not yet implemented")
             
        raise TaskLoadError(f"Unsupported task source: {config.source}")

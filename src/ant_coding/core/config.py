"""
Configuration and environment loading utilities.
"""

import os
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


def get_env(key: str, default: str = None) -> str:
    """
    Get an environment variable or raise ConfigError if it's missing and no default is provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value


# --- Config Models ---


class ModelConfig(BaseModel):
    name: str
    litellm_model: str
    api_key_env: str
    max_tokens: int = 8192
    temperature: float = 0.0


class MemoryMode(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"
    HYBRID = "hybrid"


class MemoryConfig(BaseModel):
    mode: MemoryMode
    shared_keys: List[str] = Field(default_factory=list)


class TasksConfig(BaseModel):
    source: str  # e.g., "swe-bench", "custom"
    subset: Optional[str] = None
    task_ids: Optional[List[str]] = None
    limit: Optional[int] = None


class ExecutionConfig(BaseModel):
    max_workers: int = 1
    timeout_seconds: int = 1800
    max_iterations: int = 10


class EvalConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["pass@1"])
    eval_model: Optional[str] = None


class OutputConfig(BaseModel):
    dir: str = "results"
    save_traces: bool = True


class ExperimentConfig(BaseModel):
    name: str
    model: Union[str, ModelConfig]
    memory: Union[str, MemoryConfig]
    tasks: TasksConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# --- Loaders ---


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML file {path}: {e}")


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    data = _load_yaml(path)
    return ModelConfig(**data)


def load_memory_config(path: Union[str, Path]) -> MemoryConfig:
    data = _load_yaml(path)
    return MemoryConfig(**data)


def load_experiment_config(path: Union[str, Path]) -> ExperimentConfig:
    data = _load_yaml(path)

    # Resolve nested model config if it's a string (path)
    if isinstance(data.get("model"), str):
        model_path = Path("configs/models") / f"{data['model']}.yaml"
        if not model_path.exists():
            raise ConfigError(f"Model config not found for: {data['model']}")
        data["model"] = load_model_config(model_path)

    # Resolve nested memory config if it's a string (path)
    if isinstance(data.get("memory"), str):
        memory_path = Path("configs/memory") / f"{data['memory']}.yaml"
        if not memory_path.exists():
            raise ConfigError(f"Memory config not found for: {data['memory']}")
        data["memory"] = load_memory_config(memory_path)

    return ExperimentConfig(**data)

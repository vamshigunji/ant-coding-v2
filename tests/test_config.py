import pytest
from pathlib import Path
from pydantic import ValidationError
from ant_coding.core.config import (
    load_model_config, 
    load_memory_config, 
    load_experiment_config, 
    ConfigError,
    MemoryMode
)

def test_load_model_config_success():
    config = load_model_config("configs/models/claude-sonnet.yaml")
    assert config.name == "claude-sonnet"
    assert config.max_tokens == 8192

def test_load_model_config_invalid():
    # Create temporary invalid yaml
    path = Path("configs/models/invalid.yaml")
    path.write_text("name: 'invalid'") # missing litellm_model and api_key_env
    with pytest.raises(ValidationError):
        load_model_config(path)
    path.unlink()

def test_load_memory_config_success():
    config = load_memory_config("configs/memory/hybrid.yaml")
    assert config.mode == MemoryMode.HYBRID
    assert "implementation_plan" in config.shared_keys

def test_load_memory_config_invalid_mode():
    path = Path("configs/memory/invalid.yaml")
    path.write_text("mode: 'invalid_mode'")
    with pytest.raises(ValidationError):
        load_memory_config(path)
    path.unlink()

def test_load_experiment_config_success():
    config = load_experiment_config("configs/experiments/baseline-sequential.yaml")
    assert config.name == "baseline-sequential"
    assert config.model.name == "claude-sonnet"
    assert config.memory.mode == MemoryMode.SHARED

def test_load_experiment_config_missing_model():
    path = Path("configs/experiments/missing_model.yaml")
    path.write_text("""
name: "missing-model"
model: "nonexistent-model"
memory: "shared"
tasks:
  source: "test"
""")
    with pytest.raises(ConfigError) as excinfo:
        load_experiment_config(path)
    assert "nonexistent-model" in str(excinfo.value)
    path.unlink()

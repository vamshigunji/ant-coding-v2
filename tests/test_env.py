import pytest
import os
from ant_coding.core.config import get_env, ConfigError

def test_get_env_success(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test-value")
    assert get_env("TEST_KEY") == "test-value"

def test_get_env_missing_raises():
    if "NON_EXISTENT_KEY" in os.environ:
        del os.environ["NON_EXISTENT_KEY"]
    with pytest.raises(ConfigError) as excinfo:
        get_env("NON_EXISTENT_KEY")
    assert "NON_EXISTENT_KEY" in str(excinfo.value)

def test_get_env_with_default():
    assert get_env("NON_EXISTENT_KEY", default="default-value") == "default-value"

import pytest
from ant_coding.core.config import MemoryConfig, MemoryMode
from ant_coding.memory.manager import MemoryManager

def test_memory_mode_enum():
    assert MemoryMode.SHARED == "shared"
    assert MemoryMode.ISOLATED == "isolated"
    assert MemoryMode.HYBRID == "hybrid"
    assert len(list(MemoryMode)) == 3

def test_resolve_key_shared():
    config = MemoryConfig(mode=MemoryMode.SHARED)
    manager = MemoryManager(config)
    assert manager._resolve_key("agent1", "plan") == "app:plan"

def test_resolve_key_isolated():
    config = MemoryConfig(mode=MemoryMode.ISOLATED)
    manager = MemoryManager(config)
    assert manager._resolve_key("agent1", "plan") == "temp:agent1:plan"

def test_resolve_key_hybrid():
    config = MemoryConfig(mode=MemoryMode.HYBRID, shared_keys=["plan"])
    manager = MemoryManager(config)
    # Shared key
    assert manager._resolve_key("agent1", "plan") == "app:plan"
    # Private key
    assert manager._resolve_key("agent1", "scratch") == "temp:agent1:scratch"

def test_shared_visibility():
    config = MemoryConfig(mode=MemoryMode.SHARED)
    manager = MemoryManager(config)
    manager.write("planner", "plan", "step 1")
    assert manager.read("coder", "plan") == "step 1"
    
    manager.write("reviewer", "plan", "step 2")
    assert manager.read("coder", "plan") == "step 2"
    
    assert manager.read("coder", "missing") is None
    assert "plan" in manager.list_keys("coder")

def test_isolated_visibility():
    config = MemoryConfig(mode=MemoryMode.ISOLATED)
    manager = MemoryManager(config)
    manager.write("planner", "plan", "private plan")
    assert manager.read("coder", "plan") is None
    assert manager.read("planner", "plan") == "private plan"
    
    manager.write("coder", "plan", "coder plan")
    assert manager.read("coder", "plan") == "coder plan"
    assert manager.read("planner", "plan") == "private plan"

def test_hybrid_visibility():
    config = MemoryConfig(mode=MemoryMode.HYBRID, shared_keys=["plan"])
    manager = MemoryManager(config)
    
    # Shared key write
    manager.write("planner", "plan", "shared plan")
    assert manager.read("coder", "plan") == "shared plan"
    
    # Private key write
    manager.write("planner", "scratch", "private thought")
    assert manager.read("coder", "scratch") is None
    assert manager.read("planner", "scratch") == "private thought"
    
    visible = manager.list_keys("coder")
    assert "plan" in visible
    assert "scratch" not in visible

def test_access_logging():
    config = MemoryConfig(mode=MemoryMode.SHARED)
    manager = MemoryManager(config)
    manager.write("agent1", "k1", "v1")
    manager.read("agent2", "k1")
    manager.read("agent2", "k2") # not found
    
    log = manager.get_access_log()
    assert len(log) == 3
    assert log[0]["action"] == "write"
    assert log[1]["action"] == "read"
    assert log[1]["found"] is True
    assert log[2]["found"] is False

def test_snapshot_and_reset():
    config = MemoryConfig(mode=MemoryMode.SHARED)
    manager = MemoryManager(config)
    manager.write("agent1", "k1", "v1")
    
    assert manager.get_state_snapshot() == {"app:k1": "v1"}
    manager.reset()
    assert manager.get_state_snapshot() == {}
    assert manager.get_access_log() == []

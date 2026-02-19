"""
Memory management system for agents with configurable visibility modes.
"""

import logging
from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union

from ant_coding.core.config import MemoryConfig, MemoryMode, ConfigError

if TYPE_CHECKING:
    from ant_coding.observability.event_logger import EventLogger

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages state for agents with support for Shared, Isolated, and Hybrid modes.
    Enforces visibility rules via key-prefix routing.
    """

    def __init__(
        self,
        config: MemoryConfig,
        event_logger: Optional["EventLogger"] = None,
        experiment_id: str = "",
        task_id: str = "",
    ):
        self.mode = config.mode
        self.shared_keys = config.shared_keys or []
        self._state: Dict[str, Any] = {}
        self._access_log: List[Dict[str, Any]] = []
        self._event_logger = event_logger
        self._experiment_id = experiment_id
        self._task_id = task_id

    def _resolve_key(self, agent_id: str, key: str) -> str:
        """
        Resolve a user-provided key to a prefixed state key based on memory mode.
        
        Args:
            agent_id: The ID of the agent performing the operation.
            key: The key provided by the agent.
            
        Returns:
            A string key with the appropriate prefix.
        """
        if self.mode == MemoryMode.SHARED:
            return f"app:{key}"
        
        if self.mode == MemoryMode.ISOLATED:
            return f"temp:{agent_id}:{key}"
        
        if self.mode == MemoryMode.HYBRID:
            if key in self.shared_keys:
                return f"app:{key}"
            else:
                return f"temp:{agent_id}:{key}"
        
        raise ValueError(f"Unsupported memory mode: {self.mode}")

    def write(self, agent_id: str, key: str, value: Any):
        """Write a value to memory."""
        resolved_key = self._resolve_key(agent_id, key)
        self._state[resolved_key] = value
        self._log_access("write", agent_id, key, resolved_key)
        self._log_memory_event("write", agent_id, key, resolved_key, value=value)

    def read(self, agent_id: str, key: str) -> Any:
        """Read a value from memory."""
        resolved_key = self._resolve_key(agent_id, key)
        value = self._state.get(resolved_key)
        found = resolved_key in self._state
        self._log_access("read", agent_id, key, resolved_key, found=found)
        self._log_memory_event("read", agent_id, key, resolved_key, found=found)
        return value

    def list_keys(self, agent_id: str) -> List[str]:
        """List all keys visible to the given agent."""
        visible_keys = []
        for resolved_key in self._state.keys():
            if resolved_key.startswith("app:"):
                # Shared/App keys are visible to everyone
                visible_keys.append(resolved_key[4:])
            elif resolved_key.startswith(f"temp:{agent_id}:"):
                # Private keys only visible to owner
                prefix_len = len(f"temp:{agent_id}:")
                visible_keys.append(resolved_key[prefix_len:])
        return sorted(list(set(visible_keys)))

    def _log_access(self, action: str, agent: str, key: str, resolved_key: str, found: Optional[bool] = None):
        """Record memory access operation."""
        entry = {
            "action": action,
            "agent": agent,
            "key": key,
            "resolved_key": resolved_key
        }
        if found is not None:
            entry["found"] = found
        self._access_log.append(entry)

    def get_access_log(self) -> List[Dict[str, Any]]:
        """Return the history of memory operations."""
        return self._access_log

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a copy of the full internal state."""
        return self._state.copy()

    def reset(self):
        """Clear all state and logs."""
        self._state.clear()
        self._access_log.clear()

    def set_context(self, task_id: str, experiment_id: str) -> None:
        """Set the current task/experiment context for event logging."""
        self._task_id = task_id
        self._experiment_id = experiment_id

    def _log_memory_event(
        self,
        action: str,
        agent_id: str,
        key: str,
        resolved_key: str,
        value: Any = None,
        found: Optional[bool] = None,
    ) -> None:
        """Log a MEMORY_READ or MEMORY_WRITE event if event_logger is configured."""
        if self._event_logger is None:
            return

        from ant_coding.observability.event_logger import Event, EventType

        event_type = EventType.MEMORY_WRITE if action == "write" else EventType.MEMORY_READ
        payload: Dict[str, Any] = {
            "agent": agent_id,
            "key": key,
            "resolved_key": resolved_key,
        }
        if action == "write" and value is not None:
            payload["value_size"] = len(str(value))
        if found is not None:
            payload["found"] = found

        self._event_logger.log(Event(
            type=event_type,
            task_id=self._task_id,
            experiment_id=self._experiment_id,
            agent_id=agent_id,
            payload=payload,
        ))

    @classmethod
    def from_config(cls, config: Union[MemoryConfig, Dict[str, Any]]) -> "MemoryManager":
        """Factory method to create a MemoryManager from config."""
        if isinstance(config, dict):
            try:
                config = MemoryConfig(**config)
            except Exception as e:
                raise ConfigError(f"Invalid memory config: {e}")
        return cls(config)

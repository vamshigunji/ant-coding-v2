"""
Event logging system for experiment observability.

Provides an append-only JSONL event log that records every LLM call,
tool call, memory access, and task lifecycle event during experiments.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """All observable event types in the experiment lifecycle."""

    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    ERROR = "error"
    TASK_START = "task_start"
    TASK_END = "task_end"


@dataclass
class Event:
    """A single observable event during experiment execution."""

    type: EventType
    task_id: str
    experiment_id: str
    agent_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventLogger:
    """
    Append-only event logger that writes to JSONL files.

    Each experiment gets its own events.jsonl file. Events are written
    immediately on log() and can be queried in-memory via get_events()
    and get_token_breakdown().
    """

    def __init__(
        self,
        experiment_id: str,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the EventLogger.

        Args:
            experiment_id: Unique identifier for this experiment run.
            output_dir: Base directory for output files. If provided,
                events are written to {output_dir}/{experiment_id}/events.jsonl.
                If None, events are only stored in memory.
        """
        self.experiment_id = experiment_id
        self._events: List[Event] = []
        self._output_path: Optional[Path] = None

        if output_dir:
            dir_path = Path(output_dir) / experiment_id
            dir_path.mkdir(parents=True, exist_ok=True)
            self._output_path = dir_path / "events.jsonl"

    def _serialize_event(self, event: Event) -> Dict[str, Any]:
        """Convert an Event to a JSON-serializable dict."""
        data = asdict(event)
        # Convert enum to its value string
        data["type"] = event.type.value
        # Convert datetime to ISO format string
        data["timestamp"] = event.timestamp.isoformat()
        return data

    def log(self, event: Event) -> None:
        """
        Log an event. Appends to in-memory list and JSONL file.

        Args:
            event: The event to log.
        """
        self._events.append(event)

        if self._output_path:
            line = json.dumps(self._serialize_event(event), default=str)
            with open(self._output_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def get_events(
        self,
        agent_name: Optional[str] = None,
        event_type: Optional[EventType] = None,
        task_id: Optional[str] = None,
    ) -> List[Event]:
        """
        Retrieve logged events with optional filtering.

        Args:
            agent_name: Filter by agent_id.
            event_type: Filter by EventType.
            task_id: Filter by task_id.

        Returns:
            List of matching events in chronological order.
        """
        events = self._events

        if agent_name is not None:
            events = [e for e in events if e.agent_id == agent_name]

        if event_type is not None:
            events = [e for e in events if e.type == event_type]

        if task_id is not None:
            events = [e for e in events if e.task_id == task_id]

        return events

    def get_token_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate per-agent token usage from LLM_CALL events.

        Returns:
            Dict mapping agent_id to token counts:
            {
                "planner": {"prompt": N, "completion": N, "total": N},
                "coder": {"prompt": N, "completion": N, "total": N},
            }
        """
        breakdown: Dict[str, Dict[str, int]] = {}

        for event in self._events:
            if event.type != EventType.LLM_CALL:
                continue

            agent = event.agent_id or "unknown"
            if agent not in breakdown:
                breakdown[agent] = {"prompt": 0, "completion": 0, "total": 0}

            payload = event.payload
            breakdown[agent]["prompt"] += payload.get("prompt_tokens", 0)
            breakdown[agent]["completion"] += payload.get("completion_tokens", 0)
            breakdown[agent]["total"] += payload.get("total_tokens", 0)

        return breakdown

    def clear(self) -> None:
        """Clear all in-memory events. Does not delete the JSONL file."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        """Return the number of logged events."""
        return len(self._events)

    @property
    def output_path(self) -> Optional[Path]:
        """Return the path to the JSONL output file, if configured."""
        return self._output_path

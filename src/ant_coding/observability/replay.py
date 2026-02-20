"""
Session replay for experiment events.

Loads events from a JSONL file and provides stepping, state reconstruction,
and cumulative token curve for post-hoc analysis.

Reference: Sprint-6-Epic-1.md (S6-E1-S03)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ant_coding.observability.event_logger import Event, EventType


class SessionReplay:
    """
    Replay an experiment session from an events.jsonl file.

    Supports stepping through events, reconstructing memory state
    at any point, and computing cumulative token curves.
    """

    def __init__(self, events_path: str):
        """
        Load events from a JSONL file.

        Args:
            events_path: Path to an events.jsonl file from a completed experiment.

        Raises:
            FileNotFoundError: If the events file doesn't exist.
        """
        self._events: List[Event] = []
        self._cursor: int = 0

        path = Path(events_path)
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = self._deserialize_event(json.loads(line))
                self._events.append(event)

    @property
    def total_events(self) -> int:
        """Return the total number of events loaded."""
        return len(self._events)

    @property
    def cursor(self) -> int:
        """Return the current cursor position."""
        return self._cursor

    def reset(self) -> None:
        """Reset the cursor to the beginning."""
        self._cursor = 0

    def step(self, count: int = 1) -> List[Event]:
        """
        Return the next `count` events from the current cursor position.

        Advances the cursor by the number of events returned.

        Args:
            count: Number of events to return.

        Returns:
            List of events (may be fewer than `count` if near the end).
        """
        start = self._cursor
        end = min(start + count, len(self._events))
        events = self._events[start:end]
        self._cursor = end
        return events

    def state_at(self, event_index: int) -> Dict[str, Any]:
        """
        Reconstruct the memory state at a given event index.

        Replays all MEMORY_WRITE events up to (and including) event_index
        to build the state dictionary.

        Args:
            event_index: The event index up to which to reconstruct state.

        Returns:
            Dict of all memory keys and their values at that point.
        """
        state: Dict[str, Any] = {}
        end = min(event_index + 1, len(self._events))

        for event in self._events[:end]:
            if event.type == EventType.MEMORY_WRITE:
                key = event.payload.get("key", "")
                value = event.payload.get("value")
                if key:
                    state[key] = value

        return state

    def token_curve(self) -> List[Tuple[int, int]]:
        """
        Compute the cumulative token curve.

        Returns a list of (event_index, cumulative_tokens) tuples,
        one entry per LLM_CALL event.

        Returns:
            List of (event_index, cumulative_tokens) tuples.
        """
        curve: List[Tuple[int, int]] = []
        cumulative = 0

        for i, event in enumerate(self._events):
            if event.type == EventType.LLM_CALL:
                tokens = event.payload.get("total_tokens", 0)
                cumulative += tokens
                curve.append((i, cumulative))

        return curve

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        task_id: Optional[str] = None,
    ) -> List[Event]:
        """
        Get all events with optional filtering.

        Args:
            event_type: Filter by EventType.
            task_id: Filter by task_id.

        Returns:
            Filtered list of events.
        """
        events = self._events

        if event_type is not None:
            events = [e for e in events if e.type == event_type]

        if task_id is not None:
            events = [e for e in events if e.task_id == task_id]

        return events

    @staticmethod
    def _deserialize_event(data: Dict[str, Any]) -> Event:
        """
        Convert a JSON dict back into an Event object.

        Args:
            data: Dict from JSON parsing.

        Returns:
            Event instance.
        """
        event_type = EventType(data["type"])
        timestamp = datetime.fromisoformat(data["timestamp"])

        return Event(
            type=event_type,
            task_id=data["task_id"],
            experiment_id=data["experiment_id"],
            agent_id=data.get("agent_id"),
            payload=data.get("payload", {}),
            timestamp=timestamp,
        )

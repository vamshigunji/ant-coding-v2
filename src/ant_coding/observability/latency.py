"""
Latency tracking utilities for experiment observability.

Provides helpers to measure and extract timing data from event logs,
including per-task wall time, LLM call latency, and tool call latency.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from ant_coding.observability.event_logger import Event, EventType


def get_task_wall_time(events: List[Event], task_id: str) -> Optional[float]:
    """
    Calculate wall time for a task from TASK_START to TASK_END events.

    Args:
        events: List of events to search.
        task_id: The task ID to calculate wall time for.

    Returns:
        Wall time in seconds, or None if start/end events not found.
    """
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    for event in events:
        if event.task_id != task_id:
            continue
        if event.type == EventType.TASK_START and start_time is None:
            start_time = event.timestamp.timestamp()
        elif event.type == EventType.TASK_END:
            end_time = event.timestamp.timestamp()

    if start_time is not None and end_time is not None:
        return end_time - start_time
    return None


def get_llm_latencies(events: List[Event], task_id: Optional[str] = None) -> List[float]:
    """
    Extract LLM call latencies from events.

    Args:
        events: List of events to search.
        task_id: Optional task ID filter.

    Returns:
        List of duration_ms values from LLM_CALL events.
    """
    latencies = []
    for event in events:
        if event.type != EventType.LLM_CALL:
            continue
        if task_id is not None and event.task_id != task_id:
            continue
        duration = event.payload.get("duration_ms")
        if duration is not None:
            latencies.append(float(duration))
    return latencies


def get_tool_latencies(events: List[Event], task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract tool call latencies from events.

    Args:
        events: List of events to search.
        task_id: Optional task ID filter.

    Returns:
        List of dicts with tool_name, method, duration_ms, and success.
    """
    results = []
    for event in events:
        if event.type != EventType.TOOL_CALL:
            continue
        if task_id is not None and event.task_id != task_id:
            continue
        payload = event.payload
        results.append({
            "tool_name": payload.get("tool_name", ""),
            "method": payload.get("method", ""),
            "duration_ms": payload.get("duration_ms", 0.0),
            "success": payload.get("success", False),
        })
    return results


def get_latency_summary(events: List[Event], task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute a latency summary across LLM and tool calls.

    Args:
        events: List of events to analyze.
        task_id: Optional task ID filter.

    Returns:
        Dict with total_llm_ms, avg_llm_ms, total_tool_ms, avg_tool_ms,
        llm_call_count, and tool_call_count.
    """
    llm_latencies = get_llm_latencies(events, task_id)
    tool_entries = get_tool_latencies(events, task_id)
    tool_latencies = [t["duration_ms"] for t in tool_entries]

    return {
        "llm_call_count": len(llm_latencies),
        "total_llm_ms": sum(llm_latencies),
        "avg_llm_ms": sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0,
        "tool_call_count": len(tool_latencies),
        "total_tool_ms": sum(tool_latencies),
        "avg_tool_ms": sum(tool_latencies) / len(tool_latencies) if tool_latencies else 0.0,
    }


@contextmanager
def measure_duration_ms() -> Generator[Dict[str, float], None, None]:
    """
    Context manager to measure elapsed time in milliseconds.

    Usage:
        with measure_duration_ms() as timing:
            # do work
        print(timing["duration_ms"])
    """
    result: Dict[str, float] = {"duration_ms": 0.0}
    start = time.time()
    try:
        yield result
    finally:
        result["duration_ms"] = (time.time() - start) * 1000

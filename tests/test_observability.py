"""
Comprehensive tests for the observability layer: EventLogger, latency tracking.
"""

import json
import time
from datetime import datetime, timedelta


from ant_coding.observability.event_logger import Event, EventLogger, EventType
from ant_coding.observability.latency import (
    get_latency_summary,
    get_llm_latencies,
    get_task_wall_time,
    get_tool_latencies,
    measure_duration_ms,
)


# ── Helpers ──


def _make_event(
    event_type: EventType,
    task_id: str = "task-1",
    experiment_id: str = "exp-1",
    agent_id: str = None,
    payload: dict = None,
    timestamp: datetime = None,
) -> Event:
    return Event(
        type=event_type,
        task_id=task_id,
        experiment_id=experiment_id,
        agent_id=agent_id,
        payload=payload or {},
        timestamp=timestamp or datetime.now(),
    )


def _make_llm_event(
    agent_id: str,
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
    duration_ms: float = 200.0,
    task_id: str = "task-1",
) -> Event:
    return _make_event(
        EventType.LLM_CALL,
        task_id=task_id,
        agent_id=agent_id,
        payload={
            "model": "gpt-4",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": duration_ms,
        },
    )


# ── EventLogger JSONL Writing Tests ──


def test_event_logger_writes_jsonl(tmp_path):
    """Events are written to a JSONL file."""
    logger = EventLogger("exp-001", output_dir=str(tmp_path))
    event = _make_event(EventType.TASK_START)
    logger.log(event)

    jsonl_path = tmp_path / "exp-001" / "events.jsonl"
    assert jsonl_path.exists()

    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["type"] == "task_start"
    assert data["task_id"] == "task-1"


def test_event_logger_appends_multiple_events(tmp_path):
    """Multiple events are appended line by line."""
    logger = EventLogger("exp-002", output_dir=str(tmp_path))
    for i in range(5):
        logger.log(_make_event(EventType.LLM_CALL, agent_id=f"agent-{i}"))

    jsonl_path = tmp_path / "exp-002" / "events.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 5
    # Each line is valid JSON
    for line in lines:
        data = json.loads(line)
        assert "type" in data
        assert "timestamp" in data


def test_event_logger_jsonl_format_valid(tmp_path):
    """Every line in the JSONL file is a valid JSON object with required fields."""
    logger = EventLogger("exp-format", output_dir=str(tmp_path))
    logger.log(_make_event(EventType.TASK_START))
    logger.log(_make_llm_event("planner"))
    logger.log(_make_event(EventType.MEMORY_WRITE, agent_id="planner"))
    logger.log(_make_event(EventType.TOOL_CALL, agent_id="coder"))
    logger.log(_make_event(EventType.TASK_END))

    jsonl_path = tmp_path / "exp-format" / "events.jsonl"
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 5

    required_fields = {"type", "task_id", "experiment_id", "timestamp"}
    for line in lines:
        data = json.loads(line)
        assert required_fields.issubset(data.keys())


def test_event_logger_memory_only():
    """EventLogger works without output_dir (memory-only mode)."""
    logger = EventLogger("mem-only")
    logger.log(_make_event(EventType.TASK_START))
    logger.log(_make_event(EventType.TASK_END))

    assert logger.event_count == 2
    assert logger.output_path is None


# ── Event Filtering Tests ──


def test_get_events_all():
    """get_events() returns all events in chronological order."""
    logger = EventLogger("exp-all")
    for i in range(10):
        logger.log(_make_event(EventType.LLM_CALL, agent_id=f"a{i}"))

    events = logger.get_events()
    assert len(events) == 10


def test_get_events_filter_by_agent():
    """get_events(agent_name=...) returns only that agent's events."""
    logger = EventLogger("exp-filter")
    logger.log(_make_event(EventType.LLM_CALL, agent_id="planner"))
    logger.log(_make_event(EventType.LLM_CALL, agent_id="coder"))
    logger.log(_make_event(EventType.LLM_CALL, agent_id="planner"))

    planner_events = logger.get_events(agent_name="planner")
    assert len(planner_events) == 2
    assert all(e.agent_id == "planner" for e in planner_events)


def test_get_events_filter_by_type():
    """get_events(event_type=...) returns only matching event types."""
    logger = EventLogger("exp-type")
    logger.log(_make_event(EventType.TASK_START))
    logger.log(_make_event(EventType.LLM_CALL, agent_id="a"))
    logger.log(_make_event(EventType.MEMORY_WRITE, agent_id="a"))
    logger.log(_make_event(EventType.TASK_END))

    llm_events = logger.get_events(event_type=EventType.LLM_CALL)
    assert len(llm_events) == 1
    assert llm_events[0].type == EventType.LLM_CALL


def test_get_events_filter_by_task():
    """get_events(task_id=...) returns only events for that task."""
    logger = EventLogger("exp-task")
    logger.log(_make_event(EventType.TASK_START, task_id="t1"))
    logger.log(_make_event(EventType.TASK_START, task_id="t2"))
    logger.log(_make_event(EventType.TASK_END, task_id="t1"))

    t1_events = logger.get_events(task_id="t1")
    assert len(t1_events) == 2
    assert all(e.task_id == "t1" for e in t1_events)


# ── Token Breakdown Tests ──


def test_token_breakdown_single_agent():
    """Token breakdown for a single agent."""
    logger = EventLogger("exp-tokens")
    logger.log(_make_llm_event("planner", prompt_tokens=100, completion_tokens=200))
    logger.log(_make_llm_event("planner", prompt_tokens=50, completion_tokens=75))

    breakdown = logger.get_token_breakdown()
    assert "planner" in breakdown
    assert breakdown["planner"]["prompt"] == 150
    assert breakdown["planner"]["completion"] == 275
    assert breakdown["planner"]["total"] == 425


def test_token_breakdown_multiple_agents():
    """Token breakdown separates agents correctly."""
    logger = EventLogger("exp-multi")
    logger.log(_make_llm_event("planner", prompt_tokens=100, completion_tokens=200))
    logger.log(_make_llm_event("coder", prompt_tokens=80, completion_tokens=300))
    logger.log(_make_llm_event("planner", prompt_tokens=50, completion_tokens=100))

    breakdown = logger.get_token_breakdown()
    assert len(breakdown) == 2
    assert breakdown["planner"]["total"] == 450
    assert breakdown["coder"]["total"] == 380


def test_token_breakdown_ignores_non_llm_events():
    """Token breakdown only counts LLM_CALL events."""
    logger = EventLogger("exp-ignore")
    logger.log(_make_event(EventType.TASK_START))
    logger.log(_make_event(EventType.MEMORY_WRITE, agent_id="a"))
    logger.log(_make_llm_event("a", prompt_tokens=10, completion_tokens=20))
    logger.log(_make_event(EventType.TOOL_CALL, agent_id="a"))

    breakdown = logger.get_token_breakdown()
    assert len(breakdown) == 1
    assert breakdown["a"]["total"] == 30


# ── Latency Tracking Tests ──


def test_task_wall_time():
    """get_task_wall_time calculates time between TASK_START and TASK_END."""
    now = datetime.now()
    events = [
        _make_event(EventType.TASK_START, task_id="t1", timestamp=now),
        _make_event(EventType.LLM_CALL, task_id="t1", timestamp=now + timedelta(seconds=1)),
        _make_event(EventType.TASK_END, task_id="t1", timestamp=now + timedelta(seconds=5)),
    ]

    wall_time = get_task_wall_time(events, "t1")
    assert wall_time is not None
    assert abs(wall_time - 5.0) < 0.01


def test_task_wall_time_missing_events():
    """get_task_wall_time returns None when events are missing."""
    events = [_make_event(EventType.TASK_START, task_id="t1")]
    assert get_task_wall_time(events, "t1") is None
    assert get_task_wall_time(events, "nonexistent") is None


def test_get_llm_latencies():
    """get_llm_latencies extracts duration_ms from LLM_CALL events."""
    events = [
        _make_llm_event("a", duration_ms=100.5),
        _make_llm_event("b", duration_ms=250.3),
        _make_event(EventType.TASK_START),  # not LLM_CALL
    ]

    latencies = get_llm_latencies(events)
    assert len(latencies) == 2
    assert latencies[0] == 100.5
    assert latencies[1] == 250.3


def test_get_tool_latencies():
    """get_tool_latencies extracts timing from TOOL_CALL events."""
    events = [
        _make_event(
            EventType.TOOL_CALL,
            agent_id="coder",
            payload={
                "tool_name": "file_ops",
                "method": "write_file",
                "duration_ms": 15.2,
                "success": True,
            },
        ),
        _make_event(EventType.LLM_CALL),  # not TOOL_CALL
    ]

    tool_lats = get_tool_latencies(events)
    assert len(tool_lats) == 1
    assert tool_lats[0]["tool_name"] == "file_ops"
    assert tool_lats[0]["duration_ms"] == 15.2
    assert tool_lats[0]["success"] is True


def test_latency_summary():
    """get_latency_summary aggregates LLM and tool latencies."""
    events = [
        _make_llm_event("a", duration_ms=100.0),
        _make_llm_event("a", duration_ms=200.0),
        _make_event(
            EventType.TOOL_CALL,
            payload={"duration_ms": 50.0, "tool_name": "x", "method": "y", "success": True},
        ),
    ]

    summary = get_latency_summary(events)
    assert summary["llm_call_count"] == 2
    assert summary["total_llm_ms"] == 300.0
    assert summary["avg_llm_ms"] == 150.0
    assert summary["tool_call_count"] == 1
    assert summary["total_tool_ms"] == 50.0


def test_measure_duration_ms():
    """measure_duration_ms context manager records elapsed time."""
    with measure_duration_ms() as timing:
        time.sleep(0.05)  # 50ms

    assert timing["duration_ms"] >= 40  # Allow some tolerance
    assert timing["duration_ms"] < 500  # But not absurdly high


# ── Agent Timeline Ordering ──


def test_events_chronological_order():
    """Events are stored and returned in chronological order."""
    logger = EventLogger("exp-order")
    now = datetime.now()

    # Log in order
    for i in range(5):
        logger.log(_make_event(
            EventType.LLM_CALL,
            agent_id=f"agent-{i}",
            timestamp=now + timedelta(milliseconds=i * 100),
        ))

    events = logger.get_events()
    for i in range(len(events) - 1):
        assert events[i].timestamp <= events[i + 1].timestamp


def test_event_logger_clear():
    """clear() removes all in-memory events."""
    logger = EventLogger("exp-clear")
    logger.log(_make_event(EventType.TASK_START))
    logger.log(_make_event(EventType.TASK_END))
    assert logger.event_count == 2

    logger.clear()
    assert logger.event_count == 0
    assert logger.get_events() == []

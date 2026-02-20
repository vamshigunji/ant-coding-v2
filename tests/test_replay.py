"""
Tests for session replay from JSONL event logs.
"""

import tempfile

import pytest

from ant_coding.observability.event_logger import Event, EventLogger, EventType
from ant_coding.observability.replay import SessionReplay


def _create_events_file(events):
    """Helper to write events to a temp JSONL file and return path."""
    tmpdir = tempfile.mkdtemp()
    logger = EventLogger("test-exp", output_dir=tmpdir)
    for event in events:
        logger.log(event)
    return str(logger.output_path)


def _make_event(event_type, task_id="t1", agent_id=None, payload=None):
    return Event(
        type=event_type,
        task_id=task_id,
        experiment_id="test-exp",
        agent_id=agent_id,
        payload=payload or {},
    )


# ── Loading ──


def test_replay_load():
    """SessionReplay loads events from JSONL."""
    events = [
        _make_event(EventType.TASK_START),
        _make_event(EventType.LLM_CALL, payload={"total_tokens": 100}),
        _make_event(EventType.TASK_END),
    ]
    path = _create_events_file(events)
    replay = SessionReplay(path)
    assert replay.total_events == 3


def test_replay_file_not_found():
    """FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        SessionReplay("/nonexistent/events.jsonl")


# ── Stepping ──


def test_replay_step():
    """step(5) returns next 5 events and advances cursor."""
    events = [_make_event(EventType.LLM_CALL) for _ in range(10)]
    path = _create_events_file(events)
    replay = SessionReplay(path)

    batch = replay.step(5)
    assert len(batch) == 5
    assert replay.cursor == 5

    batch2 = replay.step(5)
    assert len(batch2) == 5
    assert replay.cursor == 10


def test_replay_step_past_end():
    """step() past end returns remaining events."""
    events = [_make_event(EventType.LLM_CALL) for _ in range(3)]
    path = _create_events_file(events)
    replay = SessionReplay(path)

    batch = replay.step(10)
    assert len(batch) == 3
    assert replay.cursor == 3


def test_replay_reset():
    """reset() sets cursor back to 0."""
    events = [_make_event(EventType.LLM_CALL) for _ in range(5)]
    path = _create_events_file(events)
    replay = SessionReplay(path)

    replay.step(3)
    assert replay.cursor == 3
    replay.reset()
    assert replay.cursor == 0


# ── State Reconstruction ──


def test_state_at_reconstructs_memory():
    """state_at() reconstructs memory from MEMORY_WRITE events."""
    events = [
        _make_event(EventType.MEMORY_WRITE, payload={"key": "plan", "value": "v1"}),
        _make_event(EventType.LLM_CALL, payload={"total_tokens": 50}),
        _make_event(EventType.MEMORY_WRITE, payload={"key": "result", "value": "ok"}),
        _make_event(EventType.MEMORY_WRITE, payload={"key": "plan", "value": "v2"}),
    ]
    path = _create_events_file(events)
    replay = SessionReplay(path)

    # At index 1: only first MEMORY_WRITE happened
    state = replay.state_at(1)
    assert state["plan"] == "v1"
    assert "result" not in state

    # At index 3: all writes happened, plan overwritten
    state = replay.state_at(3)
    assert state["plan"] == "v2"
    assert state["result"] == "ok"


def test_state_at_empty():
    """state_at() with no MEMORY_WRITE returns empty dict."""
    events = [_make_event(EventType.LLM_CALL)]
    path = _create_events_file(events)
    replay = SessionReplay(path)
    assert replay.state_at(0) == {}


# ── Token Curve ──


def test_token_curve():
    """token_curve() returns cumulative tokens at each LLM_CALL."""
    events = [
        _make_event(EventType.TASK_START),
        _make_event(EventType.LLM_CALL, payload={"total_tokens": 100}),
        _make_event(EventType.TOOL_CALL),
        _make_event(EventType.LLM_CALL, payload={"total_tokens": 200}),
        _make_event(EventType.LLM_CALL, payload={"total_tokens": 50}),
    ]
    path = _create_events_file(events)
    replay = SessionReplay(path)

    curve = replay.token_curve()
    assert len(curve) == 3
    assert curve[0] == (1, 100)      # event index 1, 100 cumulative
    assert curve[1] == (3, 300)      # event index 3, 300 cumulative
    assert curve[2] == (4, 350)      # event index 4, 350 cumulative


def test_token_curve_no_llm_calls():
    """token_curve() returns empty list when no LLM_CALL events."""
    events = [_make_event(EventType.TASK_START), _make_event(EventType.TASK_END)]
    path = _create_events_file(events)
    replay = SessionReplay(path)
    assert replay.token_curve() == []

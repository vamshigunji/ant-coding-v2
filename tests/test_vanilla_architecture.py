"""
Tests for the vanilla_architecture roast battle system.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ant_coding.core.config import MemoryConfig, MemoryMode, ModelConfig
from ant_coding.memory.manager import MemoryManager
from ant_coding.vanilla_architecture.agent import CharacterAgent, CharacterConfig
from ant_coding.vanilla_architecture.orchestrator import RoastBattleOrchestrator
from ant_coding.vanilla_architecture.experiment_runner import ExperimentRunner


@pytest.fixture
def memory():
    return MemoryManager(MemoryConfig(mode=MemoryMode.SHARED))


@pytest.fixture
def model_config():
    return ModelConfig(
        name="test-model",
        litellm_model="openai/gpt-test",
        api_key_env="TEST_API_KEY",
        max_tokens=256,
        temperature=0.9,
    )


@pytest.fixture
def shakespeare():
    return CharacterConfig(
        name="Shakespeare",
        persona="A dramatic playwright",
        style="flowery and dramatic",
    )


@pytest.fixture
def robot():
    return CharacterConfig(
        name="Robot",
        persona="A logical AI",
        style="deadpan and clinical",
    )


def _make_mock_model(response_text: str):
    """Create a mock ModelProvider that returns a fixed response."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.name = "mock-model"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text

    model.complete = AsyncMock(return_value=mock_response)
    model.get_usage = MagicMock(return_value={
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "total_cost_usd": 0.001,
    })
    return model


class TestCharacterAgent:
    def test_agent_id_from_name(self, shakespeare, memory):
        model = _make_mock_model("To roast or not to roast!")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        assert agent.agent_id == "shakespeare"

    def test_agent_id_spaces(self, memory):
        char = CharacterConfig(name="Iron Man", persona="test", style="test")
        model = _make_mock_model("test")
        agent = CharacterAgent(character=char, model=model, memory=memory)
        assert agent.agent_id == "iron_man"

    def test_system_prompt_contains_character(self, shakespeare, memory):
        model = _make_mock_model("test")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        prompt = agent._build_system_prompt()
        assert "Shakespeare" in prompt
        assert "dramatic" in prompt.lower()

    def test_build_messages_empty_conversation(self, shakespeare, memory):
        model = _make_mock_model("test")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        messages = agent._build_messages([])
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "opening shot" in messages[1]["content"].lower()

    def test_build_messages_with_history(self, shakespeare, memory):
        model = _make_mock_model("test")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        conversation = [
            {"speaker": "Shakespeare", "text": "Thou art a fool!"},
            {"speaker": "Robot", "text": "Calculating insult quality... low."},
        ]
        messages = agent._build_messages(conversation)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"  # Shakespeare's own line
        assert messages[2]["role"] == "user"  # Robot's line

    @pytest.mark.asyncio
    async def test_respond(self, shakespeare, memory):
        model = _make_mock_model("Thou art a bucket of bolts!")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        text = await agent.respond([])
        assert text == "Thou art a bucket of bolts!"
        model.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_respond_writes_to_memory(self, shakespeare, memory):
        model = _make_mock_model("My response")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        await agent.respond([])
        # In shared mode, key is app:turn_1
        assert memory.read(agent.agent_id, "turn_1") == "My response"

    @pytest.mark.asyncio
    async def test_events_logged(self, shakespeare, memory):
        model = _make_mock_model("test")
        agent = CharacterAgent(character=shakespeare, model=model, memory=memory)
        await agent.respond([])
        event_types = [e.type.value for e in agent.events]
        assert "agent_start" in event_types
        assert "llm_call" in event_types
        assert "agent_end" in event_types


class TestRoastBattleOrchestrator:
    @pytest.mark.asyncio
    async def test_run_produces_correct_turn_count(self, shakespeare, robot, memory):
        model_a = _make_mock_model("Shakespeare's roast")
        model_b = _make_mock_model("Robot's comeback")

        agent_a = CharacterAgent(character=shakespeare, model=model_a, memory=memory)
        agent_b = CharacterAgent(character=robot, model=model_b, memory=memory)

        orchestrator = RoastBattleOrchestrator(agent_a, agent_b)
        result = await orchestrator.run(rounds=2)

        assert len(result.conversation) == 4  # 2 rounds x 2 agents
        assert result.rounds == 2

    @pytest.mark.asyncio
    async def test_alternating_speakers(self, shakespeare, robot, memory):
        model_a = _make_mock_model("A speaks")
        model_b = _make_mock_model("B speaks")

        agent_a = CharacterAgent(character=shakespeare, model=model_a, memory=memory)
        agent_b = CharacterAgent(character=robot, model=model_b, memory=memory)

        orchestrator = RoastBattleOrchestrator(agent_a, agent_b)
        result = await orchestrator.run(rounds=2)

        speakers = [t["speaker"] for t in result.conversation]
        assert speakers == ["Shakespeare", "Robot", "Shakespeare", "Robot"]

    @pytest.mark.asyncio
    async def test_usage_tracked_per_agent(self, shakespeare, robot, memory):
        model_a = _make_mock_model("A speaks")
        model_b = _make_mock_model("B speaks")

        agent_a = CharacterAgent(character=shakespeare, model=model_a, memory=memory)
        agent_b = CharacterAgent(character=robot, model=model_b, memory=memory)

        orchestrator = RoastBattleOrchestrator(agent_a, agent_b)
        result = await orchestrator.run(rounds=1)

        assert "shakespeare" in result.usage
        assert "robot" in result.usage
        assert result.usage["shakespeare"]["total_tokens"] > 0
        assert result.usage["robot"]["total_tokens"] > 0


class TestExperimentRunner:
    @pytest.mark.asyncio
    async def test_run_produces_metrics(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-run",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=2,
                output_dir=str(tmp_path),
            )
            metrics = await runner.run()

        assert metrics.experiment_id == "test-run"
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.pass_rate == 1.0
        assert metrics.total_tokens == 300  # 150 per agent
        assert metrics.total_cost == 0.002  # 0.001 per agent
        assert metrics.avg_duration > 0
        assert metrics.metadata["rounds"] == 2
        assert metrics.metadata["total_turns"] == 4

    @pytest.mark.asyncio
    async def test_persists_events_jsonl(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-events",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=1,
                output_dir=str(tmp_path),
            )
            await runner.run()

        events_path = tmp_path / "test-events" / "events.jsonl"
        assert events_path.exists()

        events = []
        with open(events_path) as f:
            for line in f:
                events.append(json.loads(line))

        # task_start + (agent_start, llm_call, agent_end)*2 agents*1 round + task_end
        # = 1 + 6 + 1 = 8
        assert len(events) == 8
        assert events[0]["type"] == "task_start"
        assert events[-1]["type"] == "task_end"

        # All events have timestamps and experiment_id
        for e in events:
            assert e["experiment_id"] == "test-events"
            assert "timestamp" in e

    @pytest.mark.asyncio
    async def test_persists_metrics_json(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-metrics",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=1,
                output_dir=str(tmp_path),
            )
            await runner.run()

        metrics_path = tmp_path / "test-metrics" / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            data = json.load(f)

        assert data["experiment_id"] == "test-metrics"
        assert data["total_tokens"] == 300
        assert data["pass_rate"] == 1.0
        assert "per_agent_usage" in data["metadata"]
        assert "shakespeare" in data["metadata"]["per_agent_usage"]
        assert "robot" in data["metadata"]["per_agent_usage"]

    @pytest.mark.asyncio
    async def test_persists_memory_log(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-memory",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=1,
                output_dir=str(tmp_path),
            )
            await runner.run()

        memory_path = tmp_path / "test-memory" / "memory_log.json"
        assert memory_path.exists()

        with open(memory_path) as f:
            data = json.load(f)

        assert "access_log" in data
        assert "final_state" in data
        assert len(data["access_log"]) > 0
        assert len(data["final_state"]) > 0

    @pytest.mark.asyncio
    async def test_persists_report_md(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-report",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=1,
                output_dir=str(tmp_path),
            )
            await runner.run()

        report_path = tmp_path / "test-report" / "report.md"
        assert report_path.exists()

        content = report_path.read_text()
        assert "# Experiment Report: test-report" in content
        assert "## Metrics" in content
        assert "## Per-Agent Usage" in content
        assert "## Transcript" in content
        assert "## Memory State" in content
        assert "shakespeare" in content
        assert "robot" in content

    @pytest.mark.asyncio
    async def test_output_directory_created(self, shakespeare, robot, model_config, tmp_path):
        with patch("ant_coding.vanilla_architecture.experiment_runner.ModelProvider") as MockMP:
            MockMP.side_effect = lambda cfg: _make_mock_model("Mock response")

            runner = ExperimentRunner(
                experiment_id="test-dirs",
                model_config=model_config,
                char_a=shakespeare,
                char_b=robot,
                rounds=1,
                output_dir=str(tmp_path),
            )
            await runner.run()

        out = tmp_path / "test-dirs"
        assert out.is_dir()
        assert (out / "events.jsonl").exists()
        assert (out / "metrics.json").exists()
        assert (out / "memory_log.json").exists()
        assert (out / "report.md").exists()

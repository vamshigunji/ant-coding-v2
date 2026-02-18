"""
ExperimentRunner: orchestrates a roast battle and persists all
observability artifacts (events, metrics, memory logs, report).

Output structure:
    results/{experiment_id}/
    ├── metrics.json
    ├── events.jsonl
    ├── report.md
    └── memory_log.json
"""

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ant_coding.core.config import ModelConfig, MemoryConfig, MemoryMode
from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.memory.manager import MemoryManager
from ant_coding.models.provider import ModelProvider
from ant_coding.observability.event_logger import Event, EventType
from ant_coding.vanilla_architecture.agent import CharacterAgent, CharacterConfig
from ant_coding.vanilla_architecture.orchestrator import RoastBattleOrchestrator, BattleResult

logger = logging.getLogger(__name__)


def _serialize_event(event: Event) -> dict:
    """Convert an Event dataclass to a JSON-safe dict."""
    d = asdict(event)
    d["type"] = event.type.value
    d["timestamp"] = event.timestamp.isoformat()
    return d


class ExperimentRunner:
    """
    Full-lifecycle runner that:
      1. Sets up agents, memory, and model providers
      2. Runs the orchestration (roast battle)
      3. Computes ExperimentMetrics
      4. Persists events.jsonl, metrics.json, memory_log.json, report.md
    """

    def __init__(
        self,
        experiment_id: str,
        model_config: ModelConfig,
        char_a: CharacterConfig,
        char_b: CharacterConfig,
        memory_mode: str = "shared",
        rounds: int = 3,
        output_dir: str = "results",
    ):
        self.experiment_id = experiment_id
        self.model_config = model_config
        self.char_a = char_a
        self.char_b = char_b
        self.memory_mode = memory_mode
        self.rounds = rounds
        self.output_dir = Path(output_dir) / experiment_id

        # Initialised during run()
        self.memory: Optional[MemoryManager] = None
        self.agent_a: Optional[CharacterAgent] = None
        self.agent_b: Optional[CharacterAgent] = None
        self.result: Optional[BattleResult] = None
        self.metrics: Optional[ExperimentMetrics] = None
        self.duration_seconds: float = 0.0
        self._task_start_event: Optional[Event] = None
        self._task_end_event: Optional[Event] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> ExperimentMetrics:
        """Execute the full experiment lifecycle and return metrics."""
        logger.info(f"Experiment '{self.experiment_id}' starting")
        start = time.time()

        # 1. Setup
        self._task_start_event = Event(
            type=EventType.TASK_START,
            task_id="roast-battle",
            experiment_id=self.experiment_id,
            payload={"rounds": self.rounds, "characters": [self.char_a.name, self.char_b.name]},
        )
        self.memory = MemoryManager(MemoryConfig(mode=MemoryMode(self.memory_mode)))
        model_a = ModelProvider(self.model_config)
        model_b = ModelProvider(self.model_config)
        self.agent_a = CharacterAgent(
            character=self.char_a, model=model_a,
            memory=self.memory, experiment_id=self.experiment_id,
        )
        self.agent_b = CharacterAgent(
            character=self.char_b, model=model_b,
            memory=self.memory, experiment_id=self.experiment_id,
        )

        # 2. Run battle
        orchestrator = RoastBattleOrchestrator(self.agent_a, self.agent_b)
        self.result = await orchestrator.run(rounds=self.rounds)
        self.duration_seconds = time.time() - start

        # 3. Compute metrics
        self.metrics = self._compute_metrics()
        self._task_end_event = Event(
            type=EventType.TASK_END,
            task_id="roast-battle",
            experiment_id=self.experiment_id,
            payload={
                "total_tokens": self.metrics.total_tokens,
                "total_cost_usd": self.metrics.total_cost,
                "duration_seconds": self.duration_seconds,
            },
        )

        # 4. Persist everything
        self._persist()

        logger.info(
            f"Experiment '{self.experiment_id}' complete — "
            f"{self.metrics.total_tokens} tokens, "
            f"${self.metrics.total_cost:.4f}, "
            f"{self.duration_seconds:.1f}s"
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> ExperimentMetrics:
        total_tokens = sum(u["total_tokens"] for u in self.result.usage.values())
        total_cost = sum(u["total_cost_usd"] for u in self.result.usage.values())

        per_agent = {}
        for agent_id, usage in self.result.usage.items():
            per_agent[agent_id] = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "cost_usd": usage["total_cost_usd"],
            }

        return ExperimentMetrics(
            experiment_id=self.experiment_id,
            total_tasks=1,  # one battle = one task
            successful_tasks=1,
            failed_tasks=0,
            pass_rate=1.0,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_duration=self.duration_seconds,
            metadata={
                "rounds": self.result.rounds,
                "total_turns": len(self.result.conversation),
                "model": self.model_config.name,
                "memory_mode": self.memory_mode,
                "characters": [self.char_a.name, self.char_b.name],
                "per_agent_usage": per_agent,
            },
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_events()
        self._write_metrics()
        self._write_memory_log()
        self._write_report()
        logger.info(f"Artifacts saved to {self.output_dir}")

    def _collect_all_events(self) -> List[Event]:
        events = [self._task_start_event]
        events.extend(self.agent_a.events)
        events.extend(self.agent_b.events)
        events.append(self._task_end_event)
        # Sort by timestamp (task_start was created before agents ran)
        events.sort(key=lambda e: e.timestamp)
        return events

    def _write_events(self):
        path = self.output_dir / "events.jsonl"
        events = self._collect_all_events()
        with open(path, "w") as f:
            for event in events:
                f.write(json.dumps(_serialize_event(event)) + "\n")
        logger.info(f"  events.jsonl: {len(events)} events")

    def _write_metrics(self):
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(asdict(self.metrics), f, indent=2)
        logger.info(f"  metrics.json written")

    def _write_memory_log(self):
        path = self.output_dir / "memory_log.json"
        data = {
            "access_log": self.memory.get_access_log(),
            "final_state": self.memory.get_state_snapshot(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"  memory_log.json: {len(data['access_log'])} operations")

    def _write_report(self):
        path = self.output_dir / "report.md"
        m = self.metrics
        pa = m.metadata.get("per_agent_usage", {})

        lines = [
            f"# Experiment Report: {self.experiment_id}",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Model:** {self.model_config.name} (`{self.model_config.litellm_model}`)  ",
            f"**Memory mode:** {self.memory_mode}  ",
            f"**Rounds:** {self.result.rounds}  ",
            f"**Duration:** {self.duration_seconds:.1f}s  ",
            "",
            "---",
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total tokens | {m.total_tokens:,} |",
            f"| Total cost | ${m.total_cost:.4f} |",
            f"| Pass rate | {m.pass_rate:.0%} |",
            f"| Duration | {m.avg_duration:.1f}s |",
            f"| Turns | {m.metadata.get('total_turns', 0)} |",
            "",
        ]

        # Per-agent table
        if pa:
            lines += [
                "## Per-Agent Usage",
                "",
                "| Agent | Prompt | Completion | Total | Cost |",
                "|-------|--------|------------|-------|------|",
            ]
            for agent_id, u in pa.items():
                lines.append(
                    f"| {agent_id} | {u['prompt_tokens']:,} | "
                    f"{u['completion_tokens']:,} | {u['total_tokens']:,} | "
                    f"${u['cost_usd']:.4f} |"
                )
            lines += [""]

        # Transcript
        lines += [
            "---",
            "",
            "## Transcript",
            "",
        ]
        for i, turn in enumerate(self.result.conversation, 1):
            lines.append(f"**[Turn {i}] {turn['speaker']}:**")
            lines.append(f"> {turn['text']}")
            lines.append("")

        # Memory snapshot
        lines += [
            "---",
            "",
            "## Memory State (final)",
            "",
            "| Key | Value (truncated) |",
            "|-----|-------------------|",
        ]
        for key, value in self.memory.get_state_snapshot().items():
            val_str = str(value)
            display = val_str[:80] + "..." if len(val_str) > 80 else val_str
            lines.append(f"| `{key}` | {display} |")
        lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"  report.md written")

"""
Orchestrator that runs a turn-based roast battle between two CharacterAgents.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field

from ant_coding.vanilla_architecture.agent import CharacterAgent
from ant_coding.models.provider import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class BattleResult:
    """Result of a completed roast battle."""
    conversation: List[Dict[str, str]]
    rounds: int
    usage: Dict[str, Dict[str, Any]]  # per-agent usage stats


class RoastBattleOrchestrator:
    """
    Coordinates a turn-based conversation between two CharacterAgents.
    Agent A opens, Agent B responds, and they alternate for N rounds.
    """

    def __init__(self, agent_a: CharacterAgent, agent_b: CharacterAgent):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.conversation: List[Dict[str, str]] = []

    async def run(self, rounds: int = 3) -> BattleResult:
        """
        Execute the roast battle for the given number of rounds.

        Each round consists of one turn per agent (A then B).

        Args:
            rounds: Number of full rounds (each round = 2 turns).

        Returns:
            BattleResult with conversation and usage stats.
        """
        logger.info(
            f"Roast battle: {self.agent_a.character.name} vs "
            f"{self.agent_b.character.name} — {rounds} rounds"
        )

        for round_num in range(1, rounds + 1):
            logger.info(f"--- Round {round_num} ---")

            # Agent A's turn
            response_a = await self.agent_a.respond(self.conversation)
            self.conversation.append({
                "speaker": self.agent_a.character.name,
                "text": response_a,
            })
            logger.info(f"[{self.agent_a.character.name}]: {response_a}")

            # Agent B's turn
            response_b = await self.agent_b.respond(self.conversation)
            self.conversation.append({
                "speaker": self.agent_b.character.name,
                "text": response_b,
            })
            logger.info(f"[{self.agent_b.character.name}]: {response_b}")

        result = BattleResult(
            conversation=self.conversation,
            rounds=rounds,
            usage={
                self.agent_a.agent_id: self.agent_a.model.get_usage(),
                self.agent_b.agent_id: self.agent_b.model.get_usage(),
            },
        )

        self._print_summary(result)
        return result

    def _print_summary(self, result: BattleResult):
        """Print a human-readable summary of the battle."""
        total_tokens = 0
        total_cost = 0.0
        for agent_id, usage in result.usage.items():
            total_tokens += usage["total_tokens"]
            total_cost += usage["total_cost_usd"]

        logger.info("=" * 50)
        logger.info("BATTLE COMPLETE")
        logger.info(f"Rounds: {result.rounds}")
        logger.info(f"Total turns: {len(result.conversation)}")
        logger.info(f"Total tokens: {total_tokens}")
        logger.info(f"Total cost: ${total_cost:.4f}")
        for agent_id, usage in result.usage.items():
            logger.info(
                f"  {agent_id}: {usage['total_tokens']} tokens, "
                f"${usage['total_cost_usd']:.4f}"
            )
        logger.info("=" * 50)

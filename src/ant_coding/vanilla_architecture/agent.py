"""
Character-driven agent that uses a persona to generate responses.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager
from ant_coding.observability.event_logger import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class CharacterConfig:
    """Defines a character's personality for the agent."""
    name: str
    persona: str
    style: str  # e.g. "sarcastic", "deadpan", "over-the-top"


class CharacterAgent:
    """
    An agent with a distinct character persona that generates responses
    via an LLM and reads/writes conversation history through shared memory.
    """

    def __init__(
        self,
        character: CharacterConfig,
        model: ModelProvider,
        memory: MemoryManager,
        experiment_id: str = "roast-battle",
    ):
        self.character = character
        self.agent_id = character.name.lower().replace(" ", "_")
        self.model = model
        self.memory = memory
        self.experiment_id = experiment_id
        self.events: List[Event] = []

    def _build_system_prompt(self) -> str:
        return (
            f"You are {self.character.name}.\n"
            f"Personality: {self.character.persona}\n"
            f"Style: {self.character.style}\n\n"
            "You are in a friendly roast battle with another character. "
            "Your job is to playfully make fun of them based on what they say. "
            "Keep it light-hearted, witty, and funny — never mean-spirited. "
            "Respond in 2-3 sentences max. Stay in character at all times."
        )

    def _build_messages(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        for entry in conversation:
            if entry["speaker"] == self.character.name:
                messages.append({"role": "assistant", "content": entry["text"]})
            else:
                messages.append({"role": "user", "content": entry["text"]})
        if not conversation:
            messages.append({
                "role": "user",
                "content": "The roast battle has started! Fire the opening shot.",
            })
        return messages

    async def respond(self, conversation: List[Dict[str, str]]) -> str:
        """
        Generate a response given the conversation history.

        Args:
            conversation: List of {"speaker": ..., "text": ...} entries.

        Returns:
            The agent's response text.
        """
        self._log_event(EventType.AGENT_START, {"action": "respond"})

        messages = self._build_messages(conversation)

        self._log_event(EventType.LLM_CALL, {
            "model": self.model.config.name,
            "message_count": len(messages),
        })

        response = await self.model.complete(
            messages=messages,
            temperature=0.9,
            max_tokens=256,
        )

        text = response.choices[0].message.content.strip()

        # Write our response to shared memory so the other agent can see it
        turn_number = len(conversation) + 1
        self.memory.write(self.agent_id, f"turn_{turn_number}", text)

        usage = self.model.get_usage()
        self._log_event(EventType.AGENT_END, {
            "response_length": len(text),
            "total_tokens": usage["total_tokens"],
            "total_cost_usd": usage["total_cost_usd"],
        })

        return text

    def _log_event(self, event_type: EventType, payload: Dict[str, Any]):
        event = Event(
            type=event_type,
            task_id="roast-battle",
            experiment_id=self.experiment_id,
            agent_id=self.agent_id,
            payload=payload,
        )
        self.events.append(event)
        logger.debug(f"[{self.agent_id}] {event_type.value}: {payload}")

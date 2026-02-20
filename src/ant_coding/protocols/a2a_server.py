"""
A2A (Agent-to-Agent) server for exposing orchestration patterns as agents.

Registers orchestration patterns as discoverable A2A agents with Agent Cards
that describe capabilities, input/output schemas, and metadata.

Reference: docs/prd.md Section 8, Sprint-6-Epic-2.md (S6-E2-S02)
"""

import logging
from typing import Any, Dict, List, Optional

from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry

logger = logging.getLogger(__name__)


class AgentCard:
    """
    A2A Agent Card describing an agent's capabilities.

    Follows the A2A protocol convention for agent discovery.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Agent Card to a dict."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }


# Standard schemas for A2A task submission/response
TASK_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_description": {
            "type": "string",
            "description": "Description of the software engineering task to solve",
        },
        "repo_url": {
            "type": "string",
            "description": "Git repository URL (optional)",
        },
        "test_command": {
            "type": "string",
            "description": "Command to run tests (optional)",
        },
    },
    "required": ["task_description"],
}

TASK_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "patch": {"type": "string", "description": "Generated diff/patch"},
        "total_tokens": {"type": "integer"},
        "total_cost": {"type": "number"},
        "duration_seconds": {"type": "number"},
    },
}


class A2AServer:
    """
    A2A-compliant agent server for orchestration patterns.

    Exposes registered patterns as discoverable agents and routes
    incoming tasks to the specified agent.
    """

    def __init__(self):
        """Initialize the A2A server."""
        self._agent_cards: Dict[str, AgentCard] = {}

    def register_pattern(self, pattern: OrchestrationPattern) -> AgentCard:
        """
        Register an orchestration pattern as an A2A agent.

        Generates an Agent Card from the pattern's metadata.

        Args:
            pattern: The orchestration pattern to register.

        Returns:
            The generated AgentCard.
        """
        agents = pattern.get_agent_definitions()
        agent_names = [a.get("name", "unknown") for a in agents]

        card = AgentCard(
            name=pattern.name(),
            description=pattern.description(),
            capabilities=[
                "software_engineering",
                "code_generation",
                "bug_fixing",
                "test_execution",
            ],
            input_schema=TASK_INPUT_SCHEMA,
            output_schema=TASK_OUTPUT_SCHEMA,
            metadata={
                "agent_count": len(agents),
                "agent_names": agent_names,
                "framework": "ant-coding",
            },
        )

        self._agent_cards[pattern.name()] = card
        return card

    def register_all(self) -> List[AgentCard]:
        """
        Register all patterns from the OrchestrationRegistry.

        Returns:
            List of generated AgentCards.
        """
        cards = []
        for name in OrchestrationRegistry.list_available():
            pattern = OrchestrationRegistry.get(name)
            card = self.register_pattern(pattern)
            cards.append(card)
        return cards

    def discover(self) -> List[Dict[str, Any]]:
        """
        Handle A2A discovery request.

        Returns:
            List of Agent Card dicts for all registered agents.
        """
        return [card.to_dict() for card in self._agent_cards.values()]

    def get_agent(self, name: str) -> Optional[AgentCard]:
        """
        Look up an agent by name.

        Args:
            name: Agent/pattern name.

        Returns:
            AgentCard if found, None otherwise.
        """
        return self._agent_cards.get(name)

    async def submit_task(
        self,
        agent_name: str,
        task: Dict[str, Any],
        model: Any = None,
        memory: Any = None,
        tools: Any = None,
        workspace_dir: str = "",
    ) -> Dict[str, Any]:
        """
        Route an A2A task submission to the specified agent.

        Args:
            agent_name: Name of the agent/pattern to route to.
            task: Task dict with at least "task_description".
            model: ModelProvider instance.
            memory: MemoryManager instance.
            tools: ToolRegistry instance.
            workspace_dir: Workspace directory path.

        Returns:
            Dict with task result or error.
        """
        card = self._agent_cards.get(agent_name)
        if not card:
            return {"error": f"Agent '{agent_name}' not found", "success": False}

        try:
            pattern = OrchestrationRegistry.get(agent_name)
            result = await pattern.solve(
                task=task,
                model=model,
                memory=memory,
                tools=tools,
                workspace_dir=workspace_dir,
            )
            return {
                "success": result.success,
                "task_id": result.task_id,
                "total_tokens": result.total_tokens,
                "total_cost": result.total_cost,
                "duration_seconds": result.duration_seconds,
            }
        except Exception as e:
            logger.warning(f"A2A task submission to '{agent_name}' failed: {e}")
            return {"error": str(e), "success": False}

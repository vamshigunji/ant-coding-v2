"""
Entry point for running a roast battle between two character agents.

Usage:
    python -m ant_coding.vanilla_architecture.runner
    python -m ant_coding.vanilla_architecture.runner --config configs/experiments/roast-battle.yaml
    python -m ant_coding.vanilla_architecture.runner --rounds 5
"""

import asyncio
import argparse
import logging
from pathlib import Path

import yaml

from ant_coding.core.config import ModelConfig, MemoryConfig, MemoryMode
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager
from ant_coding.vanilla_architecture.agent import CharacterAgent, CharacterConfig
from ant_coding.vanilla_architecture.orchestrator import RoastBattleOrchestrator

# Default characters
SHAKESPEARE = CharacterConfig(
    name="Shakespeare",
    persona=(
        "A dramatic Elizabethan playwright who speaks in iambic pentameter "
        "and references his own plays constantly. Thinks modern technology "
        "is witchcraft and is baffled by anyone who can't quote Hamlet."
    ),
    style="flowery, dramatic, archaic English with theatrical flair",
)

ROBOT = CharacterConfig(
    name="Robot",
    persona=(
        "A hyper-logical AI robot from the year 3000 who finds human "
        "emotions inefficient. Communicates in a deadpan, matter-of-fact "
        "tone and constantly runs 'calculations' on how suboptimal humans are."
    ),
    style="deadpan, clinical, sprinkled with fake statistics and beep-boop sounds",
)


def load_characters_from_config(path: str) -> tuple:
    """Load character and model config from a YAML file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    characters = []
    for char_cfg in cfg.get("characters", []):
        characters.append(CharacterConfig(**char_cfg))

    model_cfg = cfg.get("model", {})
    rounds = cfg.get("rounds", 3)
    memory_mode = cfg.get("memory_mode", "shared")

    return characters, model_cfg, rounds, memory_mode


async def run_battle(
    model_config: ModelConfig,
    char_a: CharacterConfig,
    char_b: CharacterConfig,
    memory_mode: str = "shared",
    rounds: int = 3,
):
    """Set up and run a roast battle."""
    # Shared memory so both agents can see the full conversation state
    memory = MemoryManager(MemoryConfig(mode=MemoryMode(memory_mode)))

    # Each agent gets its own ModelProvider instance for independent token tracking
    model_a = ModelProvider(model_config)
    model_b = ModelProvider(model_config)

    agent_a = CharacterAgent(character=char_a, model=model_a, memory=memory)
    agent_b = CharacterAgent(character=char_b, model=model_b, memory=memory)

    orchestrator = RoastBattleOrchestrator(agent_a, agent_b)
    result = await orchestrator.run(rounds=rounds)

    # Print the conversation
    print("\n" + "=" * 60)
    print("  ROAST BATTLE TRANSCRIPT")
    print("=" * 60)
    for i, turn in enumerate(result.conversation, 1):
        print(f"\n[Turn {i}] {turn['speaker']}:")
        print(f"  {turn['text']}")

    # Print token usage
    print("\n" + "-" * 60)
    print("  TOKEN USAGE")
    print("-" * 60)
    for agent_id, usage in result.usage.items():
        print(
            f"  {agent_id}: "
            f"{usage['prompt_tokens']} prompt + "
            f"{usage['completion_tokens']} completion = "
            f"{usage['total_tokens']} total "
            f"(${usage['total_cost_usd']:.4f})"
        )
    total_tokens = sum(u["total_tokens"] for u in result.usage.values())
    total_cost = sum(u["total_cost_usd"] for u in result.usage.values())
    print(f"  {'TOTAL':>10}: {total_tokens} tokens, ${total_cost:.4f}")
    print("-" * 60)

    # Show memory state
    print("\n  MEMORY STATE")
    print("-" * 60)
    snapshot = memory.get_state_snapshot()
    for key, value in snapshot.items():
        print(f"  {key}: {value[:80]}..." if len(str(value)) > 80 else f"  {key}: {value}")
    print("-" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a roast battle between two AI characters")
    parser.add_argument("--config", type=str, help="Path to battle config YAML")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--model", type=str, default="claude-sonnet", help="Model name")
    parser.add_argument("--memory", type=str, default="shared", choices=["shared", "isolated", "hybrid"],
                        help="Memory mode (default: shared)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.config:
        characters, model_cfg_dict, rounds, memory_mode = load_characters_from_config(args.config)
        char_a, char_b = characters[0], characters[1]
        model_config = ModelConfig(**model_cfg_dict)
        rounds = args.rounds or rounds
        memory_mode = args.memory or memory_mode
    else:
        # Use defaults
        char_a, char_b = SHAKESPEARE, ROBOT
        rounds = args.rounds
        memory_mode = args.memory

        # Load model config from configs/ directory
        model_path = Path("configs/models") / f"{args.model}.yaml"
        if model_path.exists():
            with open(model_path) as f:
                model_config = ModelConfig(**yaml.safe_load(f))
        else:
            # Fallback: construct a minimal config
            model_config = ModelConfig(
                name=args.model,
                litellm_model="anthropic/claude-3-5-sonnet-20241022",
                api_key_env="ANTHROPIC_API_KEY",
            )

    asyncio.run(run_battle(model_config, char_a, char_b, memory_mode, rounds))


if __name__ == "__main__":
    main()

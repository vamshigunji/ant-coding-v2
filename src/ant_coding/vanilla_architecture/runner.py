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
from datetime import datetime
from pathlib import Path

import yaml

from ant_coding.core.config import ModelConfig
from ant_coding.vanilla_architecture.agent import CharacterConfig
from ant_coding.vanilla_architecture.experiment_runner import ExperimentRunner

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


def _print_live_output(runner: ExperimentRunner):
    """Print a live summary to stdout after the experiment completes."""
    result = runner.result
    metrics = runner.metrics

    print("\n" + "=" * 60)
    print("  ROAST BATTLE TRANSCRIPT")
    print("=" * 60)
    for i, turn in enumerate(result.conversation, 1):
        print(f"\n[Turn {i}] {turn['speaker']}:")
        print(f"  {turn['text']}")

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
    print(f"  {'TOTAL':>10}: {metrics.total_tokens} tokens, ${metrics.total_cost:.4f}")
    print("-" * 60)

    print("\n  MEMORY STATE")
    print("-" * 60)
    snapshot = runner.memory.get_state_snapshot()
    for key, value in snapshot.items():
        print(f"  {key}: {value[:80]}..." if len(str(value)) > 80 else f"  {key}: {value}")
    print("-" * 60)

    print(f"\n  Artifacts saved to: {runner.output_dir}/")
    print(f"    - events.jsonl   (agent & LLM events)")
    print(f"    - metrics.json   (ExperimentMetrics)")
    print(f"    - memory_log.json (access log + final state)")
    print(f"    - report.md      (full human-readable report)")
    print("-" * 60)


async def run_experiment(
    model_config: ModelConfig,
    char_a: CharacterConfig,
    char_b: CharacterConfig,
    memory_mode: str = "shared",
    rounds: int = 3,
    output_dir: str = "results",
    experiment_id: str = None,
):
    """Set up and run a full experiment with persistence."""
    if experiment_id is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_id = f"roast-battle-{timestamp}"

    runner = ExperimentRunner(
        experiment_id=experiment_id,
        model_config=model_config,
        char_a=char_a,
        char_b=char_b,
        memory_mode=memory_mode,
        rounds=rounds,
        output_dir=output_dir,
    )

    metrics = await runner.run()
    _print_live_output(runner)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run a roast battle between two AI characters")
    parser.add_argument("--config", type=str, help="Path to battle config YAML")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds (default: 3)")
    parser.add_argument("--model", type=str, default="claude-sonnet", help="Model name")
    parser.add_argument("--memory", type=str, default="shared", choices=["shared", "isolated", "hybrid"],
                        help="Memory mode (default: shared)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID (auto-generated if omitted)")
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
        char_a, char_b = SHAKESPEARE, ROBOT
        rounds = args.rounds
        memory_mode = args.memory

        model_path = Path("configs/models") / f"{args.model}.yaml"
        if model_path.exists():
            with open(model_path) as f:
                model_config = ModelConfig(**yaml.safe_load(f))
        else:
            model_config = ModelConfig(
                name=args.model,
                litellm_model="anthropic/claude-sonnet-4-5-20250929",
                api_key_env="ANTHROPIC_API_KEY",
            )

    asyncio.run(run_experiment(
        model_config, char_a, char_b, memory_mode, rounds,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
    ))


if __name__ == "__main__":
    main()

"""
CLI entry point for running experiments.

Usage:
    python -m ant_coding.cli.run <config_path> [--pattern NAME] [--output DIR]
"""

import argparse
import asyncio
import logging
import sys

from ant_coding.runner.experiment import ExperimentRunner
from ant_coding.runner.output import ResultWriter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ant-coding",
        description="Run an ant-coding experiment from a YAML config file.",
    )
    parser.add_argument(
        "config",
        help="Path to the experiment config YAML file.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Override the orchestration pattern name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override the output directory.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    """Run the experiment asynchronously."""
    runner = ExperimentRunner.from_config_file(
        args.config,
        pattern_name=args.pattern,
        output_dir=args.output,
    )

    results = await runner.run()
    summary = runner.get_summary()

    # Save results
    writer = ResultWriter(
        output_dir=runner.output_dir,
        experiment_name=runner.config.name,
    )
    output_path = writer.save_all(
        config=runner.config.model_dump(),
        results=results,
        summary=summary,
        events=runner.events,
    )

    # Print summary
    print(f"\nExperiment: {summary['experiment_name']}")
    print(f"Pattern:    {summary['pattern']}")
    print(f"Tasks:      {summary['successful_tasks']}/{summary['total_tasks']} passed")
    print(f"Pass rate:  {summary['pass_rate']:.1%}")
    print(f"Tokens:     {summary['total_tokens']:,}")
    print(f"Cost:       ${summary['total_cost_usd']:.4f}")
    print(f"Avg time:   {summary['avg_duration_seconds']:.1f}s")
    print(f"Output:     {output_path}")

    return 0 if summary["pass_rate"] > 0 else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

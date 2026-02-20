#!/usr/bin/env python3
"""
Cross-experiment comparison CLI.

Usage:
    python scripts/compare_results.py results/exp-a
    python scripts/compare_results.py results/exp-a results/exp-b
    python scripts/compare_results.py results/exp-a results/exp-b results/exp-c

Single experiment: prints all 4 tiers of metrics.
Two+ experiments: prints pairwise comparison tables with statistical tests.
Saves comparison_report.md to the current directory.

Reference: Sprint-6-Epic-1.md (S6-E1-S04)
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ant_coding.eval.comparison import compare_experiments, ComparisonResult
from ant_coding.eval.metrics import ExperimentMetrics
from ant_coding.eval.report import (
    generate_comparison_markdown,
    generate_markdown,
    metrics_from_json,
)


def load_metrics(result_dir: str) -> ExperimentMetrics:
    """
    Load ExperimentMetrics from a result directory.

    Expects a metrics.json file in the directory.

    Args:
        result_dir: Path to the experiment result directory.

    Returns:
        ExperimentMetrics instance.

    Raises:
        FileNotFoundError: If metrics.json doesn't exist.
    """
    metrics_path = Path(result_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {result_dir}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        return metrics_from_json(f.read())


def main() -> None:
    """Run the comparison CLI."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_results.py <result_dir> [result_dir2] ...")
        print()
        print("  Single dir:  prints all 4 tiers of metrics")
        print("  Two+ dirs:   prints pairwise comparison with statistical tests")
        print()
        print("Saves comparison_report.md to current directory.")
        sys.exit(1)

    result_dirs = sys.argv[1:]

    # Load metrics from each directory
    all_metrics = []
    for rd in result_dirs:
        try:
            metrics = load_metrics(rd)
            all_metrics.append(metrics)
            print(f"Loaded: {metrics.experiment_id} from {rd}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

    if not all_metrics:
        print("No metrics loaded.")
        sys.exit(1)

    # Single experiment
    if len(all_metrics) == 1:
        report = generate_markdown(all_metrics[0])
        print()
        print(report)
        _save_report(report)
        return

    # Multiple experiments: pairwise comparisons
    comparisons = []
    for i in range(len(all_metrics)):
        for j in range(i + 1, len(all_metrics)):
            comp = compare_experiments(
                results_a=[],  # No per-task results from metrics.json alone
                results_b=[],
                metrics_a=all_metrics[i],
                metrics_b=all_metrics[j],
            )
            comparisons.append(comp)

    report = generate_comparison_markdown(all_metrics, comparisons)
    print()
    print(report)
    _save_report(report)


def _save_report(report: str) -> None:
    """Save report to comparison_report.md."""
    output_path = Path("comparison_report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

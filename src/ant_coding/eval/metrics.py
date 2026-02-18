"""
Metric definitions for evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


def _default_failure_categories() -> Dict[str, int]:
    """Default failure category counters."""
    return {
        "planning": 0,
        "implementation": 0,
        "integration": 0,
        "hallucination_cascade": 0,
        "timeout": 0,
        "tool_failure": 0,
    }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment run."""

    experiment_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    pass_rate: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PRD+ Tier 1 — Primary
    cost_per_resolution: float = 0.0

    # PRD+ Tier 2 — Efficiency
    useful_token_ratio: float = 0.0
    overhead_ratio: float = 0.0
    tokens_per_resolution: float = 0.0

    # PRD+ Tier 3 — Quality
    avg_patch_quality: float = 0.0
    avg_patch_size_ratio: float = 0.0

    # PRD+ Tier 4 — Robustness
    resolution_variance_cv: float = 0.0
    error_recovery_rate: float = 0.0
    failure_categories: Dict[str, int] = field(default_factory=_default_failure_categories)

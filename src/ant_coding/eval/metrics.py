"""
Metric definitions for evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ExperimentMetrics:
    experiment_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    pass_rate: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

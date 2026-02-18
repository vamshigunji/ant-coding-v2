"""
Core type definitions for tasks and task results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class TaskSource(str, Enum):
    SWE_BENCH = "swe-bench"
    CUSTOM = "custom"
    GAIA = "gaia"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Task:
    id: str
    description: str
    source: TaskSource
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    max_tokens_budget: int = 100_000
    timeout_seconds: int = 600
    files_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


VALID_FAILURE_CATEGORIES = frozenset({
    "planning",
    "implementation",
    "integration",
    "hallucination_cascade",
    "timeout",
    "tool_failure",
})


@dataclass
class TaskResult:
    """Result of running a single task in an experiment."""

    task_id: str
    experiment_id: str
    success: bool
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_seconds: float = 0.0
    agent_traces: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PRD+ fields
    intermediate_test_results: List[bool] = field(default_factory=list)
    failure_category: Optional[str] = None
    generated_patch_lines: int = 0
    gold_patch_lines: int = 0
    judge_scores: Optional[Dict[str, Any]] = None

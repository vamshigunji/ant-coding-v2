"""
Experiment registry for tracking experiment lineage.

Enables experiment tracking: each experiment records its parent, the single
variable changed, a hypothesis, and post-run insights. Stored as a YAML file.

Reference: docs/experimentation-playbook.md, Sprint-6-Epic-1.md (S6-E1-S05)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ant_coding.eval.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = "experiments/registry.yml"


class ExperimentRegistry:
    """
    YAML-based experiment registry for lineage tracking.

    Each experiment entry records its parent, variable changed, hypothesis,
    config path, status, outcome metrics, and post-run insights.
    """

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        """
        Initialize the registry, creating the file if it doesn't exist.

        Args:
            registry_path: Path to the registry YAML file.
        """
        self._path = Path(registry_path)
        self._data: Dict[str, Any] = {"experiments": []}

        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                self._data = loaded if "experiments" in loaded else {"experiments": []}
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        """Write the registry back to disk."""
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    def add_experiment(
        self,
        experiment_id: str,
        config_path: str = "",
        parent: Optional[str] = None,
        variable_changed: Optional[str] = None,
        hypothesis: str = "",
    ) -> Dict[str, Any]:
        """
        Register a new planned experiment.

        Args:
            experiment_id: Unique identifier for the experiment.
            config_path: Path to the experiment config YAML.
            parent: Parent experiment ID (None for baselines).
            variable_changed: Description of the single variable changed.
            hypothesis: What you expect to happen.

        Returns:
            The created experiment entry dict.
        """
        entry = {
            "id": experiment_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "parent": parent,
            "variable_changed": variable_changed,
            "hypothesis": hypothesis,
            "config": config_path,
            "results": f"results/{experiment_id}",
            "status": "planned",
            "outcome": {
                "pass_rate": None,
                "total_tokens": None,
                "total_cost": None,
                "cost_per_resolution": None,
                "useful_token_ratio": None,
                "overhead_ratio": None,
                "tokens_per_resolution": None,
                "avg_patch_quality": None,
                "avg_patch_size_ratio": None,
                "resolution_variance_cv": None,
                "error_recovery_rate": None,
            },
            "insight": "",
            "next_action": "",
        }

        self._data["experiments"].append(entry)
        self._save()
        return entry

    def update_status(self, experiment_id: str, status: str) -> None:
        """
        Update the status of an experiment.

        Args:
            experiment_id: The experiment to update.
            status: New status ("planned", "running", "complete").
        """
        entry = self._find(experiment_id)
        if entry:
            entry["status"] = status
            self._save()

    def update_outcome(
        self,
        experiment_id: str,
        metrics: ExperimentMetrics,
    ) -> None:
        """
        Populate the outcome section with experiment metrics.

        Sets status to "complete" automatically.

        Args:
            experiment_id: The experiment to update.
            metrics: The computed ExperimentMetrics.
        """
        entry = self._find(experiment_id)
        if not entry:
            logger.warning(f"Experiment '{experiment_id}' not found in registry.")
            return

        entry["outcome"] = {
            "pass_rate": metrics.pass_rate,
            "total_tokens": metrics.total_tokens,
            "total_cost": metrics.total_cost,
            "cost_per_resolution": (
                metrics.cost_per_resolution
                if metrics.cost_per_resolution != float("inf")
                else None
            ),
            "useful_token_ratio": metrics.useful_token_ratio,
            "overhead_ratio": metrics.overhead_ratio,
            "tokens_per_resolution": (
                metrics.tokens_per_resolution
                if metrics.tokens_per_resolution != float("inf")
                else None
            ),
            "avg_patch_quality": metrics.avg_patch_quality,
            "avg_patch_size_ratio": metrics.avg_patch_size_ratio,
            "resolution_variance_cv": metrics.resolution_variance_cv,
            "error_recovery_rate": metrics.error_recovery_rate,
        }
        entry["status"] = "complete"
        self._save()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up an experiment by ID.

        Args:
            experiment_id: The experiment to find.

        Returns:
            The experiment entry dict, or None if not found.
        """
        return self._find(experiment_id)

    def get_lineage(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Return the chain of parent experiments back to the root baseline.

        Args:
            experiment_id: Starting experiment ID.

        Returns:
            List of experiment entries from root to the given experiment.
        """
        chain = []
        current_id = experiment_id
        seen = set()

        while current_id and current_id not in seen:
            seen.add(current_id)
            entry = self._find(current_id)
            if not entry:
                break
            chain.append(entry)
            current_id = entry.get("parent")

        chain.reverse()  # Root first
        return chain

    def validate(self) -> List[str]:
        """
        Validate the registry for common issues.

        Checks:
        - Non-baseline experiments have exactly one variable_changed
        - Warns about experiments with status="planned" older than 7 days

        Returns:
            List of warning/error messages. Empty if valid.
        """
        warnings = []
        cutoff = datetime.now() - timedelta(days=7)

        for entry in self._data["experiments"]:
            exp_id = entry.get("id", "unknown")
            parent = entry.get("parent")
            variable = entry.get("variable_changed")
            status = entry.get("status", "")
            date_str = entry.get("date", "")

            # Non-baseline experiments must have variable_changed
            if parent and not variable:
                warnings.append(
                    f"Experiment '{exp_id}' has a parent but no variable_changed."
                )

            # Stale planned experiments
            if status == "planned" and date_str:
                try:
                    exp_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if exp_date < cutoff:
                        warnings.append(
                            f"Experiment '{exp_id}' has been 'planned' since {date_str} (>7 days)."
                        )
                except ValueError:
                    pass

        return warnings

    def suggest_id(
        self, parent: str, variable_changed: str
    ) -> str:
        """
        Suggest an experiment ID based on parent and variable changed.

        Convention: {parent}--{variable_changed_slug}

        Args:
            parent: Parent experiment ID.
            variable_changed: Description of what changed.

        Returns:
            Suggested experiment ID string.
        """
        slug = variable_changed.lower()
        # Replace common separators with hyphens
        for char in [" → ", " -> ", ": ", " ", "/", "_"]:
            slug = slug.replace(char, "-")
        # Remove non-alphanumeric (except hyphens)
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        # Collapse multiple hyphens
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug.strip("-")

        return f"{parent}--{slug}"

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Return all experiments in the registry."""
        return list(self._data["experiments"])

    def _find(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Find an experiment entry by ID."""
        for entry in self._data["experiments"]:
            if entry.get("id") == experiment_id:
                return entry
        return None

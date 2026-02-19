"""
Result output structure for experiment results.

Creates a timestamped output directory with config, results, and events.
"""

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ant_coding.observability.event_logger import Event
from ant_coding.tasks.types import TaskResult


class ResultWriter:
    """
    Writes experiment results to a structured output directory.

    Output structure:
        <output_dir>/<experiment_id>/
            config.json          — Copy of experiment config
            results.json         — All task results
            summary.json         — Aggregate metrics
            events.jsonl         — Event log (one JSON per line)
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Initialize the result writer.

        Args:
            output_dir: Base output directory.
            experiment_name: Name of the experiment.
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.experiment_id = f"{timestamp}_{experiment_name}"
        self.output_path = Path(output_dir) / self.experiment_id
        self.output_path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save experiment configuration.

        Args:
            config: Experiment config as dict.

        Returns:
            Path to saved config file.
        """
        path = self.output_path / "config.json"
        path.write_text(json.dumps(config, indent=2, default=str))
        return path

    def save_results(self, results: List[TaskResult]) -> Path:
        """
        Save all task results.

        Args:
            results: List of TaskResult objects.

        Returns:
            Path to saved results file.
        """
        path = self.output_path / "results.json"
        data = [asdict(r) for r in results]
        path.write_text(json.dumps(data, indent=2, default=str))
        return path

    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Save experiment summary.

        Args:
            summary: Summary dict from ExperimentRunner.get_summary().

        Returns:
            Path to saved summary file.
        """
        path = self.output_path / "summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        return path

    def save_events(self, events: List[Event]) -> Path:
        """
        Save events in JSONL format (one JSON object per line).

        Args:
            events: List of Event objects.

        Returns:
            Path to saved events file.
        """
        path = self.output_path / "events.jsonl"
        with open(path, "w") as f:
            for event in events:
                line = json.dumps(asdict(event), default=str)
                f.write(line + "\n")
        return path

    def save_all(
        self,
        config: Dict[str, Any],
        results: List[TaskResult],
        summary: Dict[str, Any],
        events: List[Event],
    ) -> Path:
        """
        Save all experiment outputs at once.

        Args:
            config: Experiment config dict.
            results: List of TaskResult objects.
            summary: Summary dict.
            events: List of Event objects.

        Returns:
            Path to the output directory.
        """
        self.save_config(config)
        self.save_results(results)
        self.save_summary(summary)
        self.save_events(events)
        return self.output_path

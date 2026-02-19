"""
ExperimentRunner: the central orchestrator that wires all layers together.

Manages the outer loop (config loading, task iteration, evaluation) while
delegating the inner loop (agent coordination) to OrchestrationPattern plugins.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ant_coding.core.config import (
    ExperimentConfig,
    ModelConfig,
    MemoryConfig,
    load_experiment_config,
)
from ant_coding.memory.manager import MemoryManager
from ant_coding.models.provider import ModelProvider
from ant_coding.observability.event_logger import Event, EventLogger, EventType
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.loader import TaskLoader
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.tasks.workspace import TaskWorkspace
from ant_coding.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs a complete experiment: loads config, iterates tasks, delegates to
    orchestration patterns, and collects results.

    The runner never directly calls agents — it delegates to
    OrchestrationPattern.solve() for the inner agent loop.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        pattern_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the ExperimentRunner.

        Args:
            config: The experiment configuration.
            pattern_name: Override orchestration pattern name (defaults to config name).
            output_dir: Override output directory.
        """
        self.config = config
        self.pattern_name = pattern_name or config.name
        self.output_dir = Path(output_dir or config.output.dir)
        self.results: List[TaskResult] = []
        self.events: List[Event] = []
        self._start_time: Optional[float] = None
        self.event_logger = EventLogger(
            experiment_id=config.name,
            output_dir=str(self.output_dir),
        )

    @classmethod
    def from_config_file(cls, path: str, **kwargs: Any) -> "ExperimentRunner":
        """
        Create an ExperimentRunner from a YAML config file.

        Args:
            path: Path to the experiment config YAML.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            Configured ExperimentRunner instance.
        """
        config = load_experiment_config(path)
        return cls(config, **kwargs)

    def _log_event(
        self,
        event_type: EventType,
        task_id: str,
        agent_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an event via both in-memory list and EventLogger."""
        event = Event(
            type=event_type,
            task_id=task_id,
            experiment_id=self.config.name,
            agent_id=agent_id,
            payload=payload or {},
        )
        self.events.append(event)
        self.event_logger.log(event)

    def _init_model(self) -> ModelProvider:
        """Initialize the model provider from config with event logging."""
        model_config = self.config.model
        if isinstance(model_config, str):
            raise ValueError(
                f"Model config must be resolved before running. Got string: {model_config}"
            )
        return ModelProvider(
            model_config,
            event_logger=self.event_logger,
            experiment_id=self.config.name,
        )

    def _init_memory(self) -> MemoryManager:
        """Initialize the memory manager from config with event logging."""
        memory_config = self.config.memory
        if isinstance(memory_config, str):
            raise ValueError(
                f"Memory config must be resolved before running. Got string: {memory_config}"
            )
        return MemoryManager(
            memory_config,
            event_logger=self.event_logger,
            experiment_id=self.config.name,
        )

    async def _run_task(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        pattern_name: str,
    ) -> TaskResult:
        """
        Run a single task through the orchestration pattern.

        Args:
            task: The task to solve.
            model: LLM provider.
            memory: Memory manager (reset per task).
            pattern_name: Name of the orchestration pattern to use.

        Returns:
            TaskResult from the orchestration pattern.
        """
        self._log_event(EventType.TASK_START, task.id)
        start_time = time.time()

        # Set up workspace
        workspace = TaskWorkspace(task)
        await workspace.setup()

        try:
            # Initialize tools scoped to workspace with event logging
            tools = ToolRegistry(
                workspace.workspace_dir,
                event_logger=self.event_logger,
                experiment_id=self.config.name,
                task_id=task.id,
            )

            # Get orchestration pattern
            pattern = OrchestrationRegistry.get(pattern_name)

            # Reset memory and model usage for this task
            memory.reset()
            model.reset_usage()

            # Set task context for event logging
            model.set_context(task_id=task.id, experiment_id=self.config.name)
            memory.set_context(task_id=task.id, experiment_id=self.config.name)

            # Delegate to orchestration pattern
            result = await pattern.solve(
                task=task,
                model=model,
                memory=memory,
                tools=tools.as_dict(),
                workspace_dir=str(workspace.workspace_dir),
            )

            # Update duration
            result.duration_seconds = time.time() - start_time

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            result = TaskResult(
                task_id=task.id,
                experiment_id=self.config.name,
                success=False,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )
        finally:
            await workspace.teardown()

        self._log_event(
            EventType.TASK_END,
            task.id,
            payload={
                "success": result.success,
                "tokens": result.total_tokens,
                "duration": result.duration_seconds,
            },
        )

        return result

    async def run(self) -> List[TaskResult]:
        """
        Execute the full experiment pipeline.

        1. Load tasks from config
        2. Initialize model and memory
        3. For each task, delegate to orchestration pattern
        4. Collect and return results

        Returns:
            List of TaskResult objects.
        """
        self._start_time = time.time()
        logger.info(f"Starting experiment: {self.config.name}")

        # Load tasks
        loader = TaskLoader()
        tasks = loader.load_from_config(self.config.tasks)
        logger.info(f"Loaded {len(tasks)} tasks")

        # Initialize layers
        model = self._init_model()
        memory = self._init_memory()

        # Run tasks
        self.results = []
        for i, task in enumerate(tasks):
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task.id}")
            result = await self._run_task(task, model, memory, self.pattern_name)
            self.results.append(result)

        elapsed = time.time() - self._start_time
        successful = sum(1 for r in self.results if r.success)
        logger.info(
            f"Experiment complete: {successful}/{len(self.results)} tasks passed "
            f"in {elapsed:.1f}s"
        )

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the experiment results.

        Returns:
            Dict with experiment metadata and aggregate stats.
        """
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        total_tokens = sum(r.total_tokens for r in self.results)
        total_cost = sum(r.total_cost for r in self.results)
        avg_duration = (
            sum(r.duration_seconds for r in self.results) / total if total > 0 else 0.0
        )

        return {
            "experiment_name": self.config.name,
            "pattern": self.pattern_name,
            "total_tasks": total,
            "successful_tasks": successful,
            "failed_tasks": total - successful,
            "pass_rate": successful / total if total > 0 else 0.0,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_duration_seconds": avg_duration,
            "total_events": len(self.events),
        }

# Developer Guide

This guide covers how to extend ant-coding with new orchestration patterns, tools, and experiment configurations.

## Creating a New Orchestration Pattern

### 1. Subclass OrchestrationPattern

Create a new file in `src/ant_coding/orchestration/examples/`:

```python
# src/ant_coding/orchestration/examples/my_pattern.py

from typing import Dict, Any, List
from ant_coding.orchestration.base import OrchestrationPattern
from ant_coding.orchestration.registry import OrchestrationRegistry
from ant_coding.tasks.types import Task, TaskResult
from ant_coding.models.provider import ModelProvider
from ant_coding.memory.manager import MemoryManager


@OrchestrationRegistry.register
class MyPattern(OrchestrationPattern):
    """My custom orchestration pattern."""

    def name(self) -> str:
        return "my-pattern"

    def description(self) -> str:
        return "Description of what this pattern does."

    def get_agent_definitions(self) -> List[Dict[str, str]]:
        return [
            {"name": "Planner", "role": "Plans the approach"},
            {"name": "Coder", "role": "Implements the solution"},
        ]

    async def solve(
        self,
        task: Task,
        model: ModelProvider,
        memory: MemoryManager,
        tools: Dict[str, Any],
        workspace_dir: str,
    ) -> TaskResult:
        # 1. Use model.complete() for LLM calls
        response = await model.complete([
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": task.description},
        ])
        plan = response.choices[0].message.content

        # 2. Use memory for inter-agent communication
        memory.write("planner", "plan", plan)

        # 3. Second agent reads the plan
        plan_from_memory = memory.read("coder", "plan")

        # 4. Return TaskResult
        usage = model.get_usage()
        return TaskResult(
            task_id=task.id,
            experiment_id=self.name(),
            success=True,
            total_tokens=usage["total_tokens"],
            total_cost=usage["total_cost_usd"],
            agent_traces=[
                {"agent": "Planner", "action": "plan", "output": plan},
            ],
        )
```

### 2. Register It

The `@OrchestrationRegistry.register` decorator automatically registers your pattern. To use it in an experiment, reference it by name:

```yaml
# In your experiment config or CLI
python scripts/run_experiment.py config.yaml --pattern my-pattern
```

### 3. Available Patterns

| Pattern | Name | Agents | Description |
|---------|------|--------|-------------|
| SingleAgent | `single-agent` | 1 | Baseline control group |
| MinimalSequential | `minimal-sequential` | 2 | Planner → Coder pipeline |
| MinimalParallel | `minimal-parallel` | 2 | Parallel agents with merge |
| MinimalLoop | `minimal-loop` | 2 | Iterative refinement loop |

## Adding New Tools

### 1. Implement the Tool

Create a new tool class in `src/ant_coding/tools/`:

```python
# src/ant_coding/tools/my_tool.py

class MyTool:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    def execute(self, input_data: str) -> str:
        # Your tool logic here
        return "result"
```

### 2. Register in ToolRegistry

Add your tool to `src/ant_coding/tools/registry.py`:

```python
class ToolRegistry:
    def __init__(self, workspace_dir, ...):
        # ... existing tools ...
        self.my_tool = MyTool(workspace_dir)

    def as_dict(self) -> Dict[str, Any]:
        return {
            # ... existing tools ...
            "my_tool": self.my_tool,
        }
```

### 3. Expose via MCP (Optional)

Add a tool definition and handler in `src/ant_coding/protocols/mcp_server.py`:

```python
TOOL_DEFINITIONS.append({
    "name": "my_tool",
    "description": "What my tool does.",
    "input_schema": {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input data"},
        },
        "required": ["input"],
    },
})
```

## Experiment Configuration

### YAML Config Reference

```yaml
name: experiment-name              # Unique experiment identifier

model:
  name: gpt-4                     # Display name
  litellm_model: gpt-4            # LiteLLM model ID
  api_key_env: OPENAI_API_KEY     # Env var with API key
  max_tokens: 8192                # Max tokens per request
  temperature: 0.0                # Sampling temperature

memory:
  mode: shared                    # shared | isolated | hybrid
  shared_keys: []                 # Keys shared across agents (hybrid mode)

tasks:
  source: custom                  # custom | swe-bench | gaia
  subset: tasks/custom/my.yaml   # Path or subset name
  limit: 10                      # Max tasks to run (optional)
  task_ids: [task-1, task-2]     # Specific task IDs (optional)

execution:
  max_workers: 1                  # Parallel task workers
  timeout_seconds: 1800           # Per-task timeout
  max_iterations: 10              # Max orchestration iterations

eval:
  metrics: [pass@1]              # Metrics to compute
  eval_model: null               # Override judge model (optional)

output:
  dir: results                   # Output directory
  save_traces: true              # Save agent traces

baseline_experiment_id: null     # Reference for overhead_ratio (PRD+)
```

### Memory Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **shared** | All agents read/write the same memory space | Cooperative multi-agent patterns |
| **isolated** | Each agent has its own private memory | Independent agent patterns |
| **hybrid** | Some keys shared, rest isolated | Selective information sharing |

### Configuring Baselines (PRD+)

To compute `overhead_ratio`, set `baseline_experiment_id` to a previous single-agent experiment:

```yaml
baseline_experiment_id: "2024-01-15T10-30-00_single-agent-baseline"
```

This tells the evaluation harness to compare token usage against the baseline.

## Experiment Registry

Track experiment lineage in `experiments/registry.yml`:

```yaml
experiments:
  baseline-gpt4:
    config_path: configs/experiments/baseline.yaml
    status: completed  # planned | running | completed | failed
    parent: null
    variable_changed: null
    hypothesis: "Establish single-agent baseline"

  baseline-gpt4--shared-memory:
    config_path: configs/experiments/shared-mem.yaml
    status: planned
    parent: baseline-gpt4
    variable_changed: memory_mode
    hypothesis: "Shared memory improves pass rate for sequential pattern"
```

### Naming Convention

`{parent}--{variable-slug}` — e.g., `baseline-gpt4--shared-memory`

### Using the Registry API

```python
from ant_coding.core.experiment_registry import ExperimentRegistry

registry = ExperimentRegistry("experiments/registry.yml")

# Add experiment
registry.add_experiment(
    experiment_id="baseline-gpt4",
    config_path="configs/experiments/baseline.yaml",
    hypothesis="Establish baseline",
)

# Track lineage
lineage = registry.get_lineage("baseline-gpt4--shared-memory")
# Returns: [{"experiment_id": "baseline-gpt4", ...}, {"experiment_id": "baseline-gpt4--shared-memory", ...}]

# Suggest next ID
next_id = registry.suggest_id("baseline-gpt4", "temperature")
# Returns: "baseline-gpt4--temperature"
```

## Running Statistical Comparisons

```python
from ant_coding.eval.comparison import compare_experiments, generate_comparison_report
from ant_coding.eval.harness import calculate_metrics

# Calculate metrics for each experiment
metrics_a = calculate_metrics(results_a, "exp-a")
metrics_b = calculate_metrics(results_b, "exp-b")

# Run paired statistical tests
comparison = compare_experiments(results_a, results_b, metrics_a, metrics_b)

# Generate report
report = generate_comparison_report(metrics_a, metrics_b, comparison)
print(report)
```

### Statistical Tests Used

| Test | Purpose | When Used |
|------|---------|-----------|
| McNemar's | Pass rate difference significance | Binary pass/fail comparison |
| Wilcoxon signed-rank | Continuous metric differences | Token usage, cost, duration |
| Cohen's d | Effect size magnitude | All paired comparisons |
| Bootstrap CI | Confidence intervals | Pass rate, cost differences |

## Session Replay

Replay experiment events for debugging:

```python
from ant_coding.observability.replay import SessionReplay

replay = SessionReplay("results/my-exp/events.jsonl")

# Step through events
events = replay.step(5)  # Next 5 events

# Reconstruct memory state at event N
state = replay.state_at(10)

# Get cumulative token curve
curve = replay.token_curve()  # [(event_idx, cumulative_tokens), ...]

# Filter events
llm_calls = replay.get_events(event_type=EventType.LLM_CALL)
```

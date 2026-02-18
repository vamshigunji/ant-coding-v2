# AgentForge — Product Requirements Document

**Version:** 1.0
**Date:** February 18, 2026
**Author:** Vamshi (Research Lead)
**Status:** Implementation-Ready

---

## 1. Purpose & Scope

AgentForge is a research framework that empirically compares **shared memory** vs **isolated memory** architectures in multi-agent systems solving real-world software engineering tasks.

### 1.1 What This Document Covers

This PRD defines **every component except agent orchestration logic**. The orchestration layer (how agents coordinate, delegate, and collaborate) is the researcher's domain — built manually using Google ADK primitives.

Everything else — task loading, model abstraction, tool interfaces, memory backends, evaluation harness, observability, configuration, and CLI — should be built by a coding agent from this specification.

### 1.2 Core Research Question

> Do shared memory architectures (all agents read/write a common knowledge store) outperform isolated memory architectures (each agent maintains private context) in multi-agent software engineering, measured by token efficiency, output quality, and task completion rate?

### 1.3 Non-Goals

- Building a production SaaS product
- Creating a general-purpose agent framework
- Competing with existing tools (Aider, SWE-agent)
- Supporting non-Python codebases (SWE-bench is Python-only)

---

## 2. System Architecture

The system is composed of 8 independent, swappable modules ("Lego blocks"). Each block communicates through defined interfaces. Replacing any single block should not require changes to other blocks.

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentForge System                        │
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  1. Task     │───▶│ 3. Orchestr. │───▶│ 7. Evaluation     │  │
│  │  Loader      │    │ (MANUAL)     │    │    Harness        │  │
│  └─────────────┘    └──────┬───────┘    └───────────────────┘  │
│                            │                                    │
│                     ┌──────┼──────┐                             │
│                     │      │      │                             │
│                     ▼      ▼      ▼                             │
│              ┌──────┐ ┌────┐ ┌──────┐                          │
│              │2.Model│ │4.Mem│ │5.Tool│                         │
│              │ Layer │ │ory │ │ Layer│                          │
│              └──────┘ └────┘ └──────┘                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐ │
│  │ 6. Protocol   │    │ 8. Observ.   │    │  9. Config &      │ │
│  │    Layer      │    │    Layer     │    │     CLI           │ │
│  └──────────────┘    └──────────────┘    └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Block Ownership

| Block | Owner | This PRD Specifies |
|-------|-------|--------------------|
| 1. Task Loader | Coding Agent | Full implementation |
| 2. Model Layer | Coding Agent | Full implementation |
| 3. Orchestration | Researcher (Vamshi) | Interface contracts only |
| 4. Memory Layer | Coding Agent | Full implementation |
| 5. Tool Layer | Coding Agent | Full implementation |
| 6. Protocol Layer | Coding Agent | Full implementation |
| 7. Evaluation Harness | Coding Agent | Full implementation |
| 8. Observability Layer | Coding Agent | Full implementation |
| 9. Config & CLI | Coding Agent | Full implementation |

---

## 3. Technology Stack

### 3.1 Core Dependencies

```
# requirements.txt
google-adk>=1.25.0          # Agent orchestration framework
litellm>=1.80.0             # Model abstraction (100+ providers)
swebench>=2.1.0             # Benchmark dataset and evaluation harness
datasets>=2.20.0            # HuggingFace dataset loading
docker>=7.0.0               # Container management for SWE-bench eval
pyyaml>=6.0                 # Configuration files
pydantic>=2.0               # Data validation and schemas
click>=8.0                  # CLI framework
rich>=13.0                  # Terminal output formatting
opentelemetry-api>=1.20     # Tracing standard
opentelemetry-sdk>=1.20     # Tracing implementation
structlog>=24.0             # Structured logging
duckdb>=0.10.0              # Local analytics database for eval results
httpx>=0.27.0               # Async HTTP client
```

### 3.2 Python Version

Python 3.11+ (required by google-adk)

### 3.3 External Services Required

| Service | Purpose | Required |
|---------|---------|----------|
| Google AI Studio API key | Gemini models via ADK | Yes (free tier works) |
| OpenAI API key | GPT models via LiteLLM | Optional |
| Anthropic API key | Claude models via LiteLLM | Optional |
| Docker | SWE-bench evaluation containers | Yes |

---

## 4. Directory Structure

```
agentforge/
├── pyproject.toml                    # Project metadata and dependencies
├── README.md
├── .env.example                      # Template for API keys
│
├── configs/                          # All YAML configuration files
│   ├── default.yaml                  # Base configuration
│   ├── experiments/
│   │   ├── shared_memory_gpt4o.yaml
│   │   ├── isolated_memory_gpt4o.yaml
│   │   ├── shared_memory_sonnet.yaml
│   │   └── isolated_memory_sonnet.yaml
│   └── models/
│       ├── gemini.yaml
│       ├── gpt4o.yaml
│       ├── claude_sonnet.yaml
│       └── deepseek.yaml
│
├── src/
│   └── agentforge/
│       ├── __init__.py
│       ├── core/                     # Shared types and interfaces
│       │   ├── __init__.py
│       │   ├── types.py              # Pydantic models for all data types
│       │   ├── interfaces.py         # Abstract base classes (contracts)
│       │   ├── events.py             # Event system for observability
│       │   └── exceptions.py         # Custom exception hierarchy
│       │
│       ├── tasks/                    # Block 1: Task Loader
│       │   ├── __init__.py
│       │   ├── loader.py             # Load SWE-bench tasks
│       │   ├── sampler.py            # Task sampling strategies
│       │   └── formatter.py          # Format tasks for agent consumption
│       │
│       ├── models/                   # Block 2: Model Layer
│       │   ├── __init__.py
│       │   ├── provider.py           # LiteLLM wrapper with cost tracking
│       │   ├── registry.py           # Model registry from config
│       │   └── token_counter.py      # Token counting utilities
│       │
│       ├── orchestration/            # Block 3: Researcher's Domain
│       │   ├── __init__.py
│       │   ├── base.py               # Base classes for orchestration patterns
│       │   └── README.md             # "This is YOUR playground"
│       │
│       ├── memory/                   # Block 4: Memory Layer
│       │   ├── __init__.py
│       │   ├── shared.py             # Shared memory backend
│       │   ├── isolated.py           # Isolated memory backend
│       │   ├── hybrid.py             # Hybrid memory backend
│       │   └── base.py               # Abstract memory interface
│       │
│       ├── tools/                    # Block 5: Tool Layer
│       │   ├── __init__.py
│       │   ├── code_executor.py      # Sandboxed code execution
│       │   ├── file_ops.py           # File read/write/edit/search
│       │   ├── git_ops.py            # Git diff/commit/branch
│       │   ├── search.py             # Codebase search (grep, ast)
│       │   └── registry.py           # Tool registry for agent binding
│       │
│       ├── protocols/                # Block 6: Protocol Layer
│       │   ├── __init__.py
│       │   └── mcp_bridge.py         # MCP tool integration bridge
│       │
│       ├── evaluation/               # Block 7: Evaluation Harness
│       │   ├── __init__.py
│       │   ├── harness.py            # Main evaluation orchestrator
│       │   ├── metrics.py            # Metric computation
│       │   ├── swebench_runner.py    # SWE-bench integration
│       │   ├── judge.py              # LLM-as-judge scorer
│       │   └── reporter.py           # Results formatting and export
│       │
│       ├── observability/            # Block 8: Observability Layer
│       │   ├── __init__.py
│       │   ├── tracer.py             # OpenTelemetry tracing
│       │   ├── cost_tracker.py       # Token/cost aggregation
│       │   ├── event_logger.py       # Structured event logging
│       │   └── session_recorder.py   # Full session recording for replay
│       │
│       └── cli/                      # Block 9: CLI
│           ├── __init__.py
│           ├── main.py               # Click CLI entry point
│           ├── run.py                # `agentforge run` command
│           ├── eval.py               # `agentforge eval` command
│           └── report.py             # `agentforge report` command
│
├── experiments/                      # Experiment outputs (gitignored)
│   └── .gitkeep
│
└── tests/
    ├── test_tasks/
    ├── test_models/
    ├── test_memory/
    ├── test_tools/
    ├── test_evaluation/
    └── fixtures/
        └── sample_swebench_task.json
```

---

## 5. Block 1 — Task Loader

### 5.1 Purpose

Load, sample, and format SWE-bench tasks into a standardized structure that the orchestration layer consumes.

### 5.2 Data Models

```python
# src/agentforge/core/types.py

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class TaskDifficulty(str, Enum):
    EASY = "easy"           # <15 min estimated fix
    MEDIUM = "medium"       # 15-60 min estimated fix
    HARD = "hard"           # >60 min estimated fix

class Task(BaseModel):
    """A single coding task derived from SWE-bench."""
    instance_id: str                          # e.g., "astropy__astropy-14539"
    repo: str                                 # e.g., "astropy/astropy"
    base_commit: str                          # Commit hash for checkout
    problem_statement: str                    # The GitHub issue text
    hints_text: Optional[str] = None          # Comments before PR
    version: str                              # Package version
    test_patch: str                           # Tests that verify the fix
    gold_patch: Optional[str] = None          # Reference solution (hidden during runs)
    fail_to_pass: list[str] = Field(default_factory=list)  # Tests that must flip
    pass_to_pass: list[str] = Field(default_factory=list)  # Tests that must stay passing
    difficulty: Optional[TaskDifficulty] = None
    created_at: Optional[str] = None

class TaskBatch(BaseModel):
    """A collection of tasks for an experiment run."""
    tasks: list[Task]
    dataset_name: str                         # e.g., "SWE-bench_Lite"
    sample_size: int
    sampling_strategy: str                    # e.g., "random", "stratified", "difficulty_balanced"
    random_seed: int
```

### 5.3 Interface Contract

```python
# src/agentforge/core/interfaces.py

from abc import ABC, abstractmethod
from agentforge.core.types import Task, TaskBatch

class TaskLoaderInterface(ABC):
    @abstractmethod
    def load_dataset(self, dataset_name: str) -> list[Task]:
        """Load all tasks from a named SWE-bench dataset variant."""
        ...

    @abstractmethod
    def sample(
        self,
        tasks: list[Task],
        n: int,
        strategy: str = "random",
        seed: int = 42,
        difficulty: str | None = None,
    ) -> TaskBatch:
        """Sample n tasks using the specified strategy."""
        ...

    @abstractmethod
    def format_for_agent(self, task: Task) -> dict:
        """Format a task into the prompt structure agents receive."""
        ...
```

### 5.4 Implementation Requirements

1. **`loader.py`**: Use HuggingFace `datasets` library to load from `princeton-nlp/SWE-bench_Lite` (300 tasks, default), `SWE-bench/SWE-bench_Verified` (500 tasks), or `princeton-nlp/SWE-bench` (2,294 tasks). Map HuggingFace fields to the `Task` Pydantic model. Cache downloaded datasets locally in `~/.cache/agentforge/datasets/`.

2. **`sampler.py`**: Three sampling strategies:
   - `random`: Uniform random sample with seed
   - `stratified`: Maintain repo distribution proportionally
   - `difficulty_balanced`: Equal samples from easy/medium/hard (requires SWE-bench Verified annotations)

3. **`formatter.py`**: Convert `Task` into the agent-facing prompt. The format must include:
   - Repository name and version
   - Problem statement (issue text)
   - Hints (if available and configured)
   - Instructions for producing a git diff patch
   - Do NOT include gold_patch or test_patch (these are for evaluation only)

### 5.5 Configuration

```yaml
# In experiment YAML
tasks:
  dataset: "SWE-bench_Lite"        # Which SWE-bench variant
  sample_size: 50                   # How many tasks per run
  sampling_strategy: "random"       # random | stratified | difficulty_balanced
  seed: 42                          # Reproducibility
  include_hints: false              # Whether to provide issue comments
  difficulty_filter: null            # null | easy | medium | hard
```

---

## 6. Block 2 — Model Layer

### 6.1 Purpose

Provide a unified interface for calling any LLM (Gemini, GPT, Claude, DeepSeek, Grok, local via Ollama) through LiteLLM, integrated with Google ADK's `LiteLlm` wrapper. Track tokens and costs per call.

### 6.2 Data Models

```python
# src/agentforge/core/types.py (additions)

class ModelCall(BaseModel):
    """Record of a single LLM API call."""
    call_id: str                              # UUID
    model: str                                # e.g., "openai/gpt-4o"
    agent_name: str                           # Which agent made this call
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float                           # Estimated cost
    latency_ms: float                         # Time to first token
    timestamp: str                            # ISO 8601
    task_instance_id: str                     # Which task this belongs to
    experiment_id: str                        # Which experiment run

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str                                 # Human-readable name
    litellm_model_string: str                 # e.g., "openai/gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_base: str | None = None               # For proxy/Ollama
    api_key_env_var: str | None = None         # Env var name holding key
    supports_function_calling: bool = True
    supports_structured_output: bool = False
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
```

### 6.3 Interface Contract

```python
# src/agentforge/core/interfaces.py (additions)

class ModelProviderInterface(ABC):
    @abstractmethod
    def get_adk_model(self, model_name: str) -> "LiteLlm":
        """Return a Google ADK LiteLlm wrapper instance for the named model.

        Usage in orchestration:
            model = provider.get_adk_model("gpt4o")
            agent = LlmAgent(model=model, name="planner", ...)
        """
        ...

    @abstractmethod
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Return configuration for a named model."""
        ...

    @abstractmethod
    def list_available_models(self) -> list[str]:
        """List all configured model names."""
        ...
```

### 6.4 Implementation Requirements

1. **`provider.py`**: Wraps LiteLLM with `google.adk.models.lite_llm.LiteLlm`. Must:
   - Read model configs from YAML
   - Set API keys from environment variables
   - Instantiate `LiteLlm(model="<litellm_string>")` for each configured model
   - Hook into LiteLLM callbacks for token counting and cost tracking
   - Emit `ModelCall` events to the observability layer after each call

2. **`registry.py`**: Loads model configurations from `configs/models/*.yaml`. Validates that required API keys are present in environment.

3. **`token_counter.py`**: Utility to count tokens using `litellm.token_counter()` for pre-call estimation and post-call verification.

### 6.5 Configuration

```yaml
# configs/models/gpt4o.yaml
name: "gpt4o"
litellm_model_string: "openai/gpt-4o"
temperature: 0.0
max_tokens: 4096
api_key_env_var: "OPENAI_API_KEY"
supports_function_calling: true
supports_structured_output: true
cost_per_1k_input_tokens: 0.0025
cost_per_1k_output_tokens: 0.01
```

```yaml
# configs/models/gemini.yaml
name: "gemini_flash"
litellm_model_string: "gemini/gemini-2.5-flash"
temperature: 0.0
max_tokens: 8192
api_key_env_var: "GOOGLE_API_KEY"
supports_function_calling: true
supports_structured_output: true
cost_per_1k_input_tokens: 0.00015
cost_per_1k_output_tokens: 0.0006
```

```yaml
# configs/models/claude_sonnet.yaml
name: "claude_sonnet"
litellm_model_string: "anthropic/claude-sonnet-4-5-20250929"
temperature: 0.0
max_tokens: 4096
api_key_env_var: "ANTHROPIC_API_KEY"
supports_function_calling: true
supports_structured_output: false
cost_per_1k_input_tokens: 0.003
cost_per_1k_output_tokens: 0.015
```

```yaml
# configs/models/deepseek.yaml
name: "deepseek_coder"
litellm_model_string: "deepseek/deepseek-coder"
temperature: 0.0
max_tokens: 4096
api_key_env_var: "DEEPSEEK_API_KEY"
supports_function_calling: true
supports_structured_output: false
cost_per_1k_input_tokens: 0.00014
cost_per_1k_output_tokens: 0.00028
```

```yaml
# configs/models/ollama_local.yaml
name: "local_codellama"
litellm_model_string: "ollama/codellama:34b"
temperature: 0.0
max_tokens: 4096
api_base: "http://localhost:11434"
supports_function_calling: false
supports_structured_output: false
cost_per_1k_input_tokens: 0.0    # Free (local)
cost_per_1k_output_tokens: 0.0
```

### 6.6 ADK Integration Pattern

The orchestration layer (Vamshi's code) will use the model layer like this:

```python
from google.adk.agents import LlmAgent
from agentforge.models.provider import ModelProvider

provider = ModelProvider(config_dir="configs/models/")

planner = LlmAgent(
    name="planner",
    model=provider.get_adk_model("gpt4o"),   # Returns LiteLlm instance
    instruction="You are a senior software architect...",
    tools=[...],
)
```

---

## 7. Block 3 — Orchestration Layer (Interface Only)

This block is **Vamshi's manual implementation zone**. The PRD defines only the interfaces that the orchestration layer must implement and the ADK primitives available to it.

### 7.1 Base Classes for Experiments

```python
# src/agentforge/orchestration/base.py

from abc import ABC, abstractmethod
from agentforge.core.types import Task, ExperimentResult

class OrchestrationPattern(ABC):
    """Base class all orchestration patterns must implement.

    The coding agent builds this base class.
    Vamshi creates subclasses for each architecture being tested.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a unique name for this pattern (e.g., 'sequential_shared')."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this pattern."""
        ...

    @abstractmethod
    async def solve(self, task: Task) -> "AgentPatch":
        """Given a task, coordinate agents to produce a solution patch.

        This is where the multi-agent orchestration logic lives.
        Use Google ADK primitives (SequentialAgent, ParallelAgent, LoopAgent)
        to define the agent workflow.

        Returns:
            AgentPatch with the generated git diff
        """
        ...

    @abstractmethod
    def get_memory_type(self) -> str:
        """Return the memory architecture type: 'shared' | 'isolated' | 'hybrid'."""
        ...

class AgentPatch(BaseModel):
    """The output of an orchestration pattern solving a task."""
    instance_id: str
    model_name_or_path: str                   # For SWE-bench submission
    model_patch: str                          # The git diff content
    agent_trace: list[dict]                   # Sequence of agent actions
    total_tokens: int
    total_cost_usd: float
    solve_time_seconds: float
```

### 7.2 Available ADK Primitives

The researcher has access to these Google ADK constructs:

```python
from google.adk.agents import (
    LlmAgent,           # An agent backed by an LLM
    SequentialAgent,     # Runs sub-agents in order (A → B → C)
    ParallelAgent,       # Runs sub-agents simultaneously
    LoopAgent,           # Repeats sub-agents until escalate=True
    BaseAgent,           # For custom agent logic
)
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
```

### 7.3 Example: How Vamshi Uses the Framework

```python
# This is an EXAMPLE of what Vamshi writes manually.
# The coding agent does NOT build this — it builds everything around it.

from agentforge.orchestration.base import OrchestrationPattern, AgentPatch
from agentforge.models.provider import ModelProvider
from agentforge.memory.shared import SharedMemoryBackend
from agentforge.tools.registry import ToolRegistry

class SequentialSharedMemory(OrchestrationPattern):
    def name(self) -> str:
        return "sequential_shared"

    def description(self) -> str:
        return "Sequential 3-agent pipeline with shared memory, GPT-4o"

    def get_memory_type(self) -> str:
        return "shared"

    async def solve(self, task: Task) -> AgentPatch:
        provider = ModelProvider(config_dir="configs/models/")
        tools = ToolRegistry(workspace=f"/tmp/agentforge/{task.instance_id}")
        memory = SharedMemoryBackend()

        planner = LlmAgent(
            name="planner",
            model=provider.get_adk_model("gpt4o"),
            instruction="Analyze the issue and produce a fix plan...",
            output_key="app:plan",             # Writes to shared state
            tools=[tools.get("file_read"), tools.get("codebase_search")],
        )

        coder = LlmAgent(
            name="coder",
            model=provider.get_adk_model("gpt4o"),
            instruction="Read the plan from state and implement the fix...",
            output_key="app:patch",
            tools=[tools.get("file_write"), tools.get("file_edit")],
        )

        reviewer = LlmAgent(
            name="reviewer",
            model=provider.get_adk_model("gpt4o"),
            instruction="Review the patch for correctness and style...",
            output_key="app:review",
            tools=[tools.get("file_read"), tools.get("code_execute")],
        )

        pipeline = SequentialAgent(
            name="pipeline",
            sub_agents=[planner, coder, reviewer],
        )

        # ... run pipeline and collect results ...
```

---

## 8. Block 4 — Memory Layer

### 8.1 Purpose

Implement three memory architectures that the orchestration layer can plug into. These map to Google ADK's `session.state` with scoped prefixes.

### 8.2 Core Concepts

Google ADK state prefixes:
- **`app:`** — Persists for the session, shared across all agents. Resets between sessions.
- **`user:`** — Persists across sessions for a given user. Long-term memory.
- **`temp:`** — Temporary, cleared after each agent invocation. Private scratch space.
- **No prefix** — Same as `app:` behavior (persists for the session).

### 8.3 Interface Contract

```python
# src/agentforge/memory/base.py

from abc import ABC, abstractmethod

class MemoryBackendInterface(ABC):
    """Abstract interface for memory architectures.

    Each implementation configures how ADK session.state keys
    are scoped and accessed by agents.
    """

    @abstractmethod
    def get_state_prefix_config(self) -> dict:
        """Return the state prefix configuration for this memory type.

        Returns a dict mapping logical keys to ADK state prefixes.
        Example: {"plan": "app:plan", "code": "app:code"}  # shared
        Example: {"plan": "temp:planner_plan", ...}          # isolated
        """
        ...

    @abstractmethod
    def get_agent_instruction_addendum(self, agent_name: str) -> str:
        """Return additional instructions for an agent about how to
        read/write state under this memory architecture."""
        ...

    @abstractmethod
    def get_output_key(self, agent_name: str, data_key: str) -> str:
        """Return the correct ADK output_key for an agent writing a data key.

        Args:
            agent_name: The agent writing (e.g., "planner")
            data_key: What it's writing (e.g., "plan")

        Returns:
            Fully qualified state key (e.g., "app:plan" for shared,
            "temp:planner_plan" for isolated)
        """
        ...

    @abstractmethod
    def get_read_keys(self, agent_name: str) -> list[str]:
        """Return the state keys this agent is allowed to read.

        For shared: all keys.
        For isolated: only that agent's own keys.
        """
        ...

    @abstractmethod
    def get_memory_type(self) -> str:
        """Return 'shared', 'isolated', or 'hybrid'."""
        ...
```

### 8.4 Implementation: Shared Memory

```python
# src/agentforge/memory/shared.py

class SharedMemoryBackend(MemoryBackendInterface):
    """All agents read/write to a common state namespace.

    Uses ADK `app:` prefix so all agents in the session can
    read and write all keys.

    State flow example with 3 agents (Planner, Coder, Reviewer):
      - Planner writes: app:plan, app:analysis
      - Coder reads app:plan, writes: app:patch, app:files_modified
      - Reviewer reads app:plan + app:patch + app:files_modified,
        writes: app:review, app:approved
    """
```

Key behaviors:
- All output_keys use `app:` prefix
- All agents can read all `app:` keys
- Agent instructions include: "You can read all shared state keys"
- This is the "blackboard" architecture

### 8.5 Implementation: Isolated Memory

```python
# src/agentforge/memory/isolated.py

class IsolatedMemoryBackend(MemoryBackendInterface):
    """Each agent maintains private state. Inter-agent communication
    happens only through explicit handoff messages.

    Uses ADK `temp:` prefix scoped per agent so state is cleared
    after each invocation and not visible to other agents.

    State flow example:
      - Planner writes: temp:planner_plan (visible only to planner)
      - To pass info to Coder, the orchestrator must explicitly
        extract planner output and include it in coder's prompt
      - Coder writes: temp:coder_patch (visible only to coder)
    """
```

Key behaviors:
- All output_keys use `temp:<agent_name>_` prefix
- Each agent can only read its own `temp:` keys
- Inter-agent data transfer requires orchestration-layer intervention
- Agent instructions include: "You have access only to your own workspace"

### 8.6 Implementation: Hybrid Memory

```python
# src/agentforge/memory/hybrid.py

class HybridMemoryBackend(MemoryBackendInterface):
    """Agents have both private scratch space AND shared communication channels.

    - `app:shared_*` keys are readable/writable by all agents
    - `temp:<agent>_*` keys are private to each agent
    - `user:*` keys persist across sessions (long-term memory)

    This mirrors real team dynamics: shared docs + private notes.
    """
```

Key behaviors:
- Shared keys: `app:shared_plan`, `app:shared_patch`, etc.
- Private keys: `temp:planner_notes`, `temp:coder_scratch`, etc.
- Long-term keys: `user:learned_patterns`, `user:repo_knowledge`
- Agent instructions differentiate between reading shared vs private state

### 8.7 Configuration

```yaml
# In experiment YAML
memory:
  type: "shared"          # shared | isolated | hybrid
  # Hybrid-specific config:
  shared_keys:            # Only used when type=hybrid
    - "plan"
    - "patch"
    - "review"
  long_term_enabled: false  # Whether to use user: prefix keys
```

---

## 9. Block 5 — Tool Layer

### 9.1 Purpose

Provide coding tools (file I/O, code execution, search, git) as Google ADK `FunctionTool` instances that any agent can use.

### 9.2 Interface Contract

```python
# src/agentforge/core/interfaces.py (additions)

class ToolRegistryInterface(ABC):
    @abstractmethod
    def get(self, tool_name: str) -> callable:
        """Return an ADK-compatible tool function by name."""
        ...

    @abstractmethod
    def get_all(self) -> list[callable]:
        """Return all registered tool functions."""
        ...

    @abstractmethod
    def get_for_role(self, role: str) -> list[callable]:
        """Return tools appropriate for an agent role.
        Roles: 'planner', 'coder', 'reviewer', 'debugger', 'tester'
        """
        ...

    @abstractmethod
    def set_workspace(self, path: str) -> None:
        """Set the working directory for all file operations."""
        ...
```

### 9.3 Tool Specifications

Each tool must be a Python function with a docstring (ADK uses docstrings for tool descriptions). Tools must be sandboxed to the workspace directory.

#### 9.3.1 File Operations

```python
# src/agentforge/tools/file_ops.py

def file_read(file_path: str) -> dict:
    """Read the contents of a file in the repository.

    Args:
        file_path: Relative path from repository root (e.g., "src/utils.py")

    Returns:
        dict with keys:
          - status: "success" | "error"
          - content: The file content as a string
          - line_count: Number of lines
          - error_message: Error description if status is "error"
    """

def file_write(file_path: str, content: str) -> dict:
    """Write content to a file, creating it if it doesn't exist.

    Args:
        file_path: Relative path from repository root
        content: The full file content to write

    Returns:
        dict with status and bytes_written
    """

def file_edit(file_path: str, old_text: str, new_text: str) -> dict:
    """Replace a specific text fragment in a file. The old_text must
    appear exactly once in the file.

    Args:
        file_path: Relative path from repository root
        old_text: The exact text to find (must be unique in the file)
        new_text: The replacement text

    Returns:
        dict with status, lines_changed count
    """

def file_list(directory: str = ".") -> dict:
    """List files and directories in the given path.

    Args:
        directory: Relative path from repository root (default: root)

    Returns:
        dict with status and entries list (name, type, size)
    """
```

#### 9.3.2 Code Execution

```python
# src/agentforge/tools/code_executor.py

def execute_command(command: str, timeout: int = 30) -> dict:
    """Execute a shell command in the repository workspace.

    The command runs in an isolated subprocess with:
    - Working directory set to the repository root
    - Timeout of 30 seconds (configurable)
    - No network access
    - No write access outside the workspace

    Args:
        command: Shell command to execute (e.g., "python -m pytest tests/test_utils.py")
        timeout: Maximum execution time in seconds

    Returns:
        dict with keys:
          - status: "success" | "error" | "timeout"
          - stdout: Standard output
          - stderr: Standard error
          - return_code: Process exit code
          - execution_time_ms: How long it took
    """

def run_tests(test_path: str = "", test_names: list[str] = None) -> dict:
    """Run pytest on specified tests or the full test suite.

    Args:
        test_path: Path to test file or directory (default: run all)
        test_names: Specific test function names to run

    Returns:
        dict with:
          - status: "success" | "error"
          - passed: Number of tests passed
          - failed: Number of tests failed
          - errors: Number of test errors
          - output: Full pytest output
    """
```

#### 9.3.3 Codebase Search

```python
# src/agentforge/tools/search.py

def grep_search(pattern: str, file_glob: str = "**/*.py", max_results: int = 50) -> dict:
    """Search for a regex pattern across repository files.

    Args:
        pattern: Regex pattern to search for
        file_glob: Glob pattern to filter files (default: all Python files)
        max_results: Maximum matches to return

    Returns:
        dict with matches list (file_path, line_number, line_content)
    """

def find_definition(symbol_name: str) -> dict:
    """Find where a function, class, or variable is defined.

    Uses Python AST parsing for accurate results.

    Args:
        symbol_name: The name to find (e.g., "calculate_distance")

    Returns:
        dict with definitions list (file_path, line_number, type, context)
    """

def find_references(symbol_name: str) -> dict:
    """Find all usages of a symbol across the codebase.

    Args:
        symbol_name: The name to search for

    Returns:
        dict with references list (file_path, line_number, context)
    """
```

#### 9.3.4 Git Operations

```python
# src/agentforge/tools/git_ops.py

def git_diff() -> dict:
    """Show the current unstaged changes as a unified diff.

    Returns:
        dict with diff content string
    """

def git_log(n: int = 10) -> dict:
    """Show recent commit history.

    Args:
        n: Number of commits to show

    Returns:
        dict with commits list (hash, author, date, message)
    """

def git_show(file_path: str) -> dict:
    """Show the original version of a file (before any modifications).

    Args:
        file_path: Relative path to file

    Returns:
        dict with the original file content from HEAD
    """
```

### 9.4 Sandbox Requirements

- All file operations MUST be restricted to the workspace directory (prevent path traversal)
- Code execution MUST use `subprocess` with timeout
- No network access during code execution
- Workspace is a temporary directory per task, created by cloning the repo at `base_commit`

### 9.5 Tool-to-Role Mapping

```yaml
# Default tool assignments per role
roles:
  planner:
    - file_read
    - file_list
    - grep_search
    - find_definition
    - find_references
    - git_log
    - git_show
  coder:
    - file_read
    - file_write
    - file_edit
    - file_list
    - grep_search
    - find_definition
    - execute_command
  reviewer:
    - file_read
    - file_list
    - grep_search
    - git_diff
    - run_tests
    - execute_command
  debugger:
    - file_read
    - file_edit
    - grep_search
    - execute_command
    - run_tests
    - find_definition
    - find_references
  tester:
    - file_read
    - run_tests
    - execute_command
    - grep_search
```

### 9.6 ADK Integration Pattern

Tools become ADK agent tools like this:

```python
from agentforge.tools.registry import ToolRegistry

tools = ToolRegistry(workspace="/tmp/agentforge/astropy__astropy-14539")

planner = LlmAgent(
    name="planner",
    model=provider.get_adk_model("gpt4o"),
    instruction="...",
    tools=tools.get_for_role("planner"),  # Returns list of functions
)
```

---

## 10. Block 6 — Protocol Layer

### 10.1 Purpose

Bridge MCP (Model Context Protocol) tools into the ADK tool ecosystem, enabling external tool servers.

### 10.2 Implementation Requirements

1. **`mcp_bridge.py`**: Wrap MCP tool servers as ADK-compatible functions
   - Read MCP server configurations from YAML
   - Start/stop MCP servers as subprocess
   - Convert MCP tool schemas to ADK function signatures
   - Forward tool calls and return results

### 10.3 Configuration

```yaml
# In experiment YAML
protocols:
  mcp_servers: []                  # List of MCP server configs
  # Example:
  # - name: "filesystem"
  #   command: "npx"
  #   args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
```

### 10.4 Note

For the initial implementation, the Protocol Layer is minimal. The built-in tools (Block 5) handle all SWE-bench requirements. MCP becomes relevant when extending to custom tool servers later.

---

## 11. Block 7 — Evaluation Harness

### 11.1 Purpose

Run the complete evaluation pipeline: take agent-generated patches, evaluate them against SWE-bench, compute metrics, and produce reports.

### 11.2 Data Models

```python
# src/agentforge/core/types.py (additions)

class PatchResult(BaseModel):
    """Result of evaluating a single agent-generated patch."""
    instance_id: str
    resolved: bool                            # Did the patch fix the issue?
    resolution_status: str                    # RESOLVED_FULL | RESOLVED_PARTIAL | NOT_RESOLVED
    tests_passed: int
    tests_failed: int
    tests_errored: int
    fail_to_pass_results: dict[str, bool]     # Test name -> passed?
    pass_to_pass_results: dict[str, bool]     # Test name -> still passing?

class ExperimentResult(BaseModel):
    """Aggregated results for an entire experiment run."""
    experiment_id: str
    experiment_name: str                      # e.g., "sequential_shared_gpt4o"
    orchestration_pattern: str                # e.g., "sequential_shared"
    model_name: str
    memory_type: str                          # shared | isolated | hybrid
    timestamp: str

    # Task-level results
    patch_results: list[PatchResult]

    # Aggregate metrics
    resolution_rate: float                    # Fraction of tasks fully resolved
    partial_resolution_rate: float
    avg_tokens_per_task: float
    median_tokens_per_task: float
    total_tokens: int
    total_cost_usd: float
    avg_solve_time_seconds: float
    avg_agent_turns: float                    # Number of agent invocations

    # Per-agent token breakdown
    tokens_by_agent: dict[str, int]           # agent_name -> total tokens

    # Statistical measures
    pass_at_1: float                          # pass@1 score
    pass_at_3: float | None = None            # pass@3 if multiple runs
    confidence_interval_95: tuple[float, float] | None = None
```

### 11.3 Interface Contract

```python
# src/agentforge/core/interfaces.py (additions)

class EvaluationHarnessInterface(ABC):
    @abstractmethod
    async def evaluate_patch(self, patch: "AgentPatch", task: Task) -> PatchResult:
        """Evaluate a single agent-generated patch against SWE-bench.

        This runs the Docker-based SWE-bench evaluation harness.
        """
        ...

    @abstractmethod
    async def evaluate_batch(
        self,
        patches: list["AgentPatch"],
        tasks: list[Task],
    ) -> list[PatchResult]:
        """Evaluate a batch of patches in parallel."""
        ...

    @abstractmethod
    def compute_metrics(
        self,
        patch_results: list[PatchResult],
        model_calls: list[ModelCall],
        experiment_config: dict,
    ) -> ExperimentResult:
        """Compute aggregate metrics from patch-level results."""
        ...

    @abstractmethod
    def compare_experiments(
        self,
        results: list[ExperimentResult],
    ) -> "ComparisonReport":
        """Statistical comparison between experiment configurations.

        Computes:
        - Resolution rate differences with p-values
        - Token efficiency ratios
        - Cost comparisons
        - Effect sizes (Cohen's d)
        """
        ...
```

### 11.4 Implementation Requirements

1. **`swebench_runner.py`**: Interface with SWE-bench's Docker evaluation harness
   - Accept `AgentPatch` -> write to JSONL format expected by SWE-bench
   - Run `swebench.harness.run_evaluation` programmatically
   - Parse evaluation logs to extract per-test results
   - Map results back to `PatchResult` objects
   - Requirements: Docker must be running, at least 16GB RAM, 120GB disk

2. **`metrics.py`**: Compute all metrics defined in `ExperimentResult`
   - `resolution_rate`: Count of `RESOLVED_FULL` / total tasks
   - `pass_at_k`: Use the unbiased estimator: `1 - C(n-c, k) / C(n, k)` where n=total, c=correct, k=attempts
   - `confidence_interval_95`: Bootstrap confidence interval for resolution rate
   - Token breakdowns from `ModelCall` records

3. **`judge.py`**: LLM-as-judge scoring for patch quality (beyond binary pass/fail)
   - Given: task description, gold patch, agent patch
   - Score on dimensions: correctness (0-5), completeness (0-5), code quality (0-5), efficiency (0-5)
   - Use a separate LLM call (configurable model) for judging
   - Average 3 judge calls for reliability

4. **`reporter.py`**: Generate reports in multiple formats
   - Markdown summary table
   - JSON for programmatic consumption
   - DuckDB insert for longitudinal analysis
   - CSV export for statistical tools (R, SPSS)

### 11.5 Configuration

```yaml
# In experiment YAML
evaluation:
  swebench_max_workers: 4          # Parallel Docker containers
  swebench_cache_level: "env"      # none | base | env | instance
  llm_judge_enabled: true
  llm_judge_model: "gemini_flash"  # Cheap model for judging
  llm_judge_runs: 3                # Evaluations per patch for reliability
  pass_at_k: [1, 3]                # Which pass@k values to compute
  output_formats: ["markdown", "json", "csv"]
```

---

## 12. Block 8 — Observability Layer

### 12.1 Purpose

Record everything that happens during an experiment run so results are explainable and reproducible.

### 12.2 Data Models

```python
# src/agentforge/core/events.py

from pydantic import BaseModel
from enum import Enum

class EventType(str, Enum):
    EXPERIMENT_START = "experiment.start"
    EXPERIMENT_END = "experiment.end"
    TASK_START = "task.start"
    TASK_END = "task.end"
    AGENT_INVOCATION = "agent.invocation"
    AGENT_RESPONSE = "agent.response"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    MODEL_CALL = "model.call"
    MODEL_RESPONSE = "model.response"
    STATE_UPDATE = "state.update"
    ERROR = "error"

class Event(BaseModel):
    """Immutable event record."""
    event_id: str                             # UUID
    event_type: EventType
    timestamp: str                            # ISO 8601
    experiment_id: str
    task_instance_id: str | None = None
    agent_name: str | None = None
    data: dict                                # Event-specific payload
    parent_event_id: str | None = None        # For hierarchical tracing
```

### 12.3 Implementation Requirements

1. **`event_logger.py`**: Central event bus
   - Accept `Event` objects and write them to:
     - Structured log file (JSONL format, one event per line)
     - In-memory buffer for real-time queries
   - Thread-safe for concurrent agent execution
   - File path: `experiments/<experiment_id>/events.jsonl`

2. **`tracer.py`**: OpenTelemetry integration
   - Create spans for: experiment -> task -> agent invocation -> tool call
   - Attach token counts and costs as span attributes
   - Export to OTLP endpoint if configured, otherwise local Jaeger or console

3. **`cost_tracker.py`**: Real-time cost aggregation
   - Subscribe to `MODEL_CALL` events
   - Maintain running totals: per-agent, per-task, per-experiment
   - Expose `get_cost_summary()` method
   - Emit warnings if cost exceeds configurable threshold

4. **`session_recorder.py`**: Full session recording
   - Record the complete ADK session (all events from `runner.run_async`)
   - Store as a replayable JSON file
   - Enable session rewind using ADK's rewind capability
   - Path: `experiments/<experiment_id>/sessions/<task_id>.json`

### 12.4 Configuration

```yaml
# In experiment YAML
observability:
  log_level: "INFO"                 # DEBUG | INFO | WARNING | ERROR
  events_file: true                 # Write events.jsonl
  session_recording: true           # Record full sessions for replay
  cost_warning_threshold_usd: 10.0  # Warn if experiment exceeds this
  opentelemetry:
    enabled: false                  # Enable OTLP tracing
    endpoint: "http://localhost:4317"
```

---

## 13. Block 9 — Configuration & CLI

### 13.1 Master Configuration Schema

Every experiment is defined by a single YAML file that composes all blocks.

```yaml
# configs/experiments/shared_memory_gpt4o.yaml

experiment:
  name: "sequential_shared_gpt4o"
  description: "Sequential 3-agent pipeline with shared memory, GPT-4o"
  tags: ["shared", "sequential", "gpt4o"]
  repeat_runs: 3                    # Run each task N times for pass@k

tasks:
  dataset: "SWE-bench_Lite"
  sample_size: 50
  sampling_strategy: "random"
  seed: 42
  include_hints: false
  difficulty_filter: null

model:
  name: "gpt4o"                     # References configs/models/gpt4o.yaml

memory:
  type: "shared"

orchestration:
  pattern: "sequential_shared"      # Maps to a registered OrchestrationPattern
  agents:                           # Agent definitions (used by orchestration)
    - name: "planner"
      role: "planner"
      instruction_template: "planner_v1"
    - name: "coder"
      role: "coder"
      instruction_template: "coder_v1"
    - name: "reviewer"
      role: "reviewer"
      instruction_template: "reviewer_v1"

evaluation:
  swebench_max_workers: 4
  swebench_cache_level: "env"
  llm_judge_enabled: true
  llm_judge_model: "gemini_flash"
  llm_judge_runs: 3
  pass_at_k: [1, 3]
  output_formats: ["markdown", "json", "csv"]

observability:
  log_level: "INFO"
  events_file: true
  session_recording: true
  cost_warning_threshold_usd: 10.0
  opentelemetry:
    enabled: false

workspace:
  base_dir: "/tmp/agentforge"       # Where repos are cloned
  cleanup_after_task: true          # Delete workspace after each task
```

### 13.2 CLI Commands

```bash
# Run a complete experiment
agentforge run --config configs/experiments/shared_memory_gpt4o.yaml

# Run a single task for debugging
agentforge run --config configs/experiments/shared_memory_gpt4o.yaml \
               --task-id "astropy__astropy-14539" \
               --verbose

# Evaluate existing patches (re-run SWE-bench without re-generating)
agentforge eval --experiment-dir experiments/2026-02-18_sequential_shared_gpt4o/

# Generate comparison report between experiments
agentforge report compare \
    experiments/2026-02-18_sequential_shared_gpt4o/ \
    experiments/2026-02-18_sequential_isolated_gpt4o/ \
    --output report.md

# List available orchestration patterns
agentforge patterns list

# Validate a config file
agentforge validate --config configs/experiments/shared_memory_gpt4o.yaml

# Show cost estimate for an experiment (dry run)
agentforge estimate --config configs/experiments/shared_memory_gpt4o.yaml
```

### 13.3 CLI Implementation Requirements

Use the `click` library. Each command maps to a module in `src/agentforge/cli/`.

**`run.py`** — The main execution flow:

```
1. Load and validate experiment config
2. Initialize model provider (verify API keys)
3. Load and sample tasks
4. Set up observability (event logger, cost tracker)
5. For each task:
   a. Clone repo at base_commit into workspace
   b. Initialize tool registry with workspace path
   c. Initialize memory backend
   d. Instantiate orchestration pattern
   e. Call pattern.solve(task) -> AgentPatch
   f. Save patch to experiments/<id>/patches/<task_id>.json
   g. Record session
   h. Clean up workspace (if configured)
6. Run SWE-bench evaluation on all patches
7. Compute metrics
8. Generate reports
9. Print summary to console
```

### 13.4 Experiment Output Structure

```
experiments/
└── 2026-02-18T14-30-00_sequential_shared_gpt4o/
    ├── config.yaml                  # Copy of experiment config (reproducibility)
    ├── events.jsonl                 # All events
    ├── patches/
    │   ├── astropy__astropy-14539.json
    │   ├── django__django-16379.json
    │   └── ...
    ├── sessions/
    │   ├── astropy__astropy-14539.json
    │   └── ...
    ├── evaluation/
    │   ├── predictions.jsonl        # SWE-bench format predictions
    │   ├── results.json             # Per-task evaluation results
    │   └── swebench_logs/           # Raw Docker evaluation logs
    ├── results/
    │   ├── metrics.json             # Computed aggregate metrics
    │   ├── summary.md               # Human-readable summary
    │   ├── metrics.csv              # For external analysis
    │   └── cost_breakdown.json      # Detailed cost analysis
    └── analytics.duckdb             # DuckDB with all structured data
```

---

## 14. Workspace Management

### 14.1 Purpose

For each task, create an isolated workspace by cloning the target repository at the correct commit.

### 14.2 Implementation Requirements

Create a `WorkspaceManager` class (in `src/agentforge/core/workspace.py`):

```python
class WorkspaceManager:
    def __init__(self, base_dir: str = "/tmp/agentforge"):
        ...

    async def create_workspace(self, task: Task) -> str:
        """Clone the repo at base_commit into a temp directory.

        Steps:
        1. Create directory: base_dir/<instance_id>/
        2. git clone the repo (shallow clone for speed)
        3. git checkout <base_commit>
        4. Return the workspace path

        Returns:
            Absolute path to the workspace root
        """

    async def get_generated_diff(self, workspace_path: str) -> str:
        """Generate a git diff of all changes made by agents.

        This is the model_patch that gets submitted to SWE-bench.
        """

    async def cleanup_workspace(self, workspace_path: str) -> None:
        """Remove the workspace directory."""
```

### 14.3 Important Notes

- Use shallow clone (`--depth 1`) then fetch the specific commit for speed
- The SWE-bench repos are: astropy, django, flask, matplotlib, pylint, pytest, requests, scikit-learn, seaborn, sphinx, sympy, xarray
- Some repos are large (django ~300MB); cache bare clones in `~/.cache/agentforge/repos/`

---

## 15. Error Handling

### 15.1 Exception Hierarchy

```python
# src/agentforge/core/exceptions.py

class AgentForgeError(Exception):
    """Base exception for all AgentForge errors."""

class ConfigurationError(AgentForgeError):
    """Invalid configuration file or missing required values."""

class ModelError(AgentForgeError):
    """LLM API call failure (rate limit, auth, timeout)."""

class ToolExecutionError(AgentForgeError):
    """Tool failed during execution (file not found, timeout, etc.)."""

class WorkspaceError(AgentForgeError):
    """Git clone, checkout, or workspace setup failure."""

class EvaluationError(AgentForgeError):
    """SWE-bench Docker evaluation failure."""

class BudgetExceededError(AgentForgeError):
    """Experiment cost exceeded the configured threshold."""
```

### 15.2 Error Recovery Strategy

- **Model API failures**: Retry 3 times with exponential backoff (1s, 4s, 16s). After 3 failures, skip the task and log it as `NOT_RESOLVED` with error metadata.
- **Tool execution timeout**: Kill the process, return error dict, let the agent decide next action.
- **Workspace failures**: Log error, skip the task, continue with remaining tasks.
- **Budget exceeded**: Gracefully stop the experiment, evaluate completed patches, generate partial report.

---

## 16. Testing Requirements

### 16.1 Unit Tests

Each block must have unit tests covering:

- **Task Loader**: Loading from a mock dataset, sampling strategies, formatting
- **Model Layer**: Config loading, LiteLlm wrapper creation (mock API calls)
- **Memory Layer**: State prefix generation for each memory type, read/write key scoping
- **Tool Layer**: File operations (mock filesystem), search (mock AST), sandbox path validation
- **Evaluation**: Metric computation from mock results, pass@k calculation, report generation
- **Observability**: Event emission, cost aggregation, JSONL formatting
- **Config**: YAML parsing, validation, merging defaults

### 16.2 Integration Tests

- **End-to-end dry run**: Load config -> load tasks -> create workspace -> run tools -> generate diff -> evaluate (using a minimal fixture task, not full SWE-bench)
- **Model integration**: Verify LiteLlm wrapper works with at least one real API (use cheapest model)
- **SWE-bench integration**: Run evaluation on a single gold patch to verify Docker harness works

### 16.3 Test Fixture

```json
{
  "instance_id": "test__test-001",
  "repo": "test/test-repo",
  "base_commit": "abc123",
  "problem_statement": "The calculate_sum function returns incorrect results for negative numbers.",
  "version": "1.0.0",
  "test_patch": "diff --git a/tests/test_math.py ...",
  "fail_to_pass": ["tests/test_math.py::test_negative_sum"],
  "pass_to_pass": ["tests/test_math.py::test_positive_sum"]
}
```

---

## 17. Build Order

Implement blocks in this order, as each depends on the previous:

```
Phase 1: Foundation (no external dependencies)
  1. core/types.py          — All Pydantic models
  2. core/interfaces.py     — All abstract base classes
  3. core/exceptions.py     — Exception hierarchy
  4. core/events.py         — Event system types

Phase 2: Infrastructure
  5. configs/               — YAML config schemas and defaults
  6. core/workspace.py      — Workspace manager
  7. observability/         — Event logger, cost tracker

Phase 3: Data & Model
  8. tasks/                 — Task loader, sampler, formatter
  9. models/                — Model provider, registry

Phase 4: Capabilities
  10. tools/                — All tool implementations
  11. memory/               — All memory backends
  12. protocols/            — MCP bridge (minimal)

Phase 5: Evaluation
  13. evaluation/           — Harness, metrics, judge, reporter
  14. orchestration/base.py — Base class only (Vamshi fills in)

Phase 6: CLI
  15. cli/                  — All CLI commands
```

---

## 18. Acceptance Criteria

The system is ready for orchestration experimentation when:

1. `agentforge validate --config configs/experiments/shared_memory_gpt4o.yaml` passes without errors
2. `agentforge estimate --config configs/experiments/shared_memory_gpt4o.yaml` shows expected cost
3. A sample orchestration pattern (provided as reference in `orchestration/base.py` docstring) can:
   - Load a task
   - Create a workspace
   - Use tools to read/modify files
   - Generate a diff
   - Pass the diff to SWE-bench evaluation
   - Produce a metrics report
4. All three memory backends (shared, isolated, hybrid) produce different state key configurations when given the same agent list
5. The model layer can swap between at least 2 different providers (e.g., Gemini and GPT-4o) by changing only the config YAML
6. The observability layer records events, tracks costs, and produces a readable cost summary
7. All unit tests pass
8. `agentforge run --config ... --task-id <single-task>` completes end-to-end

---

## 19. Glossary

| Term | Definition |
|------|-----------|
| **Shared memory** | Architecture where all agents read/write a common state store (ADK `app:` prefix) |
| **Isolated memory** | Architecture where each agent has private state (ADK `temp:<agent>_` prefix) |
| **Orchestration pattern** | A specific multi-agent topology (e.g., sequential pipeline, parallel fan-out) |
| **AgentPatch** | The git diff generated by the multi-agent system to fix an issue |
| **pass@k** | Probability of at least one correct solution in k attempts |
| **SWE-bench** | Benchmark of real GitHub issues for evaluating coding agents |
| **ADK** | Google Agent Development Kit — the orchestration framework |
| **LiteLLM** | Library providing unified API for 100+ LLM providers |
| **MCP** | Model Context Protocol — standard for tool integration |
| **A2A** | Agent-to-Agent Protocol — standard for inter-agent communication |

---

## 20. References

- Google ADK Documentation: https://google.github.io/adk-docs/
- Google ADK Python SDK: https://github.com/google/adk-python
- ADK + LiteLLM Integration: https://google.github.io/adk-docs/agents/models/litellm/
- LiteLLM Documentation: https://docs.litellm.ai/
- SWE-bench: https://www.swebench.com/SWE-bench/
- SWE-bench GitHub: https://github.com/SWE-bench/SWE-bench
- SWE-bench Datasets: https://www.swebench.com/SWE-bench/guides/datasets/
- SWE-bench Evaluation: https://www.swebench.com/SWE-bench/guides/evaluation/
- A2A Protocol: https://github.com/a2aproject/A2A
- MCP Specification: https://modelcontextprotocol.io/

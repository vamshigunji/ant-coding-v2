# Documentation

## Getting Started

| Document | Description |
|----------|-------------|
| [README](../README.md) | Project overview, quickstart, architecture diagram |
| [Developer Guide](guides/developer-guide.md) | Extending the framework with new patterns, tools, and configs |

## Specifications

| Document | Description |
|----------|-------------|
| [PRD](spec/prd.md) | Full product requirements — all 8 framework layers |
| [PRD+](spec/prd-plus.md) | Extended metrics, 4-tier evaluation, failure classification |
| [Success Metrics](spec/success-metrics.md) | The 11 metrics across 4 tiers and why they matter |

## Architecture

| Document | Description |
|----------|-------------|
| [System Overview](architecture/system-overview.md) | High-level layer diagram and responsibilities |
| [Layer Interactions](architecture/layer-interactions.md) | Data flow between layers during experiment execution |
| [Memory Architecture](architecture/memory-architecture.md) | Shared, isolated, and hybrid memory modes |
| [Experiment Lifecycle](architecture/experiment-lifecycle.md) | End-to-end phases: input, init, execution, collection, evaluation |

## Guides

| Document | Description |
|----------|-------------|
| [Developer Guide](guides/developer-guide.md) | How to create patterns, add tools, configure experiments |
| [Experimentation Playbook](guides/experimentation-playbook.md) | Research methodology — experiment ladders, one variable at a time |
| [Contributing](guides/contributing.md) | Git workflow, branching, commit conventions |

## Project Resources

| Resource | Location | Description |
|----------|----------|-------------|
| Sprint Tracker | [`.agent/sprint.yml`](../.agent/sprint.yml) | Development progress — source of truth |
| Epic Files | [`stories_and_epics/`](../stories_and_epics/) | Acceptance criteria (Given/When/Then) per story |
| Experiment Configs | [`configs/experiments/`](../configs/experiments/) | YAML templates for experiments |
| Model Configs | [`configs/models/`](../configs/models/) | LLM provider configurations |
| Memory Configs | [`configs/memory/`](../configs/memory/) | Memory mode configurations |
| Experiment Registry | [`experiments/registry.yml`](../experiments/registry.yml) | Experiment lineage tracking |

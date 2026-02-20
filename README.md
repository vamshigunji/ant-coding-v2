# ant-coding

A framework for benchmarking agentic coding architectures. Compare single-agent vs. multi-agent approaches on software engineering tasks with rigorous statistical evaluation.

## Overview

ant-coding provides a complete pipeline for running controlled experiments on agentic coding systems:

1. **Configure** experiments via YAML (model, memory mode, orchestration pattern, tasks)
2. **Run** tasks through pluggable orchestration patterns (single-agent, sequential, parallel, loop)
3. **Evaluate** with 4-tier PRD+ metrics (primary, efficiency, quality, robustness)
4. **Compare** experiments with paired statistical tests (McNemar's, Wilcoxon, bootstrap CI)
5. **Report** results in Markdown, JSON, or CSV

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│  YAML Config │───►│  Experiment  │───►│  Orchestration   │
│              │    │    Runner    │    │    Pattern        │
└─────────────┘    └──────┬───────┘    │  (single-agent,  │
                          │            │   sequential,     │
                          │            │   parallel, loop) │
                          │            └────────┬─────────┘
                          │                     │
                   ┌──────▼───────┐    ┌────────▼─────────┐
                   │   Evaluation  │    │    Tool Layer     │
                   │   Harness     │    │  (code exec,     │
                   │  (4-tier      │    │   file ops, git,  │
                   │   metrics)    │    │   search)         │
                   └──────┬───────┘    └──────────────────┘
                          │
                   ┌──────▼───────┐    ┌──────────────────┐
                   │  Statistical  │    │  Memory Layer     │
                   │  Comparison   │    │  (shared,         │
                   └──────┬───────┘    │   isolated,       │
                          │            │   hybrid)          │
                   ┌──────▼───────┐    └──────────────────┘
                   │    Reports    │
                   │  (MD/JSON/CSV)│
                   └──────────────┘
```

### Layer Modules

| Module | Purpose |
|--------|---------|
| `core/` | Config loading (Pydantic), type definitions, experiment registry |
| `models/` | LiteLLM model abstraction with token budget enforcement |
| `memory/` | Shared, isolated, and hybrid memory modes |
| `tasks/` | Task loading (YAML, SWE-bench), workspace management |
| `tools/` | Code execution, file ops, git ops, codebase search |
| `orchestration/` | Plugin interface + reference patterns |
| `runner/` | Experiment runner, result output |
| `eval/` | 4-tier metrics, LLM judge, failure classifier, comparison, reports |
| `observability/` | Event logging (JSONL), session replay |
| `protocols/` | MCP tool server, A2A agent registration |

## Quickstart

### Install

```bash
pip install -e ".[dev]"
```

### Configure

Create an experiment config YAML:

```yaml
# configs/experiments/my-experiment.yaml
name: single-agent-baseline
model:
  name: gpt-4
  litellm_model: gpt-4
  api_key_env: OPENAI_API_KEY
memory:
  mode: shared
tasks:
  source: custom
  subset: tasks/custom/my-tasks.yaml
execution:
  max_workers: 1
  timeout_seconds: 1800
output:
  dir: results
```

### Run

```bash
python scripts/run_experiment.py configs/experiments/my-experiment.yaml
```

### Compare Two Experiments

```bash
python scripts/compare_results.py results/exp-a results/exp-b
```

## 4-Tier Metrics Framework

| Tier | Metrics | Purpose |
|------|---------|---------|
| **1 - Primary** | pass_rate, cost_per_resolution | Core success and cost |
| **2 - Efficiency** | useful_token_ratio, overhead_ratio, tokens_per_resolution | Token and cost efficiency |
| **3 - Quality** | avg_patch_quality, avg_patch_size_ratio | Code quality via LLM judge |
| **4 - Robustness** | resolution_variance_cv, error_recovery_rate, failure_categories | Reliability and consistency |

## Running Tests

```bash
# Full suite
pytest tests/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=ant_coding --cov-report=term-missing
```

## Project Structure

```
ant-coding/
├── configs/              # Experiment and model configs (YAML)
├── docs/                 # PRD, architecture docs, experimentation playbook
├── experiments/          # Experiment registry (registry.yml)
├── scripts/              # CLI scripts (run_experiment, compare_results)
├── src/ant_coding/       # Source code
│   ├── cli/              # CLI entry point
│   ├── core/             # Config, types, experiment registry
│   ├── eval/             # Metrics, judge, classifier, comparison, reports
│   ├── memory/           # Memory manager (shared/isolated/hybrid)
│   ├── models/           # LiteLLM model provider
│   ├── observability/    # Event logger, session replay
│   ├── orchestration/    # Pattern base class, registry, reference impls
│   ├── protocols/        # MCP tool server, A2A agent server
│   ├── runner/           # Experiment runner, result writer
│   ├── tasks/            # Task loading, workspace setup
│   └── tools/            # Code exec, file ops, git ops, search
├── stories_and_epics/    # Sprint planning docs
└── tests/                # Unit and integration tests
    └── integration/      # Full pipeline and edge case tests
```

## Documentation

- [Developer Guide](docs/developer-guide.md) — Extending the framework
- [PRD](docs/prd.md) — Product requirements
- [PRD+](docs/prd-plus.md) — Extended metrics and evaluation
- [Experimentation Playbook](docs/experimentation-playbook.md) — Running experiments
- [Success Metrics](docs/success-metrics.md) — What we measure and why

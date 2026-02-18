# Layer Interactions — ant-coding

## How Layers Communicate

```mermaid
sequenceDiagram
    participant R as ExperimentRunner
    participant T as TaskLoader
    participant M as ModelProvider
    participant O as OrchestrationPattern
    participant Mem as MemoryManager
    participant Tools as ToolRegistry
    participant E as EventLogger
    participant Eval as EvalHarness

    R->>T: load_tasks(config)
    T-->>R: list[Task]

    R->>M: init(model_config)
    R->>Mem: init(memory_mode)
    R->>Tools: init(workspace_dir)
    R->>E: init(experiment_id)

    loop For each Task
        R->>T: workspace.setup()
        T-->>R: workspace_dir

        R->>E: log(TASK_START)
        R->>O: solve(task, model, memory, tools, workspace_dir)

        Note over O: Orchestration owns the agent loop.<br/>Everything below happens inside solve().

        O->>E: log(AGENT_START, "Planner")
        O->>M: complete(messages, tools)
        M->>E: log(LLM_CALL, tokens, cost)
        M-->>O: response

        O->>Mem: write("planner", "plan", plan_text)
        Mem->>E: log(MEMORY_WRITE)

        O->>E: log(AGENT_START, "Coder")
        O->>Mem: read("coder", "plan")
        Mem->>E: log(MEMORY_READ)
        Mem-->>O: plan_text (or None if isolated)

        O->>M: complete(messages, tools)
        M->>E: log(LLM_CALL, tokens, cost)
        M-->>O: response with tool calls

        O->>Tools: file_ops.edit_file(path, old, new)
        Tools->>E: log(TOOL_CALL)
        Tools-->>O: success

        O->>Tools: code_executor.run_command("pytest")
        Tools->>E: log(TOOL_CALL)
        Tools-->>O: test_output

        O-->>R: TaskResult

        R->>E: log(TASK_END)
        R->>Eval: evaluate(task, result)
        Eval-->>R: scores
    end

    R->>Eval: aggregate_metrics()
    R->>Eval: generate_report()
```

## Dependency Matrix

Which layer calls which. Read as "Row calls Column".

```mermaid
graph LR
    subgraph "Dependency Direction"
        direction LR
        A["ExperimentRunner"] --> B["All Layers"]
        C["Orchestration"] --> D["Models"]
        C --> E["Memory"]
        C --> F["Tools"]
        G["Tools"] --> H["TaskWorkspace"]
        I["Eval"] --> J["Models (for LLM Judge)"]
        K["Observability"] --> L["Receives from all layers"]
    end
```

| Caller ↓ / Callee → | Tasks | Models | Orchestration | Memory | Tools | Eval | Observability |
|----------------------|-------|--------|---------------|--------|-------|------|---------------|
| **ExperimentRunner** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Orchestration** | — | ✅ | — | ✅ | ✅ | — | ✅ |
| **Eval** | ✅ | ✅ | — | — | — | — | — |
| **Tools** | ✅* | — | — | — | — | — | ✅ |
| **Memory** | — | — | — | — | — | — | ✅ |

*Tools depend on TaskWorkspace for filesystem scoping.

## Key Design Rule

**Orchestration never imports from Eval, and Eval never imports from Orchestration.** The ExperimentRunner is the only component that sees both. This ensures orchestration patterns can't game the evaluation.
